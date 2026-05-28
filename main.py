import os
import time
import csv
from datetime import datetime

from absl import app, flags
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax
import gzip

import tqdm
from src.agents import hiql as learner
from src import viz_utils
from src.gc_dataset import GCSDataset

from jaxrl_m.wandb import setup_wandb, default_wandb_config
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor

from ml_collections import config_flags
import pickle

from src.utils import record_video, CsvLogger

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
flags.DEFINE_string('save_dir', f'experiment_output/', '')
flags.DEFINE_string('run_group', 'Debug', '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('eval_episodes', 50, '')
flags.DEFINE_integer('num_video_episodes', 2, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('pretrain_steps', 0, '')

flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('use_rep', 0, '')
flags.DEFINE_integer('rep_dim', None, '')
flags.DEFINE_enum('rep_type', 'state', ['state', 'diff', 'concat'], '')
flags.DEFINE_integer('policy_train_rep', 0, '')
flags.DEFINE_integer('use_waypoints', 0, '')
flags.DEFINE_integer('way_steps', 1, '')

flags.DEFINE_float('pretrain_expectile', 0.7, '')
flags.DEFINE_float('p_randomgoal', 0.3, '')
flags.DEFINE_float('p_trajgoal', 0.5, '')
flags.DEFINE_float('p_currgoal', 0.2, '')
flags.DEFINE_float('high_p_randomgoal', 0., '')
flags.DEFINE_integer('geom_sample', 1, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('temperature', 1, '')
flags.DEFINE_float('high_temperature', 1, '')
flags.DEFINE_integer('use_reachability', 0, '')
flags.DEFINE_integer('reachability_horizon', 25, '')
flags.DEFINE_float('reachability_anchor_eps', 0.75, '')
flags.DEFINE_float('reachability_far_eps', 5.0, '')
flags.DEFINE_integer('reachability_negative_horizon', 200, '')
flags.DEFINE_float('reachability_hard_negative_prob', 0.75, '')
flags.DEFINE_integer('reachability_anchor_max_candidates', 64, '')
flags.DEFINE_integer('reachability_anchor_sample_attempts', 16, '')
flags.DEFINE_float('reachability_loss_weight', 1.0, '')
flags.DEFINE_float('reachability_threshold', 0.5, '')
flags.DEFINE_integer('reachability_resample_attempts', 10, '')
flags.DEFINE_integer('reachability_filter_policy', 0, '')
flags.DEFINE_integer('reachability_pretrain_steps', 0, '')
flags.DEFINE_integer('freeze_reachability_after_pretrain', 0, '')
flags.DEFINE_integer('reachability_start_step', -1, '')
flags.DEFINE_float('reachability_start_frac', 0.0, '')
flags.DEFINE_integer('use_reachability_mod_adv', 0, '')
flags.DEFINE_float('reachability_alpha', 0.1, '')
flags.DEFINE_integer('reachability_alpha_warmup_steps', 0, '')
flags.DEFINE_float('reachability_alpha_max', -1.0, '')
flags.DEFINE_float('reachability_alpha_min', -1.0, '')
flags.DEFINE_integer('reachability_alpha_decay_steps', 0, '')

flags.DEFINE_integer('visual', 0, '')
flags.DEFINE_string('encoder', 'impala', '')

flags.DEFINE_string('algo_name', None, '')  # Not used, only for logging

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'hiql',
    'group': 'Debug',
    'name': '{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)

gcdataset_config = GCSDataset.get_default_config()
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)


@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g):
        return agent.network(s, g, info=True, method='value')

    s = batch['observations']
    g = batch['goals']

    info = get_info(s, g)

    stats = {}

    stats.update({
        'v': info['v'].mean(),
    })

    return stats


@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2


def get_v(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }


def _prob_summary(prefix, probs, threshold):
    if probs.size == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_p10': np.nan,
            f'{prefix}_p50': np.nan,
            f'{prefix}_p90': np.nan,
            f'{prefix}_below_threshold_frac': np.nan,
            f'{prefix}_count': 0,
        }
    return {
        f'{prefix}_mean': probs.mean(),
        f'{prefix}_min': probs.min(),
        f'{prefix}_p10': np.percentile(probs, 10),
        f'{prefix}_p50': np.percentile(probs, 50),
        f'{prefix}_p90': np.percentile(probs, 90),
        f'{prefix}_below_threshold_frac': (probs < threshold).mean(),
        f'{prefix}_count': probs.shape[0],
    }


def _safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def get_eval_reachability_stats(agent, trajs, use_rep=False, reachability_alpha_effective=0.0):
    valid_trajs = [
        (traj_id, traj, np.sum(traj['reward']) > 0)
        for traj_id, traj in enumerate(trajs)
        if 'subgoal' in traj and len(traj['subgoal']) > 0 and 'episode_goal' in traj
    ]
    if not valid_trajs:
        return {}, []

    observations = np.concatenate([np.asarray(traj['observation']) for _, traj, _ in valid_trajs], axis=0)
    next_observations = np.concatenate([np.asarray(traj['next_observation']) for _, traj, _ in valid_trajs], axis=0)
    subgoals = np.concatenate([np.asarray(traj['subgoal']) for _, traj, _ in valid_trajs], axis=0)
    episode_goals = np.concatenate([np.asarray(traj['episode_goal']) for _, traj, _ in valid_trajs], axis=0)
    horizon = int(agent.config['way_steps'])
    future_observations = []
    for _, traj, _ in valid_trajs:
        traj_observations = np.asarray(traj['observation'])
        traj_next_observations = np.asarray(traj['next_observation'])
        traj_len = len(traj_observations)
        for t in range(traj_len):
            future_t = t + horizon
            if future_t < traj_len:
                future_observations.append(traj_observations[future_t])
            else:
                future_observations.append(traj_next_observations[-1])
    future_observations = np.asarray(future_observations)

    success_mask = np.concatenate([
        np.full(len(traj['subgoal']), success, dtype=bool)
        for _, traj, success in valid_trajs
    ])
    traj_ids = np.concatenate([
        np.full(len(traj['subgoal']), traj_id, dtype=np.int32)
        for traj_id, traj, _ in valid_trajs
    ])
    time_indices = np.concatenate([
        np.arange(len(traj['subgoal']), dtype=np.int32)
        for _, traj, _ in valid_trajs
    ])

    subgoal_probs = np.asarray(agent.predict_reachability(
        observations=observations,
        goals=subgoals,
        low_dim_goals=use_rep,
    ))

    if use_rep:
        final_goals_for_r = np.asarray(agent.get_value_goal_rep(
            targets=episode_goals,
            bases=observations,
        ))
        subgoal_distance = np.linalg.norm(subgoals, axis=-1)
        final_goal_distance = np.linalg.norm(final_goals_for_r, axis=-1)
    else:
        final_goals_for_r = episode_goals
        subgoal_distance = np.linalg.norm(subgoals[..., :2] - observations[..., :2], axis=-1)
        final_goal_distance = np.linalg.norm(episode_goals[..., :2] - observations[..., :2], axis=-1)

    final_goal_probs = np.asarray(agent.predict_reachability(
        observations=observations,
        goals=final_goals_for_r,
        low_dim_goals=use_rep,
    ))
    value_adv = np.asarray(agent.predict_value_advantage(
        observations=observations,
        next_observations=future_observations,
        goals=episode_goals,
    ))
    mod_adv = value_adv + reachability_alpha_effective * subgoal_probs * (value_adv > 0)

    if observations.shape[-1] >= 2 and episode_goals.shape[-1] >= 2:
        raw_final_goal_distance = np.linalg.norm(episode_goals[..., :2] - observations[..., :2], axis=-1)
        future_final_goal_distance = np.linalg.norm(episode_goals[..., :2] - future_observations[..., :2], axis=-1)
    else:
        raw_final_goal_distance = np.linalg.norm(episode_goals - observations, axis=-1)
        future_final_goal_distance = np.linalg.norm(episode_goals - future_observations, axis=-1)
    distance_progress = raw_final_goal_distance - future_final_goal_distance

    threshold = float(agent.config['reachability_threshold'])
    stats = {
        'reachability/success_traj_count': sum(success for _, _, success in valid_trajs),
        'reachability/failure_traj_count': sum(not success for _, _, success in valid_trajs),
        'reachability/distance_progress_horizon': horizon,
        'reachability/alpha_effective': reachability_alpha_effective,
    }
    for name, mask in [('success', success_mask), ('failure', ~success_mask), ('all', np.ones_like(success_mask, dtype=bool))]:
        stats.update(_prob_summary(f'reachability/{name}_subgoal_prob', subgoal_probs[mask], threshold))
        stats.update(_prob_summary(f'reachability/{name}_final_goal_prob', final_goal_probs[mask], threshold))
        if mask.any():
            stats[f'reachability/{name}_subgoal_distance_mean'] = subgoal_distance[mask].mean()
            stats[f'reachability/{name}_final_goal_distance_mean'] = final_goal_distance[mask].mean()
            stats[f'reachability/{name}_raw_final_goal_distance_mean'] = raw_final_goal_distance[mask].mean()
            stats[f'reachability/{name}_future_final_goal_distance_mean'] = future_final_goal_distance[mask].mean()
            stats[f'reachability/{name}_distance_progress_mean'] = distance_progress[mask].mean()
            stats[f'reachability/{name}_value_adv_mean'] = value_adv[mask].mean()
            stats[f'reachability/{name}_mod_adv_mean'] = mod_adv[mask].mean()
            stats[f'reachability/{name}_r_vs_distance_progress_corr'] = _safe_corr(subgoal_probs[mask], distance_progress[mask])
            stats[f'reachability/{name}_r_vs_subgoal_distance_corr'] = _safe_corr(subgoal_probs[mask], subgoal_distance[mask])
            stats[f'reachability/{name}_adv_vs_r_corr'] = _safe_corr(value_adv[mask], subgoal_probs[mask])
            stats[f'reachability/{name}_mod_adv_vs_r_corr'] = _safe_corr(mod_adv[mask], subgoal_probs[mask])
            stats[f'reachability/{name}_mod_adv_vs_distance_progress_corr'] = _safe_corr(mod_adv[mask], distance_progress[mask])
        else:
            stats[f'reachability/{name}_subgoal_distance_mean'] = np.nan
            stats[f'reachability/{name}_final_goal_distance_mean'] = np.nan
            stats[f'reachability/{name}_raw_final_goal_distance_mean'] = np.nan
            stats[f'reachability/{name}_future_final_goal_distance_mean'] = np.nan
            stats[f'reachability/{name}_distance_progress_mean'] = np.nan
            stats[f'reachability/{name}_value_adv_mean'] = np.nan
            stats[f'reachability/{name}_mod_adv_mean'] = np.nan
            stats[f'reachability/{name}_r_vs_distance_progress_corr'] = np.nan
            stats[f'reachability/{name}_r_vs_subgoal_distance_corr'] = np.nan
            stats[f'reachability/{name}_adv_vs_r_corr'] = np.nan
            stats[f'reachability/{name}_mod_adv_vs_r_corr'] = np.nan
            stats[f'reachability/{name}_mod_adv_vs_distance_progress_corr'] = np.nan

    rows = []
    for idx in range(subgoal_probs.shape[0]):
        rows.append({
            'traj_id': int(traj_ids[idx]),
            't': int(time_indices[idx]),
            'success': int(success_mask[idx]),
            'subgoal_r': float(subgoal_probs[idx]),
            'final_goal_r': float(final_goal_probs[idx]),
            'subgoal_distance': float(subgoal_distance[idx]),
            'final_goal_distance': float(final_goal_distance[idx]),
            'raw_final_goal_distance': float(raw_final_goal_distance[idx]),
            'future_final_goal_distance': float(future_final_goal_distance[idx]),
            'distance_progress': float(distance_progress[idx]),
            'value_adv': float(value_adv[idx]),
            'mod_adv': float(mod_adv[idx]),
        })
    return stats, rows


def append_reachability_samples(path, step, rows):
    if not rows:
        return
    file_exists = os.path.exists(path)
    fieldnames = [
        'step', 'traj_id', 't', 'success',
        'subgoal_r', 'final_goal_r',
        'subgoal_distance', 'final_goal_distance',
        'raw_final_goal_distance', 'future_final_goal_distance',
        'distance_progress', 'value_adv', 'mod_adv',
    ]
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({'step': step, **row})

###################################修改数据集#################################
def load_calvin_dataset(data_dir):
    episodes = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    for f in files:
        path = os.path.join(data_dir, f)
        data = np.load(path, allow_pickle=True)

        episode = {}
        episode['obs'] = data['obs']
        episode['actions'] = data['actions']
        episode['rewards'] = data['rewards'] if 'rewards' in data else np.zeros(len(data['actions']))
        episode['dones'] = data['dones'] if 'dones' in data else np.zeros(len(data['actions']))

        episodes.append(episode)

    return episodes
###################################修改数据集#################################




def main(_):
    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'
    exp_name += f'_{FLAGS.wandb["name"]}'

    FLAGS.gcdataset['p_randomgoal'] = FLAGS.p_randomgoal
    FLAGS.gcdataset['p_trajgoal'] = FLAGS.p_trajgoal
    FLAGS.gcdataset['p_currgoal'] = FLAGS.p_currgoal
    FLAGS.gcdataset['geom_sample'] = FLAGS.geom_sample
    FLAGS.gcdataset['high_p_randomgoal'] = FLAGS.high_p_randomgoal
    FLAGS.gcdataset['way_steps'] = FLAGS.way_steps
    FLAGS.gcdataset['discount'] = FLAGS.discount
    FLAGS.gcdataset['reachability_horizon'] = FLAGS.reachability_horizon
    FLAGS.gcdataset['reachability_anchor_eps'] = FLAGS.reachability_anchor_eps
    FLAGS.gcdataset['reachability_far_eps'] = FLAGS.reachability_far_eps
    FLAGS.gcdataset['reachability_negative_horizon'] = FLAGS.reachability_negative_horizon
    FLAGS.gcdataset['reachability_hard_negative_prob'] = FLAGS.reachability_hard_negative_prob
    FLAGS.gcdataset['reachability_anchor_max_candidates'] = FLAGS.reachability_anchor_max_candidates
    FLAGS.gcdataset['reachability_anchor_sample_attempts'] = FLAGS.reachability_anchor_sample_attempts
    FLAGS.config['pretrain_expectile'] = FLAGS.pretrain_expectile
    FLAGS.config['discount'] = FLAGS.discount
    FLAGS.config['temperature'] = FLAGS.temperature
    FLAGS.config['high_temperature'] = FLAGS.high_temperature
    FLAGS.config['use_waypoints'] = FLAGS.use_waypoints
    FLAGS.config['way_steps'] = FLAGS.way_steps
    FLAGS.config['value_hidden_dims'] = (FLAGS.value_hidden_dim,) * FLAGS.value_num_layers
    FLAGS.config['use_rep'] = FLAGS.use_rep
    FLAGS.config['rep_dim'] = FLAGS.rep_dim
    FLAGS.config['policy_train_rep'] = FLAGS.policy_train_rep
    FLAGS.config['use_reachability'] = FLAGS.use_reachability
    FLAGS.config['reachability_loss_weight'] = FLAGS.reachability_loss_weight
    FLAGS.config['reachability_threshold'] = FLAGS.reachability_threshold
    FLAGS.config['reachability_resample_attempts'] = FLAGS.reachability_resample_attempts
    FLAGS.config['use_reachability_mod_adv'] = FLAGS.use_reachability_mod_adv
    FLAGS.config['reachability_alpha'] = FLAGS.reachability_alpha

    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    params_dict['reachability_filter_policy'] = FLAGS.reachability_filter_policy
    params_dict['reachability_pretrain_steps'] = FLAGS.reachability_pretrain_steps
    params_dict['freeze_reachability_after_pretrain'] = FLAGS.freeze_reachability_after_pretrain
    params_dict['reachability_start_step'] = FLAGS.reachability_start_step
    params_dict['reachability_start_frac'] = FLAGS.reachability_start_frac
    params_dict['reachability_alpha_warmup_steps'] = FLAGS.reachability_alpha_warmup_steps
    params_dict['reachability_alpha_max'] = FLAGS.reachability_alpha_max
    params_dict['reachability_alpha_min'] = FLAGS.reachability_alpha_min
    params_dict['reachability_alpha_decay_steps'] = FLAGS.reachability_alpha_decay_steps
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    goal_info = None
    discrete = False
    if 'antmaze' in FLAGS.env_name:
        from src import d4rl_utils, d4rl_ant, ant_diagnostics
        env_name = FLAGS.env_name

        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    elif 'kitchen' in FLAGS.env_name:
        from src import d4rl_utils
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    elif 'calvin' in FLAGS.env_name:
        from src import d4rl_utils
        from src.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from src.envs.gym_env import GymWrapper
        from src.envs.gym_env import wrap_env
        initialize(config_path='src/envs/conf')
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)

        #data = pickle.load(gzip.open('data/calvin.gz', "rb"))
        data_path = 'data/calvin/dataset'
        data = load_calvin_dataset(data_path)

        ds = []
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset = d4rl_utils.get_dataset(None, FLAGS.env_name, dataset=dataset)
    elif 'procgen' in FLAGS.env_name:
        from src.envs.procgen_env import ProcgenWrappedEnv, get_procgen_dataset
        import matplotlib

        matplotlib.use('Agg')

        n_processes = 1
        env_name = 'maze'
        env = ProcgenWrappedEnv(n_processes, env_name, 1, 1)

        if FLAGS.env_name == 'procgen-500':
            dataset = get_procgen_dataset('data/procgen/level500.npz', state_based=('state' in FLAGS.env_name))
            min_level, max_level = 0, 499
        elif FLAGS.env_name == 'procgen-1000':
            dataset = get_procgen_dataset('data/procgen/level1000.npz', state_based=('state' in FLAGS.env_name))
            min_level, max_level = 0, 999
        else:
            raise NotImplementedError

        # Test on large levels having >=20 border states
        large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430, 468, 470, 476, 491] + [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335, 5342, 5366, 5375, 5413, 5430, 5474, 5491]
        goal_infos = []
        goal_infos.append({'eval_level': [level for level in large_levels if min_level <= level <= max_level], 'eval_level_name': 'train'})
        goal_infos.append({'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'})

        dones_float = 1.0 - dataset['masks']
        dones_float[-1] = 1.0
        dataset = dataset.copy({
            'dones_float': dones_float
        })

        discrete = True
        example_action = np.max(dataset['actions'], keepdims=True)
    else:
        raise NotImplementedError

    env.reset()

    pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())
    total_steps = FLAGS.reachability_pretrain_steps + FLAGS.pretrain_steps
    if FLAGS.reachability_start_step >= 0:
        reachability_start_step = FLAGS.reachability_start_step
    else:
        reachability_start_step = int(FLAGS.pretrain_steps * FLAGS.reachability_start_frac)
    example_batch = dataset.sample(1)
    agent = learner.create_learner(FLAGS.seed,
                                   example_batch['observations'],
                                   example_batch['actions'] if not discrete else example_action,
                                   visual=FLAGS.visual,
                                   encoder=FLAGS.encoder,
                                   discrete=discrete,
                                   use_layer_norm=FLAGS.use_layer_norm,
                                   rep_type=FLAGS.rep_type,
                                   **FLAGS.config)

    # For debugging metrics
    if 'antmaze' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
    elif 'kitchen' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'calvin' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'procgen-500' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    elif 'procgen-1000' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    else:
        raise NotImplementedError

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        reachability_pretraining = i <= FLAGS.reachability_pretrain_steps
        policy_step = i - FLAGS.reachability_pretrain_steps
        if reachability_pretraining:
            reachability_update = True
            reachability_mod_adv = False
            reachability_alpha_effective = 0.0
        else:
            scheduled_reachability = (
                FLAGS.use_reachability
                and policy_step >= reachability_start_step
            )
            reachability_frozen_after_pretrain = (
                FLAGS.freeze_reachability_after_pretrain
                and FLAGS.reachability_pretrain_steps > 0
            )
            reachability_update = scheduled_reachability and not reachability_frozen_after_pretrain
            reachability_mod_adv = scheduled_reachability and FLAGS.use_reachability_mod_adv
            if scheduled_reachability:
                if FLAGS.reachability_alpha_decay_steps > 0:
                    alpha_max = (
                        FLAGS.reachability_alpha_max
                        if FLAGS.reachability_alpha_max >= 0
                        else FLAGS.reachability_alpha
                    )
                    alpha_min = (
                        FLAGS.reachability_alpha_min
                        if FLAGS.reachability_alpha_min >= 0
                        else 0.0
                    )
                    decay_progress = (
                        (policy_step - reachability_start_step)
                        / FLAGS.reachability_alpha_decay_steps
                    )
                    decay_progress = float(np.clip(decay_progress, 0.0, 1.0))
                    reachability_alpha_effective = alpha_max + (alpha_min - alpha_max) * decay_progress
                elif FLAGS.reachability_alpha_warmup_steps > 0:
                    warmup_progress = (
                        (policy_step - reachability_start_step)
                        / FLAGS.reachability_alpha_warmup_steps
                    )
                    reachability_alpha_effective = FLAGS.reachability_alpha * float(np.clip(warmup_progress, 0.0, 1.0))
                else:
                    reachability_alpha_effective = FLAGS.reachability_alpha
            else:
                reachability_alpha_effective = 0.0

        pretrain_batch = pretrain_dataset.sample(
            FLAGS.batch_size,
            sample_reachability=reachability_update,
        )
        if reachability_pretraining:
            agent, update_info = supply_rng(agent.pretrain_update)(
                pretrain_batch,
                value_update=False,
                actor_update=False,
                high_actor_update=False,
                reachability_update=True,
                reachability_mod_adv=reachability_mod_adv,
                reachability_alpha_effective=reachability_alpha_effective,
            )
        else:
            agent, update_info = supply_rng(agent.pretrain_update)(
                pretrain_batch,
                reachability_update=reachability_update,
                reachability_mod_adv=reachability_mod_adv,
                reachability_alpha_effective=reachability_alpha_effective,
            )

        if i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['training/phase/reachability_pretraining'] = float(reachability_pretraining)
            train_metrics['training/phase/reachability_update'] = float(reachability_update)
            train_metrics['training/phase/reachability_mod_adv'] = float(reachability_mod_adv)
            train_metrics['training/phase/reachability_start_step'] = reachability_start_step
            train_metrics['training/phase/reachability_alpha_effective'] = reachability_alpha_effective
            train_metrics['training/phase/policy_step'] = max(policy_step, 0)
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if policy_step > 0 and (policy_step == 1 or policy_step % FLAGS.eval_interval == 0):
            policy_fn = partial(supply_rng(agent.sample_actions), discrete=discrete)
            if FLAGS.use_reachability and FLAGS.use_waypoints and FLAGS.reachability_filter_policy:
                high_policy_fn = partial(supply_rng(agent.sample_reachable_high_actions))
            else:
                high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            policy_rep_fn = agent.get_policy_rep
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            if 'procgen' in FLAGS.env_name:
                eval_metrics = {}
                for goal_info in goal_infos:
                    eval_info, trajs, renders = evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=FLAGS.env_name, num_episodes=FLAGS.eval_episodes,
                        base_observation=base_observation, num_video_episodes=0,
                        use_waypoints=FLAGS.use_waypoints,
                        eval_temperature=0, epsilon=0.05,
                        goal_info=goal_info, config=FLAGS.config,
                    )
                    eval_metrics.update({f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            else:
                eval_info, trajs, renders = evaluate_with_trajectories(
                    policy_fn, high_policy_fn, policy_rep_fn, env, env_name=FLAGS.env_name, num_episodes=FLAGS.eval_episodes,
                    base_observation=base_observation, num_video_episodes=FLAGS.num_video_episodes,
                    use_waypoints=FLAGS.use_waypoints,
                    eval_temperature=0,
                    goal_info=goal_info, config=FLAGS.config,
                )
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
                if FLAGS.use_reachability and FLAGS.use_waypoints:
                    reachability_stats, reachability_rows = get_eval_reachability_stats(
                        agent,
                        trajs,
                        use_rep=FLAGS.use_rep,
                        reachability_alpha_effective=reachability_alpha_effective,
                    )
                    eval_metrics.update({
                        f'evaluation/{k}': v
                        for k, v in reachability_stats.items()
                    })
                    append_reachability_samples(
                        os.path.join(FLAGS.save_dir, 'reachability_eval_subgoals.csv'),
                        i,
                        reachability_rows,
                    )

                if FLAGS.num_video_episodes > 0:
                    video = record_video('Video', i, renders=renders)
                    eval_metrics['video'] = video

            traj_metrics = get_traj_v(agent, example_trajectory)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
             )
            eval_metrics['value_traj_viz'] = wandb.Image(value_viz)

            if 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name and FLAGS.env_name.startswith('antmaze'):
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update({
                    f'debugging/{k}': v for k, v in new_metrics_dist.items()})

                image_v = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    partial(get_v, agent),
                )
                eval_metrics['v'] = wandb.Image(image_v)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if policy_step > 0 and policy_step % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict()
            )

            fname = os.path.join(FLAGS.save_dir, f'params_{policy_step}.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
