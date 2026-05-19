import pandas as pd
import matplotlib.pyplot as plt

# ===== 读取数据 =====
hiql = pd.read_csv('experiment_output/hiql/EXP/sd000_1775052050_antmaze-large-diverse-v2_20260401_220050_HIQL/eval_hiql.csv')
hiql_wo = pd.read_csv('experiment_output/hiql/EXP/sd000_1775383090_antmaze-large-diverse-v2_20260405_175810_HIQL(wo repr)/eval_hiql(wo repr).csv')
hgcbc = pd.read_csv('experiment_output/hiql/EXP/sd000_1775460130_antmaze-large-diverse-v2_20260406_152210_HGCBC/eval_hgcbc.csv')
iql = pd.read_csv('experiment_output/hiql/EXP/sd000_1775526940_antmaze-large-diverse-v2_20260407_095540_IQL/eval_iql.csv')
por = pd.read_csv('experiment_output/hiql/EXP/sd000_1775647074_antmaze-large-diverse-v2_20260408_191754_POR/eval_por.csv')
# ===== 关键列 =====
x_key = 'step'
y_return = 'evaluation/episode.normalized_return'
y_success = 'debugging/pct_within_10'
y_final = 'evaluation/final.episode.normalized_return'  # 更严谨

# ===== 平滑函数 =====
def smooth(y, window=10):
    return y.rolling(window, min_periods=1).mean()

# ===== 自动统一成功率尺度（关键）=====
def normalize_success(df):
    y = df[y_success].copy()
    if y.max() <= 1.0:   # 如果是0~1
        y = y * 100.0    # 转成0~100
    return y

# ===== 创建 subplot =====
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# =========================
# (1) Normalized Return
# =========================
axs[0].plot(hiql[x_key], smooth(hiql[y_return]), label='HIQL')
axs[0].plot(hiql_wo[x_key], smooth(hiql_wo[y_return]), label='WO REPR')
axs[0].plot(hgcbc[x_key], smooth(hgcbc[y_return]), label='HGCBC')
axs[0].plot(iql[x_key], smooth(iql[y_return]), label='IQL')
axs[0].plot(por[x_key], smooth(por[y_return]), label='POR')


axs[0].set_title('Normalized Return')
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Return')
axs[0].legend()
axs[0].grid()

# =========================
# (2) Success Rate（已统一尺度）
# =========================
axs[1].plot(hiql[x_key], smooth(normalize_success(hiql)), label='HIQL')
axs[1].plot(hiql_wo[x_key], smooth(normalize_success(hiql_wo)), label='WO REPR')
axs[1].plot(hgcbc[x_key], smooth(normalize_success(hgcbc)), label='HGCBC')
axs[1].plot(iql[x_key], smooth(normalize_success(iql)), label='IQL')
axs[1].plot(por[x_key], smooth(normalize_success(por)), label='POR')

axs[1].set_title('Success Rate (%)')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Success Rate (0~100)')
axs[1].legend()
axs[1].grid()

# =========================
# (3) Final Performance
# =========================
# final_values = [
#     hiql[y_final].iloc[-1],
#     hiql_wo[y_final].iloc[-1],
#     hgcbc[y_final].iloc[-1],
#     iql[y_final].iloc[-1]
# ]
#
# labels = ['HIQL', 'WO REPR', 'HGCBC', 'IQL']
#
# axs[2].bar(labels, final_values)
# axs[2].set_title('Final Performance')
# axs[2].set_ylabel('Normalized Return')

# ===== 布局 & 保存 =====
plt.tight_layout()
plt.savefig('antmaze_final_subplot.png', dpi=300)
plt.show()