import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(7)

# -----------------------
# 設定
# -----------------------
BARS = 1200
OPPORTUNITIES = 220

base_points = np.sort(
    np.random.choice(np.arange(20, BARS - 20), size=OPPORTUNITIES, replace=False)
)

# 判断確率（その日の気分・体調）
p1 = np.clip(0.58 + np.random.normal(0, 0.06, OPPORTUNITIES), 0.05, 0.95)
p2 = np.clip(0.58 + np.random.normal(0, 0.06, OPPORTUNITIES), 0.05, 0.95)

take1 = np.random.rand(OPPORTUNITIES) < p1
take2 = np.random.rand(OPPORTUNITIES) < p2

# エントリー位置の解釈ズレ（±2本）
jitter1 = np.round(np.random.normal(0, 2, OPPORTUNITIES)).astype(int)
jitter2 = np.round(np.random.normal(0, 2, OPPORTUNITIES)).astype(int)

entry1 = np.where(take1, base_points + jitter1, np.nan)
entry2 = np.where(take2, base_points + jitter2, np.nan)

x, y, c = [], [], []

for i in range(OPPORTUNITIES):
    e1, e2 = entry1[i], entry2[i]

    if np.isnan(e1) and np.isnan(e2):
        continue  # 両方スルーは除外

    x.append(base_points[i])

    if not np.isnan(e1) and not np.isnan(e2):
        delta = e2 - e1
        y.append(delta)
        c.append("tab:blue" if delta == 0 else "tab:red")
    else:
        y.append(10)   # 片方だけ取った
        c.append("tab:red")

# -----------------------
# 描画
# -----------------------
plt.figure(figsize=(10, 5))
plt.scatter(x, y, c=c, alpha=0.75)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(10, linestyle=":", linewidth=1)
plt.title("Self-Inconsistency in Manual Backtesting")
plt.xlabel("Opportunity (time)")
plt.ylabel("Entry mismatch (bars)")

legend = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="tab:blue", markersize=8,
           label="Match (same decision)"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="tab:red", markersize=8,
           label="Mismatch (different / only one took)")
]
plt.legend(handles=legend)
plt.tight_layout()

plt.savefig("backtest_self_inconsistency.png", dpi=150)
plt.close()

print("saved: backtest_self_inconsistency.png")

