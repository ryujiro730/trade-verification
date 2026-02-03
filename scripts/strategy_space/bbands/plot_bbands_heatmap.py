import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "output" / "strategy_space"

CSV_PATH = OUT_DIR / "EURUSD_bbands_M15_M1_5y_1pip.csv"

df = pd.read_csv(CSV_PATH)

# ===============================
# Expectancy Heatmap
# ===============================
pivot_exp = df.pivot(index="sl_pips", columns="tp_pips", values="expectancy")

plt.figure(figsize=(10, 8))
plt.imshow(pivot_exp.values, origin="lower", aspect="auto")
plt.colorbar(label="Expectancy (pips / trade)")
plt.xlabel("TP (pips)")
plt.ylabel("SL (pips)")
plt.title("EURUSD BBands Expectancy Heatmap (5Y, 1pip)")
plt.tight_layout()
plt.savefig(OUT_DIR / "expectancy_heatmap_5y.png", dpi=160)
plt.close()

# ===============================
# Max DD Heatmap
# ===============================
pivot_dd = df.pivot(index="sl_pips", columns="tp_pips", values="max_dd")

plt.figure(figsize=(10, 8))
plt.imshow(pivot_dd.values, origin="lower", aspect="auto")
plt.colorbar(label="Max Drawdown (pips)")
plt.xlabel("TP (pips)")
plt.ylabel("SL (pips)")
plt.title("EURUSD BBands MaxDD Heatmap (5Y)")
plt.tight_layout()
plt.savefig(OUT_DIR / "maxdd_heatmap_5y.png", dpi=160)
plt.close()

# ===============================
# Expectancy Distribution
# ===============================
plt.figure(figsize=(8, 5))
plt.hist(df["expectancy"], bins=80)
plt.axvline(0, linestyle="--")
plt.xlabel("Expectancy (pips / trade)")
plt.ylabel("Count")
plt.title("Expectancy Distribution (5Y, All TP × SL)")
plt.tight_layout()
plt.savefig(OUT_DIR / "expectancy_distribution_5y.png", dpi=160)
plt.close()

print("Saved:")
print(" - expectancy_heatmap_5y.png")
print(" - maxdd_heatmap_5y.png")
print(" - expectancy_distribution_5y.png")

