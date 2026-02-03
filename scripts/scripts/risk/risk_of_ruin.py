"""
risk_of_ruin.py
=================================
Risk-of-ruin analysis based on losing streaks.

Purpose:
- Quantify capital drawdown caused by consecutive losses
- Evaluate survivability under (risk_per_trade × leverage)
- Produce a ruin heatmap and hard thresholds

This is NOT a strategy backtest.
This is a survival constraint generator.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================
# Paths
# =============================
DATA_PATH = Path("~/trade/data/parquet/ndx.parquet").expanduser()
OUT_DIR = Path("~/trade/results/risk_of_ruin").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# Parameters (Grid)
# =============================
RISK_PER_TRADE = np.array([0.0025, 0.005, 0.01, 0.02, 0.03, 0.05])  # 0.25% .. 5%
LEVERAGE = np.array([1, 1.5, 2, 3, 4, 5])
RUIN_THRESHOLD = -0.5  # -50% capital = practical ruin


# =============================
# Data
# =============================
def load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df


def compute_losing_streaks(df: pd.DataFrame) -> pd.Series:
    is_loss = df["return"] < 0

    streaks = []
    current = 0
    for v in is_loss:
        if v:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    return pd.Series(streaks)


# =============================
# Risk of Ruin
# =============================
def capital_after_streak(
    streak_len: int, risk: float, leverage: float
) -> float:
    """
    Remaining capital after N consecutive losses.
    """
    loss_per_trade = risk * leverage
    return (1 - loss_per_trade) ** streak_len


def evaluate_grid(streaks: pd.Series) -> pd.DataFrame:
    records = []

    worst_streak = int(streaks.max())
    p99_streak = int(streaks.quantile(0.99))
    p95_streak = int(streaks.quantile(0.95))

    for risk in RISK_PER_TRADE:
        for lev in LEVERAGE:
            cap_worst = capital_after_streak(worst_streak, risk, lev)
            cap_p99 = capital_after_streak(p99_streak, risk, lev)
            cap_p95 = capital_after_streak(p95_streak, risk, lev)

            records.append(
                {
                    "risk_per_trade": risk,
                    "leverage": lev,
                    "capital_after_worst": cap_worst,
                    "capital_after_p99": cap_p99,
                    "capital_after_p95": cap_p95,
                    "ruin_worst": cap_worst <= (1 + RUIN_THRESHOLD),
                    "ruin_p99": cap_p99 <= (1 + RUIN_THRESHOLD),
                }
            )

    return pd.DataFrame(records)


# =============================
# Visualization
# =============================
def plot_heatmap(df: pd.DataFrame, col: str, title: str, fname: str):
    pivot = df.pivot(
        index="risk_per_trade", columns="leverage", values=col
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Remaining Capital"},
    )
    plt.title(title)
    plt.ylabel("Risk per Trade")
    plt.xlabel("Leverage")
    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


# =============================
# Main
# =============================
def main():
    print("=" * 80)
    print("☠️  RISK OF RUIN ANALYSIS (NDX)")
    print("=" * 80)

    df = load_returns(DATA_PATH)
    streaks = compute_losing_streaks(df)

    print("\n[1] Losing Streak Facts")
    print(f"  max streak : {streaks.max()}")
    print(f"  99% streak : {int(streaks.quantile(0.99))}")
    print(f"  95% streak : {int(streaks.quantile(0.95))}")

    grid = evaluate_grid(streaks)

    print("\n[2] Ruin Conditions (Worst Case)")
    print(
        grid[
            ["risk_per_trade", "leverage", "capital_after_worst", "ruin_worst"]
        ]
    )

    print("\n[3] Generating Heatmaps")
    plot_heatmap(
        grid,
        "capital_after_worst",
        "Capital After Worst Losing Streak",
        "capital_after_worst.png",
    )
    plot_heatmap(
        grid,
        "capital_after_p99",
        "Capital After 99% Losing Streak",
        "capital_after_p99.png",
    )

    print("\n✅ Risk of ruin analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

