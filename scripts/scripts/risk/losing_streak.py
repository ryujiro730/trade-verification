"""
losing_streak.py
========================
Losing streak analysis for NDX (cash index).

Purpose:
- Quantify consecutive losing days
- Identify worst losing streaks
- Evaluate streak distribution
- Provide hard constraints for survival-based strategy design

Key idea:
"How many losses in a row can the market realistically produce?"
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = Path("~/trade/data/parquet/ndx.parquet").expanduser()


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Losing Streak Computation
# -----------------------------------------------------------------------------
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df


def compute_losing_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes consecutive losing-day streaks.

    Losing day = return < 0
    """
    df = df.copy()
    df["is_loss"] = df["return"] < 0

    streaks = []
    streak_len = 0
    start_date = None

    for _, row in df.iterrows():
        if row["is_loss"]:
            if streak_len == 0:
                start_date = row["date"]
            streak_len += 1
        else:
            if streak_len > 0:
                streaks.append(
                    {
                        "start_date": start_date,
                        "end_date": row["date"],
                        "length": streak_len,
                    }
                )
            streak_len = 0
            start_date = None

    # handle case where series ends with a losing streak
    if streak_len > 0:
        streaks.append(
            {
                "start_date": start_date,
                "end_date": df.iloc[-1]["date"],
                "length": streak_len,
            }
        )

    return pd.DataFrame(streaks)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_streak_distribution(streaks: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.hist(streaks["length"], bins=50)
    plt.xlabel("Consecutive Losing Days")
    plt.ylabel("Frequency")
    plt.title("Distribution of Losing Streak Lengths (NDX)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "losing_streak_distribution.png")
    plt.close()


def plot_worst_streaks(streaks: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    worst = streaks.sort_values("length", ascending=False).head(top_n)

    plt.figure(figsize=(10, 4))
    plt.barh(worst["start_date"].astype(str), worst["length"])
    plt.xlabel("Consecutive Losing Days")
    plt.ylabel("Streak Start Date")
    plt.title(f"Worst {top_n} Losing Streaks (NDX)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "worst_losing_streaks.png")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("📉 LOSING STREAK ANALYSIS (NDX)")
    print("=" * 80)

    df = load_data(DATA_PATH)
    df = compute_returns(df)

    streaks = compute_losing_streaks(df)

    print("\n[1] Losing Streak Summary")
    print(streaks["length"].describe(percentiles=[0.9, 0.95, 0.99]))

    print("\n[2] Worst Losing Streaks")
    print(streaks.sort_values("length", ascending=False).head(10))

    print("\n[3] Probability View")
    for k in [3, 5, 7, 10, 15, 20]:
        p = (streaks["length"] >= k).mean()
        print(f"  P(streak ≥ {k} days): {p:.2%}")

    OUT_DIR = Path("~/trade/results/losing_streak").expanduser()

    plot_streak_distribution(streaks, OUT_DIR)
    plot_worst_streaks(streaks, OUT_DIR)

    print("\n✅ Losing streak analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
