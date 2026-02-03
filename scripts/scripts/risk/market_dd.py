"""
market_dd.py
=================================
Market drawdown analysis for NDX (cash index).

Outputs:
- Max Drawdown
- Drawdown distribution
- Time To Recovery (TTR)
- Equity & Drawdown plots (PNG)

Designed for:
- Survival analysis
- Leverage feasibility checks
- Worst-case focused evaluation
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================
# Paths
# =============================
DATA_PATH = Path("~/trade/data/parquet/ndx.parquet").expanduser()
OUTPUT_DIR = Path("~/trade/results/market_dd").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# Data Loading
# =============================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =============================
# Core Calculations
# =============================
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    return df


def compute_equity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["equity"] = (1.0 + df["return"]).cumprod()
    df.loc[df.index[0], "equity"] = 1.0
    return df


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1.0
    return df


# =============================
# Time To Recovery
# =============================
def compute_time_to_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    One record per completed drawdown cycle.
    """
    records = []
    in_dd = False

    peak_date = None
    trough_date = None
    trough_dd = None

    for row in df.itertuples():
        if row.drawdown < 0 and not in_dd:
            in_dd = True
            peak_date = row.date
            trough_date = row.date
            trough_dd = row.drawdown

        if in_dd:
            if row.drawdown < trough_dd:
                trough_dd = row.drawdown
                trough_date = row.date

            if row.drawdown == 0:
                records.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": row.date,
                        "max_drawdown": trough_dd,
                        "ttr_days": (row.date - trough_date).days,
                    }
                )
                in_dd = False

    return pd.DataFrame(records)


# =============================
# Plotting (Save Only)
# =============================
def plot_equity_and_drawdown(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df["date"], df["equity"])
    axes[0].set_title("NDX Equity Curve (Normalized)")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True)

    axes[1].plot(df["date"], df["drawdown"], color="red")
    axes[1].set_title("NDX Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    plt.tight_layout()
    out = OUTPUT_DIR / "equity_drawdown.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


def plot_drawdown_distribution(df: pd.DataFrame) -> None:
    dd = df["drawdown"].dropna()

    plt.figure(figsize=(12, 4))
    plt.hist(dd, bins=120)
    plt.title("Drawdown Distribution (NDX)")
    plt.xlabel("Drawdown")
    plt.ylabel("Frequency")
    plt.grid(True)

    out = OUTPUT_DIR / "drawdown_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


# =============================
# Main
# =============================
def main() -> None:
    print("=" * 80)
    print("📉 MARKET DRAWDOWN ANALYSIS : NDX")
    print("=" * 80)

    df = load_data(DATA_PATH)
    df = compute_returns(df)
    df = compute_equity(df)
    df = compute_drawdown(df)

    # ---- Max DD
    max_dd = df["drawdown"].min()
    print("\n[1] Max Drawdown")
    print(f"  max DD : {max_dd:.2%}")

    # ---- DD Stats
    print("\n[2] Drawdown Distribution")
    print(df["drawdown"].describe(percentiles=[0.01, 0.05, 0.1]))

    # ---- TTR
    ttr = compute_time_to_recovery(df)
    print("\n[3] Time To Recovery (days)")
    if not ttr.empty:
        print(ttr["ttr_days"].describe())
        print("\nWorst recoveries:")
        print(ttr.sort_values("ttr_days", ascending=False).head(5))
    else:
        print("  no completed drawdown cycles")

    # ---- Plots
    print("\n[4] Generating plots")
    plot_equity_and_drawdown(df)
    plot_drawdown_distribution(df)

    print("\n✅ Market drawdown analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
