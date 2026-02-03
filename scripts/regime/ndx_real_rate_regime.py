"""
ndx_real_rate_regime.py
====================================
NDX × Real Interest Rate Regime Analysis

Purpose:
- Identify market regimes where drawdowns and losing streaks increase
- Focus on survival / risk characteristics, not strategy PnL
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ========================
# PATHS
# ========================
NDX_PATH = Path("~/trade/data/parquet/ndx.parquet").expanduser()
REAL_RATE_PATH = Path("~/trade/data/parquet/macro/real_rate.parquet").expanduser()
OUT_DIR = Path("~/trade/results/regime/ndx_real_rate").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ========================
# LOAD & MERGE
# ========================
def load_and_merge() -> pd.DataFrame:
    ndx = pd.read_parquet(NDX_PATH)[["date", "close"]]
    macro = pd.read_parquet(REAL_RATE_PATH)[["date", "real_rate"]]

    df = pd.merge(ndx, macro, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ========================
# FEATURE ENGINEERING
# ========================
def add_returns_and_dd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["equity"] = (1 + df["ret"]).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1
    return df


def add_real_rate_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real rate regimes (can be tuned later):
      A: real_rate < 0
      B: 0 <= real_rate < 1
      C: 1 <= real_rate < 2
      D: real_rate >= 2
    """
    bins = [-np.inf, 0, 1, 2, np.inf]
    labels = [
        "RealRate < 0",
        "0 ≤ RealRate < 1",
        "1 ≤ RealRate < 2",
        "RealRate ≥ 2",
    ]
    df = df.copy()
    df["regime"] = pd.cut(df["real_rate"], bins=bins, labels=labels)
    return df


# ========================
# REGIME ANALYSIS
# ========================
def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("regime")
        .agg(
            days=("ret", "count"),
            avg_return=("ret", "mean"),
            vol=("ret", "std"),
            max_dd=("drawdown", "min"),
            avg_dd=("drawdown", "mean"),
        )
        .sort_values("max_dd")
    )
    return summary


def losing_streaks_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for regime, g in df.groupby("regime"):
        g = g.dropna(subset=["ret"])
        is_loss = g["ret"] < 0

        streak = 0
        streaks = []

        for v in is_loss:
            if v:
                streak += 1
            else:
                if streak > 0:
                    streaks.append(streak)
                streak = 0

        if streak > 0:
            streaks.append(streak)

        if streaks:
            records.append(
                {
                    "regime": regime,
                    "max_streak": max(streaks),
                    "p95_streak": np.percentile(streaks, 95),
                    "p99_streak": np.percentile(streaks, 99),
                }
            )

    return pd.DataFrame(records).set_index("regime")


# ========================
# MAIN
# ========================
def main():
    print("=" * 80)
    print("📊 NDX × REAL RATE REGIME ANALYSIS")
    print("=" * 80)

    df = load_and_merge()
    df = add_returns_and_dd(df)
    df = add_real_rate_regime(df)

    print("\n[1] Regime Summary (Risk View)")
    summary = regime_summary(df)
    print(summary)

    print("\n[2] Losing Streaks by Regime")
    streaks = losing_streaks_by_regime(df)
    print(streaks)

    summary.to_csv(OUT_DIR / "regime_risk_summary.csv")
    streaks.to_csv(OUT_DIR / "regime_losing_streaks.csv")

    print("\nSaved to:", OUT_DIR)
    print("\n✅ Regime analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

