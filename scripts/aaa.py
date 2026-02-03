"""
Reason ②: Does discretionary filtering really remove bad trades?
Compare:
- All rule-based trades
- Trades likely to be skipped by discretionary filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Config
# ===============================
PAIR = "EURUSD"
TIMEFRAME = "H1"
DATA_PATH = Path("data/parquet") / f"{PAIR}_{TIMEFRAME}.parquet"
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 20

# 仮定：裁量で「見送られやすい」条件
SKIP_ATR_THRESHOLD = 1.0    # ATR比が大きい
SKIP_DELAY_THRESHOLD = 3   # ブレイク後3本以上遅れ

# ===============================
# Load data
# ===============================
df = pd.read_parquet(DATA_PATH).copy()

# ===============================
# ATR
# ===============================
tr = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    )
)
df["atr"] = tr.rolling(ATR_PERIOD).mean()

# ===============================
# Breakout definition
# ===============================
df["prev_high"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift()
df["is_breakout"] = df["close"] > df["prev_high"]
df = df.dropna()

# ===============================
# Build trades
# ===============================
trades = []

HOLD_BARS = 10  # 固定保有（簡易）

for idx in df.index[df["is_breakout"]]:
    loc = df.index.get_loc(idx)
    if loc + HOLD_BARS >= len(df):
        continue

    entry_price = df.iloc[loc]["close"]
    exit_price = df.iloc[loc + HOLD_BARS]["close"]
    atr = df.iloc[loc]["atr"]

    r = (exit_price - entry_price) / atr

    # 疑似「裁量の遅れ」
    delay = np.random.randint(0, 6)

    trades.append({
        "r": r,
        "delay": delay,
        "abs_r": abs(r),
    })

trades_df = pd.DataFrame(trades)

# ===============================
# Define skipped trades (discretionary filter)
# ===============================
trades_df["is_skipped"] = (
    (trades_df["abs_r"] > SKIP_ATR_THRESHOLD) |
    (trades_df["delay"] >= SKIP_DELAY_THRESHOLD)
)

# ===============================
# Metrics
# ===============================
def summarize(df):
    wins = df[df["r"] > 0]["r"]
    losses = df[df["r"] <= 0]["r"]

    pf = (
        wins.sum() / abs(losses.sum())
        if len(wins) > 0 and len(losses) > 0
        else float("nan")
    )

    return {
        "count": len(df),
        "winrate": (df["r"] > 0).mean(),
        "avg_r": df["r"].mean(),
        "pf": pf,
    }


summary_all = summarize(trades_df)
summary_skipped = summarize(trades_df[trades_df["is_skipped"]])

summary = pd.DataFrame.from_dict(
    {
        "全トレード": summary_all,
        "見送られたトレード": summary_skipped,
    },
    orient="index"
)

print(summary)

