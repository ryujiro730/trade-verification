from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd
from numba import njit


# =====================================================
# Config
# =====================================================

PAIR = "EURUSD"
ENTRY_TF = "M15"
EXIT_TF = "M1"

BB_N = 20
BB_K = 2.0

TP_RANGE = range(5, 101)   # 1pip刻み
SL_RANGE = range(5, 101)

SPREAD_PIPS = 0.7
PIP_SIZE = 0.0001

PROCESSES = max(1, cpu_count() - 1)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "parquet"
OUT_DIR = BASE_DIR / "output" / "strategy_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Utils
# =====================================================

def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name or "date"
        df = df.reset_index().rename(columns={idx_name: "date"})

    if "date" not in df.columns:
        raise RuntimeError(f"'date' column not found: {path}")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def bollinger(close: np.ndarray, n: int, k: float):
    s = pd.Series(close)
    ma = s.rolling(n).mean().to_numpy()
    sd = s.rolling(n).std(ddof=0).to_numpy()
    return ma, ma + k * sd, ma - k * sd


# =====================================================
# Trade Definition
# =====================================================

@dataclass(frozen=True)
class Trade:
    entry_i: int
    side: int  # +1 long, -1 short


# =====================================================
# Entry Logic
# =====================================================

def build_trades(
    df15: pd.DataFrame,
    df1: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Trade]:

    df15 = df15[(df15["date"] >= start) & (df15["date"] <= end)]
    df1  = df1[(df1["date"] >= start) & (df1["date"] <= end)]

    close = df15["close"].to_numpy()
    _, upper, lower = bollinger(close, BB_N, BB_K)

    m1_times = df1["date"].values.astype("datetime64[ns]").astype("int64")

    trades = []

    for i in range(len(df15) - 1):
        if close[i] < lower[i]:
            side = +1
        elif close[i] > upper[i]:
            side = -1
        else:
            continue

        next_t = df15.iloc[i + 1]["date"].value
        j = np.searchsorted(m1_times, next_t, side="left")

        if 0 <= j < len(df1):
            trades.append(Trade(j, side))

    # 同一M1エントリー重複排除
    seen = set()
    uniq = []
    for t in trades:
        if t.entry_i not in seen:
            seen.add(t.entry_i)
            uniq.append(t)

    return uniq


# =====================================================
# Numba-accelerated Simulation
# =====================================================

@njit
def simulate_one(
    tp_pips: int,
    sl_pips: int,
    entry_idx: np.ndarray,
    sides: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> tuple:

    tp = tp_pips * PIP_SIZE
    sl = sl_pips * PIP_SIZE
    spread = SPREAD_PIPS * PIP_SIZE * 2

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    wins = 0
    gp = 0.0
    gl = 0.0
    n = entry_idx.shape[0]

    for k in range(n):
        i0 = entry_idx[k]
        side = sides[k]
        entry = o[i0]

        exit_price = c[-1]

        if side == 1:
            tp_p = entry + tp
            sl_p = entry - sl
            for i in range(i0, len(o)):
                if l[i] <= sl_p:
                    exit_price = sl_p
                    break
                if h[i] >= tp_p:
                    exit_price = tp_p
                    break
            pnl = (exit_price - entry) - spread
        else:
            tp_p = entry - tp
            sl_p = entry + sl
            for i in range(i0, len(o)):
                if h[i] >= sl_p:
                    exit_price = sl_p
                    break
                if l[i] <= tp_p:
                    exit_price = tp_p
                    break
            pnl = (entry - exit_price) - spread

        pnl_pips = pnl / PIP_SIZE

        equity += pnl_pips
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

        if pnl_pips > 0:
            wins += 1
            gp += pnl_pips
        elif pnl_pips < 0:
            gl += -pnl_pips

    expectancy = equity / n if n > 0 else 0.0
    pf = gp / gl if gl > 0 else 1e9

    return wins / n, expectancy, pf, max_dd


def worker(args):
    tp, sl, entry_idx, sides, o, h, l, c = args
    win_rate, exp, pf, max_dd = simulate_one(
        tp, sl, entry_idx, sides, o, h, l, c
    )
    return {
        "tp_pips": tp,
        "sl_pips": sl,
        "win_rate": win_rate,
        "expectancy": exp,
        "pf": pf,
        "max_dd": max_dd,
    }


# =====================================================
# Main
# =====================================================

def main():
    df15 = load(DATA_DIR / f"{PAIR}_{ENTRY_TF}.parquet")
    df1  = load(DATA_DIR / f"{PAIR}_{EXIT_TF}.parquet")

    # === 直近5年に限定 ===
    end = min(df15["date"].max(), df1["date"].max())
    start = end - pd.DateOffset(years=5)

    print("TEST RANGE:", start, "→", end)

    trades = build_trades(df15, df1, start, end)
    print("Total trades:", len(trades))

    entry_idx = np.array([t.entry_i for t in trades], dtype=np.int64)
    sides     = np.array([t.side for t in trades], dtype=np.int8)

    o = df1["open"].to_numpy()
    h = df1["high"].to_numpy()
    l = df1["low"].to_numpy()
    c = df1["close"].to_numpy()

    params = [
        (tp, sl, entry_idx, sides, o, h, l, c)
        for tp in TP_RANGE
        for sl in SL_RANGE
    ]

    with Pool(PROCESSES) as p:
        results = list(
            tqdm(
                p.imap_unordered(worker, params, chunksize=8),
                total=len(params),
                desc="Bruteforce (5y, 1pip)"
            )
        )

    out = pd.DataFrame(results)
    out_path = OUT_DIR / f"{PAIR}_bbands_M15_M1_5y_1pip.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()

