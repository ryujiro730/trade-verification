from __future__ import annotations
import matplotlib.pyplot as plt
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

# =====================================================
# Config
# =====================================================

PAIR = "EURUSD"
ENTRY_TF = "M15"
EXIT_TF = "M1"

BB_N = 20
BB_K = 2.0

TP_RANGE = range(5, 101)   # pips
SL_RANGE = range(5, 101)

SPREAD_PIPS = 0.7          # 0 or 1.0
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

    # DatetimeIndex → date 列に強制統一
    if isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name or "date"
        df = df.reset_index().rename(columns={idx_name: "date"})

    # 念のため保険
    if "date" not in df.columns:
        raise RuntimeError(f"'date' column not found after loading {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

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
# Entry: M15 → M1
# =====================================================

def build_trades(
    df15: pd.DataFrame,
    df1: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Trade]:

    df15 = df15[(df15["date"] >= start) & (df15["date"] <= end)].copy()
    df1  = df1[(df1["date"] >= start) & (df1["date"] <= end)].copy()

    close = df15["close"].to_numpy()
    _, upper, lower = bollinger(close, BB_N, BB_K)

    m1_times = df1["date"].values.astype("datetime64[ns]").astype("int64")

    trades: List[Trade] = []

    for i in range(len(df15) - 1):
        side = 0
        if close[i] < lower[i]:
            side = +1
        elif close[i] > upper[i]:
            side = -1
        else:
            continue

        next_m15_time = df15.iloc[i + 1]["date"].value
        j = np.searchsorted(m1_times, next_m15_time, side="left")

        if 0 <= j < len(df1):
            trades.append(Trade(entry_i=j, side=side))

    # 同一M1 entry 重複排除
    seen = set()
    uniq = []
    for t in trades:
        if t.entry_i not in seen:
            seen.add(t.entry_i)
            uniq.append(t)

    return uniq


# =====================================================
# TP / SL Simulation
# =====================================================

def simulate_param(args):
    tp_pips, sl_pips, trades, o, h, l, c = args

    tp = tp_pips * PIP_SIZE
    sl = sl_pips * PIP_SIZE
    spread = SPREAD_PIPS * PIP_SIZE * 2

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    wins = 0
    gp = 0.0
    gl = 0.0
    pnls = []

    for tr in trades:
        i0 = tr.entry_i
        entry = o[i0]
        side = tr.side

        exit_price = None

        if side == +1:
            tp_p = entry + tp
            sl_p = entry - sl
            for i in range(i0, len(o)):
                if l[i] <= sl_p:
                    exit_price = sl_p
                    break
                if h[i] >= tp_p:
                    exit_price = tp_p
                    break
            if exit_price is None:
                exit_price = c[-1]
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
            if exit_price is None:
                exit_price = c[-1]
            pnl = (entry - exit_price) - spread

        pnl_pips = pnl / PIP_SIZE
        pnls.append(pnl_pips)

        equity += pnl_pips
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

        if pnl_pips > 0:
            wins += 1
            gp += pnl_pips
        elif pnl_pips < 0:
            gl += -pnl_pips

    if not pnls:
        return None

    return {
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "trades": len(pnls),
        "win_rate": wins / len(pnls),
        "expectancy": float(np.mean(pnls)),
        "pf": gp / gl if gl > 0 else np.inf,
        "max_dd": max_dd,
    }


# =====================================================
# Main
# =====================================================

def main():
    df15 = load(DATA_DIR / f"{PAIR}_{ENTRY_TF}.parquet")
    df1  = load(DATA_DIR / f"{PAIR}_{EXIT_TF}.parquet")

    # ✅ 安全な共通期間
    start = max(df15["date"].min(), df1["date"].min())
    end   = min(df15["date"].max(), df1["date"].max())

    print("SAFE RANGE:", start, "→", end)

    trades = build_trades(df15, df1, start, end)
    print("Total trades:", len(trades))

    o = df1["open"].to_numpy()
    h = df1["high"].to_numpy()
    l = df1["low"].to_numpy()
    c = df1["close"].to_numpy()

    params = [
        (tp, sl, trades, o, h, l, c)
        for tp in TP_RANGE
        for sl in SL_RANGE
    ]

    with Pool(PROCESSES) as p:
        results = list(filter(None, p.map(simulate_param, params)))

    out = pd.DataFrame(results)
    out_path = OUT_DIR / f"{PAIR}_bbands_M15_M1_bruteforce.csv"
    out.to_csv(out_path, index=False)
    
    visualize(out)
    
    print("Saved:")
    print(" -", out_path)
    print(" - expectancy_heatmap.png")
    print(" - maxdd_heatmap.png")
    print(" - expectancy_distribution.png")

def visualize(df: pd.DataFrame):
    # --- Expectancy Heatmap ---
    pivot_exp = df.pivot(index="sl_pips", columns="tp_pips", values="expectancy")

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_exp.values, origin="lower", aspect="auto")
    plt.colorbar(label="Expectancy (pips / trade)")
    plt.xlabel("TP (pips)")
    plt.ylabel("SL (pips)")
    plt.title("EURUSD BBands Expectancy Heatmap (M15 entry / M1 exit)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expectancy_heatmap.png", dpi=160)
    plt.close()

    # --- Max DD Heatmap ---
    pivot_dd = df.pivot(index="sl_pips", columns="tp_pips", values="max_dd")

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_dd.values, origin="lower", aspect="auto")
    plt.colorbar(label="Max Drawdown (pips)")
    plt.xlabel("TP (pips)")
    plt.ylabel("SL (pips)")
    plt.title("EURUSD BBands MaxDD Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "maxdd_heatmap.png", dpi=160)
    plt.close()

    # --- Expectancy Distribution ---
    plt.figure(figsize=(8, 5))
    plt.hist(df["expectancy"], bins=80)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Expectancy (pips / trade)")
    plt.ylabel("Count")
    plt.title("Expectancy Distribution (All TP × SL)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expectancy_distribution.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()

