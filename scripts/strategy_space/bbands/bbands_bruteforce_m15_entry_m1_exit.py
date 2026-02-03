# scripts/strategy_space/bbands_bruteforce_m15_entry_m1_exit.py
from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


# =========================
# Config
# =========================

PIP_SIZE_EURUSD = 0.0001  # EURUSD pip

@dataclass(frozen=True)
class Trade:
    entry_i: int          # M1 index for entry
    side: int             # +1 long, -1 short


@dataclass
class ResultRow:
    tp_pips: int
    sl_pips: int
    total_trades: int
    win_rate: float
    expectancy_pips: float
    pf: float
    max_dd_pips: float


# =========================
# Data Loading
# =========================

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    return df


def to_m1_arrays(df_m1: pd.DataFrame):
    # Use int64 nanoseconds for fast searchsorted
    t = df_m1["date"].values.astype("datetime64[ns]").astype(np.int64)
    o = df_m1["open"].to_numpy(dtype=np.float64)
    h = df_m1["high"].to_numpy(dtype=np.float64)
    l = df_m1["low"].to_numpy(dtype=np.float64)
    c = df_m1["close"].to_numpy(dtype=np.float64)
    return t, o, h, l, c


# =========================
# Indicators / Signals
# =========================

def bollinger_bands(close: np.ndarray, n: int, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close)
    ma = s.rolling(n).mean().to_numpy()
    sd = s.rolling(n).std(ddof=0).to_numpy()
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def build_trades_m15_entry_m1_open(
    df_m15: pd.DataFrame,
    m1_time_i64: np.ndarray,
    n: int,
    k: float,
    start: Optional[str],
    end: Optional[str],
) -> List[Trade]:
    df = df_m15.copy()

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    close = df["close"].to_numpy(dtype=np.float64)
    ma, upper, lower = bollinger_bands(close, n=n, k=k)

    # Signals: mean reversion
    sig_long = close < lower
    sig_short = close > upper

    # Entry time: next M15 bar time (or current bar time + 15min),
    # then map to first M1 bar >= that time
    # We assume df["date"] is bar OPEN time. So "next bar open" is shift(-1) of date.
    dt = df["date"].values.astype("datetime64[ns]")
    next_dt = np.roll(dt, -1)
    next_dt[-1] = np.datetime64("NaT")

    trades: List[Trade] = []
    next_dt_i64 = next_dt.astype("datetime64[ns]").astype(np.int64)

    for i in range(len(df) - 1):  # last one has no next bar
        side = 0
        if sig_long[i]:
            side = +1
        elif sig_short[i]:
            side = -1
        else:
            continue

        t_entry = next_dt_i64[i]
        if np.isnan(t_entry):
            continue

        j = np.searchsorted(m1_time_i64, t_entry, side="left")
        if j <= 0 or j >= len(m1_time_i64):
            continue

        trades.append(Trade(entry_i=int(j), side=side))

    # Optional: de-duplicate if multiple signals lead to same M1 entry index
    # Keep first occurrence
    seen = set()
    uniq: List[Trade] = []
    for tr in trades:
        if tr.entry_i in seen:
            continue
        seen.add(tr.entry_i)
        uniq.append(tr)

    return uniq


# =========================
# Backtest core (TP/SL on M1)
# =========================

def simulate_param(
    trades: List[Trade],
    m1_open: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    m1_close: np.ndarray,
    tp_pips: int,
    sl_pips: int,
    spread_pips: float,
    pip_size: float = PIP_SIZE_EURUSD,
) -> ResultRow:
    if not trades:
        return ResultRow(tp_pips, sl_pips, 0, 0.0, 0.0, float("nan"), 0.0)

    tp = tp_pips * pip_size
    sl = sl_pips * pip_size
    spr = spread_pips * pip_size

    pnls_pips: List[float] = []

    # sequential equity for max DD
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0

    n_m1 = len(m1_open)

    for tr in trades:
        i0 = tr.entry_i
        if i0 >= n_m1:
            continue

        # Fill at M1 open with spread cost:
        # long pays spread on entry, short pays spread on entry (simplified: full spread as cost)
        entry = m1_open[i0]
        side = tr.side

        # Apply spread as immediate cost in pips:
        # pnl = side*(exit-entry) - spread
        # (This approximates paying spread once; if you want entry+exit spread, multiply by 2)
        spread_cost = spr

        if side == +1:
            tp_price = entry + tp
            sl_price = entry - sl
            hit = None
            exit_price = None

            for i in range(i0, n_m1):
                hi = m1_high[i]
                lo = m1_low[i]

                hit_tp = hi >= tp_price
                hit_sl = lo <= sl_price

                if hit_tp and hit_sl:
                    # pessimistic: SL first
                    hit = "SL"
                    exit_price = sl_price
                    break
                if hit_sl:
                    hit = "SL"
                    exit_price = sl_price
                    break
                if hit_tp:
                    hit = "TP"
                    exit_price = tp_price
                    break

            if exit_price is None:
                exit_price = m1_close[-1]

            pnl = (exit_price - entry) - spread_cost
        else:
            # short
            tp_price = entry - tp
            sl_price = entry + sl

            exit_price = None
            for i in range(i0, n_m1):
                hi = m1_high[i]
                lo = m1_low[i]

                hit_tp = lo <= tp_price
                hit_sl = hi >= sl_price

                if hit_tp and hit_sl:
                    # pessimistic: SL first
                    exit_price = sl_price
                    break
                if hit_sl:
                    exit_price = sl_price
                    break
                if hit_tp:
                    exit_price = tp_price
                    break

            if exit_price is None:
                exit_price = m1_close[-1]

            pnl = (entry - exit_price) - spread_cost

        pnl_pips = pnl / pip_size
        pnls_pips.append(pnl_pips)

        # stats
        equity += pnl_pips
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

        if pnl_pips > 0:
            wins += 1
            gross_profit += pnl_pips
        elif pnl_pips < 0:
            gross_loss += -pnl_pips

    total = len(pnls_pips)
    if total == 0:
        return ResultRow(tp_pips, sl_pips, 0, 0.0, 0.0, float("nan"), 0.0)

    win_rate = wins / total
    expectancy = float(np.mean(pnls_pips))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return ResultRow(
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        total_trades=total,
        win_rate=win_rate,
        expectancy_pips=expectancy,
        pf=pf,
        max_dd_pips=float(max_dd),
    )


# =========================
# Parallel runner
# =========================

_GLOBAL = {}

def _init_worker(payload):
    # Avoid copying huge arrays per task; store once per worker
    _GLOBAL.update(payload)

def _worker_task(args):
    tp_pips, sl_pips = args
    return simulate_param(
        trades=_GLOBAL["trades"],
        m1_open=_GLOBAL["m1_open"],
        m1_high=_GLOBAL["m1_high"],
        m1_low=_GLOBAL["m1_low"],
        m1_close=_GLOBAL["m1_close"],
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        spread_pips=_GLOBAL["spread_pips"],
        pip_size=_GLOBAL["pip_size"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--entry-tf", default="M15")
    ap.add_argument("--exit-tf", default="M1")
    ap.add_argument("--data-dir", default="../data/parquet")
    ap.add_argument("--out-dir", default="../output/strategy_space")
    ap.add_argument("--bb-n", type=int, default=20)
    ap.add_argument("--bb-k", type=float, default=2.0)
    ap.add_argument("--start", default=None)  # e.g. 2001-01-01
    ap.add_argument("--end", default=None)    # e.g. 2026-01-01
    ap.add_argument("--tp-min", type=int, default=5)
    ap.add_argument("--tp-max", type=int, default=100)
    ap.add_argument("--sl-min", type=int, default=5)
    ap.add_argument("--sl-max", type=int, default=100)
    ap.add_argument("--spread-pips", type=float, default=0.0)
    ap.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    m15_path = os.path.join(args.data_dir, f"{args.pair}_{args.entry_tf}.parquet")
    m1_path  = os.path.join(args.data_dir, f"{args.pair}_{args.exit_tf}.parquet")

    print("[load]", m15_path)
    df_m15 = load_parquet(m15_path)
    print("[load]", m1_path)
    df_m1 = load_parquet(m1_path)

    # Use arrays for speed
    m1_time_i64, m1_open, m1_high, m1_low, m1_close = to_m1_arrays(df_m1)

    trades = build_trades_m15_entry_m1_open(
        df_m15=df_m15,
        m1_time_i64=m1_time_i64,
        n=args.bb_n,
        k=args.bb_k,
        start=args.start,
        end=args.end,
    )
    print(f"[trades] {len(trades)} signals (unique M1 entries)")

    # Param grid (TP x SL)
    grid = [(tp, sl) for tp in range(args.tp_min, args.tp_max + 1)
                    for sl in range(args.sl_min, args.sl_max + 1)]
    print(f"[grid] {len(grid)} params")

    payload = {
        "trades": trades,
        "m1_open": m1_open,
        "m1_high": m1_high,
        "m1_low": m1_low,
        "m1_close": m1_close,
        "spread_pips": args.spread_pips,
        "pip_size": PIP_SIZE_EURUSD if args.pair.endswith("USD") else PIP_SIZE_EURUSD,
    }

    with Pool(processes=args.processes, initializer=_init_worker, initargs=(payload,)) as pool:
        rows: List[ResultRow] = list(pool.imap_unordered(_worker_task, grid, chunksize=args.chunksize))

    # Save CSV
    out_csv = os.path.join(
        args.out_dir,
        f"{args.pair}_bbands_entry{args.entry_tf}_exit{args.exit_tf}_tp{args.tp_min}-{args.tp_max}_sl{args.sl_min}-{args.sl_max}_spread{args.spread_pips}.csv"
    )
    df_out = pd.DataFrame([r.__dict__ for r in rows])
    df_out.to_csv(out_csv, index=False)
    print("[saved]", out_csv)

    # Heatmap (expectancy)
    tp_vals = np.arange(args.tp_min, args.tp_max + 1)
    sl_vals = np.arange(args.sl_min, args.sl_max + 1)

    pivot = df_out.pivot(index="sl_pips", columns="tp_pips", values="expectancy_pips").reindex(index=sl_vals, columns=tp_vals)

    plt.figure()
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xlabel("TP (pips)")
    plt.ylabel("SL (pips)")
    plt.title(f"{args.pair} BBands expectancy (pips/trade) | entry {args.entry_tf}, exit {args.exit_tf}, spread {args.spread_pips} pips")
    plt.colorbar()
    out_png = out_csv.replace(".csv", "_heatmap_expectancy.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print("[saved]", out_png)


if __name__ == "__main__":
    main()

