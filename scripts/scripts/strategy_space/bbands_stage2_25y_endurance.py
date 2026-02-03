from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit

# =====================================================
# Config
# =====================================================

PAIR = "EURUSD"
ENTRY_TF = "M15"
EXIT_TF = "M1"

BB_N = 20
BB_K = 2.0

# --- スプレッド等 ---
SPREAD_PIPS = 0.7
PIP_SIZE = 0.0001

# --- 抽出ルール（ここだけ変えれば良い） ---
TOP_N = 12                 # 25年耐久に回す候補数
MIN_TRADES_5Y = 400        # 5年で取引が少ない偶然を排除（trades列があれば）
MAX_DD_PIPS_5Y = 5000      # 5年でDDが異常に大きいのを除外（pips）

# スコア：期待値を主軸にしつつ、DDが小さいものを優先
DD_PENALTY = 0.30

PROCESSES = max(1, cpu_count() - 1)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "parquet"
OUT_DIR = BASE_DIR / "output" / "strategy_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_5Y = OUT_DIR / f"{PAIR}_bbands_M15_M1_5y_1pip.csv"

# --- 約定価格の1pip量子化モード ---
# "floor"  : ロング利確は不利・ショート利確は有利寄り
# "round"  : 中立（おすすめ）
# "ceil"   : ロング利確は有利・ショート利確は不利寄り
EXEC_Q_MODE = "round"


# =====================================================
# Data loading
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
# Entry: M15 signal -> M1 entry index
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

    trades: List[Trade] = []

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

    # 同一M1 entry 重複排除
    seen = set()
    uniq = []
    for t in trades:
        if t.entry_i not in seen:
            seen.add(t.entry_i)
            uniq.append(t)

    return uniq


# =====================================================
# Numba helpers: 1 pip quantization
# =====================================================

@njit
def quantize_1pip(price: float, pip_size: float, mode: int) -> float:
    """
    mode:
      0 = floor
      1 = round
      2 = ceil
    """
    x = price / pip_size
    if mode == 0:
        return np.floor(x) * pip_size
    elif mode == 2:
        return np.ceil(x) * pip_size
    else:
        return np.round(x) * pip_size


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
    q_mode: int,
) -> Tuple[float, float, float, float, float, float, float, int]:

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

        # エントリーは open で固定
        entry = o[i0]

        exit_price = c[-1]

        if side == 1:
            tp_p = entry + tp
            sl_p = entry - sl

            # --- M1 OHLCで到達判定 ---
            for i in range(i0, len(o)):
                hit_sl = (l[i] <= sl_p)
                hit_tp = (h[i] >= tp_p)

                if hit_sl and hit_tp:
                    # 同一バー内で両方ヒットした場合は順序不明
                    # → 中立処理：より近い方を先に到達したとみなす（反論が出づらい）
                    # 距離が同じなら SL 優先（保守的）
                    dist_tp = tp_p - entry
                    dist_sl = entry - sl_p
                    if dist_tp < dist_sl:
                        exit_price = tp_p
                    else:
                        exit_price = sl_p
                    break
                elif hit_sl:
                    exit_price = sl_p
                    break
                elif hit_tp:
                    exit_price = tp_p
                    break

            # --- 約定価格を1pip単位に量子化 ---
            exit_price = quantize_1pip(exit_price, PIP_SIZE, q_mode)

            pnl = (exit_price - entry) - spread

        else:
            tp_p = entry - tp
            sl_p = entry + sl

            for i in range(i0, len(o)):
                hit_sl = (h[i] >= sl_p)
                hit_tp = (l[i] <= tp_p)

                if hit_sl and hit_tp:
                    dist_tp = entry - tp_p
                    dist_sl = sl_p - entry
                    if dist_tp < dist_sl:
                        exit_price = tp_p
                    else:
                        exit_price = sl_p
                    break
                elif hit_sl:
                    exit_price = sl_p
                    break
                elif hit_tp:
                    exit_price = tp_p
                    break

            exit_price = quantize_1pip(exit_price, PIP_SIZE, q_mode)

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

    win_rate = wins / n if n > 0 else 0.0
    expectancy = equity / n if n > 0 else 0.0
    pf = gp / gl if gl > 0 else 1e9

    total_pnl = equity  # pips
    return win_rate, expectancy, pf, max_dd, total_pnl, gp, gl, n


def worker(args):
    tp, sl, entry_idx, sides, o, h, l, c, q_mode = args
    win_rate, expectancy, pf, max_dd, total_pnl, gp, gl, n = simulate_one(
        tp, sl, entry_idx, sides, o, h, l, c, q_mode
    )
    return {
        "tp_pips": tp,
        "sl_pips": sl,
        "trades": int(n),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "pf": float(pf),
        "max_dd": float(max_dd),
        "total_pnl_pips": float(total_pnl),
        "gross_profit_pips": float(gp),
        "gross_loss_pips": float(gl),
    }


# =====================================================
# Selection logic (from 5y CSV)
# =====================================================

def pick_candidates(df5: pd.DataFrame) -> pd.DataFrame:
    df = df5.copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if "trades" in df.columns:
        df = df[df["trades"] >= MIN_TRADES_5Y]

    df = df[df["max_dd"] <= MAX_DD_PIPS_5Y]

    if "trades" in df.columns:
        dd_per_trade = df["max_dd"] / df["trades"]
    else:
        dd_per_trade = df["max_dd"]

    df["score"] = df["expectancy"] - DD_PENALTY * dd_per_trade

    df = df.sort_values("score", ascending=False).head(TOP_N)

    keep_cols = ["tp_pips", "sl_pips", "expectancy", "pf", "max_dd", "score"]
    if "trades" in df.columns:
        keep_cols.insert(2, "trades")

    return df[keep_cols]


# =====================================================
# Plot helpers
# =====================================================

def plot_candidates_table(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.3 * len(df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_endurance_bar(df: pd.DataFrame, path: Path):
    labels = [f"TP{int(r.tp_pips)}/SL{int(r.sl_pips)}" for r in df.itertuples()]
    vals = df["total_pnl_pips"].to_numpy()

    plt.figure(figsize=(12, 5))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total PnL (pips) over 25y")
    plt.title("25y Endurance (1pip execution): Total PnL by Candidate")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# =====================================================
# Main
# =====================================================

def main():
    if not CSV_5Y.exists():
        raise RuntimeError(f"5y CSV not found: {CSV_5Y}")

    df5 = pd.read_csv(CSV_5Y)
    candidates = pick_candidates(df5)

    cand_path = OUT_DIR / f"{PAIR}_bbands_candidates_from_5y.csv"
    candidates.to_csv(cand_path, index=False)

    print("Picked candidates (from 5y):")
    show_cols = ["tp_pips", "sl_pips", "score", "expectancy", "max_dd"]
    if "trades" in candidates.columns:
        show_cols.append("trades")
    print(candidates[show_cols].to_string(index=False))
    print("Saved:", cand_path)

    plot_candidates_table(candidates.round(4), OUT_DIR / f"{PAIR}_bbands_candidates_table.png")

    # --- Load full data for 25y ---
    df15 = load(DATA_DIR / f"{PAIR}_{ENTRY_TF}.parquet")
    df1  = load(DATA_DIR / f"{PAIR}_{EXIT_TF}.parquet")

    end = min(df15["date"].max(), df1["date"].max())
    start = end - pd.DateOffset(years=25)

    print("25Y RANGE:", start, "→", end)

    trades_25y = build_trades(df15, df1, start, end)
    print("Total trades (25y):", len(trades_25y))

    entry_idx = np.array([t.entry_i for t in trades_25y], dtype=np.int64)
    sides     = np.array([t.side for t in trades_25y], dtype=np.int8)

    o = df1["open"].to_numpy()
    h = df1["high"].to_numpy()
    l = df1["low"].to_numpy()
    c = df1["close"].to_numpy()

    # qmode
    q_mode = 1
    if EXEC_Q_MODE == "floor":
        q_mode = 0
    elif EXEC_Q_MODE == "ceil":
        q_mode = 2

    params = []
    for r in candidates.itertuples(index=False):
        params.append((int(r.tp_pips), int(r.sl_pips), entry_idx, sides, o, h, l, c, q_mode))

    with Pool(PROCESSES) as p:
        results = list(
            tqdm(
                p.imap_unordered(worker, params, chunksize=1),
                total=len(params),
                desc="Endurance 25y (1pip exec)"
            )
        )

    out = pd.DataFrame(results)
    out_path = OUT_DIR / f"{PAIR}_bbands_25y_endurance_from_5y_1pip_exec.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)

    plot_endurance_bar(
        out.sort_values("total_pnl_pips", ascending=False),
        OUT_DIR / f"{PAIR}_bbands_25y_endurance_total_pnl_1pip_exec.png"
    )

    print("Saved plots:")
    print(" -", OUT_DIR / f"{PAIR}_bbands_candidates_table.png")
    print(" -", OUT_DIR / f"{PAIR}_bbands_25y_endurance_total_pnl_1pip_exec.png")


if __name__ == "__main__":
    main()

