import pandas as pd
import numpy as np

# scripts/overview/inspect_m15_m1_alignment.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # ~/trade
DATA_DIR = BASE_DIR / "data" / "parquet"

PAIR = "EURUSD"
M15_PATH = DATA_DIR / f"{PAIR}_M15.parquet"
M1_PATH  = DATA_DIR / f"{PAIR}_M1.parquet"

def load(path):
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    df15 = load(M15_PATH)
    df1  = load(M1_PATH)

    print("=" * 80)
    print("BASIC INFO")
    print("=" * 80)

    print("M15 rows:", len(df15))
    print("M1  rows:", len(df1))

    print("\nM15 date range:", df15["date"].min(), "→", df15["date"].max())
    print("M1  date range:", df1["date"].min(),  "→", df1["date"].max())

    print("\nTIMEZONE CHECK")
    print("M15 tz:", df15["date"].dt.tz)
    print("M1  tz:", df1["date"].dt.tz)

    print("\nFIRST / LAST ROWS (M15)")
    print(df15.head(3)[["date", "open", "close"]])
    print(df15.tail(3)[["date", "open", "close"]])

    print("\nFIRST / LAST ROWS (M1)")
    print(df1.head(3)[["date", "open", "close"]])
    print(df1.tail(3)[["date", "open", "close"]])

    print("\n" + "=" * 80)
    print("M15 → M1 ALIGNMENT CHECK")
    print("=" * 80)

    # M15の次バー時刻
    df15["next_m15"] = df15["date"].shift(-1)

    m1_times = df1["date"].values.astype("datetime64[ns]").astype("int64")

    missing = 0
    offsets = []

    for t in df15["next_m15"].dropna().head(2000):
        t_ns = t.value
        idx = np.searchsorted(m1_times, t_ns, side="left")

        if idx >= len(m1_times):
            missing += 1
            continue

        diff_sec = (m1_times[idx] - t_ns) / 1e9
        offsets.append(diff_sec)

    offsets = np.array(offsets)

    print(f"Checked samples : {len(offsets)}")
    print(f"Missing M1 bars: {missing}")
    print("Offset seconds stats:")
    print("  min :", offsets.min())
    print("  max :", offsets.max())
    print("  mean:", offsets.mean())

    print("\nEXPECTED:")
    print("  min = 0 sec")
    print("  max ≈ 60 sec")

    print("\n" + "=" * 80)
    print("DATE GAP CHECK (M1)")
    print("=" * 80)

    gaps = df1["date"].diff().dt.total_seconds().value_counts().head(10)
    print(gaps)


if __name__ == "__main__":
    main()

