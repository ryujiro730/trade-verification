"""
fetch_real_rate.py
========================
Fetch and construct US real interest rate (10Y).

Real Rate = DGS10 - T10YIE
Source     = FRED (official)
"""

from pathlib import Path
import os
import pandas as pd
from fredapi import Fred

OUT_PATH = Path("~/trade/data/parquet/macro/real_rate.parquet").expanduser()


def main() -> None:
    fred = Fred(api_key=os.environ["FRED_API_KEY"])

    # Fetch data
    dgs10 = fred.get_series("DGS10")
    t10yie = fred.get_series("T10YIE")

    df = pd.concat(
        [dgs10, t10yie],
        axis=1,
        keys=["nominal_10y", "inflation_10y"]
    )

    df = df.dropna()
    df["real_rate"] = df["nominal_10y"] - df["inflation_10y"]
    df = df.reset_index().rename(columns={"index": "date"})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("=" * 80)
    print("📉 REAL INTEREST RATE DATA READY")
    print("=" * 80)
    print(df.head())
    print(df.tail())
    print(f"\nSaved to: {OUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
