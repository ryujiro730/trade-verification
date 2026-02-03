# scripts/inspect_parquet.py
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


PARQUET_PATH = "~/trade/data/parquet/ndx.parquet"

def main():
    print("=" * 80)
    print("📦 PARQUET FILE INSPECTION")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Arrowレベルのメタ情報
    # --------------------------------------------------
    table = pq.read_table(PARQUET_PATH)
    schema = table.schema

    print("\n[1] Arrow Schema")
    print(schema)

    print("\n[2] Row Groups")
    pf = pq.ParquetFile(PARQUET_PATH)
    print(f"  row_groups: {pf.num_row_groups}")
    print(f"  rows      : {pf.metadata.num_rows}")
    print(f"  columns   : {pf.metadata.num_columns}")

    # --------------------------------------------------
    # 2. pandasでロード
    # --------------------------------------------------
    df = pd.read_parquet(PARQUET_PATH)

    print("\n[3] DataFrame Info")
    print(df.info())

    # --------------------------------------------------
    # 3. 基本統計量
    # --------------------------------------------------
    print("\n[4] Describe (numeric)")
    print(df.describe().T)

    # --------------------------------------------------
    # 4. 日付の健全性
    # --------------------------------------------------
    print("\n[5] Date Range Check")
    print("  min date :", df["date"].min())
    print("  max date :", df["date"].max())
    print("  rows     :", len(df))

    # 欠損日チェック（営業日ベースではなく連番）
    date_diff = df["date"].diff().dt.days
    gaps = date_diff[date_diff > 1]

    print("\n[6] Date Gaps (>1 day)")
    if len(gaps) == 0:
        print("  ✅ No gaps")
    else:
        print(f"  ⚠ gaps found: {len(gaps)}")
        print(gaps.head(10))

    # --------------------------------------------------
    # 5. NaN / Inf チェック
    # --------------------------------------------------
    print("\n[7] NaN / Inf Check")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes("number").columns
    inf_count = (~np.isfinite(df[numeric_cols])).sum()
    print("\n[8] Inf Count")
    print(inf_count)

    # --------------------------------------------------
    # 6. 価格の論理整合性
    # --------------------------------------------------
    print("\n[9] Price Sanity Checks")

    bad_high_low = df[df["high"] < df["low"]]
    print(f"  high < low      : {len(bad_high_low)}")

    bad_open_range = df[
        (df["open"] > df["high"]) | (df["open"] < df["low"])
    ]
    print(f"  open out of range: {len(bad_open_range)}")

    bad_close_range = df[
        (df["close"] > df["high"]) | (df["close"] < df["low"])
    ]
    print(f"  close out of range: {len(bad_close_range)}")

    # --------------------------------------------------
    # 7. ボリュームチェック
    # --------------------------------------------------
    print("\n[10] Volume Checks")
    print("  min volume:", df["volume"].min())
    print("  zero volume rows:", (df["volume"] == 0).sum())

    # --------------------------------------------------
    # 8. 重複チェック
    # --------------------------------------------------
    print("\n[11] Duplicate Dates")
    dup = df["date"].duplicated().sum()
    print(f"  duplicated dates: {dup}")

    # --------------------------------------------------
    # 9. 最初と最後の実データ
    # --------------------------------------------------
    print("\n[12] Head / Tail (5 rows)")
    print(df.head())
    print(df.tail())

    print("\n✅ INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
