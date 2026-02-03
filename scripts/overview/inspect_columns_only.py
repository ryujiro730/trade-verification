# scripts/overview/inspect_columns_only.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "parquet"

for tf in ["M15", "M1"]:
    path = DATA_DIR / f"EURUSD_{tf}.parquet"
    print("=" * 80)
    print(path.name)
    df = pd.read_parquet(path)
    print("columns:", df.columns.tolist())
    print("index  :", type(df.index))
    print("index name:", df.index.name)
    print(df.head(2))

