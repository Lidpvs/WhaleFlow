# src/merge_parts.py

from __future__ import annotations
from pathlib import Path
import pandas as pd

# Eigene Module
from src.config import Config
from src.utils_io import save_parquet, load_parquet

def main():
    raw_dir = Path(Config.RAW_DIR)
    files = sorted(raw_dir.glob("btc_whale_tx_part*50blocks_100btc.parquet"))

    if not files:
        raise FileNotFoundError("No part files found. Expected btc_whale_tx_part*_50blocks_100btc.parquet in data/raw")
    
    dfs = []
    for f in files:
        df = load_parquet(f)
        df["source_file"] = f.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True)
    all_df = all_df.sort_values("timestamp").drop_duplicates(subset=["txid"], keep="first").reset_index(drop=True)

    out_path = raw_dir / "btc_whale_tx_10parts_50blocks_100btc_dedup.parquet"
    save_parquet(all_df, out_path)

    print("Merged parts:", len(files))
    print("Rows after dedup:", len(all_df))
    print("Time range:", all_df["timestamp"].min(), all_df["timestamp"].max())
    print("Saved ->", out_path)

if __name__ == "__main__":
    main()