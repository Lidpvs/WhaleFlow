
import pandas as pd

from src.config import Config
from src.utils_io import load_parquet

def main():
    path = f"{Config.RAW_DIR}/btc_whale_tx_last30blocks_50btc.parquet"
    #path = f"{Config.RAW_DIR}/btc_whale_tx_part09_50blocks_100btc.parquet"
    df = load_parquet(path)

    
    print("Rows:", len(df))
    print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())

    print("\nTop 10 by max output:")
    print(df.sort_values("whale_max_output_btc", ascending=False).head(10))

    print("\nDescribe:")
    print(df["whale_max_output_btc"].describe())

if __name__ == "__main__":
    main()