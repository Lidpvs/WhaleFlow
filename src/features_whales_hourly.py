# src/features_whales_hourly.py

from __future__ import annotations
import pandas as pd

#Eigene Module
from src.config import Config
from src.utils_io import load_parquet, save_parquet

def main():
    in_path = f"{Config.RAW_DIR}/btc_whale_tx_10parts_50blocks_100btc_dedup.parquet"
    df = load_parquet(in_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)


    df["hour"] = df["timestamp"].dt.floor("h")


    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            whale_tx_count=("txid", "count"),
            whale_sum_btc=("whale_outputs_sum_btc", "sum"),
            whale_max_btc=("whale_max_output_btc", "max"),
            whale_mean_btc=("whale_max_output_btc", "mean")
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    if len(hourly) > 0:
        q = hourly["whale_sum_btc"].quantile(0.90)
        hourly["spike_flag"] = (hourly["whale_sum_btc"] >= q).astype(int)
    else:
        hourly["spike_flag"] = 0

    hourly["whale_sum_3h"] = hourly["whale_sum_btc"].rolling(3, min_periods=1).sum()
    hourly["spike_cluster"] = hourly["spike_flag"].rolling(3, min_periods=1).max().astype(int)


    out_path = f"{Config.PROCESSED_DIR}/btc_whales_hourly_10parts_100btc.parquet"
    save_parquet(hourly, out_path)

    print("Hourly rows:", len(hourly))
    print("Time range:", hourly["hour"].min(), "->", hourly["hour"].max())
    print("Spike count:", int(hourly["spike_flag"].sum()))
    print("Saved ->", out_path)



if __name__ == "__main__":
    main()