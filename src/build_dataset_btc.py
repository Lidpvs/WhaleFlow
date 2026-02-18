# src/build_dataset_btc.py

from __future__ import annotations
import numpy as np
import pandas as pd

# Eigene Module
from src.config import Config
from src.utils_io import load_parquet, save_parquet

def main():
    price_path = f"{Config.RAW_DIR}/btcusdt_1h.parquet"
    whales_path = f"{Config.PROCESSED_DIR}/btc_whales_hourly_10parts_100btc.parquet"

    px = load_parquet(price_path)
    w = load_parquet(whales_path)

    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
    px = px.sort_values("timestamp").reset_index(drop=True)

    w["hour"] = pd.to_datetime(w["hour"], utc=True)
    w = w.sort_values("hour").reset_index(drop=True)

    px["hour"] = px["timestamp"].dt.floor("h")

    ds = px.merge(w, on="hour", how="left")

    for col in [
        "whale_tx_count",
        "whale_sum_btc", 
        "whale_max_btc", 
        "whale_mean_btc", 
        "spike_flag",
        "whale_sum_3h",
        "spike_cluster"
    ]:
        
        ds[col] = ds[col].fillna(0)

    ds["ret_1h"] = ds["close"].pct_change().fillna(0)
    ds["vol_24h"] = ds["ret_1h"].rolling(24).std().fillna(0)

    ds["ma_24"] = ds["close"].rolling(24).mean()
    ds["trend_regime"] = np.where(ds["close"] > ds["ma_24"], 1, np.where(ds["close"] < ds["ma_24"], -1, 0))
    ds["trend_regime"] = ds["trend_regime"].fillna(0).astype(int)

    med = ds["vol_24h"].rolling(72).median()
    ds["vol_regime"] = np.where(ds["vol_24h"] > med, 1, 0)
    ds["vol_regime"] = ds["vol_regime"].fillna(0).astype(int)

    out_path = f"{Config.PROCESSED_DIR}/btc_dataset_10parts_100btc.parquet"
    save_parquet(ds, out_path)

    print("Dataset rows:", len(ds))
    print("Time range:", ds["timestamp"].min(), "->", ds["timestamp"].max())
    print("Saved ->", out_path)


if __name__ == "__main__":
    main()