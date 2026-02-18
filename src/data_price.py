# src/data_price.py

from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

# Eigene Module
from src.config import Config
from src.utils_io import save_parquet

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore"
]

def to_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    base_url: str = Config.BINANCE_BASE_URL,
    limit: int = 1000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    
    url = f"{base_url}/api/v3/klines"

    start_ms = to_millis(start_dt)
    end_ms = to_millis(end_dt)

    all_rows: list[list] = []
    cur_start = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start = last_open_time + 1
        if next_start >= end_ms:
            break

        cur_start = next_start
        time.sleep(sleep_s)

        if len(all_rows) > 2_000_000:
            raise RuntimeError("Too many rows fetched; check parameters.")

   
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop(columns=["open_time"]).sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)

def main(days_back: int = Config.DAYS_BACK) -> None:
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)

    for symbol in [Config.BTC_SYMBOL, Config.SOL_SYMBOL]:
        print(f"Fetching {symbol} {Config.INTERVAL} from {start_dt} to {end_dt}...")
        df = fetch_binance_klines(symbol=symbol, interval=Config.INTERVAL, start_dt=start_dt, end_dt=end_dt)
        out_path = f"{Config.RAW_DIR}/{symbol.lower()}_{Config.INTERVAL}.parquet"
        save_parquet(df, out_path)
        print(f"Saved {len(df)} rows -> {out_path}")





if __name__ == "__main__":
    main()


