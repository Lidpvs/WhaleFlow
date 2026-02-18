# src/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    INTERVAL: str = "1h"
    DAYS_BACK: int = 30
    BINANCE_BASE_URL: str = "https://api.binance.com"
    BTC_SYMBOL: str = "BTCUSDT"
    SOL_SYMBOL: str = "SOLUSDT"
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"

    WHALE_ALERT_API_KEY: str = ""
    WHALE_THRESHOLD_USD: int = 1_000_000
    
    BTC_RPC_URL: str = ""       # quicknode api
