# src/data_chain_btc.py

import requests
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

# EIGENE MODULE
from src.config import Config
from src.utils_io import save_parquet

def btc_rpc(method, params=None):
    url = Config.BTC_RPC_URL

    payload = {
        "jsonrpc": "1.0",
        "id": "whaleflow",
        "method": method,
        "params": params or []
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    # print("RPC call:", method)

    if "error" in data and data["error"]:
        raise Exception(f"RPC error: {data['error']}")
    
    

    return data["result"]


def get_latest_block_height():
    return btc_rpc("getblockcount")

def get_block_hash(height: int):
    return btc_rpc("getblockhash", [height])

def get_block(block_hash: str):
    return btc_rpc("getblock", [block_hash, 2])

def extract_whales_from_block(block, whale_threshold_btc=50):
    whales = []
    block_time = datetime.fromtimestamp(block["time"], tz=timezone.utc)

    for tx in block["tx"]:
        txid = tx["txid"]

        for vout in tx.get("vout", []):
            value_btc = vout.get("value", 0)

            if value_btc >= whale_threshold_btc:
                whales.append({
                    "timestamp": block_time,
                    "txid": txid,
                    "value_btc": value_btc
                })
    return whales

def scan_recent_blocks(n_blocks=50, whale_threshold_btc=50, start_height=None):
    latest = get_latest_block_height()
    start = start_height if start_height is not None else latest
    print(f"Latest block height {latest}")
    print(f"Start height: {start}  (scanning {n_blocks} blocks downwards)")

    target_height = start - n_blocks + 1
    all_whales = []

    for h in range(start, target_height - 1, -1):
        print(f"Scanning block {h}")

        block_hash = get_block_hash(h)
        block = get_block(block_hash)

        whales = extract_whales_from_block(block, whale_threshold_btc)
        all_whales.extend(whales)

    print(f"Total whales events found: {len(all_whales)}")
    return all_whales

def whales_to_df(whales: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(whales)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def outputs_to_tx_level(df_outputs: pd.DataFrame) -> pd.DataFrame:
    if df_outputs.empty:
        return pd.DataFrame(columns=["timestamp", "txid", "whale_max_output_btc", "whale_outputs_count", "whale_outputs_sum_btc"])
    
    df_tx = (
        df_outputs.groupby(["timestamp", "txid"], as_index=False)
        .agg(
            whale_max_output_btc=("value_btc", "max"),
            whale_outputs_count=("value_btc", "size"),
            whale_outputs_sum_btc=("value_btc", "sum")
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df_tx







if __name__ == "__main__":
    N_BLOCKS = 50
    THRESH = 100
    
    RUN_TAG = "part12"

    base_path = Path(f"{Config.RAW_DIR}/_base_height.txt")
    latest = get_latest_block_height()
    print("Using latest:", latest)

    PART_INDEX = int(RUN_TAG.replace("part", ""))
    START_HEIGHT = latest - (PART_INDEX - 1) * N_BLOCKS - (PART_INDEX - 1)

    out_outputs = Path(f"{Config.RAW_DIR}/btc_whale_outputs_{RUN_TAG}_{N_BLOCKS}blocks_{THRESH}btc.parquet")
    out_tx = Path(f"{Config.RAW_DIR}/btc_whale_tx_{RUN_TAG}_{N_BLOCKS}blocks_{THRESH}btc.parquet")
    
    
    if out_outputs.exists() and out_tx.exists():
        print("Cache found, skipping RPC calls.")
        print("Cached files:")
        print("-", out_outputs)
        print("-", out_tx)
    else:
        whales = scan_recent_blocks(n_blocks=N_BLOCKS, whale_threshold_btc=THRESH, start_height=START_HEIGHT)

        df_outputs = whales_to_df(whales)
        df_tx = outputs_to_tx_level(df_outputs)

        print("\n--- BASIC CHECKS ---")
        print("rows (whale outputs):", len(df_outputs))
        print("unique txids (outputs):", df_outputs["txid"].nunique() if not df_outputs.empty else 0)
        print("rows (whale tx):", len(df_tx))
        print("unique txids (tx-level):", df_tx["txid"].nunique() if not df_tx.empty else 0)
        print("time range:", df_tx["timestamp"].min() if not df_tx.empty else None, "->", df_tx["timestamp"].max() if not df_tx.empty else None)

        save_parquet(df_outputs, out_outputs)
        save_parquet(df_tx, out_tx)
        print("\nSaved:")
        print("-", out_outputs)
        print("-", out_tx)