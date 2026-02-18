# src/impact_check.py

from __future__ import annotations
import pandas as pd
import numpy as np

# Eigene Module
from src.config import Config
from src.utils_io import load_parquet

def future_return(df: pd.DataFrame, horizon_h: int) -> pd.Series:
    return df["close"].shift(-horizon_h) / df["close"] - 1

def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 3 or len(b) < 3:
        return None

    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())

    diffs = np.array(diffs)
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    mu = float(diffs.mean())

    return lo, hi, mu


def main():
    path = f"{Config.PROCESSED_DIR}/btc_dataset_10parts_100btc.parquet"
    df = load_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for h in [1, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        df[f"fut_ret_{h}h"] = future_return(df, h)

    spikes = df[df["spike_flag"] == 1].copy()
    non = df[df["spike_flag"] == 0].copy()

    print("Spike hours:", len(spikes), "Non spike hours:", len(non))

   
    for h in [1, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        s_mean = spikes[f"fut_ret_{h}h"].mean()
        n_mean = non[f"fut_ret_{h}h"].mean()

        print(f"\nHorizon {h}h:")
        print("     mean future return on spike:", s_mean)
        print("     mean future return non-spike:", n_mean)

        print("  median future return on spike:", float(spikes[f"fut_ret_{h}h"].median()))
        print("  median future return non-spike:", float(non[f"fut_ret_{h}h"].median()))


        ci = bootstrap_diff(
            spikes[f"fut_ret_{h}h"].to_numpy(),
            non[f"fut_ret_{h}h"].to_numpy()
        )

        if ci:
            lo, hi, mu = ci
            print(f"  bootstrap diff(mean spike - mean non) ≈ {mu:.6f}")
            print(f"  95% CI: [{lo:.6f}, {hi:.6f}]")
        else:
            print("  Not enough data for bootstrap.")



if __name__ == "__main__":
    main()