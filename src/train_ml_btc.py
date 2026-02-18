# src/train_ml_btc.py

from __future__ import annotations
import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Eigene Module
from src.config import Config
from src.utils_io import load_parquet

HORIZON_H = 5

def make_target(df: pd.DataFrame, horizon_h: int) -> pd.Series:
    fut_ret = df["close"].shift(-horizon_h) / df["close"] - 1
    return (fut_ret > 0).astype(int), fut_ret

def simple_backtest_spike(df: pd.DataFrame, horizon_h: int) -> None:
    fut_ret = df["close"].shift(-horizon_h) / df["close"] - 1

    spike = df["spike_flag"] == 1
    strat_ret = np.where(spike, fut_ret, 0.0)

    n_trades = int(spike.sum())
    avg_trade = float(np.nanmean(fut_ret[spike])) if n_trades > 0 else 0.0
    hitrate = float(np.nanmean(fut_ret[spike] > 0).astype(float)) if n_trades > 0 else 0.0
    total_ret = float(np.nansum(strat_ret))

    print("\n--- SIMPLE STRATEGY BACKTEST ---")
    print(f"Horizon: {horizon_h}h | Trades (spikes): {n_trades}")
    print(f"Avg trade return: {avg_trade:.4%}")
    print(f"Hitrate: {hitrate:.2%}")
    print(f"Sum of returns (not compounded): {total_ret:.4%}")

def main():
    path = f"{Config.PROCESSED_DIR}/btc_dataset_10parts_100btc.parquet"
    df = load_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.dropna(subset=["ma_24"]).copy()

    y, fut_ret = make_target(df, HORIZON_H)
    df["fut_ret"] = fut_ret
    df["y_up"] = y

    feature_cols = [
        "whale_tx_count",
        "whale_sum_btc",
        "whale_max_btc",
        "whale_mean_btc",
        "whale_sum_3h",
        "spike_flag",
        "spike_cluster",
        "vol_24h",
        "trend_regime",
        "vol_regime"
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in dataset: {missing}")
    
    X = df[feature_cols].astype(float).copy()
    y = df["y_up"].astype(int).copy()

    valid = df["fut_ret"].notna()
    X = X[valid]
    y = y[valid]
    df_valid = df.loc[valid].copy()

    print("Rows for ML:", len(df_valid))
    print("Time range:", df_valid["timestamp"].min(), "->", df_valid["timestamp"].max())
    print("Positive class rate:", float(y.mean()))

    baseline_pred = (y.mean() >= 0.5)
    baseline_acc = float((y == int(baseline_pred)).mean())
    print("Baseline (always predict majority class) acc:", baseline_acc)

    tscv = TimeSeriesSplit(n_splits=5)
    accs, f1s = [], []

    fold = 0
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        accs.append(acc)
        f1s.append(f1)


        print(f"\nFold {fold}: acc={acc:.3f} f1={f1:.3f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    print("\n--- CV SUMMARY ---")
    print("Mean acc:", float(np.mean(accs)))
    print("Mean f1 :", float(np.mean(f1s)))



    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    final_model.fit(X, y)

    coefs = pd.Series(final_model.named_steps["clf"].coef_[0], index=feature_cols)\
                        .sort_values(key=np.abs, ascending=False)

    print("\n--- FINAL MODEL (trained on all data) ---")
    print("Top coefficients (by abs value):")
    print(coefs.head(10))

    simple_backtest_spike(df_valid, HORIZON_H)


    out_json = Path(f"{Config.PROCESSED_DIR}/ml_results_btc.json")

    results = {
        "rows_ml": int(len(df_valid)),
        "time_min": str(df_valid["timestamp"].min()),
        "time_max": str(df_valid["timestamp"].max()),
        "positive_class_rate": float(y.mean()),
        "cv_mean_acc": float(np.mean(accs)),
        "cv_mean_f1": float(np.mean(f1s)),
        "fold_acc": [float(a) for a in accs],
        "fold_f1": [float(f) for f in f1s],
        "top_coeffs": coefs.head(10).to_dict(),
        "horizon_h": int(HORIZON_H),
    }

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved ML results ->", str(out_json))

if __name__ == "__main__":
    main()