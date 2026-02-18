# app.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json
from pathlib import Path

DATA_PATH = "data/processed/btc_dataset_10parts_100btc.parquet"
ML_PATH = "data/processed/ml_results_btc.json"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data
def load_ml_results(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def add_future_return(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    out = df.copy()
    out["fut_ret"] = out["close"].shift(-horizon_h) / out["close"] - 1
    out["fut_ret"] = out["fut_ret"].replace([np.inf, -np.inf], np.nan)
    return out


def strategy_metrics(df: pd.DataFrame, horizon_h: int) -> dict:
    fut_ret = df["close"].shift(-horizon_h) / df["close"] - 1
    fut_ret = fut_ret.replace([np.inf, -np.inf], np.nan)

    signal = df["spike_flag"] == 1
    trades = fut_ret[signal].dropna()

    n_trades = int(signal.sum())
    avg_trade = float(trades.mean()) if len(trades) else np.nan
    hitrate = float((trades > 0).mean()) if len(trades) else np.nan
    total_ret = float(np.nansum(np.where(signal, fut_ret, 0.0)))

    
    mean_spike = float(trades.mean()) if len(trades) else np.nan
    mean_non = float(fut_ret[~signal].dropna().mean()) if len(fut_ret[~signal].dropna()) else np.nan
    diff = mean_spike - mean_non if (np.isfinite(mean_spike) and np.isfinite(mean_non)) else np.nan

    return {
        "h": horizon_h,
        "trades": n_trades,
        "avg_trade": avg_trade,
        "hitrate": hitrate,
        "sum_ret": total_ret,
        "mean_spike": mean_spike,
        "mean_non": mean_non,
        "diff_mean": diff,
    }


def plot_price_with_spikes(df: pd.DataFrame, n_last: int = 250):
    d = df.tail(n_last).copy()

    fig, ax = plt.subplots()
    ax.plot(d["timestamp"], d["close"])

    spikes = d[d["spike_flag"] == 1]
    for t in spikes["timestamp"]:
        ax.axvline(t, linestyle="--", linewidth=1)

    ax.set_title("BTC close (last window) with whale spikes")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close price")
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_sumret_vs_horizon(cmp: pd.DataFrame):
    
    d = cmp.sort_values("h").copy()

    fig, ax = plt.subplots()
    ax.plot(d["h"], d["sum_ret"] * 100, marker="o")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Strategy total return vs horizon (spike → long)")
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("Sum of returns, % (not compounded)")
    st.pyplot(fig)



def main():
    st.set_page_config(page_title="WhaleFlow BTC", layout="wide")
    st.title("WhaleFlow: BTC whale spikes vs price (MVP)")

    st.write("Data source: Binance OHLCV + Bitcoin on-chain whale aggregation (hourly)")

    with st.sidebar:
        st.header("Controls")
        horizons = st.multiselect(
            "Horizons (hours) to compare",
            [1, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            default=[4, 5, 6, 7, 10, 12],
        )
        horizon_main = st.selectbox("Main horizon for table/chart", horizons, index=horizons.index(5) if 5 in horizons else 0)
        last_n = st.slider("Show last N hours on chart", 100, 721, 250, step=25)

    df = load_data(DATA_PATH)

    st.subheader("Horizon comparison (strategy + simple impact)")
    rows = [strategy_metrics(df, h) for h in horizons]
    cmp = pd.DataFrame(rows).sort_values("h")

    cmp_show = cmp.copy()
    cmp_show["avg_trade_%"] = cmp_show["avg_trade"] * 100
    cmp_show["sum_ret_%"] = cmp_show["sum_ret"] * 100
    cmp_show["hitrate_%"] = cmp_show["hitrate"] * 100
    cmp_show["mean_spike_%"] = cmp_show["mean_spike"] * 100
    cmp_show["mean_non_%"] = cmp_show["mean_non"] * 100
    cmp_show["diff_mean_%"] = cmp_show["diff_mean"] * 100

    st.dataframe(
        cmp_show[["h","trades","avg_trade_%","hitrate_%","sum_ret_%","mean_spike_%","mean_non_%","diff_mean_%"]],
        use_container_width=True
    )

    st.subheader("Sum of returns vs horizon")
    plot_sumret_vs_horizon(cmp)


    df_main = add_future_return(df, horizon_h=horizon_main)

    n_spikes = int((df_main["spike_flag"] == 1).sum())
    st.subheader("Quick stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_main)}")
    c2.metric("Spike hours", f"{n_spikes}")
    c3.metric("Time range", f"{df_main['timestamp'].min()} -> {df_main['timestamp'].max()}")
    c4.metric("Horizon", f"{horizon_main}h")


    st.subheader("ML results (saved from training)")
    ml = load_ml_results(ML_PATH)

    if not ml:
        st.warning("ML results file not found")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CV mean acc", f"{ml.get('cv_mean_acc', 0):.3f}")
        c2.metric("CV mean F1", f"{ml.get('cv_mean_f1', 0):.3f}")
        c3.metric("Class + rate", f"{ml.get('positive_class_rate', 0)*100:.1f}%")
        c4.metric("Horizon", f"{ml.get('horizon_h', '')}h")

        st.caption(f"ML rows: {ml.get('rows_ml')} | {ml.get('time_min')} -> {ml.get('time_max')}")

        st.write("Fold scores:")
        st.write({"acc": ml.get("fold_acc", []), "f1": ml.get("fold_f1", [])})

        st.write("Top coefficients (LogReg):")
        st.dataframe(pd.DataFrame(
            {"feature": list(ml.get("top_coeffs", {}).keys()),
             "coef": list(ml.get("top_coeffs", {}).values())}
        ).sort_values("coef", key=lambda s: s.abs(), ascending=False), use_container_width=True)
    


    st.subheader("Price chart with spike markers")
    plot_price_with_spikes(df_main, n_last=last_n)

    st.subheader("Top spikes (by whale_sum_btc)")
    spikes = df_main[df_main["spike_flag"] == 1].copy()
    cols = [
        "timestamp",
        "close",
        "whale_tx_count",
        "whale_sum_btc",
        "whale_sum_3h",
        "spike_cluster",
        "trend_regime",
        "vol_regime",
        "fut_ret"
    ]
    spikes_view = spikes[cols].sort_values("whale_sum_btc", ascending=False).head(15)
    st.dataframe(spikes_view, use_container_width=True)

    st.subheader("Simple strategy summary (spike -> long for horizon)")
    fut_ret = df_main["fut_ret"]
    signal = (df_main["spike_flag"] == 1)
    valid_trades = fut_ret[signal].dropna()

    if len(valid_trades) == 0:
        st.warning("No valid spike trades in the selected horizon window")
    else:
        avg_trade = float(valid_trades.min())
        hitrate = float((valid_trades > 0).mean())
        total_ret = float(np.nansum(np.where(signal, fut_ret, 0.0)))

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", f"{int(signal.sum())}")
        c2.metric("Avg trade return", f"{avg_trade*100:.2f}%")
        c3.metric("Hitrate", f"{hitrate:.1%}")

        st.caption(f"Sum of returns (not compounded): {total_ret*100:.2f}%")

    st.divider()
    st.caption("MVP UI. Next: add ML probability signal + more history")


if __name__ == "__main__":
    main()