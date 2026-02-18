# WhaleFlow — BTC Whale Activity vs Price (ML + Strategy)

Simple research project exploring how **Bitcoin whale activity** affects short-term price movement.

The project combines:

- On-chain whale transaction aggregation  
- Market data (Binance OHLCV)  
- Statistical impact analysis  
- Machine Learning (Logistic Regression)  
- Simple trading strategy backtest  
- Streamlit visualization  

---

## Idea

Large Bitcoin transfers ("whales") often precede market moves.  
This project tests:

> Does whale activity statistically precede price growth?

And:

> Can we build a simple predictive model and trading signal from it?

---

## Data

Sources:

- **Binance OHLCV** — BTCUSDT 1h candles  
- **Bitcoin blockchain** — whale transactions ≥ 100 BTC  

Aggregated into hourly features.

---

## Features

Main engineered features:

- `whale_tx_count` — number of whale transactions per hour  
- `whale_sum_btc` — total BTC moved by whales  
- `whale_max_btc` — largest whale transfer  
- `whale_mean_btc` — average whale size  
- `whale_sum_3h` — rolling whale activity (3h window)  
- `spike_flag` — whale activity spike (top 10%)  
- `spike_cluster` — cluster of recent spikes  
- `vol_24h` — rolling volatility  
- `trend_regime` — market trend state  
- `vol_regime` — volatility regime  

---

## Project Structure

src/
├── data_price.py # Download BTC/SOL price data
├── features_whales_hourly.py # Build whale features
├── build_dataset_btc.py # Merge price + whale data
├── impact_check.py # Statistical spike impact test
├── train_ml_btc.py # ML model training + evaluation
├── app.py # Streamlit dashboard
├── utils_io.py
└── config.py

data/
├── raw/
└── processed/


---

## Installation

```bash
pip install -r requirements.txt


Main dependencies:

pandas

numpy

scikit-learn

matplotlib

streamlit

pyarrow

Pipeline

Run step by step:

python -m src.data_price
python -m src.features_whales_hourly
python -m src.build_dataset_btc
python -m src.impact_check
python -m src.train_ml_btc


Run dashboard:

streamlit run src/app.py

Strategy

Simple rule:

If whale spike → open long → hold N hours


Tested horizons: 1–12 hours

Results (example)

Best region observed:

Horizon	Avg Trade	Hitrate	Sum Return
4h	~0.59%	62%	~4.7%
5h	~0.58%	62%	~4.6%
7h	~0.55%	75%	~4.4%

Statistical test:

Mean return after spikes > non-spikes

Confidence intervals often positive

Indicates weak but real signal

Machine Learning

Model: Logistic Regression

Target:

Will price be higher after N hours?


Evaluation:

TimeSeriesSplit (5 folds)

Accuracy ≈ 0.45–0.52

F1 ≈ 0.42–0.52

Weak predictive signal (expected for markets)

Most important features typically:

whale_sum_3h

volatility

whale_tx_count

spike_flag

Visualization

Streamlit app shows:

Price chart with whale spikes

Top whale events

Strategy performance

Horizon comparison

ML metrics

Feature importance

Interpretation

The project shows:

Whale activity correlates with short-term market moves

Strongest effect appears within 4–7 hours

Signal is weak but non-random

ML alone is insufficient, but helps explain behavior

This is a research / exploratory project, not a trading system.

Future Improvements

Detect whale direction (exchange inflow/outflow)

Add orderbook / funding rate

Use XGBoost / RandomForest

Add probabilistic trading filter (ML + spike)

Increase dataset size

Walk-forward backtest

Real-time pipeline

Author

Lidiia Petrovska
AI / Data Science / Blockchain

Disclaimer

This project is for research and educational purposes only.
Not financial advice.