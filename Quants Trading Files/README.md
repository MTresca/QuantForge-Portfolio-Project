# QuantForge-Toolkit 🏗️📈

A professional-grade **Quantitative Finance** toolkit implementing algorithmic trading, portfolio optimization, and financial sentiment analysis.

---

## Architecture

```
quantforge-toolkit/
├── quantforge/
│   ├── __init__.py
│   ├── trading_engine/          # Task 1: Algorithmic Trading (Mean Reversion + ML)
│   │   ├── __init__.py
│   │   ├── data_sourcing.py     # yfinance data fetcher with caching
│   │   ├── feature_engineering.py # Bollinger Bands, RSI, log returns, ML targets
│   │   ├── ml_filter.py         # Random Forest classifier for signal filtering
│   │   ├── strategy.py          # Core strategy logic (BB + ML filter)
│   │   ├── backtester.py        # VectorBT-based backtesting engine
│   │   └── run_backtest.py      # Entry point: end-to-end pipeline
│   │
│   ├── portfolio_optimizer/     # Task 2: Portfolio Optimization (MPT + Black-Litterman)
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Multi-asset data retrieval
│   │   ├── markowitz.py         # Mean-Variance Optimization (Markowitz)
│   │   ├── black_litterman.py   # Black-Litterman model
│   │   ├── risk_metrics.py      # Sharpe, Sortino, Max Drawdown, CVaR
│   │   ├── visualization.py     # Plotly efficient frontier & weight charts
│   │   └── run_optimizer.py     # Entry point: compare EW vs Max-Sharpe vs BL
│   │
│   ├── sentiment_analyzer/      # Task 3: FinBERT Sentiment Analysis
│   │   ├── __init__.py
│   │   ├── finbert_model.py     # HuggingFace FinBERT wrapper
│   │   ├── headline_scraper.py  # Financial headline data loader
│   │   ├── correlation.py       # Sentiment-to-price correlation engine
│   │   └── run_sentiment.py     # Entry point: analyze & correlate
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Centralized logging configuration
│
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

---

## Theoretical Background

### Task 1 — Mean Reversion with ML Signal Filtering

**Mean Reversion** posits that asset prices tend to revert to a long-term mean. Bollinger Bands quantify this by plotting ±2σ envelopes around a 20-period SMA. When price touches the lower band, the asset is considered "oversold."

**The Problem:** Raw Bollinger Band signals generate significant false positives in trending markets. A stock crashing through the lower band in a downtrend is *not* mean-reverting — it's falling.

**Our Solution:** We train a `RandomForestClassifier` on engineered features (RSI, log returns, volatility regime, volume Z-score) to predict whether a lower-band touch will actually result in a positive return over the next N days. The ML model acts as a **filter**, suppressing entries during adverse regimes.

**Backtesting:** We use VectorBT for vectorized backtesting, which is orders of magnitude faster than event-driven frameworks for this type of strategy.

### Task 2 — Modern Portfolio Theory & Black-Litterman

**Markowitz (1952)** showed that optimal portfolios lie on an "efficient frontier" in risk-return space. The key insight: diversification reduces portfolio variance below the weighted average of individual variances when correlations < 1.

**The Problem:** Markowitz optimization is notoriously sensitive to expected return estimates, producing extreme allocations from small input changes.

**Black-Litterman (1990)** solves this by starting from **market equilibrium returns** (implied by market-cap weights via reverse optimization) and blending in investor "views" using Bayesian updating. This produces far more stable, intuitive allocations.

We compare three strategies: Equal Weight (1/N), Maximum Sharpe Ratio (Markowitz), and Black-Litterman.

### Task 3 — Financial Sentiment via FinBERT

**FinBERT** (ProsusAI) is a BERT model fine-tuned on 10K+ financial articles. Unlike generic sentiment models, it understands domain-specific language: "the stock tanked" → Negative; "revenue beat expectations" → Positive.

We map FinBERT's three-class output (Positive/Negative/Neutral) to a continuous score [-1, +1] and correlate it with subsequent price movements to measure **sentiment alpha**.

---

## Risk Metrics

All modules compute standard quantitative risk metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (Rp - Rf) / σp | Risk-adjusted return per unit of total volatility |
| **Sortino Ratio** | (Rp - Rf) / σ_downside | Like Sharpe but only penalizes downside volatility |
| **Max Drawdown** | max(peak - trough) / peak | Worst peak-to-trough loss |
| **CVaR (ES)** | E[R \| R < VaR_α] | Expected loss in the worst α% of scenarios |
| **Win Rate** | #winning_trades / #total_trades | Percentage of profitable trades |
| **Calmar Ratio** | Ann. Return / Max Drawdown | Return per unit of tail risk |

---

## Installation

```bash
git clone https://github.com/your-username/quantforge-toolkit.git
cd quantforge-toolkit
pip install -r requirements.txt
```

## Quick Start

```bash
# Task 1: Run the ML-filtered mean reversion backtest
python -m quantforge.trading_engine.run_backtest

# Task 2: Run portfolio optimization
python -m quantforge.portfolio_optimizer.run_optimizer

# Task 3: Run sentiment analysis
python -m quantforge.sentiment_analyzer.run_sentiment
```

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for full dependency list

## License

MIT License — see `LICENSE` for details.
