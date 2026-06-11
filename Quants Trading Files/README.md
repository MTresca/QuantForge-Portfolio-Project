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
│   ├── portfolio_simulator/     # Task 4: Portfolio Simulation & Assistant
│   │   ├── __init__.py
│   │   ├── config.py            # All constants and configuration
│   │   ├── portfolio.py         # Portfolio, Position, MonthlySnapshot classes
│   │   ├── simulator.py         # Month-by-month simulation engine
│   │   ├── rebalancer.py        # Periodic rebalancing with BL/Markowitz
│   │   ├── risk_dashboard.py    # Risk metrics panel + Plotly charts
│   │   ├── monte_carlo.py       # GBM forward projection (1 000 paths)
│   │   ├── reporter.py          # Monthly HTML report generator
│   │   ├── assistant.py         # Interactive CLI portfolio assistant
│   │   └── run_simulator.py     # Entry point: simulate + assistant
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Centralized logging configuration
│
├── tests/
│   └── test_portfolio_simulator.py
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

# Task 4: Run portfolio simulation (2022–today, quarterly rebalance)
python -m quantforge.portfolio_simulator.run_simulator simulate \
    --start 2022-01-01 \
    --tickers AAPL MSFT SPY QQQ BND \
    --weights 0.25 0.20 0.25 0.15 0.15 \
    --rebalance quarterly \
    --assistant

# Task 4: Launch assistant on a saved portfolio
python -m quantforge.portfolio_simulator.run_simulator assistant \
    --state portfolio_state.json
```

---

## Task 4 — Portfolio Simulator & Assistant

### Overview

Simulates a real **€100,000 virtual portfolio** over any historical window
and provides an intelligent CLI assistant for interactive analysis.

### Features

| Feature | Description |
|---------|-------------|
| **Portfolio State** | Persistent JSON state (positions, cash, P&L, history) |
| **Monthly Simulation** | Step month-by-month with real yfinance OHLCV prices |
| **Rebalancing** | Monthly/quarterly rebalancing using Max-Sharpe or Black-Litterman |
| **Risk Dashboard** | Sharpe, Sortino, Max Drawdown, VaR, CVaR, rolling metrics |
| **Monte Carlo** | GBM forward projection (1 000 paths, 6M/12M/24M horizons) |
| **HTML Reports** | Self-contained monthly report with embedded Plotly charts |
| **CLI Assistant** | Natural-language CLI for portfolio Q&A and chart generation |

### CLI Assistant Commands

```
how is my portfolio          — current risk dashboard
worst month / best month     — historical extremes
should i rebalance?          — BL/Markowitz rebalancing recommendation
show me my drawdown chart    — equity + drawdown interactive chart
show allocation              — asset allocation pie chart
what is my exposure to tech? — sector breakdown
run sentiment analysis       — FinBERT on current holdings
simulate next 6 months       — Monte Carlo GBM fan chart
generate report              — monthly HTML report
help                         — list all commands
```

### Theoretical Background

**Monthly Simulation Engine:**
The simulator fetches end-of-month adjusted close prices and uses them
to mark the portfolio to market at each period. End-of-month pricing
mirrors standard fund NAV reporting and smooths intra-month noise.

**GBM Monte Carlo:**
Forward projections use the log-normal GBM model:
`V_{t+1} = V_t × exp[(μ - σ²/2) + σ × ε]`
where μ and σ are estimated from the portfolio's own monthly return history.
The 10th/50th/90th percentile fan quantifies the range of plausible outcomes.

**Dynamic Rebalancing:**
When `--rebalance-mode dynamic` is selected, the rebalancer estimates
expected returns and covariances from the trailing 12 months of snapshots,
then calls either `MarkowitzOptimizer.maximize_sharpe()` or
`BlackLittermanModel.optimize()` to derive new target weights each period.

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for full dependency list

## License

MIT License — see `LICENSE` for details.
