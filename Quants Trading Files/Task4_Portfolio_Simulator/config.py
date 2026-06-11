"""
Portfolio Simulator — Configuration
=====================================
All module-level constants. Import from here; never hardcode values
in the implementation files.
"""

from typing import Dict, List

# ── Capital ──────────────────────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL: float = 100_000.0      # €100,000 virtual capital

# ── Default asset universe ────────────────────────────────────────────────────
DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "BND"
]

# ── Transaction costs ─────────────────────────────────────────────────────────
TRANSACTION_COST_RATE: float = 0.001            # 0.10% per trade (one-way)

# ── Market conventions ────────────────────────────────────────────────────────
TRADING_DAYS: int = 252
RISK_FREE_RATE: float = 0.04                    # Annual risk-free rate (decimal)

# ── Rebalancing ───────────────────────────────────────────────────────────────
REBALANCE_FREQUENCIES: tuple = ("monthly", "quarterly", "never")
REBALANCE_DRIFT_THRESHOLD: float = 0.05        # Rebalance if weight drifts > 5 pp

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_PATHS: int = 1_000
MC_HORIZONS_MONTHS: List[int] = [6, 12, 24]
MC_PERCENTILES: List[int] = [10, 50, 90]

# ── Persistence ───────────────────────────────────────────────────────────────
DEFAULT_STATE_FILE: str = "portfolio_state.json"
DEFAULT_OUTPUT_DIR: str = "output"

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_TICKER: str = "SPY"

# ── Sector mapping (used by assistant's "exposure to X" command) ──────────────
SECTOR_MAP: Dict[str, str] = {
    "AAPL":    "Technology",
    "MSFT":    "Technology",
    "GOOGL":   "Technology",
    "AMZN":    "Consumer Discretionary",
    "NVDA":    "Technology",
    "META":    "Technology",
    "TSLA":    "Consumer Discretionary",
    "SPY":     "ETF (Broad Market)",
    "QQQ":     "ETF (Tech-Heavy)",
    "BND":     "ETF (Bonds)",
    "GLD":     "Commodities",
    "JPM":     "Financials",
    "BAC":     "Financials",
    "BRK-B":   "Financials",
    "JNJ":     "Healthcare",
    "UNH":     "Healthcare",
    "XOM":     "Energy",
    "CVX":     "Energy",
    "BTC-USD": "Crypto",
    "ETH-USD": "Crypto",
}

# ── Chart colour palette (matches existing visualization.py) ──────────────────
CHART_COLORS: List[str] = [
    "#2962FF",   # blue
    "#FF6D00",   # orange
    "#4CAF50",   # green
    "#9C27B0",   # purple
    "#F44336",   # red
    "#00BCD4",   # cyan
    "#FFC107",   # amber
    "#607D8B",   # blue-grey
]
