"""
Task 4 — Portfolio Simulator & Assistant
=========================================
Simulates a €100,000 virtual portfolio over time, providing:
- Month-by-month P&L tracking with real market data
- Periodic rebalancing via Black-Litterman / Max-Sharpe optimization
- Risk dashboard (Sharpe, Sortino, Max Drawdown, VaR, CVaR)
- Monte Carlo GBM forward projection
- Interactive CLI assistant
- Monthly HTML report generation
"""

from quantforge.portfolio_simulator.portfolio import Portfolio, Position, MonthlySnapshot
from quantforge.portfolio_simulator.simulator import PortfolioSimulator

__all__ = [
    "Portfolio",
    "Position",
    "MonthlySnapshot",
    "PortfolioSimulator",
]
