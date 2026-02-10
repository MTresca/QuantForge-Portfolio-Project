"""
Risk Metrics Module
===================
Comprehensive suite of quantitative risk metrics for portfolio analysis.
All metrics follow CFA Institute / GIPS standards.
"""

from typing import Dict

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def compute_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    name: str = "Portfolio",
) -> Dict[str, float]:
    """
    Compute a full suite of risk-adjusted performance metrics.

    Args:
        returns: Series of daily log or simple returns.
        risk_free_rate: Annual risk-free rate (decimal).
        name: Portfolio name for logging.

    Returns:
        Dictionary of named metrics.
    """
    returns = returns.dropna()
    n = len(returns)

    if n < 2:
        logger.warning("Insufficient data for %s (%d observations)", name, n)
        return {}

    # Cumulative return
    cum_return = (1 + returns).cumprod()
    total_return = cum_return.iloc[-1] - 1.0

    # Annualized return (geometric)
    ann_return = (1 + total_return) ** (TRADING_DAYS / n) - 1

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    # Maximum Drawdown
    peak = cum_return.cummax()
    drawdown = (cum_return - peak) / peak
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5)

    # Conditional VaR / Expected Shortfall (95%)
    cvar_95 = returns[returns <= var_95].mean()

    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    metrics = {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility_pct": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calmar,
        "var_95_pct": var_95 * 100,
        "cvar_95_pct": cvar_95 * 100 if not np.isnan(cvar_95) else 0.0,
        "skewness": skew,
        "kurtosis": kurt,
    }

    logger.info("Risk Metrics for %s:", name)
    for key, val in metrics.items():
        logger.info("  %-28s: %+.4f", key, val)

    return metrics


def compare_portfolios(
    portfolio_returns: Dict[str, pd.Series],
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Compare multiple portfolios side-by-side.

    Args:
        portfolio_returns: Dict mapping portfolio name → daily return series.
        risk_free_rate: Annual risk-free rate.

    Returns:
        DataFrame with portfolios as columns and metrics as rows.
    """
    results = {}
    for name, returns in portfolio_returns.items():
        results[name] = compute_portfolio_metrics(returns, risk_free_rate, name)

    comparison = pd.DataFrame(results)
    logger.info("Portfolio Comparison:\n%s", comparison.round(4).to_string())

    return comparison
