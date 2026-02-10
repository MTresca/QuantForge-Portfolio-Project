"""
Run Portfolio Optimizer — Entry Point
=====================================
End-to-end pipeline comparing Equal Weight, Maximum Sharpe (Markowitz),
and Black-Litterman portfolio strategies.

Usage:
    python -m quantforge.portfolio_optimizer.run_optimizer
"""

import os

import numpy as np
import pandas as pd

from quantforge.portfolio_optimizer.black_litterman import BlackLittermanModel
from quantforge.portfolio_optimizer.data_loader import PortfolioDataLoader
from quantforge.portfolio_optimizer.markowitz import MarkowitzOptimizer
from quantforge.portfolio_optimizer.risk_metrics import compare_portfolios
from quantforge.portfolio_optimizer.visualization import (
    plot_efficient_frontier,
    plot_portfolio_performance,
    plot_weight_comparison,
)
from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


def main(output_dir: str = "output") -> None:
    """
    Execute the portfolio optimization pipeline.

    Pipeline:
        1. Load multi-asset price data
        2. Compute return statistics (μ, Σ)
        3. Optimize: Equal Weight, Max Sharpe, Black-Litterman
        4. Compute efficient frontier
        5. Compare performance metrics
        6. Generate interactive charts

    Args:
        output_dir: Directory for saving output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QuantForge Portfolio Optimizer — Starting")
    logger.info("=" * 60)

    # Step 1: Load Data
    logger.info("[1/5] Loading multi-asset data...")
    loader = PortfolioDataLoader(years=5)
    prices, returns = loader.load()
    mu, sigma = loader.compute_statistics(returns)

    # Step 2: Markowitz Optimization
    logger.info("[2/5] Running Markowitz optimization...")
    markowitz = MarkowitzOptimizer(
        expected_returns=mu,
        covariance_matrix=sigma,
        risk_free_rate=0.04,
    )

    pf_equal = markowitz.equal_weight()
    pf_max_sharpe = markowitz.maximize_sharpe()
    pf_min_var = markowitz.minimum_variance()

    # Step 3: Black-Litterman
    logger.info("[3/5] Running Black-Litterman model...")

    # Example investor views (customize as needed):
    # View 1: NVDA will outperform XOM by 10% (bullish on AI vs energy)
    # View 2: Tech sector (AAPL+MSFT) will return 12% absolute
    views = [
        {"assets": {"NVDA": 1.0, "XOM": -1.0}, "return": 0.10, "confidence": 0.7},
        {"assets": {"AAPL": 0.5, "MSFT": 0.5}, "return": 0.12, "confidence": 0.6},
    ]

    # Filter views to only include assets present in our data
    valid_assets = set(returns.columns)
    filtered_views = []
    for view in views:
        if all(a in valid_assets for a in view["assets"]):
            filtered_views.append(view)

    bl_model = BlackLittermanModel(
        covariance_matrix=sigma,
        risk_aversion=2.5,
        tau=0.05,
    )
    pf_bl = bl_model.optimize(views=filtered_views)

    # Collect all portfolios
    portfolios = {
        "Equal Weight": pf_equal,
        "Max Sharpe": pf_max_sharpe,
        "Black-Litterman": pf_bl,
    }

    # Step 4: Compute Efficient Frontier
    logger.info("[4/5] Computing efficient frontier...")
    frontier_vols, frontier_rets, _ = markowitz.compute_efficient_frontier(n_points=100)

    # Step 5: Backtest & Compare
    logger.info("[5/5] Computing portfolio performance...")

    cumulative_returns = {}
    portfolio_daily_returns = {}

    for name, pf in portfolios.items():
        weights = pf["weights"]
        # Portfolio daily return = weighted sum of individual asset returns
        pf_daily = (returns[weights.index] * weights.values).sum(axis=1)
        pf_cumulative = (1 + pf_daily).cumprod()

        cumulative_returns[name] = pf_cumulative
        portfolio_daily_returns[name] = pf_daily

    # Risk metrics comparison
    comparison = compare_portfolios(portfolio_daily_returns, risk_free_rate=0.04)
    comparison_path = os.path.join(output_dir, "portfolio_comparison.csv")
    comparison.to_csv(comparison_path)
    logger.info("Comparison table saved to %s", comparison_path)

    # Generate Charts
    plot_efficient_frontier(
        frontier_vols,
        frontier_rets,
        portfolios,
        save_path=os.path.join(output_dir, "efficient_frontier.html"),
    )

    plot_weight_comparison(
        portfolios,
        save_path=os.path.join(output_dir, "weight_comparison.html"),
    )

    plot_portfolio_performance(
        cumulative_returns,
        save_path=os.path.join(output_dir, "portfolio_performance.html"),
    )

    # Print summary table
    logger.info("=" * 60)
    logger.info("PORTFOLIO COMPARISON SUMMARY")
    logger.info("=" * 60)
    for name, pf in portfolios.items():
        logger.info(
            "%-20s | Return: %+6.2f%% | Vol: %5.2f%% | Sharpe: %.3f",
            name,
            pf["return"] * 100,
            pf["volatility"] * 100,
            pf["sharpe_ratio"],
        )

        # Top 3 holdings
        top_3 = pf["weights"].nlargest(3)
        holdings_str = ", ".join(f"{k}: {v:.1%}" for k, v in top_3.items())
        logger.info("  Top 3 holdings: %s", holdings_str)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
