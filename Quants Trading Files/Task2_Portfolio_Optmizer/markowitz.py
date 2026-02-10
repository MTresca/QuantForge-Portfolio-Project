"""
Markowitz Mean-Variance Optimization
=====================================
Implements Harry Markowitz's (1952) Modern Portfolio Theory.

Core Concept:
    An investor can construct portfolios that maximize expected return
    for a given level of risk (or minimize risk for a given return).
    The set of all such optimal portfolios forms the "Efficient Frontier."

Mathematical Formulation:
    minimize:  w' Σ w                  (portfolio variance)
    subject to: w' μ = target_return   (return constraint)
                w' 1 = 1               (fully invested)
                w ≥ 0                  (long only, no short selling)

    Where:
        w = vector of portfolio weights
        Σ = covariance matrix of asset returns
        μ = vector of expected returns
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimizer (Markowitz, 1952).

    Attributes:
        mu: Expected return vector (annualized).
        sigma: Covariance matrix (annualized).
        n_assets: Number of assets in the universe.
        risk_free_rate: Annual risk-free rate for Sharpe computation.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ):
        self.mu = expected_returns.values
        self.sigma = covariance_matrix.values
        self.asset_names = list(expected_returns.index)
        self.n_assets = len(self.mu)
        self.risk_free_rate = risk_free_rate

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Expected annual return: w' μ."""
        return float(weights @ self.mu)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Portfolio standard deviation: sqrt(w' Σ w)."""
        return float(np.sqrt(weights @ self.sigma @ weights))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Sharpe Ratio: (Return - Rf) / Volatility."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0.0

    def maximize_sharpe(self) -> Dict[str, object]:
        """
        Find the Maximum Sharpe Ratio portfolio (Tangency Portfolio).

        The tangency portfolio is the point on the efficient frontier that
        maximizes risk-adjusted return. It represents the optimal risky
        portfolio that an investor should hold (combined with the risk-free
        asset for their desired risk level).

        Returns:
            Dictionary with keys: weights, return, volatility, sharpe_ratio.
        """
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))  # Long only
        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            fun=lambda w: -self._portfolio_sharpe(w),  # Minimize negative Sharpe
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("Optimization did not converge: %s", result.message)

        weights = result.x
        weights = weights / weights.sum()  # Re-normalize

        portfolio = {
            "weights": pd.Series(weights, index=self.asset_names),
            "return": self._portfolio_return(weights),
            "volatility": self._portfolio_volatility(weights),
            "sharpe_ratio": self._portfolio_sharpe(weights),
        }

        logger.info(
            "Max Sharpe Portfolio — Return: %.2f%%, Vol: %.2f%%, Sharpe: %.3f",
            portfolio["return"] * 100,
            portfolio["volatility"] * 100,
            portfolio["sharpe_ratio"],
        )

        return portfolio

    def minimum_variance(self) -> Dict[str, object]:
        """
        Find the Minimum Variance Portfolio (leftmost point on the frontier).

        This portfolio has the lowest possible risk among all feasible
        portfolios. Useful for risk-averse investors.

        Returns:
            Dictionary with keys: weights, return, volatility, sharpe_ratio.
        """
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            fun=self._portfolio_volatility,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x
        weights = weights / weights.sum()

        return {
            "weights": pd.Series(weights, index=self.asset_names),
            "return": self._portfolio_return(weights),
            "volatility": self._portfolio_volatility(weights),
            "sharpe_ratio": self._portfolio_sharpe(weights),
        }

    def equal_weight(self) -> Dict[str, object]:
        """
        Equal Weight (1/N) portfolio — naive diversification benchmark.

        Despite its simplicity, the 1/N portfolio often outperforms
        optimized portfolios out-of-sample (DeMiguel et al., 2009)
        because it avoids estimation error in μ and Σ.

        Returns:
            Dictionary with keys: weights, return, volatility, sharpe_ratio.
        """
        weights = np.ones(self.n_assets) / self.n_assets

        return {
            "weights": pd.Series(weights, index=self.asset_names),
            "return": self._portfolio_return(weights),
            "volatility": self._portfolio_volatility(weights),
            "sharpe_ratio": self._portfolio_sharpe(weights),
        }

    def compute_efficient_frontier(
        self, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Trace the efficient frontier by solving for minimum variance
        at each target return level.

        Args:
            n_points: Number of points to compute along the frontier.

        Returns:
            Tuple of (volatilities, returns, weights_list).
        """
        # Get return range from min-variance to max-return
        min_ret = self.minimum_variance()["return"]
        max_ret = float(np.max(self.mu))
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_vols = []
        frontier_rets = []
        frontier_weights = []

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: self._portfolio_return(w) - t},
            ]
            bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
            initial = np.ones(self.n_assets) / self.n_assets

            result = minimize(
                fun=self._portfolio_volatility,
                x0=initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                frontier_vols.append(self._portfolio_volatility(result.x))
                frontier_rets.append(target)
                frontier_weights.append(result.x)

        logger.info("Computed efficient frontier with %d points", len(frontier_vols))

        return np.array(frontier_vols), np.array(frontier_rets), frontier_weights
