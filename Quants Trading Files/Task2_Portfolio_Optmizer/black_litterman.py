"""
Black-Litterman Model
=====================
Implements the Black-Litterman (1990) asset allocation model.

Motivation:
    Markowitz optimization is extremely sensitive to expected return inputs.
    Small changes in μ produce wildly different allocations. Black-Litterman
    addresses this by:
    1. Starting from EQUILIBRIUM returns (implied by market-cap weights)
    2. Blending in INVESTOR VIEWS via Bayesian updating
    3. Producing posterior returns that are stable and intuitive

Mathematical Framework:
    Prior (Equilibrium Returns):
        Π = δ Σ w_mkt
        where δ = risk aversion, Σ = covariance, w_mkt = market-cap weights

    Posterior (Blended Returns):
        μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}Π + P'Ω^{-1}Q]

        where:
        P = view pick matrix (which assets the view is about)
        Q = view vector (expected returns from views)
        Ω = view uncertainty matrix
        τ = scalar (uncertainty in prior, typically 0.05)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class BlackLittermanModel:
    """
    Black-Litterman asset allocation model.

    Combines market equilibrium with investor views to produce
    stable, intuitive portfolio allocations.

    Attributes:
        sigma: Covariance matrix (annualized).
        market_caps: Market capitalization of each asset (for equilibrium weights).
        risk_aversion: Market-implied risk aversion parameter (δ).
        tau: Uncertainty scalar for the prior distribution.
        risk_free_rate: Annual risk-free rate.
    """

    def __init__(
        self,
        covariance_matrix: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.04,
    ):
        self.sigma = covariance_matrix
        self.asset_names = list(covariance_matrix.columns)
        self.n_assets = len(self.asset_names)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        # Market-cap weights (default to equal if not provided)
        if market_caps is not None:
            self.w_mkt = (market_caps / market_caps.sum()).values
        else:
            self.w_mkt = np.ones(self.n_assets) / self.n_assets

    def equilibrium_returns(self) -> pd.Series:
        """
        Compute implied equilibrium returns via reverse optimization.

        Formula: Π = δ × Σ × w_mkt

        This extracts the returns that, when plugged into Markowitz,
        would reproduce the current market-cap weighting. The logic:
        "If the market is in equilibrium, what returns does it expect?"

        Returns:
            Series of implied equilibrium returns for each asset.
        """
        pi = self.risk_aversion * self.sigma.values @ self.w_mkt
        pi_series = pd.Series(pi, index=self.asset_names, name="Equilibrium Return")

        logger.info("Equilibrium (implied) returns:\n%s", (pi_series * 100).round(2).to_string())

        return pi_series

    def posterior_returns(
        self,
        views: List[Dict],
    ) -> pd.Series:
        """
        Compute Black-Litterman posterior expected returns.

        Blends equilibrium returns with investor views using Bayesian updating.

        Args:
            views: List of view dictionaries, each with:
                - 'assets': dict mapping asset name → weight in the view
                  (e.g., {'AAPL': 1.0} for absolute view on AAPL,
                   or {'AAPL': 1.0, 'MSFT': -1.0} for relative view)
                - 'return': expected return (annualized decimal)
                - 'confidence': confidence level 0-1 (higher = more certain)

            Example views:
                [
                    {'assets': {'AAPL': 1.0}, 'return': 0.15, 'confidence': 0.8},
                    {'assets': {'NVDA': 1, 'XOM': -1}, 'return': 0.10, 'confidence': 0.6},
                ]

        Returns:
            Series of posterior expected returns.
        """
        pi = self.equilibrium_returns().values
        sigma = self.sigma.values
        k = len(views)

        if k == 0:
            logger.info("No views provided; returning equilibrium returns.")
            return pd.Series(pi, index=self.asset_names)

        # Build P (pick matrix) and Q (view returns)
        P = np.zeros((k, self.n_assets))
        Q = np.zeros(k)
        confidences = np.zeros(k)

        for i, view in enumerate(views):
            Q[i] = view["return"]
            confidences[i] = view.get("confidence", 0.5)
            for asset, weight in view["assets"].items():
                if asset in self.asset_names:
                    j = self.asset_names.index(asset)
                    P[i, j] = weight

        # Ω = diag(uncertainty) — lower confidence → higher variance
        # Ω_ii = (1/confidence - 1) × τ × (P Σ P')_ii
        view_var = np.diag(P @ (self.tau * sigma) @ P.T)
        omega_diag = view_var * (1.0 / confidences - 1.0)
        omega_diag = np.maximum(omega_diag, 1e-10)  # Numerical stability
        omega = np.diag(omega_diag)

        # Black-Litterman master formula
        tau_sigma = self.tau * sigma
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        # Posterior precision = prior precision + view precision
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P

        # Posterior mean = precision^{-1} × (prior precision × prior mean + view precision × view mean)
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mu = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        result = pd.Series(posterior_mu, index=self.asset_names, name="BL Posterior Return")

        logger.info("Black-Litterman posterior returns:\n%s", (result * 100).round(2).to_string())

        return result

    def optimize(
        self,
        views: Optional[List[Dict]] = None,
    ) -> Dict[str, object]:
        """
        Run full Black-Litterman optimization pipeline.

        Steps:
            1. Compute equilibrium returns
            2. Blend with investor views (if any)
            3. Solve for optimal weights using posterior returns

        Args:
            views: Optional list of investor views.

        Returns:
            Dictionary with keys: weights, return, volatility, sharpe_ratio.
        """
        from quantforge.portfolio_optimizer.markowitz import MarkowitzOptimizer

        views = views or []
        posterior_mu = self.posterior_returns(views)

        # Use Markowitz optimizer with BL posterior returns
        optimizer = MarkowitzOptimizer(
            expected_returns=posterior_mu,
            covariance_matrix=self.sigma,
            risk_free_rate=self.risk_free_rate,
        )

        portfolio = optimizer.maximize_sharpe()
        portfolio["posterior_returns"] = posterior_mu

        logger.info(
            "BL Optimal Portfolio — Return: %.2f%%, Vol: %.2f%%, Sharpe: %.3f",
            portfolio["return"] * 100,
            portfolio["volatility"] * 100,
            portfolio["sharpe_ratio"],
        )

        return portfolio
