"""
Portfolio Rebalancer
=====================
Implements periodic rebalancing logic that integrates with the existing
Black-Litterman and Markowitz optimizers from Task 2.

Rebalancing strategy:
  1. Compute current weights from live prices.
  2. Either use fixed target_weights or call the optimizer to suggest new weights.
  3. Calculate required trades (delta shares) to reach the target.
  4. Execute buy/sell orders on the Portfolio, applying transaction costs.
  5. Return total costs incurred so the snapshot can record them.

Financial Context:
    Portfolio drift occurs when asset prices diverge from target weights.
    Rebalancing restores diversification and enforces the intended risk
    profile. The cost of rebalancing must be weighed against the benefit
    of mean-reverting toward the optimal allocation — this is why quarterly
    rebalancing typically outperforms monthly in cost-adjusted terms.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.config import (
    TRANSACTION_COST_RATE,
    REBALANCE_DRIFT_THRESHOLD,
    RISK_FREE_RATE,
    TRADING_DAYS,
)

logger = get_logger(__name__)


class Rebalancer:
    """
    Executes periodic portfolio rebalancing toward target weights.

    Supports two modes:
      • fixed   — rebalance toward the Portfolio's stored target_weights
      • dynamic — call the optimizer each period to re-derive optimal weights

    Attributes:
        mode:             'fixed' | 'dynamic'
        optimizer_mode:   'markowitz' | 'black_litterman' (used when mode='dynamic')
        views:            Optional investor views for Black-Litterman.
        drift_threshold:  Minimum weight drift (fraction) that triggers a trade.
                          Avoids churning for trivial deviations.
        lookback_months:  Months of history used to estimate μ and Σ for
                          optimizer re-runs (default 12).
    """

    def __init__(
        self,
        mode: str = "fixed",
        optimizer_mode: str = "markowitz",
        views: Optional[List[Dict]] = None,
        drift_threshold: float = REBALANCE_DRIFT_THRESHOLD,
        lookback_months: int = 12,
    ):
        if mode not in ("fixed", "dynamic"):
            raise ValueError("mode must be 'fixed' or 'dynamic'")
        if optimizer_mode not in ("markowitz", "black_litterman"):
            raise ValueError("optimizer_mode must be 'markowitz' or 'black_litterman'")

        self.mode = mode
        self.optimizer_mode = optimizer_mode
        self.views = views or []
        self.drift_threshold = drift_threshold
        self.lookback_months = lookback_months

    # ── Main entry point ──────────────────────────────────────────────────────

    def rebalance(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        period_label: str = "",
    ) -> float:
        """
        Rebalance the portfolio toward target weights, returning total costs.

        Steps:
          1. Determine target weights (fixed or optimizer-derived).
          2. Compute current weights and identify which assets need trading.
          3. Execute sells first (to free cash), then buys.
          4. Return cumulative transaction fees.

        Args:
            portfolio:     Portfolio instance to rebalance.
            prices:        Current end-of-period prices for all tickers.
            period_label:  Human-readable label for logging (e.g. '2024-03').

        Returns:
            Total transaction costs in euros.
        """
        if not portfolio.positions and not portfolio.target_weights:
            logger.warning("[%s] No positions or target weights — skipping rebalance.", period_label)
            return 0.0

        # ── Determine target weights ───────────────────────────────────────
        if self.mode == "dynamic" and len(portfolio.history) >= self.lookback_months:
            target_weights = self._optimizer_weights(portfolio, prices)
        else:
            target_weights = dict(portfolio.target_weights)

        if not target_weights:
            logger.warning("[%s] Empty target weights — skipping rebalance.", period_label)
            return 0.0

        # Filter target weights to tickers with available prices
        target_weights = {k: v for k, v in target_weights.items() if k in prices}
        if not target_weights:
            return 0.0

        # Renormalize after filtering
        w_sum = sum(target_weights.values())
        if w_sum > 0:
            target_weights = {k: v / w_sum for k, v in target_weights.items()}

        # Update stored target weights
        portfolio.target_weights = target_weights

        # ── Compute drift ─────────────────────────────────────────────────
        total_value = portfolio.compute_value(prices)
        current_weights = portfolio.compute_weights(prices)
        current_weights_clean = {k: v for k, v in current_weights.items() if k != "_cash"}

        deltas = self._compute_deltas(
            target_weights, current_weights_clean, total_value, prices
        )

        if not any(abs(d) > 1.0 for d in deltas.values()):
            logger.info(
                "[%s] All weights within threshold — no rebalance needed.", period_label
            )
            return 0.0

        logger.info("[%s] Rebalancing portfolio (mode=%s):", period_label, self.mode)
        self._log_weight_diff(target_weights, current_weights_clean, period_label)

        # ── Execute trades: sells first, then buys ────────────────────────
        total_costs = 0.0

        # Sells
        for ticker, delta_shares in deltas.items():
            if delta_shares < -1e-4:
                shares_to_sell = abs(delta_shares)
                proceeds = portfolio.sell(ticker, shares_to_sell, prices[ticker])
                total_costs += proceeds * TRANSACTION_COST_RATE

        # Buys
        for ticker, delta_shares in deltas.items():
            if delta_shares > 1e-4:
                cost = portfolio.buy(ticker, delta_shares, prices[ticker])
                total_costs += cost * TRANSACTION_COST_RATE

        logger.info(
            "[%s] Rebalance complete | costs €%.2f", period_label, total_costs
        )
        return total_costs

    # ── Weight calculation ────────────────────────────────────────────────────

    def _compute_deltas(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_value: float,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute the share delta for each ticker to move from current to target weights.

        Applies the drift_threshold: if the absolute weight difference is below
        the threshold, the ticker is skipped to avoid micro-trades.

        Returns:
            Dict of ticker → delta shares (positive = buy, negative = sell).
        """
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())
        deltas: Dict[str, float] = {}

        for ticker in all_tickers:
            if ticker not in prices or prices[ticker] <= 0:
                continue
            target_w = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            drift = abs(target_w - current_w)

            if drift < self.drift_threshold:
                continue  # Within tolerance — skip this ticker

            target_value = target_w * total_value
            current_value = current_w * total_value
            delta_value = target_value - current_value
            delta_shares = delta_value / prices[ticker]
            deltas[ticker] = delta_shares

        return deltas

    # ── Optimizer integration ─────────────────────────────────────────────────

    def _optimizer_weights(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Derive optimal weights by calling the Task 2 optimizers.

        Estimates annualized expected returns and covariance from the
        portfolio's historical monthly snapshots, then runs either
        Max-Sharpe (Markowitz) or Black-Litterman optimization.

        Falls back to fixed target_weights if optimization fails.

        Args:
            portfolio: Portfolio with at least lookback_months of history.
            prices:    Current prices for the universe.

        Returns:
            Dict of ticker → optimal weight.
        """
        tickers = [t for t in portfolio.positions.keys() if t in prices]
        if len(tickers) < 2:
            logger.warning("Need >= 2 tickers for optimization — using fixed weights.")
            return dict(portfolio.target_weights)

        # Extract per-ticker monthly return series from snapshot history
        returns_data: Dict[str, List[float]] = {t: [] for t in tickers}
        snapshots = portfolio.history[-self.lookback_months:]

        for i in range(1, len(snapshots)):
            prev_snap = snapshots[i - 1]
            curr_snap = snapshots[i]
            for t in tickers:
                prev_price = prev_snap.asset_prices.get(t)
                curr_price = curr_snap.asset_prices.get(t)
                if prev_price and curr_price and prev_price > 0:
                    returns_data[t].append(curr_price / prev_price - 1.0)

        # Build returns DataFrame; drop tickers with insufficient data
        returns_df = pd.DataFrame(returns_data).dropna(axis=1)
        if returns_df.shape[1] < 2 or len(returns_df) < 6:
            logger.warning("Insufficient return history for optimizer — using fixed weights.")
            return dict(portfolio.target_weights)

        # Annualized statistics (monthly → annual)
        mu = returns_df.mean() * 12
        sigma = returns_df.cov() * 12

        try:
            if self.optimizer_mode == "black_litterman":
                weights = self._run_black_litterman(mu, sigma)
            else:
                weights = self._run_markowitz(mu, sigma)

            logger.info("Optimizer suggested weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
            return weights

        except Exception as exc:  # noqa: BLE001
            logger.warning("Optimizer failed (%s) — using fixed weights.", exc)
            return dict(portfolio.target_weights)

    def _run_markowitz(
        self, mu: pd.Series, sigma: pd.DataFrame
    ) -> Dict[str, float]:
        """Call MarkowitzOptimizer.maximize_sharpe() and return weight dict."""
        from quantforge.portfolio_optimizer.markowitz import MarkowitzOptimizer

        opt = MarkowitzOptimizer(
            expected_returns=mu,
            covariance_matrix=sigma,
            risk_free_rate=RISK_FREE_RATE,
        )
        result = opt.maximize_sharpe()
        return result["weights"].to_dict()

    def _run_black_litterman(
        self, mu: pd.Series, sigma: pd.DataFrame
    ) -> Dict[str, float]:
        """Call BlackLittermanModel.optimize() and return weight dict."""
        from quantforge.portfolio_optimizer.black_litterman import BlackLittermanModel

        bl = BlackLittermanModel(
            covariance_matrix=sigma,
            risk_free_rate=RISK_FREE_RATE,
        )
        result = bl.optimize(views=self.views if self.views else None)
        return result["weights"].to_dict()

    # ── Reporting helper ──────────────────────────────────────────────────────

    def _log_weight_diff(
        self,
        target: Dict[str, float],
        current: Dict[str, float],
        period_label: str,
    ) -> None:
        """Log a formatted table of current vs. target weights."""
        all_tickers = sorted(set(target.keys()) | set(current.keys()))
        header = f"{'Ticker':<8} {'Current':>8} {'Target':>8} {'Delta':>8}"
        logger.info("[%s] %s", period_label, header)
        for t in all_tickers:
            c = current.get(t, 0.0) * 100
            tgt = target.get(t, 0.0) * 100
            delta = tgt - c
            sign = "+" if delta >= 0 else ""
            logger.info(
                "[%s]   %-8s %7.2f%% %7.2f%% %s%.2f%%",
                period_label, t, c, tgt, sign, delta,
            )

    def suggest(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Return a recommendation DataFrame without executing any trades.

        Useful for the assistant's 'Should I rebalance?' command.

        Returns:
            DataFrame with columns: ticker, current_weight, target_weight, delta_weight.
        """
        if self.mode == "dynamic" and len(portfolio.history) >= self.lookback_months:
            target = self._optimizer_weights(portfolio, prices)
        else:
            target = dict(portfolio.target_weights)

        current = {
            k: v for k, v in portfolio.compute_weights(prices).items()
            if k != "_cash"
        }

        all_tickers = sorted(set(target.keys()) | set(current.keys()))
        rows = []
        for t in all_tickers:
            c = current.get(t, 0.0)
            tgt = target.get(t, 0.0)
            rows.append(
                {
                    "ticker":         t,
                    "current_weight": round(c * 100, 2),
                    "target_weight":  round(tgt * 100, 2),
                    "delta_weight":   round((tgt - c) * 100, 2),
                    "action":         "BUY" if tgt > c + self.drift_threshold
                                     else ("SELL" if tgt < c - self.drift_threshold else "HOLD"),
                }
            )
        return pd.DataFrame(rows)
