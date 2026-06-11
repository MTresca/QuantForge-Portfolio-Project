"""
Monte Carlo Forward Projection
================================
Implements Geometric Brownian Motion (GBM) simulation to project the
portfolio's possible future values over configurable horizons.

Mathematical Framework — GBM:
    dS = μ S dt + σ S dW

    Discretised (Euler–Maruyama, monthly steps):
        S_{t+1} = S_t × exp[(μ - σ²/2) Δt + σ √Δt × ε]
        where ε ~ N(0, 1)  i.i.d.

    The drift μ and volatility σ are estimated from the portfolio's
    actual monthly return history, so the projection is grounded in
    the portfolio's own empirical behaviour rather than generic assumptions.

Financial Context:
    GBM is the canonical stochastic process underlying Black-Scholes.
    While real returns exhibit fat tails and autocorrelation, GBM provides
    a useful first-order estimate of the distribution of future outcomes.
    The 10th/50th/90th percentile fan captures meaningful downside risk
    while showing the median expected trajectory.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.config import (
    MC_N_PATHS,
    MC_HORIZONS_MONTHS,
    MC_PERCENTILES,
    CHART_COLORS,
    RISK_FREE_RATE,
)

logger = get_logger(__name__)


class MonteCarloProjector:
    """
    GBM-based Monte Carlo forward projector for portfolio value.

    Attributes:
        portfolio:  The Portfolio whose history drives μ and σ estimation.
        n_paths:    Number of simulation paths (default 1 000).
        seed:       Optional random seed for reproducibility.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        n_paths: int = MC_N_PATHS,
        seed: Optional[int] = None,
    ):
        self.portfolio = portfolio
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    # ── Parameter estimation ──────────────────────────────────────────────────

    def _estimate_params(self) -> Tuple[float, float, float]:
        """
        Estimate monthly drift and volatility from historical returns.

        Returns:
            Tuple of (mu_monthly, sigma_monthly, current_value).

        Raises:
            ValueError: If there are fewer than 3 months of history.
        """
        returns = self.portfolio.monthly_returns_series()
        if len(returns) < 3:
            raise ValueError(
                f"Need at least 3 months of history for MC projection "
                f"(have {len(returns)})."
            )
        mu_monthly = float(returns.mean())
        sigma_monthly = float(returns.std())
        current_value = self.portfolio.history[-1].portfolio_value
        return mu_monthly, sigma_monthly, current_value

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(
        self,
        horizon_months: int,
    ) -> np.ndarray:
        """
        Run N GBM paths forward for the given horizon.

        Uses the log-normal transition:
            V_{t+1} = V_t × exp[(μ - σ²/2) + σ × ε],  ε ~ N(0,1)

        Args:
            horizon_months: Number of monthly steps to project.

        Returns:
            Array of shape (horizon_months + 1, n_paths) where row 0 is
            the current value and subsequent rows are future steps.
        """
        mu, sigma, v0 = self._estimate_params()

        # GBM log-normal drift adjustment
        drift = mu - 0.5 * sigma ** 2

        # Draw all shocks upfront (shape: horizon × n_paths)
        shocks = self.rng.standard_normal((horizon_months, self.n_paths))

        # Build paths via vectorised cumsum in log-space
        log_increments = drift + sigma * shocks          # (horizon, n_paths)
        log_paths = np.cumsum(log_increments, axis=0)    # cumulative log returns
        paths = v0 * np.exp(
            np.vstack([np.zeros(self.n_paths), log_paths])
        )                                                 # (horizon+1, n_paths)

        logger.info(
            "MC simulation: %d paths × %d months | μ=%.4f σ=%.4f | start €%.2f",
            self.n_paths, horizon_months, mu, sigma, v0,
        )
        return paths

    # ── Summary statistics ────────────────────────────────────────────────────

    def percentile_bands(
        self,
        paths: np.ndarray,
        percentiles: List[int] = MC_PERCENTILES,
    ) -> pd.DataFrame:
        """
        Compute percentile bands across all paths at each time step.

        Args:
            paths:       Array of shape (steps+1, n_paths).
            percentiles: List of percentile values to compute.

        Returns:
            DataFrame indexed 0..horizon, one column per percentile
            (e.g. 'p10', 'p50', 'p90').
        """
        data = {
            f"p{p}": np.percentile(paths, p, axis=1)
            for p in percentiles
        }
        return pd.DataFrame(data)

    def probability_above(
        self,
        paths: np.ndarray,
        target_value: float,
        at_step: int = -1,
    ) -> float:
        """
        Compute P(portfolio_value > target_value) at a given step.

        Args:
            paths:        Paths array from simulate().
            target_value: Threshold value in euros.
            at_step:      Index into the time axis (-1 = final step).

        Returns:
            Probability as a float in [0, 1].
        """
        final = paths[at_step]
        prob = float(np.mean(final > target_value))
        logger.info(
            "P(value > €%.0f) at step %d = %.2f%%",
            target_value, at_step, prob * 100,
        )
        return prob

    def probability_below(
        self,
        paths: np.ndarray,
        floor_value: float,
        at_step: int = -1,
    ) -> float:
        """
        Compute P(portfolio_value < floor_value) at a given step.

        Args:
            paths:       Paths array from simulate().
            floor_value: Floor value in euros.
            at_step:     Index into the time axis (-1 = final step).

        Returns:
            Probability as a float in [0, 1].
        """
        final = paths[at_step]
        prob = float(np.mean(final < floor_value))
        logger.info(
            "P(value < €%.0f) at step %d = %.2f%%",
            floor_value, at_step, prob * 100,
        )
        return prob

    # ── Full multi-horizon run ────────────────────────────────────────────────

    def run_all_horizons(
        self,
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, Dict]:
        """
        Run simulations for multiple horizons and compile a summary.

        Args:
            horizons: List of horizon lengths in months (default: 6, 12, 24).

        Returns:
            Dict keyed by horizon_months, each value containing:
                'paths'     : np.ndarray
                'bands'     : percentile DataFrame
                'final_mean': mean final value
                'final_std' : std of final values
                'p_gain'    : P(value > initial_capital)
                'p_loss_10' : P(value < initial_capital * 0.9)
        """
        horizons = horizons or MC_HORIZONS_MONTHS
        results: Dict[int, Dict] = {}

        for h in horizons:
            paths = self.simulate(h)
            bands = self.percentile_bands(paths)
            final = paths[-1]
            ic = self.portfolio.initial_capital

            results[h] = {
                "paths":      paths,
                "bands":      bands,
                "final_mean": float(np.mean(final)),
                "final_std":  float(np.std(final)),
                "p_gain":     self.probability_above(paths, ic),
                "p_loss_10":  self.probability_below(paths, ic * 0.90),
            }

            logger.info(
                "%dM → Median: €%,.0f | P(profit): %.1f%% | P(loss>10%%): %.1f%%",
                h, bands["p50"].iloc[-1],
                results[h]["p_gain"] * 100,
                results[h]["p_loss_10"] * 100,
            )

        return results

    # ── Visualization ─────────────────────────────────────────────────────────

    def plot(
        self,
        horizons: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Generate an interactive fan chart showing percentile bands.

        Displays one sub-chart per horizon on the same figure.

        Args:
            horizons:  List of horizon lengths in months.
            save_path: Optional HTML output path.

        Returns:
            Plotly Figure.
        """
        horizons = horizons or MC_HORIZONS_MONTHS
        results = self.run_all_horizons(horizons)
        v0 = self.portfolio.history[-1].portfolio_value
        ic = self.portfolio.initial_capital

        fig = go.Figure()
        palette = [CHART_COLORS[0], CHART_COLORS[2], CHART_COLORS[1]]

        for idx, h in enumerate(horizons):
            bands = results[h]["bands"]
            color = palette[idx % len(palette)]
            months_x = list(range(h + 1))
            label = f"{h}M"

            # Shaded band between p10 and p90
            fig.add_trace(
                go.Scatter(
                    x=months_x + months_x[::-1],
                    y=list(bands["p90"]) + list(bands["p10"])[::-1],
                    fill="toself",
                    fillcolor=color.replace("FF", "33") if color.startswith("#") else color,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{label} 10–90th pct",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Median line
            fig.add_trace(
                go.Scatter(
                    x=months_x,
                    y=list(bands["p50"]),
                    mode="lines",
                    name=f"{label} median",
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"<b>{label} +%{{x}}M</b><br>Median: €%{{y:,.0f}}<extra></extra>"
                    ),
                )
            )

        # Initial capital reference
        fig.add_hline(
            y=ic,
            line_dash="dot",
            line_color="grey",
            opacity=0.7,
            annotation_text=f"Initial €{ic:,.0f}",
        )

        fig.update_layout(
            title=dict(
                text=f"Monte Carlo Forward Projection ({self.n_paths:,} paths)",
                font=dict(size=18),
            ),
            xaxis_title="Months Forward",
            yaxis_title="Portfolio Value (€)",
            yaxis_tickprefix="€",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )

        if save_path:
            fig.write_html(save_path)
            logger.info("Monte Carlo chart saved → %s", save_path)

        return fig

    # ── CLI summary ───────────────────────────────────────────────────────────

    def print_summary(
        self,
        horizons: Optional[List[int]] = None,
        target_value: Optional[float] = None,
        floor_value: Optional[float] = None,
    ) -> None:
        """
        Print a formatted Monte Carlo summary to stdout.

        Args:
            horizons:     Horizons to simulate (default: 6, 12, 24 months).
            target_value: Optional target to compute P(value > target).
            floor_value:  Optional floor to compute P(value < floor).
        """
        horizons = horizons or MC_HORIZONS_MONTHS
        results = self.run_all_horizons(horizons)
        ic = self.portfolio.initial_capital
        v0 = self.portfolio.history[-1].portfolio_value

        print("\n" + "═" * 60)
        print("  MONTE CARLO FORWARD PROJECTION")
        print(f"  Paths: {self.n_paths:,}  |  Current value: €{v0:,.2f}")
        print("═" * 60)

        for h, res in results.items():
            bands = res["bands"]
            p10_final = bands["p10"].iloc[-1]
            p50_final = bands["p50"].iloc[-1]
            p90_final = bands["p90"].iloc[-1]
            print(f"\n  Horizon: +{h} months")
            print(f"    10th pct : €{p10_final:>12,.2f}  ({(p10_final/ic - 1)*100:+.2f}%)")
            print(f"    Median   : €{p50_final:>12,.2f}  ({(p50_final/ic - 1)*100:+.2f}%)")
            print(f"    90th pct : €{p90_final:>12,.2f}  ({(p90_final/ic - 1)*100:+.2f}%)")
            print(f"    P(profit): {res['p_gain']*100:.1f}%   |   P(loss>10%): {res['p_loss_10']*100:.1f}%")

            if target_value is not None:
                p = self.probability_above(res["paths"], target_value)
                print(f"    P(> €{target_value:,.0f}): {p*100:.1f}%")
            if floor_value is not None:
                p = self.probability_below(res["paths"], floor_value)
                print(f"    P(< €{floor_value:,.0f}): {p*100:.1f}%")

        print("\n" + "═" * 60 + "\n")
