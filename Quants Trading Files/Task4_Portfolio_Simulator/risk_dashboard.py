"""
Risk Dashboard
===============
Computes and displays the full risk metrics panel for the portfolio at any
point in the simulation. Also generates the interactive Plotly equity and
drawdown chart used in the assistant and reporter.

Reuses compute_portfolio_metrics() from Task 2 to avoid duplicating formulas.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantforge.utils.logger import get_logger
from quantforge.portfolio_optimizer.risk_metrics import compute_portfolio_metrics
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.config import (
    CHART_COLORS,
    BENCHMARK_TICKER,
    RISK_FREE_RATE,
    SECTOR_MAP,
)

logger = get_logger(__name__)

# Rolling window for Sharpe calculation
ROLLING_SHARPE_WINDOW = 12   # months


class RiskDashboard:
    """
    Computes, prints, and charts the portfolio risk snapshot.

    Financial Context:
        A dashboard snapshot captures the portfolio's current state relative
        to its history. Rolling Sharpe uses a 12-month trailing window so
        it reflects recent rather than since-inception risk-adjusted performance.

    Attributes:
        portfolio: The Portfolio instance to analyse.
    """

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    # ── Core metrics ──────────────────────────────────────────────────────────

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute the full risk metric suite from the portfolio's monthly history.

        Delegates to the existing compute_portfolio_metrics() for consistency
        with the Task 2 optimizer output format.

        Returns:
            Dict of named metrics (same keys as Task 2 risk_metrics.py).
        """
        returns = self.portfolio.monthly_returns_series()
        if len(returns) < 2:
            logger.warning("Not enough history to compute metrics (need ≥ 2 months).")
            return {}
        return compute_portfolio_metrics(
            returns, risk_free_rate=RISK_FREE_RATE, name="PortfolioSimulator"
        )

    def rolling_sharpe(self, window: int = ROLLING_SHARPE_WINDOW) -> pd.Series:
        """
        Compute a rolling Sharpe ratio using monthly returns.

        Args:
            window: Look-back window in months.

        Returns:
            Series of rolling Sharpe values indexed by date.
        """
        returns = self.portfolio.monthly_returns_series()
        if len(returns) < window:
            return pd.Series(dtype=float)

        rf_monthly = RISK_FREE_RATE / 12
        excess = returns - rf_monthly
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
        sharpe.name = f"Sharpe ({window}M rolling)"
        return sharpe.dropna()

    # ── Display ───────────────────────────────────────────────────────────────

    def print_dashboard(
        self,
        prices: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Print a formatted risk dashboard to stdout.

        Args:
            prices: If provided, computes live position values; otherwise
                    the last snapshot's data is used.
        """
        if not self.portfolio.history:
            print("No simulation history yet — run the simulator first.")
            return

        snap = self.portfolio.history[-1]
        metrics = self.compute_metrics()

        # ── Header ─────────────────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  PORTFOLIO RISK DASHBOARD")
        print(f"  Inception: {self.portfolio.inception_date}  |  "
              f"Last update: {snap.date}")
        print("═" * 60)

        # ── Value summary ───────────────────────────────────────────────────
        value = snap.portfolio_value
        pnl = value - self.portfolio.initial_capital
        sign = "+" if pnl >= 0 else ""
        print(f"\n  Portfolio Value     : €{value:>12,.2f}")
        print(f"  Initial Capital     : €{self.portfolio.initial_capital:>12,.2f}")
        print(f"  Total P&L           : {sign}€{pnl:>11,.2f}  ({sign}{snap.cumulative_return_pct:.2f}%)")
        print(f"  Unrealized P&L      : €{snap.unrealized_pnl:>12,.2f}")
        print(f"  Realized P&L        : €{snap.realized_pnl:>12,.2f}")
        print(f"  Cash                : €{snap.cash:>12,.2f}")
        print(f"  Monthly Return      : {'+' if snap.monthly_return_pct >= 0 else ''}"
              f"{snap.monthly_return_pct:.2f}%")
        print(f"  Cumulative Return   : {'+' if snap.cumulative_return_pct >= 0 else ''}"
              f"{snap.cumulative_return_pct:.2f}%")
        print(f"  Drawdown from peak  : {snap.drawdown_pct:.2f}%")

        # ── Risk metrics ────────────────────────────────────────────────────
        if metrics:
            print("\n  ── Risk Metrics ─────────────────────────────────────────")
            print(f"  Annualized Volatility: {metrics.get('annualized_volatility_pct', 0):.2f}%")
            print(f"  Sharpe Ratio (all)   : {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Sortino Ratio        : {metrics.get('sortino_ratio', 0):.3f}")
            print(f"  Max Drawdown         : {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"  Calmar Ratio         : {metrics.get('calmar_ratio', 0):.3f}")
            print(f"  VaR 95% (monthly)    : {metrics.get('var_95_pct', 0):.2f}%")
            print(f"  CVaR 95% (monthly)   : {metrics.get('cvar_95_pct', 0):.2f}%")

        # ── Rolling Sharpe ──────────────────────────────────────────────────
        rs = self.rolling_sharpe()
        if not rs.empty:
            print(f"\n  Rolling Sharpe (12M) : {rs.iloc[-1]:.3f}")

        # ── Allocation ──────────────────────────────────────────────────────
        weights = snap.weights
        if weights:
            print("\n  ── Current Allocation ───────────────────────────────────")
            target = self.portfolio.target_weights
            for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
                t_w = target.get(ticker, 0.0)
                drift = (w - t_w) * 100
                drift_str = f"  (drift {'+' if drift >= 0 else ''}{drift:.1f}pp)" if abs(drift) > 0.1 else ""
                print(f"    {ticker:<8} {w * 100:>6.2f}%  (target {t_w * 100:.2f}%){drift_str}")

        # ── Best / worst ────────────────────────────────────────────────────
        if snap.asset_prices and self.portfolio.positions:
            bw = self.portfolio.best_worst_asset(snap.asset_prices)
            if bw:
                b = bw["best"]
                w_asset = bw["worst"]
                print(
                    f"\n  Best asset   : {b['ticker']} ({'+' if b['pnl_pct'] >= 0 else ''}{b['pnl_pct']:.2f}%)"
                )
                print(
                    f"  Worst asset  : {w_asset['ticker']} ({'+' if w_asset['pnl_pct'] >= 0 else ''}{w_asset['pnl_pct']:.2f}%)"
                )

        print("\n" + "═" * 60 + "\n")

    # ── Charts ────────────────────────────────────────────────────────────────

    def plot_equity_and_drawdown(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Generate a 3-panel Plotly chart: equity curve, drawdown, monthly returns.

        Args:
            benchmark_returns: Optional SPY monthly return series for comparison.
            save_path:         If provided, save the HTML chart to this path.

        Returns:
            Plotly Figure object.
        """
        history_df = self.portfolio.history_to_dataframe()
        if history_df.empty:
            logger.warning("No history to plot.")
            return go.Figure()

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=[
                "Portfolio Value (€)",
                "Drawdown from Peak (%)",
                "Monthly Return (%)",
            ],
            vertical_spacing=0.07,
        )

        dates = history_df.index

        # ── Panel 1: Portfolio equity curve ───────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history_df["portfolio_value"],
                mode="lines",
                name="Portfolio",
                line=dict(color=CHART_COLORS[0], width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>€%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

        # Benchmark overlay (rebased to initial capital)
        if benchmark_returns is not None and not benchmark_returns.empty:
            initial = self.portfolio.initial_capital
            bench_curve = (1 + benchmark_returns).cumprod() * initial
            # Align index
            bench_curve = bench_curve.reindex(dates, method="nearest")
            fig.add_trace(
                go.Scatter(
                    x=bench_curve.index,
                    y=bench_curve.values,
                    mode="lines",
                    name=f"{BENCHMARK_TICKER} (rebased)",
                    line=dict(color=CHART_COLORS[1], width=2, dash="dash"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>€%{y:,.0f}<extra></extra>",
                ),
                row=1, col=1,
            )

        # Initial capital reference line
        fig.add_hline(
            y=self.portfolio.initial_capital,
            line_dash="dot",
            line_color="grey",
            opacity=0.6,
            row=1, col=1,
        )

        # ── Panel 2: Drawdown ─────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history_df["drawdown_pct"],
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color=CHART_COLORS[4], width=1),
                fillcolor="rgba(244,67,54,0.15)",
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.2f}%<extra></extra>",
            ),
            row=2, col=1,
        )

        # ── Panel 3: Monthly returns bar chart ────────────────────────────
        colors_bar = [
            CHART_COLORS[2] if r >= 0 else CHART_COLORS[4]
            for r in history_df["monthly_return_pct"]
        ]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=history_df["monthly_return_pct"],
                name="Monthly Return",
                marker_color=colors_bar,
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.2f}%<extra></extra>",
            ),
            row=3, col=1,
        )

        # ── Layout ────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text="Portfolio Performance Dashboard",
                font=dict(size=20),
            ),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=750,
        )
        fig.update_yaxes(title_text="Value (€)", row=1, col=1, tickprefix="€")
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            logger.info("Equity chart saved → %s", save_path)

        return fig

    def plot_allocation_pie(
        self,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Generate a pie chart of the current portfolio allocation.

        Args:
            save_path: Optional path to save HTML.

        Returns:
            Plotly Figure.
        """
        if not self.portfolio.history:
            return go.Figure()

        snap = self.portfolio.history[-1]
        weights = {k: v for k, v in snap.weights.items() if k != "_cash" and v > 0.001}
        if snap.cash / snap.portfolio_value > 0.001:
            weights["Cash"] = snap.cash / snap.portfolio_value

        fig = go.Figure(
            go.Pie(
                labels=list(weights.keys()),
                values=[v * 100 for v in weights.values()],
                hole=0.4,
                marker=dict(colors=CHART_COLORS[: len(weights)]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Asset Allocation",
            template="plotly_white",
            height=450,
        )

        if save_path:
            fig.write_html(save_path)
            logger.info("Allocation chart saved → %s", save_path)

        return fig

    # ── Sector exposure ───────────────────────────────────────────────────────

    def sector_exposure(self) -> pd.DataFrame:
        """
        Group current weights by sector using the SECTOR_MAP.

        Returns:
            DataFrame with columns: sector, weight_pct.
        """
        if not self.portfolio.history:
            return pd.DataFrame()

        weights = {
            k: v for k, v in self.portfolio.history[-1].weights.items()
            if k != "_cash"
        }
        sector_weights: Dict[str, float] = {}
        for ticker, w in weights.items():
            sector = SECTOR_MAP.get(ticker, "Other")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w

        df = pd.DataFrame(
            [
                {"sector": s, "weight_pct": round(w * 100, 2)}
                for s, w in sorted(sector_weights.items(), key=lambda x: -x[1])
            ]
        )
        return df

    # ── Monthly worst/best lookup ─────────────────────────────────────────────

    def worst_month(self) -> Optional[Dict]:
        """Return the snapshot with the lowest monthly return."""
        if not self.portfolio.history:
            return None
        snap = min(self.portfolio.history, key=lambda s: s.monthly_return_pct)
        return snap.to_dict()

    def best_month(self) -> Optional[Dict]:
        """Return the snapshot with the highest monthly return."""
        if not self.portfolio.history:
            return None
        snap = max(self.portfolio.history, key=lambda s: s.monthly_return_pct)
        return snap.to_dict()
