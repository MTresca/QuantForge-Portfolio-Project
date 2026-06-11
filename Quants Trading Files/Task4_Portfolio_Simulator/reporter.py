"""
Monthly Portfolio Reporter
===========================
Generates a self-contained HTML report combining:
  • Portfolio summary table
  • Cumulative return chart vs SPY benchmark
  • Asset allocation pie chart
  • Risk metrics table
  • Top 3 rebalancing actions taken

All charts are embedded as interactive Plotly HTML so the report
requires no external assets.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.risk_dashboard import RiskDashboard
from quantforge.portfolio_simulator.config import (
    DEFAULT_OUTPUT_DIR,
    CHART_COLORS,
    BENCHMARK_TICKER,
)

logger = get_logger(__name__)


class ReportGenerator:
    """
    Produces a monthly HTML performance report for the simulated portfolio.

    Attributes:
        portfolio:   The Portfolio instance to report on.
        dashboard:   RiskDashboard used for metric computation and charts.
        output_dir:  Directory where the HTML file will be written.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.portfolio = portfolio
        self.dashboard = RiskDashboard(portfolio)
        self.output_dir = output_dir

    # ── Report generation ─────────────────────────────────────────────────────

    def generate(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate and save a full HTML report.

        Args:
            benchmark_returns: Monthly SPY returns for the benchmark overlay.
            filename:          Output filename (default: report_YYYY-MM.html).

        Returns:
            Absolute path to the generated HTML file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.portfolio.history:
            raise RuntimeError("No simulation history — run the simulator before generating a report.")

        snap = self.portfolio.history[-1]
        period = snap.date[:7]   # YYYY-MM
        filename = filename or f"report_{period}.html"
        output_path = os.path.join(self.output_dir, filename)

        metrics = self.dashboard.compute_metrics()
        history_df = self.portfolio.history_to_dataframe()

        html = self._build_html(
            snap=snap,
            metrics=metrics,
            history_df=history_df,
            benchmark_returns=benchmark_returns,
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("Report saved → %s", output_path)
        return output_path

    # ── HTML assembly ─────────────────────────────────────────────────────────

    def _build_html(
        self,
        snap,
        metrics: Dict[str, float],
        history_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
    ) -> str:
        """Assemble the complete HTML string."""
        period = snap.date[:7]
        title = f"QuantForge Portfolio Report — {period}"

        # Generate chart HTML fragments
        equity_fig = self.dashboard.plot_equity_and_drawdown(benchmark_returns)
        equity_html = equity_fig.to_html(full_html=False, include_plotlyjs=False)

        pie_fig = self.dashboard.plot_allocation_pie()
        pie_html = pie_fig.to_html(full_html=False, include_plotlyjs=False)

        summary_table = self._summary_table(snap)
        metrics_table = self._metrics_table(metrics)
        rebalance_table = self._rebalance_table(history_df)
        sector_table = self._sector_table()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f7fb; color: #1a1a2e; margin: 0; padding: 0; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: #2962FF; margin: 28px 0 10px; }}
    .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 10px; padding: 20px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                 margin-bottom: 20px; }}
    .kpi {{ background: white; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
    .kpi-label {{ font-size: 0.75rem; color: #888; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
    .positive {{ color: #4CAF50; }}
    .negative {{ color: #F44336; }}
    .neutral  {{ color: #1a1a2e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ background: #f0f4ff; padding: 8px 12px; text-align: left;
          font-weight: 600; color: #2962FF; border-bottom: 2px solid #dde4f0; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #eef0f5; }}
    tr:last-child td {{ border-bottom: none; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    footer {{ text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 32px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>QuantForge Portfolio Report</h1>
  <p class="subtitle">Period: <b>{period}</b> &nbsp;|&nbsp;
     Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b> &nbsp;|&nbsp;
     Inception: <b>{self.portfolio.inception_date}</b></p>

  <!-- KPI strip -->
  {self._kpi_strip(snap)}

  <!-- Summary table -->
  <h2>Portfolio Summary</h2>
  <div class="card">{summary_table}</div>

  <!-- Equity chart -->
  <h2>Cumulative Performance vs {BENCHMARK_TICKER}</h2>
  <div class="card">{equity_html}</div>

  <!-- Risk metrics + allocation side by side -->
  <div class="two-col">
    <div>
      <h2>Risk Metrics</h2>
      <div class="card">{metrics_table}</div>
    </div>
    <div>
      <h2>Sector Exposure</h2>
      <div class="card">{sector_table}</div>
    </div>
  </div>

  <!-- Allocation pie + rebalancing table -->
  <div class="two-col">
    <div>
      <h2>Asset Allocation</h2>
      <div class="card">{pie_html}</div>
    </div>
    <div>
      <h2>Top Rebalancing Actions</h2>
      <div class="card">{rebalance_table}</div>
    </div>
  </div>

  <footer>
    Generated by QuantForge Portfolio Simulator &nbsp;•&nbsp;
    Data sourced from Yahoo Finance via yfinance
  </footer>
</div>
</body>
</html>"""

    # ── HTML fragments ────────────────────────────────────────────────────────

    def _kpi_strip(self, snap) -> str:
        """Render the 4-column KPI strip at the top of the report."""
        value = snap.portfolio_value
        ic = self.portfolio.initial_capital
        cumul = snap.cumulative_return_pct
        monthly = snap.monthly_return_pct
        dd = snap.drawdown_pct

        def fmt_kpi(label: str, value_str: str, cls: str) -> str:
            return (
                f'<div class="kpi">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {cls}">{value_str}</div>'
                f"</div>"
            )

        cumul_cls = "positive" if cumul >= 0 else "negative"
        monthly_cls = "positive" if monthly >= 0 else "negative"
        dd_cls = "negative" if dd < -1 else "neutral"

        sign = "+" if cumul >= 0 else ""
        m_sign = "+" if monthly >= 0 else ""

        return (
            '<div class="kpi-grid">'
            + fmt_kpi("Portfolio Value", f"€{value:,.0f}", "neutral")
            + fmt_kpi("Total P&L", f"{sign}{cumul:.2f}%", cumul_cls)
            + fmt_kpi("Monthly Return", f"{m_sign}{monthly:.2f}%", monthly_cls)
            + fmt_kpi("Max Drawdown", f"{dd:.2f}%", dd_cls)
            + "</div>"
        )

    def _summary_table(self, snap) -> str:
        """Render an HTML table with the current portfolio summary."""
        rows = [
            ("Portfolio Value", f"€{snap.portfolio_value:,.2f}"),
            ("Initial Capital", f"€{self.portfolio.initial_capital:,.2f}"),
            ("Unrealized P&L", f"€{snap.unrealized_pnl:,.2f}"),
            ("Realized P&L", f"€{snap.realized_pnl:,.2f}"),
            ("Cash Balance", f"€{snap.cash:,.2f}"),
            ("Monthly Return", f"{snap.monthly_return_pct:+.2f}%"),
            ("Cumulative Return", f"{snap.cumulative_return_pct:+.2f}%"),
            ("Drawdown from Peak", f"{snap.drawdown_pct:.2f}%"),
            ("Positions", str(len(self.portfolio.positions))),
        ]
        trs = "".join(
            f"<tr><td><b>{label}</b></td><td>{val}</td></tr>" for label, val in rows
        )
        return f"<table><tbody>{trs}</tbody></table>"

    def _metrics_table(self, metrics: Dict[str, float]) -> str:
        """Render an HTML table of risk metrics."""
        if not metrics:
            return "<p>Not enough history for risk metrics.</p>"
        display = [
            ("Annualized Return",    f"{metrics.get('annualized_return_pct', 0):+.2f}%"),
            ("Annualized Volatility", f"{metrics.get('annualized_volatility_pct', 0):.2f}%"),
            ("Sharpe Ratio",         f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ("Sortino Ratio",        f"{metrics.get('sortino_ratio', 0):.3f}"),
            ("Max Drawdown",         f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ("Calmar Ratio",         f"{metrics.get('calmar_ratio', 0):.3f}"),
            ("VaR 95% (monthly)",    f"{metrics.get('var_95_pct', 0):.2f}%"),
            ("CVaR 95% (monthly)",   f"{metrics.get('cvar_95_pct', 0):.2f}%"),
        ]
        trs = "".join(
            f"<tr><td><b>{label}</b></td><td>{val}</td></tr>" for label, val in display
        )
        return f"<table><tbody>{trs}</tbody></table>"

    def _sector_table(self) -> str:
        """Render the sector exposure table."""
        df = self.dashboard.sector_exposure()
        if df.empty:
            return "<p>No sector data available.</p>"
        header = "<tr><th>Sector</th><th>Weight</th></tr>"
        rows = "".join(
            f"<tr><td>{row['sector']}</td><td>{row['weight_pct']:.2f}%</td></tr>"
            for _, row in df.iterrows()
        )
        return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"

    def _rebalance_table(self, history_df: pd.DataFrame) -> str:
        """
        Find the 3 months with the highest transaction costs and list them
        as 'rebalancing actions taken'.
        """
        if history_df.empty or "transaction_costs" not in history_df.columns:
            return "<p>No rebalancing data.</p>"

        top = (
            history_df[history_df["transaction_costs"] > 0]
            .nlargest(3, "transaction_costs")
            .reset_index()
        )

        if top.empty:
            return "<p>No rebalancing actions recorded.</p>"

        header = "<tr><th>Period</th><th>Transaction Cost</th><th>Portfolio Value</th></tr>"
        rows = "".join(
            f"<tr>"
            f"<td>{row['date'].strftime('%b %Y')}</td>"
            f"<td>€{row['transaction_costs']:,.2f}</td>"
            f"<td>€{row['portfolio_value']:,.2f}</td>"
            f"</tr>"
            for _, row in top.iterrows()
        )
        return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"
