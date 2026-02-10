"""
Portfolio Visualization Module
==============================
Interactive Plotly charts for portfolio optimization analysis.
"""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


def plot_efficient_frontier(
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    portfolios: Dict[str, Dict],
    save_path: str = "efficient_frontier.html",
) -> None:
    """
    Plot the Markowitz efficient frontier with key portfolio markers.

    Args:
        frontier_vols: Array of portfolio volatilities along the frontier.
        frontier_rets: Array of portfolio returns along the frontier.
        portfolios: Dict of named portfolios, each with 'return' and 'volatility'.
        save_path: Output file path for the HTML chart.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier_vols * 100,
            y=frontier_rets * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#2962FF", width=3),
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    colors = {
        "Equal Weight": "#FF6D00",
        "Max Sharpe": "#4CAF50",
        "Black-Litterman": "#9C27B0",
        "Min Variance": "#00BCD4",
    }

    for name, pf in portfolios.items():
        color = colors.get(name, "#666")
        fig.add_trace(
            go.Scatter(
                x=[pf["volatility"] * 100],
                y=[pf["return"] * 100],
                mode="markers+text",
                name=name,
                marker=dict(size=14, color=color, symbol="star"),
                text=[name],
                textposition="top center",
                textfont=dict(size=11, color=color),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Return: {pf['return'] * 100:.2f}%<br>"
                    f"Volatility: {pf['volatility'] * 100:.2f}%<br>"
                    f"Sharpe: {pf['sharpe_ratio']:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Efficient Frontier & Portfolio Comparison", font=dict(size=18)),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(save_path)
    logger.info("Efficient frontier chart saved to %s", save_path)


def plot_weight_comparison(
    portfolios: Dict[str, Dict],
    save_path: str = "weight_comparison.html",
) -> None:
    """
    Plot side-by-side weight distribution for each portfolio strategy.

    Args:
        portfolios: Dict of named portfolios, each with 'weights' (pd.Series).
        save_path: Output file path.
    """
    fig = go.Figure()
    colors = ["#2962FF", "#FF6D00", "#4CAF50", "#9C27B0"]

    for i, (name, pf) in enumerate(portfolios.items()):
        weights = pf["weights"]
        significant = weights[weights > 0.01].sort_values(ascending=True)

        fig.add_trace(
            go.Bar(
                y=significant.index,
                x=significant.values * 100,
                name=name,
                orientation="h",
                marker_color=colors[i % len(colors)],
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Portfolio Weight Allocation Comparison", font=dict(size=18)),
        xaxis_title="Weight (%)",
        yaxis_title="Asset",
        barmode="group",
        template="plotly_white",
        height=500,
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.write_html(save_path)
    logger.info("Weight comparison chart saved to %s", save_path)


def plot_portfolio_performance(
    cumulative_returns: Dict[str, pd.Series],
    save_path: str = "portfolio_performance.html",
) -> None:
    """
    Plot cumulative return curves for multiple portfolio strategies.

    Args:
        cumulative_returns: Dict mapping strategy name -> cumulative return series.
        save_path: Output file path.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Returns", "Rolling 60-Day Sharpe Ratio"),
        row_heights=[0.6, 0.4],
    )

    colors = ["#2962FF", "#FF6D00", "#4CAF50", "#9C27B0"]

    for i, (name, cum_ret) in enumerate(cumulative_returns.items()):
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=(cum_ret - 1) * 100,
                name=name,
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )

        daily_ret = cum_ret.pct_change().dropna()
        rolling_sharpe = (
            daily_ret.rolling(60).mean() / daily_ret.rolling(60).std()
        ) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name=f"{name} (Sharpe)",
                line=dict(color=color, width=1.5),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=dict(text="Portfolio Strategy Comparison", font=dict(size=18)),
        template="plotly_white",
        height=700,
        width=900,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Rolling Sharpe", row=2, col=1)

    fig.write_html(save_path)
    logger.info("Performance chart saved to %s", save_path)
