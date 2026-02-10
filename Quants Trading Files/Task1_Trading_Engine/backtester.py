"""
Backtesting Engine
==================
Vectorized backtesting engine that simulates trade execution and computes
comprehensive performance metrics.

Uses a vectorized approach (similar to VectorBT) for speed and scalability.
All metrics are calculated using standard quantitative finance formulas.

Performance Metrics:
    - Annualized Return: Geometric mean return scaled to 252 trading days
    - Sharpe Ratio: Risk-adjusted return (excess return / volatility)
    - Sortino Ratio: Downside-risk-adjusted return
    - Maximum Drawdown: Worst peak-to-trough decline
    - Win Rate: Percentage of profitable trades
    - Calmar Ratio: Return / Max Drawdown
"""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)

# Annualization factor: ~252 trading days per year
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate (approximate T-bill yield)


class StrategyBacktester:
    """
    Vectorized backtesting engine with comprehensive performance analytics.

    Financial Context:
        Backtesting simulates how a strategy would have performed historically.
        Key assumptions:
        - Execution at close price (no slippage model — conservative)
        - No transaction costs (can be added via cost_per_trade parameter)
        - Full capital deployment per trade (no fractional sizing)

    Attributes:
        signals_df: DataFrame with signals and positions from the strategy.
        initial_capital: Starting portfolio value.
        cost_per_trade: Transaction cost per trade (as decimal).
    """

    def __init__(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_per_trade: float = 0.001,
    ):
        """
        Initialize the backtester.

        Args:
            signals_df: DataFrame from strategy.generate_signals() with
                        columns: Close, signal, position.
            initial_capital: Starting capital in USD.
            cost_per_trade: Cost per trade as fraction (0.001 = 10 bps).
        """
        self.signals_df = signals_df.copy()
        self.initial_capital = initial_capital
        self.cost_per_trade = cost_per_trade
        self.results: Dict[str, float] = {}

    def run(self) -> pd.DataFrame:
        """
        Execute the vectorized backtest.

        Computation:
            1. Strategy returns = daily returns × position (1 if long, 0 if flat)
            2. Subtract transaction costs on signal changes
            3. Cumulative equity curve = initial_capital × cumprod(1 + strategy_returns)
            4. Benchmark = buy-and-hold from day 1

        Returns:
            DataFrame with columns: strategy_equity, benchmark_equity, daily_returns.
        """
        df = self.signals_df

        # Daily simple returns
        df["daily_return"] = df["Close"].pct_change().fillna(0)

        # Strategy returns: earn market return only when in position
        df["strategy_return"] = df["daily_return"] * df["position"].shift(1).fillna(0)

        # Transaction costs: deduct on position changes
        position_changes = df["position"].diff().abs().fillna(0)
        df["strategy_return"] -= position_changes * self.cost_per_trade

        # Equity curves
        df["strategy_equity"] = self.initial_capital * (
            1 + df["strategy_return"]
        ).cumprod()
        df["benchmark_equity"] = self.initial_capital * (
            1 + df["daily_return"]
        ).cumprod()

        # Compute all performance metrics
        self.results = self._compute_metrics(df)
        self._log_results()

        return df

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute comprehensive risk-adjusted performance metrics.

        All formulas follow CFA Institute / GIPS standards.
        """
        strategy_returns = df["strategy_return"].dropna()
        benchmark_returns = df["daily_return"].dropna()

        # --- Annualized Return ---
        # Geometric: (1 + total_return)^(252/n) - 1
        total_return = df["strategy_equity"].iloc[-1] / self.initial_capital - 1
        n_days = len(strategy_returns)
        ann_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1

        # Benchmark annualized return
        bm_total = df["benchmark_equity"].iloc[-1] / self.initial_capital - 1
        bm_ann_return = (1 + bm_total) ** (TRADING_DAYS / n_days) - 1

        # --- Volatility ---
        ann_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS)

        # --- Sharpe Ratio ---
        # Sharpe = (Ann. Return - Risk-Free Rate) / Ann. Volatility
        sharpe = (
            (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
        )

        # --- Sortino Ratio ---
        # Uses only downside deviation (negative returns)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS) if len(downside_returns) > 0 else 0.0
        sortino = (
            (ann_return - RISK_FREE_RATE) / downside_std
            if downside_std > 0
            else 0.0
        )

        # --- Maximum Drawdown ---
        equity = df["strategy_equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # --- Calmar Ratio ---
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # --- Win Rate ---
        # Identify individual trades (groups of consecutive position=1 days)
        trades = self._extract_trades(df)
        win_rate = (
            sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0
        )

        # --- Number of Trades ---
        n_trades = len(trades)

        return {
            "total_return_pct": total_return * 100,
            "annualized_return_pct": ann_return * 100,
            "benchmark_return_pct": bm_ann_return * 100,
            "annualized_volatility_pct": ann_vol * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown * 100,
            "calmar_ratio": calmar,
            "win_rate_pct": win_rate * 100,
            "n_trades": n_trades,
            "total_days": n_days,
        }

    def _extract_trades(self, df: pd.DataFrame) -> list:
        """
        Extract individual trade P&L from the position column.

        A "trade" starts when position changes from 0→1 and ends at 1→0.
        Returns a list of per-trade returns.
        """
        trades = []
        in_trade = False
        entry_price = 0.0

        for i in range(1, len(df)):
            pos = df["position"].iloc[i]
            prev_pos = df["position"].iloc[i - 1]

            if pos == 1 and prev_pos == 0:
                # Trade entry
                in_trade = True
                entry_price = df["Close"].iloc[i]
            elif pos == 0 and prev_pos == 1 and in_trade:
                # Trade exit
                exit_price = df["Close"].iloc[i]
                trade_return = (exit_price / entry_price) - 1
                trades.append(trade_return)
                in_trade = False

        return trades

    def _log_results(self) -> None:
        """Log performance metrics in a formatted table."""
        r = self.results
        logger.info("=" * 60)
        logger.info("BACKTEST PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info("Total Return:           %+.2f%%", r["total_return_pct"])
        logger.info("Annualized Return:      %+.2f%%", r["annualized_return_pct"])
        logger.info("Benchmark (B&H) Return: %+.2f%%", r["benchmark_return_pct"])
        logger.info("Annualized Volatility:   %.2f%%", r["annualized_volatility_pct"])
        logger.info("Sharpe Ratio:            %.3f", r["sharpe_ratio"])
        logger.info("Sortino Ratio:           %.3f", r["sortino_ratio"])
        logger.info("Max Drawdown:           %.2f%%", r["max_drawdown_pct"])
        logger.info("Calmar Ratio:            %.3f", r["calmar_ratio"])
        logger.info("Win Rate:               %.1f%%", r["win_rate_pct"])
        logger.info("Number of Trades:        %d", r["n_trades"])
        logger.info("=" * 60)

    def plot_results(self, df: pd.DataFrame, save_path: str = "backtest_results.html") -> None:
        """
        Generate interactive Plotly charts for backtest analysis.

        Creates a 3-panel figure:
            1. Equity Curve: Strategy vs Benchmark (Buy & Hold)
            2. Drawdown: Rolling peak-to-trough decline
            3. Bollinger Bands: Price with bands and buy/sell markers

        Args:
            df: Backtest results DataFrame.
            save_path: File path for saving the HTML chart.
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Equity Curve: Strategy vs Benchmark",
                "Drawdown",
                "Price with Bollinger Bands & Signals",
            ),
            row_heights=[0.4, 0.2, 0.4],
        )

        # Panel 1: Equity Curves
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["strategy_equity"],
                name="ML-Filtered Strategy",
                line=dict(color="#2962FF", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["benchmark_equity"],
                name="Buy & Hold",
                line=dict(color="#FF6D00", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Panel 2: Drawdown
        peak = df["strategy_equity"].cummax()
        drawdown = (df["strategy_equity"] - peak) / peak * 100
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#D32F2F", width=1),
            ),
            row=2,
            col=1,
        )

        # Panel 3: Price + Bollinger Bands + Signals
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close Price",
                line=dict(color="#333", width=1),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                name="Upper Band",
                line=dict(color="#90CAF9", width=1, dash="dot"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                name="Lower Band",
                line=dict(color="#90CAF9", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(144, 202, 249, 0.1)",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_sma"],
                name="SMA",
                line=dict(color="#FFA726", width=1, dash="dash"),
            ),
            row=3,
            col=1,
        )

        # Buy/Sell markers
        buys = df[df["signal"] == 1]
        sells = df[df["signal"] == -1]
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10, color="#4CAF50"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10, color="#F44336"),
            ),
            row=3,
            col=1,
        )

        # Layout
        fig.update_layout(
            title=dict(
                text="QuantForge — Mean Reversion + ML Filter Backtest",
                font=dict(size=18),
            ),
            height=1000,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)

        fig.write_html(save_path)
        logger.info("Backtest chart saved to %s", save_path)
