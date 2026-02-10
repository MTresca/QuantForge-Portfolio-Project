"""
Sentiment-Price Correlation Engine
===================================
Correlates FinBERT sentiment scores with subsequent asset price movements.

Methodology:
    1. Aggregate daily sentiment from all headlines for each date
    2. Fetch actual price data for the corresponding asset
    3. Compute forward returns (1-day, 3-day, 5-day) after each headline date
    4. Calculate Pearson/Spearman correlation between sentiment and returns
    5. Visualize the relationship with scatter plots and time series

The hypothesis: Positive sentiment should precede positive returns,
and negative sentiment should precede negative returns, if FinBERT
captures alpha-generating information.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPriceCorrelator:
    """
    Correlates sentiment scores with forward-looking asset returns.

    Financial Context:
        If sentiment has predictive power, we expect:
        - Positive correlation between sentiment and forward returns
        - Statistically significant p-values (< 0.05)
        - Higher correlation for shorter forward windows (sentiment decays)

    Attributes:
        ticker: Asset symbol to correlate against.
        forward_windows: List of forward periods (in trading days).
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        forward_windows: Optional[list] = None,
    ):
        self.ticker = ticker
        self.forward_windows = forward_windows or [1, 3, 5]

    def fetch_prices(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily close prices for the target asset.

        Args:
            start_date: Start date string 'YYYY-MM-DD'.
            end_date: End date string 'YYYY-MM-DD'.

        Returns:
            DataFrame with DatetimeIndex and 'Close' column.
        """
        import yfinance as yf

        # Add buffer for forward return calculation
        buffer_end = (
            pd.to_datetime(end_date) + pd.DateOffset(days=15)
        ).strftime("%Y-%m-%d")

        logger.info("Fetching %s prices from %s to %s", self.ticker, start_date, buffer_end)

        prices = yf.download(
            self.ticker,
            start=start_date,
            end=buffer_end,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        return prices[["Close"]]

    def compute_forward_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward returns for each observation date.

        Args:
            prices: DataFrame with 'Close' column and DatetimeIndex.

        Returns:
            DataFrame with forward return columns (fwd_1d, fwd_3d, fwd_5d).
        """
        for window in self.forward_windows:
            col_name = f"fwd_{window}d"
            prices[col_name] = prices["Close"].shift(-window) / prices["Close"] - 1.0

        return prices

    def merge_sentiment_and_returns(
        self,
        sentiment_df: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge daily aggregated sentiment with forward returns.

        Aggregation: Average sentiment score across all headlines on each date.

        Args:
            sentiment_df: DataFrame with 'date' and 'score' columns.
            prices: DataFrame with forward return columns.

        Returns:
            Merged DataFrame with both sentiment and return data.
        """
        # Aggregate sentiment by date
        daily_sentiment = (
            sentiment_df.groupby(sentiment_df["date"].dt.date)
            .agg(
                avg_score=("score", "mean"),
                n_headlines=("score", "count"),
                max_score=("score", "max"),
                min_score=("score", "min"),
            )
            .reset_index()
        )
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
        daily_sentiment = daily_sentiment.set_index("date")

        # Merge on date
        merged = daily_sentiment.join(prices, how="inner")

        logger.info(
            "Merged %d sentiment days with price data (%d matched)",
            len(daily_sentiment),
            len(merged),
        )

        return merged

    def compute_correlations(
        self, merged_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Pearson and Spearman correlations between sentiment and returns.

        Statistical Tests:
            - Pearson r: Linear correlation (assumes normality)
            - Spearman ρ: Rank correlation (robust to outliers)
            - p-value: Statistical significance (< 0.05 = significant)

        Args:
            merged_df: DataFrame with avg_score and forward return columns.

        Returns:
            Dictionary mapping window name → correlation statistics.
        """
        results = {}
        sentiment = merged_df["avg_score"].dropna()

        for window in self.forward_windows:
            col = f"fwd_{window}d"
            if col not in merged_df.columns:
                continue

            returns = merged_df[col].dropna()
            valid = sentiment.index.intersection(returns.index)

            if len(valid) < 5:
                logger.warning("Insufficient data for %s correlation (%d points)", col, len(valid))
                continue

            s = sentiment.loc[valid]
            r = returns.loc[valid]

            pearson_r, pearson_p = stats.pearsonr(s, r)
            spearman_r, spearman_p = stats.spearmanr(s, r)

            results[col] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_observations": len(valid),
            }

            significance = "***" if pearson_p < 0.01 else "**" if pearson_p < 0.05 else "*" if pearson_p < 0.10 else ""
            logger.info(
                "Correlation [%s]: Pearson r=%.4f (p=%.4f%s), Spearman ρ=%.4f (n=%d)",
                col,
                pearson_r,
                pearson_p,
                significance,
                spearman_r,
                len(valid),
            )

        return results

    def plot_correlation(
        self,
        merged_df: pd.DataFrame,
        correlations: Dict,
        save_path: str = "sentiment_correlation.html",
    ) -> None:
        """
        Generate interactive visualization of sentiment-price relationship.

        Creates a multi-panel figure:
            1. Time series: Sentiment score and price overlaid
            2. Scatter plots: Sentiment vs forward returns for each window

        Args:
            merged_df: Merged sentiment and returns data.
            correlations: Correlation results dictionary.
            save_path: Output file path.
        """
        n_windows = len(self.forward_windows)
        fig = make_subplots(
            rows=2,
            cols=max(n_windows, 1),
            subplot_titles=(
                [f"Sentiment & {self.ticker} Price"]
                + [""] * (max(n_windows, 1) - 1)
                + [f"Sentiment vs {w}d Forward Return" for w in self.forward_windows]
            ),
            row_heights=[0.5, 0.5],
            specs=[
                [{"colspan": max(n_windows, 1)}] + [None] * (max(n_windows, 1) - 1),
                [{}] * max(n_windows, 1),
            ],
        )

        # Panel 1: Time series overlay
        if "Close" in merged_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=merged_df.index,
                    y=merged_df["Close"],
                    name=f"{self.ticker} Price",
                    line=dict(color="#2962FF", width=2),
                ),
                row=1,
                col=1,
            )

        # Sentiment bars
        colors = merged_df["avg_score"].apply(
            lambda x: "#4CAF50" if x > 0 else "#F44336"
        )
        fig.add_trace(
            go.Bar(
                x=merged_df.index,
                y=merged_df["avg_score"],
                name="Sentiment Score",
                marker_color=colors,
                opacity=0.6,
                yaxis="y2",
            ),
            row=1,
            col=1,
        )

        # Panel 2: Scatter plots for each forward window
        for i, window in enumerate(self.forward_windows, 1):
            col_name = f"fwd_{window}d"
            if col_name not in merged_df.columns:
                continue

            valid = merged_df[["avg_score", col_name]].dropna()

            fig.add_trace(
                go.Scatter(
                    x=valid["avg_score"],
                    y=valid[col_name] * 100,
                    mode="markers",
                    name=f"{window}d Return",
                    marker=dict(
                        size=8,
                        color=valid[col_name],
                        colorscale="RdYlGn",
                        showscale=i == 1,
                    ),
                    hovertemplate="Sentiment: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
                ),
                row=2,
                col=i,
            )

            # Add trend line
            if len(valid) > 2:
                z = np.polyfit(valid["avg_score"], valid[col_name] * 100, 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid["avg_score"].min(), valid["avg_score"].max(), 50)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode="lines",
                        name=f"Trend ({window}d)",
                        line=dict(color="#FF6D00", dash="dash", width=2),
                        showlegend=False,
                    ),
                    row=2,
                    col=i,
                )

        fig.update_layout(
            title=dict(
                text=f"QuantForge — Sentiment vs {self.ticker} Price Correlation",
                font=dict(size=18),
            ),
            template="plotly_white",
            height=800,
            width=1200,
            showlegend=True,
        )

        fig.write_html(save_path)
        logger.info("Correlation chart saved to %s", save_path)
