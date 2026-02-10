"""
Portfolio Data Loader
=====================
Retrieves and validates multi-asset historical price data for portfolio
optimization. Computes returns matrices required by optimization engines.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)

# Default diversified universe spanning sectors + asset classes
DEFAULT_UNIVERSE = [
    "AAPL",   # Tech
    "MSFT",   # Tech
    "JNJ",    # Healthcare
    "JPM",    # Financials
    "XOM",    # Energy
    "PG",     # Consumer Staples
    "AMZN",   # Consumer Discretionary
    "NVDA",   # Semiconductors
    "BRK-B",  # Conglomerate
    "UNH",    # Health Insurance
]


class PortfolioDataLoader:
    """
    Loads multi-asset price data and computes return statistics.

    Financial Context:
        Portfolio optimization requires a returns matrix (T × N) where:
        - T = number of time observations
        - N = number of assets
        From this, we derive the expected return vector (μ) and the
        covariance matrix (Σ), which are inputs to the Markowitz optimizer.

    Attributes:
        tickers: List of asset symbols.
        years: Historical lookback period.
    """

    def __init__(self, tickers: Optional[List[str]] = None, years: int = 5):
        self.tickers = tickers or DEFAULT_UNIVERSE
        self.years = years

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download price data and compute daily log returns.

        Returns:
            Tuple of (prices_df, returns_df).
            prices_df: Adjusted close prices (T × N).
            returns_df: Daily log returns (T-1 × N).

        Raises:
            ValueError: If any tickers return no data.
        """
        logger.info(
            "Downloading data for %d assets over %d years: %s",
            len(self.tickers),
            self.years,
            ", ".join(self.tickers),
        )

        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=self.years)

        prices = yf.download(
            self.tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # Handle single vs multi-ticker download format
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Close"]
        elif "Close" in prices.columns:
            prices = prices[["Close"]]
            prices.columns = self.tickers

        # Drop tickers with no data
        valid_cols = prices.dropna(axis=1, how="all").columns
        dropped = set(self.tickers) - set(valid_cols)
        if dropped:
            logger.warning("Dropped tickers with no data: %s", dropped)
        prices = prices[valid_cols]

        # Forward-fill then backfill remaining gaps (holidays, delistings)
        prices = prices.ffill().bfill()

        if prices.empty or prices.shape[1] < 2:
            raise ValueError(
                "Insufficient data: need at least 2 assets with valid prices."
            )

        # Compute log returns
        returns = np.log(prices / prices.shift(1)).dropna()

        logger.info(
            "Loaded %d assets × %d observations. Returns matrix: %s",
            prices.shape[1],
            len(returns),
            returns.shape,
        )

        return prices, returns

    def compute_statistics(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute expected return vector (μ) and covariance matrix (Σ).

        Uses annualized statistics (252 trading days):
            μ = mean(daily_returns) × 252
            Σ = cov(daily_returns) × 252

        Args:
            returns: Daily log returns DataFrame.

        Returns:
            Tuple of (expected_returns, covariance_matrix).
        """
        mu = returns.mean() * 252
        sigma = returns.cov() * 252

        logger.info("Expected annual returns:\n%s", mu.round(4).to_string())

        return mu, sigma
