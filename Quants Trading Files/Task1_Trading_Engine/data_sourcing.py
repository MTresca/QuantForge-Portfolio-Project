"""
Data Sourcing Module
====================
Fetches historical OHLCV data from Yahoo Finance with robust error handling,
data validation, and optional local caching for reproducibility.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """
    Fetches and validates historical market data from Yahoo Finance.

    Financial Context:
        OHLCV (Open, High, Low, Close, Volume) data is the foundation
        of all technical analysis. We use Adjusted Close to account for
        stock splits and dividends, ensuring return calculations are accurate.

    Attributes:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY').
        start_date: Start of the historical window.
        end_date: End of the historical window.
        cache_dir: Optional directory for caching CSV files.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        years: int = 5,
        end_date: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the DataFetcher.

        Args:
            ticker: Yahoo Finance ticker symbol.
            years: Number of years of historical data to fetch.
            end_date: End date string 'YYYY-MM-DD'. Defaults to today.
            cache_dir: If set, cache downloaded data as CSV files.
        """
        self.ticker = ticker.upper()
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.start_date = (
            datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=years * 365)
        ).strftime("%Y-%m-%d")
        self.cache_dir = cache_dir

    def fetch(self) -> pd.DataFrame:
        """
        Download OHLCV data with validation and optional caching.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
            Index is a DatetimeIndex.

        Raises:
            ValueError: If downloaded data is empty or has insufficient rows.
        """
        cache_path = self._get_cache_path()

        if cache_path and os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

        logger.info(
            "Fetching %s data from %s to %s via yfinance",
            self.ticker,
            self.start_date,
            self.end_date,
        )

        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for ticker '{self.ticker}'. "
                "Verify the symbol is valid on Yahoo Finance."
            )

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validate minimum data requirements
        min_rows = 252  # ~1 trading year
        if len(df) < min_rows:
            logger.warning(
                "Only %d rows fetched (minimum recommended: %d). "
                "Results may be unreliable.",
                len(df),
                min_rows,
            )

        # Handle missing data
        missing_pct = df.isnull().sum().sum() / df.size * 100
        if missing_pct > 0:
            logger.info("Filling %.2f%% missing values via forward-fill", missing_pct)
            df = df.ffill().bfill()

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
            logger.info("Cached data to %s", cache_path)

        logger.info(
            "Fetched %d rows for %s [%s → %s]",
            len(df),
            self.ticker,
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )

        return df

    def _get_cache_path(self) -> Optional[str]:
        """Build cache file path if cache_dir is set."""
        if self.cache_dir is None:
            return None
        return os.path.join(
            self.cache_dir, f"{self.ticker}_{self.start_date}_{self.end_date}.csv"
        )
