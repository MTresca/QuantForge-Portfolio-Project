"""
Feature Engineering Module
==========================
Computes technical indicators and constructs ML-ready feature matrices.

Technical Indicators:
    - Bollinger Bands: Mean ± kσ envelope for mean-reversion signals
    - RSI (Relative Strength Index): Momentum oscillator [0, 100]
    - Log Returns: Continuously compounded returns for statistical properties
    - Volatility Regime: Rolling std of returns as a regime indicator
    - Volume Z-Score: Standardized volume to detect unusual activity
"""

from typing import Tuple

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Transforms raw OHLCV data into a feature matrix suitable for ML modeling.

    Financial Context:
        Raw price data is non-stationary and noisy. Feature engineering
        transforms it into stationary, informative signals. Each feature
        captures a different market microstructure aspect:
        - Bollinger %B: Where price sits relative to its volatility envelope
        - RSI: Overbought/oversold momentum
        - Log returns: Instantaneous growth rate (additive over time)
        - Volatility: Current risk regime
        - Volume Z-score: Participation intensity

    Attributes:
        bb_window: Bollinger Bands lookback period (default: 20).
        bb_std: Number of standard deviations for bands (default: 2.0).
        rsi_window: RSI lookback period (default: 14).
        forward_window: Periods ahead for target variable (default: 5).
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        forward_window: int = 5,
    ):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.forward_window = forward_window

    def compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and derived %B indicator.

        Bollinger Bands Theory:
            Upper Band = SMA(n) + k * σ(n)
            Lower Band = SMA(n) - k * σ(n)
            %B = (Price - Lower) / (Upper - Lower)
            When %B < 0, price is below the lower band (potential buy signal).
            When %B > 1, price is above the upper band (potential sell signal).

        Args:
            df: DataFrame with 'Close' column.

        Returns:
            DataFrame with added columns: bb_sma, bb_upper, bb_lower, bb_pct_b, bb_bandwidth.
        """
        close = df["Close"]

        df["bb_sma"] = close.rolling(window=self.bb_window).mean()
        rolling_std = close.rolling(window=self.bb_window).std()
        df["bb_upper"] = df["bb_sma"] + self.bb_std * rolling_std
        df["bb_lower"] = df["bb_sma"] - self.bb_std * rolling_std

        # %B: Normalized position within bands [0 = lower, 1 = upper]
        band_width = df["bb_upper"] - df["bb_lower"]
        df["bb_pct_b"] = (close - df["bb_lower"]) / band_width

        # Bandwidth: Volatility indicator (narrow bands = low vol = squeeze)
        df["bb_bandwidth"] = band_width / df["bb_sma"]

        logger.info("Computed Bollinger Bands (window=%d, std=%.1f)", self.bb_window, self.bb_std)
        return df

    def compute_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI).

        RSI Theory:
            RSI = 100 - 100 / (1 + RS)
            RS = Avg(Up moves over n periods) / Avg(Down moves over n periods)
            RSI < 30 is traditionally "oversold", RSI > 70 is "overbought".
            We use Wilder's smoothing (exponential) for stability.

        Args:
            df: DataFrame with 'Close' column.

        Returns:
            DataFrame with added 'rsi' column.
        """
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's exponential smoothing (equivalent to EMA with alpha = 1/n)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_window, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_window, min_periods=self.rsi_window).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        logger.info("Computed RSI (window=%d)", self.rsi_window)
        return df

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate logarithmic returns and rolling volatility.

        Financial Context:
            Log returns r_t = ln(P_t / P_{t-1}) are preferred over simple
            returns because they are:
            1. Time-additive: r(t1→t3) = r(t1→t2) + r(t2→t3)
            2. Approximately normally distributed for short horizons
            3. Symmetrical: +10% and -10% have equal magnitude

        Args:
            df: DataFrame with 'Close' column.

        Returns:
            DataFrame with added columns: log_return, volatility_20d.
        """
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["volatility_20d"] = df["log_return"].rolling(window=20).std() * np.sqrt(252)

        logger.info("Computed log returns and 20-day annualized volatility")
        return df

    def compute_volume_zscore(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate Z-score of trading volume.

        Financial Context:
            Abnormally high volume often confirms price movements.
            Low-volume breakouts are more likely to be false signals.
            Z-score standardizes volume relative to recent history.

        Args:
            df: DataFrame with 'Volume' column.
            window: Lookback period for mean/std calculation.

        Returns:
            DataFrame with added 'volume_zscore' column.
        """
        vol_mean = df["Volume"].rolling(window=window).mean()
        vol_std = df["Volume"].rolling(window=window).std()
        df["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.finfo(float).eps)

        logger.info("Computed volume Z-score (window=%d)", window)
        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary classification target for ML model.

        Target Logic:
            y = 1 if the forward N-day return is positive (price goes up)
            y = 0 otherwise

            This is the variable the Random Forest will learn to predict,
            enabling it to filter Bollinger Band signals that are likely
            to result in losing trades.

        Args:
            df: DataFrame with 'Close' column.

        Returns:
            DataFrame with added 'target' column.
        """
        forward_return = df["Close"].shift(-self.forward_window) / df["Close"] - 1.0
        df["forward_return"] = forward_return
        df["target"] = (forward_return > 0).astype(int)

        pos_rate = df["target"].mean()
        logger.info(
            "Created target variable (forward=%d days, positive_rate=%.2f%%)",
            self.forward_window,
            pos_rate * 100,
        )
        return df

    def build_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute the full feature engineering pipeline.

        Pipeline:
            1. Bollinger Bands → bb_pct_b, bb_bandwidth
            2. RSI → rsi
            3. Log Returns → log_return, volatility_20d
            4. Volume Z-Score → volume_zscore
            5. Target Variable → target (binary: up/down)
            6. Drop NaN rows from lookback periods

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            Tuple of (feature_df, target_series) with NaN rows removed.
        """
        logger.info("Building feature matrix from %d rows of raw data", len(df))

        df = self.compute_bollinger_bands(df)
        df = self.compute_rsi(df)
        df = self.compute_log_returns(df)
        df = self.compute_volume_zscore(df)
        df = self.create_target(df)

        feature_cols = [
            "bb_pct_b",
            "bb_bandwidth",
            "rsi",
            "log_return",
            "volatility_20d",
            "volume_zscore",
        ]

        # Drop rows with NaN from rolling calculations and forward-looking target
        valid_mask = df[feature_cols + ["target"]].notna().all(axis=1)
        df_clean = df.loc[valid_mask].copy()

        features = df_clean[feature_cols]
        target = df_clean["target"]

        logger.info(
            "Feature matrix ready: %d samples × %d features (dropped %d NaN rows)",
            len(features),
            len(feature_cols),
            len(df) - len(df_clean),
        )

        return features, target
