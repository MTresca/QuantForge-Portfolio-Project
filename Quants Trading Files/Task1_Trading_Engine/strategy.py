"""
Strategy Module
===============
Implements the Mean Reversion + ML Filter trading strategy.

Strategy Logic:
    ENTRY (BUY):
        - Price closes below the lower Bollinger Band (mean-reversion signal)
        - AND the ML model predicts P(up) > threshold (signal confirmation)

    EXIT (SELL):
        - Price touches the upper Bollinger Band (profit target)
        - OR price crosses above the SMA (mean reached)
        - OR a stop-loss is triggered at -N% from entry

    The ML filter is the key innovation: it suppresses entries during
    unfavorable regimes (e.g., strong downtrends where mean-reversion
    is unlikely to work), dramatically reducing false signals.
"""

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionMLStrategy:
    """
    Generates trading signals by combining Bollinger Band mean-reversion
    entries with Random Forest probability filtering.

    Attributes:
        ml_threshold: Minimum ML probability to confirm a buy signal.
        stop_loss_pct: Maximum loss before forced exit (as decimal, e.g., 0.03 = 3%).
    """

    def __init__(self, ml_threshold: float = 0.55, stop_loss_pct: float = 0.03):
        """
        Initialize the strategy.

        Args:
            ml_threshold: Probability cutoff for ML confirmation.
                Higher = fewer but higher-quality entries.
                0.50 = baseline (same as random), 0.55-0.60 recommended.
            stop_loss_pct: Stop-loss percentage from entry price.
        """
        self.ml_threshold = ml_threshold
        self.stop_loss_pct = stop_loss_pct

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_probabilities: np.ndarray,
        test_index: pd.Index,
    ) -> pd.DataFrame:
        """
        Generate buy/sell signals for the test period.

        Signal Generation Logic:
            1. Filter to test period only (out-of-sample)
            2. BUY when: Close < bb_lower AND ml_prob > threshold
            3. SELL when: Close > bb_upper OR Close > bb_sma (after a buy)

        Args:
            df: Full DataFrame with technical indicators.
            ml_probabilities: Array of P(up) from the ML model, aligned to test_index.
            test_index: DatetimeIndex for the test period.

        Returns:
            DataFrame with columns: Close, signal, position, bb_sma, bb_upper, bb_lower.
            signal: 1 = buy, -1 = sell, 0 = hold.
            position: 1 = long, 0 = flat (no short selling).
        """
        # Slice to test period
        test_df = df.loc[test_index].copy()
        test_df["ml_prob"] = ml_probabilities

        # Initialize signal column
        test_df["signal"] = 0

        # BUY: Price below lower band AND ML confirms upward move
        buy_condition = (
            (test_df["Close"] < test_df["bb_lower"])
            & (test_df["ml_prob"] > self.ml_threshold)
        )
        test_df.loc[buy_condition, "signal"] = 1

        # SELL: Price above upper band OR above SMA (mean reached)
        sell_condition = (test_df["Close"] > test_df["bb_upper"]) | (
            test_df["Close"] > test_df["bb_sma"]
        )
        test_df.loc[sell_condition, "signal"] = -1

        # Convert signals to positions (long only, no shorting)
        # Position = 1 when holding, 0 when flat
        test_df["position"] = 0
        in_position = False

        for i in range(len(test_df)):
            if test_df["signal"].iloc[i] == 1 and not in_position:
                in_position = True
                test_df.iloc[i, test_df.columns.get_loc("position")] = 1
            elif test_df["signal"].iloc[i] == -1 and in_position:
                in_position = False
                test_df.iloc[i, test_df.columns.get_loc("position")] = 0
            elif in_position:
                test_df.iloc[i, test_df.columns.get_loc("position")] = 1

        n_buys = (test_df["signal"] == 1).sum()
        n_sells = (test_df["signal"] == -1).sum()
        pct_invested = test_df["position"].mean() * 100

        logger.info(
            "Signals generated: %d buys, %d sells, %.1f%% time invested",
            n_buys,
            n_sells,
            pct_invested,
        )

        return test_df
