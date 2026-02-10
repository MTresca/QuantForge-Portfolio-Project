"""
Run Backtest — Entry Point
==========================
End-to-end pipeline that orchestrates:
    1. Data fetching from Yahoo Finance
    2. Feature engineering (Bollinger Bands, RSI, log returns)
    3. ML model training (Random Forest signal filter)
    4. Signal generation (Mean Reversion + ML confirmation)
    5. Vectorized backtesting with transaction costs
    6. Performance reporting and interactive visualization

Usage:
    python -m quantforge.trading_engine.run_backtest
"""

import os

from quantforge.trading_engine.backtester import StrategyBacktester
from quantforge.trading_engine.data_sourcing import DataFetcher
from quantforge.trading_engine.feature_engineering import FeatureEngineer
from quantforge.trading_engine.ml_filter import MLSignalFilter
from quantforge.trading_engine.strategy import MeanReversionMLStrategy
from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


def main(
    ticker: str = "SPY",
    years: int = 5,
    ml_threshold: float = 0.55,
    output_dir: str = "output",
) -> dict:
    """
    Execute the full Mean Reversion + ML Filter backtest pipeline.

    Args:
        ticker: Stock symbol to trade.
        years: Years of historical data.
        ml_threshold: ML probability threshold for trade entry.
        output_dir: Directory for saving outputs.

    Returns:
        Dictionary of performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QuantForge Trading Engine — Starting Backtest")
    logger.info("Ticker: %s | Period: %d years | ML Threshold: %.2f", ticker, years, ml_threshold)
    logger.info("=" * 60)

    # Step 1: Fetch Data
    logger.info("[1/5] Fetching market data...")
    fetcher = DataFetcher(ticker=ticker, years=years)
    df = fetcher.fetch()

    # Step 2: Feature Engineering
    logger.info("[2/5] Engineering features...")
    engineer = FeatureEngineer(
        bb_window=20,
        bb_std=2.0,
        rsi_window=14,
        forward_window=5,
    )
    features, target = engineer.build_feature_matrix(df)

    # Store the enriched DataFrame for signal generation
    df_enriched = df.loc[features.index].copy()
    df_enriched = df_enriched.join(features, rsuffix="_feat")

    # Step 3: Train ML Filter
    logger.info("[3/5] Training Random Forest signal filter...")
    ml_filter = MLSignalFilter(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        train_ratio=0.7,
    )
    ml_probs, y_test, test_index = ml_filter.train(features, target)

    # Log classification report
    report = ml_filter.get_classification_report(y_test, ml_probs)
    logger.info("Classification Report:\n%s", report)

    # Step 4: Generate Trading Signals
    logger.info("[4/5] Generating trading signals...")
    strategy = MeanReversionMLStrategy(
        ml_threshold=ml_threshold,
        stop_loss_pct=0.03,
    )
    signals_df = strategy.generate_signals(df_enriched, ml_probs, test_index)

    # Step 5: Run Backtest
    logger.info("[5/5] Running vectorized backtest...")
    backtester = StrategyBacktester(
        signals_df=signals_df,
        initial_capital=100_000.0,
        cost_per_trade=0.001,
    )
    results_df = backtester.run()

    # Generate visualization
    chart_path = os.path.join(output_dir, f"{ticker}_backtest_results.html")
    backtester.plot_results(results_df, save_path=chart_path)

    logger.info("Pipeline complete. Results saved to %s", output_dir)

    return backtester.results


if __name__ == "__main__":
    main()
