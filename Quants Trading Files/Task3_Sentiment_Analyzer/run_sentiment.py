"""
Run Sentiment Analysis — Entry Point
=====================================
End-to-end pipeline:
    1. Load financial headlines (sample or scraped)
    2. Analyze sentiment via FinBERT
    3. Fetch corresponding asset prices
    4. Correlate sentiment with forward returns
    5. Generate visualizations

Usage:
    python -m quantforge.sentiment_analyzer.run_sentiment
"""

import os

import pandas as pd

from quantforge.sentiment_analyzer.correlation import SentimentPriceCorrelator
from quantforge.sentiment_analyzer.finbert_model import FinBERTAnalyzer
from quantforge.sentiment_analyzer.headline_scraper import HeadlineLoader
from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


def main(
    ticker: str = "NVDA",
    use_sample: bool = True,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Execute the full sentiment analysis pipeline.

    Args:
        ticker: Asset to correlate sentiment against.
        use_sample: If True, use built-in sample headlines.
                    If False, attempt to scrape Yahoo Finance RSS.
        output_dir: Directory for saving outputs.

    Returns:
        DataFrame with sentiment scores and correlation data.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QuantForge Sentiment Analyzer — Starting")
    logger.info("Ticker: %s | Source: %s", ticker, "sample" if use_sample else "live")
    logger.info("=" * 60)

    # Step 1: Load Headlines
    logger.info("[1/4] Loading financial headlines...")
    loader = HeadlineLoader(ticker=ticker)

    if use_sample:
        headlines_df = loader.load_sample()
    else:
        headlines_df = loader.scrape_yahoo_rss()

    logger.info("Loaded %d headlines spanning %s to %s",
                len(headlines_df),
                headlines_df["date"].min().strftime("%Y-%m-%d"),
                headlines_df["date"].max().strftime("%Y-%m-%d"))

    # Step 2: Analyze Sentiment
    logger.info("[2/4] Running FinBERT sentiment analysis...")
    analyzer = FinBERTAnalyzer(device="cpu")
    sentiment_df = analyzer.analyze_batch(headlines_df["headline"].tolist())

    # Merge dates from headlines
    sentiment_df["date"] = headlines_df["date"].values
    sentiment_df["signal"] = sentiment_df["score"].apply(analyzer.score_to_signal)

    # Save sentiment results
    sentiment_path = os.path.join(output_dir, "sentiment_results.csv")
    sentiment_df.to_csv(sentiment_path, index=False)
    logger.info("Sentiment results saved to %s", sentiment_path)

    # Print sentiment summary
    logger.info("=" * 60)
    logger.info("SENTIMENT ANALYSIS RESULTS")
    logger.info("=" * 60)
    for _, row in sentiment_df.iterrows():
        emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(row["label"], "?")
        logger.info(
            "[%s] (%.3f) %s",
            emoji,
            row["score"],
            row["text"],
        )
    logger.info("=" * 60)
    logger.info("Average Sentiment Score: %.4f", sentiment_df["score"].mean())
    logger.info("Signal Distribution: %s", sentiment_df["signal"].value_counts().to_dict())

    # Step 3: Correlation Analysis
    logger.info("[3/4] Computing sentiment-price correlations...")
    correlator = SentimentPriceCorrelator(
        ticker=ticker,
        forward_windows=[1, 3, 5],
    )

    try:
        start_date = headlines_df["date"].min().strftime("%Y-%m-%d")
        end_date = headlines_df["date"].max().strftime("%Y-%m-%d")

        prices = correlator.fetch_prices(start_date, end_date)
        prices = correlator.compute_forward_returns(prices)

        merged = correlator.merge_sentiment_and_returns(sentiment_df, prices)
        correlations = correlator.compute_correlations(merged)

        # Step 4: Visualization
        logger.info("[4/4] Generating visualizations...")
        correlator.plot_correlation(
            merged,
            correlations,
            save_path=os.path.join(output_dir, f"{ticker}_sentiment_correlation.html"),
        )

        # Save correlation results
        if correlations:
            corr_df = pd.DataFrame(correlations).T
            corr_path = os.path.join(output_dir, "correlation_results.csv")
            corr_df.to_csv(corr_path)
            logger.info("Correlation results saved to %s", corr_path)

    except Exception as e:
        logger.warning(
            "Correlation analysis skipped: %s. "
            "Sentiment results are still available.",
            str(e),
        )

    logger.info("Pipeline complete. Results saved to %s", output_dir)

    return sentiment_df


if __name__ == "__main__":
    main()
