"""
Financial Headline Data Loader
==============================
Provides financial headline datasets for sentiment analysis.
Includes a sample dataset and a Yahoo Finance RSS scraper.
"""

from typing import List, Optional

import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


# Sample dataset of real-world-style financial headlines for demonstration
SAMPLE_HEADLINES = [
    {"date": "2024-12-01", "headline": "NVIDIA reports record quarterly revenue, beating Wall Street estimates by 12%"},
    {"date": "2024-12-01", "headline": "Tesla deliveries fall short of expectations amid increased competition in China"},
    {"date": "2024-12-02", "headline": "Federal Reserve signals potential rate cuts in early 2025, markets rally"},
    {"date": "2024-12-02", "headline": "Apple announces $100 billion share buyback program, largest in tech history"},
    {"date": "2024-12-03", "headline": "Oil prices plunge 8% on OPEC+ disagreement over production targets"},
    {"date": "2024-12-03", "headline": "Amazon Web Services wins $15 billion government cloud contract"},
    {"date": "2024-12-04", "headline": "JPMorgan CEO warns of recession risks, citing consumer credit deterioration"},
    {"date": "2024-12-04", "headline": "Microsoft Azure revenue growth accelerates to 35% year-over-year"},
    {"date": "2024-12-05", "headline": "S&P 500 hits all-time high as inflation data comes in below expectations"},
    {"date": "2024-12-05", "headline": "Bitcoin surges past $100,000 following ETF approval speculation"},
    {"date": "2024-12-06", "headline": "Meta Platforms faces $2 billion EU antitrust fine over data practices"},
    {"date": "2024-12-06", "headline": "Goldman Sachs downgrades semiconductor sector, citing inventory overhang"},
    {"date": "2024-12-07", "headline": "Berkshire Hathaway reports record cash pile of $320 billion"},
    {"date": "2024-12-07", "headline": "UnitedHealth stock drops 15% after CEO assassination shocks healthcare sector"},
    {"date": "2024-12-08", "headline": "NVIDIA launches new AI chip, stock jumps 8% in pre-market trading"},
    {"date": "2024-12-08", "headline": "Exxon Mobil cuts dividend guidance amid declining oil margins"},
    {"date": "2024-12-09", "headline": "Retail sales beat expectations as holiday shopping season gets strong start"},
    {"date": "2024-12-09", "headline": "Regional banks face pressure as commercial real estate defaults rise"},
    {"date": "2024-12-10", "headline": "Johnson & Johnson settles talc lawsuits for $8.9 billion"},
    {"date": "2024-12-10", "headline": "Procter & Gamble raises full-year guidance on strong consumer demand"},
]


class HeadlineLoader:
    """
    Loads financial headlines for sentiment analysis.

    Provides multiple data sources:
        1. Built-in sample dataset (for testing/demos)
        2. CSV file loader (for custom datasets)
        3. Yahoo Finance RSS scraper (for real-time data)

    Attributes:
        ticker: Asset symbol to filter headlines for (optional).
    """

    def __init__(self, ticker: Optional[str] = None):
        self.ticker = ticker

    def load_sample(self) -> pd.DataFrame:
        """
        Load the built-in sample headline dataset.

        Returns:
            DataFrame with columns: date, headline.
        """
        df = pd.DataFrame(SAMPLE_HEADLINES)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("Loaded %d sample headlines", len(df))
        return df

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load headlines from a CSV file.

        Expected format: columns 'date' and 'headline'.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with parsed dates and headline text.
        """
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("Loaded %d headlines from %s", len(df), filepath)
        return df

    def scrape_yahoo_rss(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Scrape financial headlines from Yahoo Finance RSS feed.

        Args:
            ticker: Stock symbol (e.g., 'NVDA'). Uses self.ticker if None.

        Returns:
            DataFrame with columns: date, headline.

        Note:
            This requires internet access and may be rate-limited.
            Falls back to sample data if scraping fails.
        """
        ticker = ticker or self.ticker or "NVDA"

        try:
            import requests
            from bs4 import BeautifulSoup

            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning(
                    "Yahoo RSS returned status %d. Falling back to sample data.",
                    response.status_code,
                )
                return self.load_sample()

            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            headlines = []
            for item in items:
                title = item.find("title")
                pub_date = item.find("pubDate")
                if title and pub_date:
                    headlines.append(
                        {
                            "date": pd.to_datetime(pub_date.text),
                            "headline": title.text,
                        }
                    )

            if not headlines:
                logger.warning("No headlines found in RSS feed. Using sample data.")
                return self.load_sample()

            df = pd.DataFrame(headlines)
            logger.info("Scraped %d headlines for %s from Yahoo RSS", len(df), ticker)
            return df

        except Exception as e:
            logger.warning("RSS scraping failed (%s). Falling back to sample data.", str(e))
            return self.load_sample()
