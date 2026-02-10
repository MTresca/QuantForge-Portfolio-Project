"""
FinBERT Model Wrapper
=====================
Wraps HuggingFace's ProsusAI/finbert for financial sentiment analysis.

Model Details:
    FinBERT is a BERT model fine-tuned on ~50,000 financial texts
    (analyst reports, earnings calls, financial news). It classifies
    text into three categories:
        - Positive: bullish/optimistic language
        - Negative: bearish/pessimistic language
        - Neutral: factual/informational language

    Unlike generic sentiment models (e.g., VADER), FinBERT understands
    financial domain nuances:
        "Revenue missed expectations" → Negative (generic model might miss this)
        "The company cut costs aggressively" → Context-dependent
        "Shares were volatile" → Neutral (not inherently positive or negative)

Scoring:
    We map the three-class output to a continuous score [-1, +1]:
        score = P(positive) - P(negative)
    This captures both direction AND intensity of sentiment.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using ProsusAI/finbert.

    Loads the model lazily on first use to avoid slow imports
    when only using other modules.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Number of texts to process at once.
        device: Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._pipeline = None

    def _load_model(self) -> None:
        """
        Lazy-load the FinBERT model via HuggingFace pipeline.

        Using pipeline() abstracts away tokenization, batching,
        and post-processing for clean inference.
        """
        if self._pipeline is not None:
            return

        logger.info("Loading FinBERT model: %s (device=%s)...", self.model_name, self.device)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=self.device if self.device == "cpu" else 0,
            truncation=True,
            max_length=512,
        )

        logger.info("FinBERT model loaded successfully.")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze a single text string for financial sentiment.

        Args:
            text: Financial text (headline, excerpt, tweet, etc.).

        Returns:
            Dictionary with keys: label, score, positive, negative, neutral.
        """
        self._load_model()

        results = self._pipeline(text, top_k=3)

        probs = {r["label"].lower(): r["score"] for r in results}
        sentiment_score = probs.get("positive", 0) - probs.get("negative", 0)

        return {
            "label": max(probs, key=probs.get),
            "score": sentiment_score,
            "positive": probs.get("positive", 0),
            "negative": probs.get("negative", 0),
            "neutral": probs.get("neutral", 0),
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts and return a structured DataFrame.

        Processes texts in batches for GPU efficiency (if available).

        Args:
            texts: List of financial text strings.

        Returns:
            DataFrame with columns: text, label, score, positive, negative, neutral.
        """
        self._load_model()

        logger.info("Analyzing %d texts in batches of %d...", len(texts), self.batch_size)

        all_results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self._pipeline(batch, top_k=3, batch_size=self.batch_size)

            for text, result in zip(batch, batch_results):
                probs = {r["label"].lower(): r["score"] for r in result}
                sentiment_score = probs.get("positive", 0) - probs.get("negative", 0)

                all_results.append(
                    {
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "label": max(probs, key=probs.get),
                        "score": sentiment_score,
                        "positive": probs.get("positive", 0),
                        "negative": probs.get("negative", 0),
                        "neutral": probs.get("neutral", 0),
                    }
                )

        df = pd.DataFrame(all_results)

        # Summary statistics
        label_counts = df["label"].value_counts()
        avg_score = df["score"].mean()
        logger.info(
            "Sentiment distribution: %s | Average score: %.3f",
            label_counts.to_dict(),
            avg_score,
        )

        return df

    def score_to_signal(self, score: float) -> str:
        """
        Map continuous sentiment score to a trading signal.

        Thresholds (conservative):
            score > 0.3  → BULLISH  (strong positive sentiment)
            score < -0.3 → BEARISH  (strong negative sentiment)
            else         → NEUTRAL  (mixed/weak signal)

        Args:
            score: Sentiment score in [-1, +1].

        Returns:
            Signal string: 'BULLISH', 'BEARISH', or 'NEUTRAL'.
        """
        if score > 0.3:
            return "BULLISH"
        elif score < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"
