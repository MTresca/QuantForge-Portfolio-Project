"""
Machine Learning Signal Filter
===============================
Implements a Random Forest classifier that learns to distinguish genuine
mean-reversion opportunities from false breakouts.

Model Architecture:
    - RandomForestClassifier: Ensemble of decision trees with bagging
    - Walk-forward validation to prevent look-ahead bias
    - Probability calibration for confidence-based position sizing

Financial Rationale:
    Raw technical signals (e.g., "price < lower Bollinger Band") have
    high false-positive rates, especially in trending markets. The ML
    filter learns regime-dependent patterns:
    - Low volatility + oversold RSI + high volume → likely reversion
    - High volatility + trend momentum + low volume → likely continuation
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from quantforge.utils.logger import get_logger

logger = get_logger(__name__)


class MLSignalFilter:
    """
    Random Forest classifier for filtering technical trading signals.

    Uses walk-forward splitting to avoid look-ahead bias: the model is
    trained only on data prior to the test period, mimicking how the
    strategy would be deployed in production.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (controls overfitting).
        min_samples_leaf: Minimum samples per leaf node.
        train_ratio: Fraction of data used for training (chronological split).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        train_ratio: float = 0.7,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.metrics: Dict[str, float] = {}

    def train(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Train the Random Forest using chronological train/test split.

        Walk-Forward Split (NOT random):
            |--- Training (70%) ---|--- Testing (30%) ---|
            t=0                   t=split              t=T

            This preserves temporal ordering and prevents look-ahead bias.
            Random cross-validation would leak future information into training.

        Args:
            features: Feature matrix (pd.DataFrame).
            target: Binary target series.

        Returns:
            Tuple of (predicted_probabilities, actual_targets, test_index)
            for the out-of-sample test period.
        """
        self.feature_names = list(features.columns)

        # Chronological split — never shuffle time series
        split_idx = int(len(features) * self.train_ratio)

        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        logger.info(
            "Walk-forward split: Train=%d samples [→ %s], Test=%d samples [%s →]",
            len(X_train),
            X_train.index[-1].strftime("%Y-%m-%d"),
            len(X_test),
            X_test.index[0].strftime("%Y-%m-%d"),
        )

        # Normalize features (fit on train, transform both)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )
        self.model.fit(X_train_scaled, y_train)

        # Out-of-sample predictions (probabilities for the positive class)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute and log metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

        logger.info(
            "Model trained — OOS Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%",
            self.metrics["accuracy"] * 100,
            self.metrics["precision"] * 100,
            self.metrics["recall"] * 100,
        )

        # Feature importances
        importances = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)
        logger.info("Feature importances:\n%s", importances.to_string())

        return y_prob, y_test.values, X_test.index

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities for new data.

        Args:
            features: Feature matrix with same columns as training data.

        Returns:
            Array of probabilities for positive class (price goes up).

        Raises:
            RuntimeError: If model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(features[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_classification_report(
        self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
    ) -> str:
        """
        Generate a detailed classification report.

        Args:
            y_true: Actual binary labels.
            y_prob: Predicted probabilities.
            threshold: Classification threshold.

        Returns:
            Formatted classification report string.
        """
        y_pred = (y_prob >= threshold).astype(int)
        return classification_report(
            y_true, y_pred, target_names=["Price Down", "Price Up"]
        )
