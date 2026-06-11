"""
Unit Tests — Portfolio Simulator (Task 4)
==========================================
Tests cover the core state management, P&L accounting, and simulation
loop logic. Network calls (yfinance) are patched so tests run offline.

Run with:
    python -m pytest tests/test_portfolio_simulator.py -v
"""

import json
import os
import tempfile
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from quantforge.portfolio_simulator.config import (
    DEFAULT_INITIAL_CAPITAL,
    TRANSACTION_COST_RATE,
)
from quantforge.portfolio_simulator.portfolio import (
    MonthlySnapshot,
    Portfolio,
    Position,
)
from quantforge.portfolio_simulator.rebalancer import Rebalancer
from quantforge.portfolio_simulator.risk_dashboard import RiskDashboard
from quantforge.portfolio_simulator.monte_carlo import MonteCarloProjector
from quantforge.portfolio_simulator.simulator import PortfolioSimulator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_portfolio() -> Portfolio:
    """A portfolio initialized with two positions at known prices."""
    p = Portfolio(initial_capital=100_000.0, inception_date="2022-01-01")
    prices = {"AAPL": 100.0, "SPY": 200.0}
    allocations = {"AAPL": 0.5, "SPY": 0.5}
    p.initialize_positions(allocations, prices)
    return p


@pytest.fixture
def portfolio_with_history(simple_portfolio: Portfolio) -> Portfolio:
    """Portfolio with 12 synthetic monthly snapshots."""
    p = simple_portfolio
    prices = {"AAPL": 100.0, "SPY": 200.0}
    for month in range(1, 13):
        # Simulate small monthly price changes
        prices = {
            "AAPL": 100.0 * (1 + month * 0.01),
            "SPY":  200.0 * (1 + month * 0.008),
        }
        p.record_snapshot(
            date=f"2022-{month:02d}-01",
            prices=prices,
            transaction_costs=0.0,
        )
    return p


# ── Position tests ─────────────────────────────────────────────────────────────

class TestPosition:
    def test_cost_basis(self):
        pos = Position("AAPL", shares=10.0, avg_cost=150.0)
        assert pos.cost_basis() == pytest.approx(1500.0)

    def test_market_value(self):
        pos = Position("AAPL", shares=10.0, avg_cost=150.0)
        assert pos.market_value(price=180.0) == pytest.approx(1800.0)

    def test_unrealized_pnl(self):
        pos = Position("AAPL", shares=10.0, avg_cost=150.0)
        assert pos.unrealized_pnl(price=180.0) == pytest.approx(300.0)

    def test_unrealized_pnl_pct(self):
        pos = Position("AAPL", shares=10.0, avg_cost=150.0)
        assert pos.unrealized_pnl_pct(price=180.0) == pytest.approx(20.0)

    def test_serialization_roundtrip(self):
        pos = Position("MSFT", shares=5.5, avg_cost=300.0)
        restored = Position.from_dict(pos.to_dict())
        assert restored.ticker == pos.ticker
        assert restored.shares == pytest.approx(pos.shares)
        assert restored.avg_cost == pytest.approx(pos.avg_cost)


# ── Portfolio initialization tests ────────────────────────────────────────────

class TestPortfolioInitialization:
    def test_initial_capital_set(self):
        p = Portfolio(initial_capital=100_000.0)
        assert p.initial_capital == pytest.approx(100_000.0)
        assert p.cash == pytest.approx(100_000.0)

    def test_initialize_positions_creates_correct_shares(self):
        p = Portfolio(initial_capital=100_000.0)
        prices = {"AAPL": 100.0, "SPY": 200.0}
        p.initialize_positions({"AAPL": 0.5, "SPY": 0.5}, prices)

        # Each gets €50k; AAPL at 100 → 500 shares; SPY at 200 → 250 shares
        assert p.positions["AAPL"].shares == pytest.approx(500.0, rel=1e-3)
        assert p.positions["SPY"].shares == pytest.approx(250.0, rel=1e-3)

    def test_initialize_positions_no_tx_costs(self):
        """Inception buy should not reduce capital via transaction costs."""
        p = Portfolio(initial_capital=100_000.0)
        prices = {"AAPL": 100.0}
        p.initialize_positions({"AAPL": 1.0}, prices)
        # Cash should be ~ 0 (all invested), no fees deducted
        assert p.cash == pytest.approx(0.0, abs=1.0)

    def test_normalize_overweight_allocations(self):
        p = Portfolio(initial_capital=100_000.0)
        prices = {"AAPL": 100.0, "SPY": 200.0}
        # Allocations sum to 1.4 → should normalize to 1.0
        p.initialize_positions({"AAPL": 0.7, "SPY": 0.7}, prices)
        weights = p.compute_weights(prices)
        total_invested = weights["AAPL"] + weights["SPY"]
        assert total_invested == pytest.approx(1.0, abs=0.01)

    def test_target_weights_stored(self):
        p = Portfolio(initial_capital=100_000.0)
        allocs = {"AAPL": 0.6, "SPY": 0.4}
        p.initialize_positions(allocs, {"AAPL": 100.0, "SPY": 200.0})
        assert p.target_weights == pytest.approx(allocs)


# ── Portfolio valuation tests ─────────────────────────────────────────────────

class TestPortfolioValuation:
    def test_compute_value_no_price_change(self, simple_portfolio):
        prices = {"AAPL": 100.0, "SPY": 200.0}
        value = simple_portfolio.compute_value(prices)
        assert value == pytest.approx(100_000.0, rel=1e-3)

    def test_compute_value_with_price_increase(self, simple_portfolio):
        prices = {"AAPL": 110.0, "SPY": 220.0}  # both +10%
        value = simple_portfolio.compute_value(prices)
        assert value > 100_000.0

    def test_compute_weights_sums_to_one(self, simple_portfolio):
        prices = {"AAPL": 100.0, "SPY": 200.0}
        weights = simple_portfolio.compute_weights(prices)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_compute_unrealized_pnl_positive_on_gain(self, simple_portfolio):
        prices = {"AAPL": 120.0, "SPY": 240.0}  # both +20%
        pnl = simple_portfolio.compute_unrealized_pnl(prices)
        assert pnl > 0

    def test_compute_drawdown_at_start(self, simple_portfolio):
        prices = {"AAPL": 100.0, "SPY": 200.0}
        dd = simple_portfolio.compute_drawdown(
            simple_portfolio.compute_value(prices)
        )
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_compute_drawdown_after_loss(self, simple_portfolio):
        # Record peak
        simple_portfolio.compute_drawdown(120_000.0)
        # Now drop to 100k
        dd = simple_portfolio.compute_drawdown(100_000.0)
        assert dd < 0
        assert dd == pytest.approx(-1 / 6, rel=1e-3)  # (100k - 120k) / 120k


# ── Buy / sell tests ──────────────────────────────────────────────────────────

class TestBuySell:
    def test_buy_creates_position(self):
        p = Portfolio(initial_capital=10_000.0)
        p.buy("AAPL", shares=10.0, price=100.0, apply_costs=False)
        assert "AAPL" in p.positions
        assert p.positions["AAPL"].shares == pytest.approx(10.0)

    def test_buy_deducts_cash(self):
        p = Portfolio(initial_capital=10_000.0)
        p.buy("AAPL", shares=10.0, price=100.0, apply_costs=False)
        assert p.cash == pytest.approx(9_000.0)

    def test_buy_applies_transaction_cost(self):
        p = Portfolio(initial_capital=10_000.0)
        cost = p.buy("AAPL", shares=10.0, price=100.0, apply_costs=True)
        expected_gross = 1_000.0
        expected_fee = expected_gross * TRANSACTION_COST_RATE
        assert cost == pytest.approx(expected_gross + expected_fee, rel=1e-5)
        assert p.cash == pytest.approx(10_000.0 - cost, rel=1e-5)

    def test_buy_averages_cost(self):
        p = Portfolio(initial_capital=10_000.0)
        p.buy("AAPL", shares=10.0, price=100.0, apply_costs=False)
        p.buy("AAPL", shares=10.0, price=120.0, apply_costs=False)
        assert p.positions["AAPL"].avg_cost == pytest.approx(110.0)
        assert p.positions["AAPL"].shares == pytest.approx(20.0)

    def test_buy_capped_at_available_cash(self):
        p = Portfolio(initial_capital=500.0)
        p.buy("AAPL", shares=100.0, price=100.0, apply_costs=False)
        # Can only buy 5 shares with 500
        assert p.positions["AAPL"].shares == pytest.approx(5.0, rel=1e-3)
        assert p.cash >= 0

    def test_sell_removes_position_when_fully_sold(self):
        p = Portfolio(initial_capital=10_000.0)
        p.buy("AAPL", shares=10.0, price=100.0, apply_costs=False)
        p.sell("AAPL", shares=10.0, price=100.0, apply_costs=False)
        assert "AAPL" not in p.positions

    def test_sell_realizes_pnl(self):
        p = Portfolio(initial_capital=10_000.0)
        p.buy("AAPL", shares=10.0, price=100.0, apply_costs=False)
        p.sell("AAPL", shares=10.0, price=120.0, apply_costs=False)
        assert p.realized_pnl == pytest.approx(200.0)  # (120-100) * 10

    def test_sell_nonexistent_ticker_is_noop(self):
        p = Portfolio(initial_capital=10_000.0)
        result = p.sell("NVDA", shares=5.0, price=400.0)
        assert result == 0.0


# ── Snapshot tests ────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_record_snapshot_appends_to_history(self, simple_portfolio):
        prices = {"AAPL": 110.0, "SPY": 220.0}
        simple_portfolio.record_snapshot("2022-02-01", prices)
        assert len(simple_portfolio.history) == 1

    def test_snapshot_calculates_correct_monthly_return(self, simple_portfolio):
        # First snapshot at +10% prices
        prices_t1 = {"AAPL": 110.0, "SPY": 220.0}
        snap = simple_portfolio.record_snapshot("2022-02-01", prices_t1)
        assert snap.monthly_return_pct == pytest.approx(10.0, abs=0.5)

    def test_snapshot_cumulative_return(self, simple_portfolio):
        prices = {"AAPL": 150.0, "SPY": 300.0}  # +50%
        snap = simple_portfolio.record_snapshot("2022-06-01", prices)
        assert snap.cumulative_return_pct == pytest.approx(50.0, abs=1.0)

    def test_history_to_dataframe(self, portfolio_with_history):
        df = portfolio_with_history.history_to_dataframe()
        assert not df.empty
        assert "portfolio_value" in df.columns
        assert "monthly_return_pct" in df.columns
        assert len(df) == 12

    def test_monthly_returns_series_length(self, portfolio_with_history):
        returns = portfolio_with_history.monthly_returns_series()
        assert len(returns) == 12

    def test_snapshot_serialization(self, simple_portfolio):
        prices = {"AAPL": 110.0, "SPY": 220.0}
        snap = simple_portfolio.record_snapshot("2022-02-01", prices)
        restored = MonthlySnapshot.from_dict(snap.to_dict())
        assert restored.portfolio_value == pytest.approx(snap.portfolio_value)
        assert restored.monthly_return_pct == pytest.approx(snap.monthly_return_pct)


# ── Persistence tests ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, portfolio_with_history):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name

        try:
            portfolio_with_history.save(path)
            loaded = Portfolio.load(path)

            assert loaded.initial_capital == pytest.approx(portfolio_with_history.initial_capital)
            assert loaded.cash == pytest.approx(portfolio_with_history.cash, rel=1e-6)
            assert len(loaded.history) == len(portfolio_with_history.history)
            assert list(loaded.positions.keys()) == list(
                portfolio_with_history.positions.keys()
            )
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Portfolio.load("/nonexistent/path/portfolio.json")

    def test_to_dict_is_json_serializable(self, portfolio_with_history):
        data = portfolio_with_history.to_dict()
        serialized = json.dumps(data, default=str)
        assert len(serialized) > 0


# ── Risk dashboard tests ──────────────────────────────────────────────────────

class TestRiskDashboard:
    def test_compute_metrics_requires_history(self, simple_portfolio):
        dash = RiskDashboard(simple_portfolio)
        # No history yet → empty dict
        metrics = dash.compute_metrics()
        assert metrics == {}

    def test_compute_metrics_with_history(self, portfolio_with_history):
        dash = RiskDashboard(portfolio_with_history)
        metrics = dash.compute_metrics()
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "annualized_volatility_pct" in metrics

    def test_rolling_sharpe_returns_series(self, portfolio_with_history):
        dash = RiskDashboard(portfolio_with_history)
        rs = dash.rolling_sharpe(window=6)
        assert isinstance(rs, pd.Series)
        assert len(rs) > 0

    def test_sector_exposure_returns_dataframe(self, portfolio_with_history):
        dash = RiskDashboard(portfolio_with_history)
        df = dash.sector_exposure()
        assert isinstance(df, pd.DataFrame)
        assert "sector" in df.columns
        assert "weight_pct" in df.columns

    def test_worst_month(self, portfolio_with_history):
        dash = RiskDashboard(portfolio_with_history)
        worst = dash.worst_month()
        assert worst is not None
        assert "monthly_return_pct" in worst

    def test_best_month(self, portfolio_with_history):
        dash = RiskDashboard(portfolio_with_history)
        best = dash.best_month()
        assert best is not None
        assert best["monthly_return_pct"] >= dash.worst_month()["monthly_return_pct"]


# ── Monte Carlo tests ─────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_simulate_returns_correct_shape(self, portfolio_with_history):
        mc = MonteCarloProjector(portfolio_with_history, n_paths=100, seed=42)
        paths = mc.simulate(horizon_months=6)
        assert paths.shape == (7, 100)  # 6 steps + initial row

    def test_first_row_equals_current_value(self, portfolio_with_history):
        mc = MonteCarloProjector(portfolio_with_history, n_paths=100, seed=42)
        v0 = portfolio_with_history.history[-1].portfolio_value
        paths = mc.simulate(horizon_months=6)
        assert paths[0].mean() == pytest.approx(v0, rel=1e-6)

    def test_percentile_bands_shape(self, portfolio_with_history):
        mc = MonteCarloProjector(portfolio_with_history, n_paths=200, seed=0)
        paths = mc.simulate(6)
        bands = mc.percentile_bands(paths)
        assert "p10" in bands.columns
        assert "p50" in bands.columns
        assert "p90" in bands.columns
        assert len(bands) == 7

    def test_p50_greater_than_p10(self, portfolio_with_history):
        mc = MonteCarloProjector(portfolio_with_history, n_paths=500, seed=7)
        paths = mc.simulate(12)
        bands = mc.percentile_bands(paths)
        assert (bands["p50"] >= bands["p10"]).all()
        assert (bands["p90"] >= bands["p50"]).all()

    def test_probability_above_below_sum_to_one(self, portfolio_with_history):
        """P(>x) + P(≤x) ≈ 1 at any threshold."""
        mc = MonteCarloProjector(portfolio_with_history, n_paths=1000, seed=42)
        paths = mc.simulate(12)
        threshold = 100_000.0
        p_above = mc.probability_above(paths, threshold)
        p_below = mc.probability_below(paths, threshold)
        assert p_above + p_below == pytest.approx(1.0, abs=0.01)

    def test_insufficient_history_raises(self):
        p = Portfolio(initial_capital=100_000.0)
        mc = MonteCarloProjector(p, n_paths=100)
        with pytest.raises(ValueError, match="at least 3 months"):
            mc.simulate(6)


# ── Rebalancer tests ──────────────────────────────────────────────────────────

class TestRebalancer:
    def test_suggest_returns_dataframe(self, portfolio_with_history):
        prices = portfolio_with_history.history[-1].asset_prices
        rebalancer = Rebalancer(mode="fixed", drift_threshold=0.0)
        df = rebalancer.suggest(portfolio_with_history, prices)
        assert isinstance(df, pd.DataFrame)
        assert "ticker" in df.columns
        assert "current_weight" in df.columns
        assert "target_weight" in df.columns

    def test_rebalance_returns_float(self, simple_portfolio):
        prices = {"AAPL": 100.0, "SPY": 200.0}
        rebalancer = Rebalancer(mode="fixed", drift_threshold=0.0)
        costs = rebalancer.rebalance(simple_portfolio, prices, "2022-01")
        assert isinstance(costs, float)
        assert costs >= 0.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            Rebalancer(mode="weekly")

    def test_no_rebalance_needed_below_threshold(self, simple_portfolio):
        """If drift is within threshold, no trades should fire."""
        prices = {"AAPL": 100.0, "SPY": 200.0}
        # Equal weights, large threshold → no trading
        rebalancer = Rebalancer(mode="fixed", drift_threshold=0.99)
        costs = rebalancer.rebalance(simple_portfolio, prices, "2022-01")
        assert costs == pytest.approx(0.0)


# ── Simulator integration test (mocked) ──────────────────────────────────────

class TestSimulatorIntegration:
    @patch("quantforge.portfolio_simulator.simulator.yf.download")
    def test_run_produces_history(self, mock_download):
        """Simulate 3 months with mocked yfinance data."""
        # Build a fake price DataFrame
        dates = pd.date_range("2022-01-01", periods=90, freq="D")
        price_data = pd.DataFrame(
            {
                "AAPL": 100.0 + np.arange(90) * 0.1,
                "SPY":  200.0 + np.arange(90) * 0.05,
                "SPY_bench": 200.0 + np.arange(90) * 0.05,  # benchmark
            },
            index=dates,
        )
        # Simulate MultiIndex columns (yfinance format)
        price_data.columns = pd.MultiIndex.from_tuples(
            [("Close", t) for t in price_data.columns]
        )
        mock_download.return_value = price_data

        p = Portfolio(initial_capital=100_000.0, inception_date="2022-01-01")
        p.initialize_positions(
            {"AAPL": 0.5, "SPY": 0.5},
            {"AAPL": 100.0, "SPY": 200.0},
        )

        sim = PortfolioSimulator(
            portfolio=p,
            tickers=["AAPL", "SPY"],
            start_date="2022-01-01",
            end_date="2022-03-31",
            rebalance_freq="never",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            state_path = tmp.name

        try:
            history_df = sim.run(state_file=state_path)
            assert not history_df.empty
            assert len(p.history) >= 1
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)
