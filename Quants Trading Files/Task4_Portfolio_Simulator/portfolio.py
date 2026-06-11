"""
Portfolio State Management
===========================
Defines the Portfolio, Position, and MonthlySnapshot classes that together
represent the complete state of a simulated portfolio.

All state is fully JSON-serializable so it can be persisted between sessions.
P&L accounting follows the FIFO cost-basis convention; fractional shares are
allowed because this is a simulation context (no fractional-share brokerage
complications to worry about).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.config import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_STATE_FILE,
    TRANSACTION_COST_RATE,
)

logger = get_logger(__name__)


# ── Position ─────────────────────────────────────────────────────────────────

class Position:
    """
    Represents a single open asset position.

    Attributes:
        ticker: Asset symbol (e.g. 'AAPL').
        shares: Number of shares held (fractional ok).
        avg_cost: Average cost per share in euros (weighted average of all buys).
    """

    def __init__(self, ticker: str, shares: float, avg_cost: float):
        self.ticker = ticker
        self.shares = shares
        self.avg_cost = avg_cost

    # ── Derived values ────────────────────────────────────────────────────────

    def cost_basis(self) -> float:
        """Total amount paid for the current number of shares."""
        return self.shares * self.avg_cost

    def market_value(self, price: float) -> float:
        """Current market value at the given price."""
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        """Mark-to-market gain/loss vs. cost basis."""
        return self.market_value(price) - self.cost_basis()

    def unrealized_pnl_pct(self, price: float) -> float:
        """Unrealized P&L as a percentage of cost basis."""
        basis = self.cost_basis()
        return (self.unrealized_pnl(price) / basis * 100) if basis > 0 else 0.0

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "ticker":   self.ticker,
            "shares":   self.shares,
            "avg_cost": self.avg_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Position":
        return cls(
            ticker=data["ticker"],
            shares=data["shares"],
            avg_cost=data["avg_cost"],
        )

    def __repr__(self) -> str:
        return f"Position({self.ticker}: {self.shares:.4f} shares @ €{self.avg_cost:.4f})"


# ── MonthlySnapshot ───────────────────────────────────────────────────────────

class MonthlySnapshot:
    """
    Immutable record of portfolio state at the end of one calendar month.

    This is the append-only history unit — one snapshot is created per
    simulation step and stored in Portfolio.history.

    Financial Context:
        Monthly snapshots give the granularity needed for rolling Sharpe
        and drawdown calculations without the noise of daily data.
    """

    def __init__(
        self,
        date: str,
        portfolio_value: float,
        cash: float,
        positions_value: float,
        weights: Dict[str, float],
        monthly_return_pct: float,
        cumulative_return_pct: float,
        drawdown_pct: float,
        unrealized_pnl: float,
        realized_pnl: float,
        transaction_costs: float,
        asset_prices: Optional[Dict[str, float]] = None,
    ):
        self.date = date
        self.portfolio_value = portfolio_value
        self.cash = cash
        self.positions_value = positions_value
        self.weights = weights
        self.monthly_return_pct = monthly_return_pct
        self.cumulative_return_pct = cumulative_return_pct
        self.drawdown_pct = drawdown_pct
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.transaction_costs = transaction_costs
        self.asset_prices = asset_prices or {}

    def to_dict(self) -> Dict:
        return {
            "date":                  self.date,
            "portfolio_value":       self.portfolio_value,
            "cash":                  self.cash,
            "positions_value":       self.positions_value,
            "weights":               self.weights,
            "monthly_return_pct":    self.monthly_return_pct,
            "cumulative_return_pct": self.cumulative_return_pct,
            "drawdown_pct":          self.drawdown_pct,
            "unrealized_pnl":        self.unrealized_pnl,
            "realized_pnl":          self.realized_pnl,
            "transaction_costs":     self.transaction_costs,
            "asset_prices":          self.asset_prices,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MonthlySnapshot":
        return cls(**data)


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Virtual portfolio with full state management and JSON persistence.

    Financial Context:
        Models a real brokerage account holding stocks and ETFs.
        Positions track average cost basis for unrealized P&L calculation.
        All transactions apply the configured transaction cost rate (0.1%).
        Drawdown is computed from the portfolio's all-time high since inception.

    Attributes:
        initial_capital:  Starting capital in euros.
        cash:             Uninvested cash currently held.
        positions:        Open positions, keyed by ticker.
        target_weights:   Desired allocation fractions (ticker → weight).
        history:          Ordered list of monthly snapshots.
        realized_pnl:     Cumulative P&L from fully or partially closed trades.
        peak_value:       Highest portfolio value ever reached (for drawdown).
        inception_date:   ISO date string when the portfolio was created.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        target_weights: Optional[Dict[str, float]] = None,
        inception_date: Optional[str] = None,
    ):
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.target_weights: Dict[str, float] = target_weights or {}
        self.history: List[MonthlySnapshot] = []
        self.realized_pnl: float = 0.0
        self.peak_value: float = initial_capital
        self.inception_date: str = inception_date or datetime.now().strftime("%Y-%m-%d")

    # ── Order execution ───────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        apply_costs: bool = True,
    ) -> float:
        """
        Execute a buy order, updating positions and deducting cash.

        If the order would exceed available cash, shares are automatically
        reduced to the maximum affordable quantity.

        Args:
            ticker:       Asset symbol.
            shares:       Desired number of shares (fractional allowed).
            price:        Execution price per share (euros).
            apply_costs:  Whether to apply the transaction cost rate.

        Returns:
            Total cash outflow (including fees).
        """
        if shares <= 0 or price <= 0:
            return 0.0

        cost_rate = TRANSACTION_COST_RATE if apply_costs else 0.0
        gross = shares * price
        fee = gross * cost_rate
        total_outflow = gross + fee

        # Cap to available cash
        if total_outflow > self.cash:
            affordable_shares = self.cash / (price * (1.0 + cost_rate))
            shares = max(0.0, affordable_shares)
            gross = shares * price
            fee = gross * cost_rate
            total_outflow = gross + fee
            if shares < 1e-8:
                return 0.0

        # Update weighted average cost
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.cost_basis() + gross) / total_shares
            pos.shares = total_shares
        else:
            self.positions[ticker] = Position(ticker, shares, price)

        self.cash -= total_outflow
        logger.info(
            "BUY  %-8s %.4f shares @ €%.4f | fee €%.2f | cash €%.2f",
            ticker, shares, price, fee, self.cash,
        )
        return total_outflow

    def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        apply_costs: bool = True,
    ) -> float:
        """
        Execute a sell order, realizing P&L and crediting cash.

        Args:
            ticker:       Asset symbol.
            shares:       Shares to sell (capped at position size).
            price:        Execution price per share (euros).
            apply_costs:  Whether to apply the transaction cost rate.

        Returns:
            Net cash inflow (proceeds minus fees).
        """
        if ticker not in self.positions or shares <= 0 or price <= 0:
            logger.warning("Cannot sell %s — no position or invalid parameters.", ticker)
            return 0.0

        pos = self.positions[ticker]
        shares = min(shares, pos.shares)
        gross = shares * price
        fee = gross * TRANSACTION_COST_RATE if apply_costs else 0.0
        net_proceeds = gross - fee

        # Realize P&L at average cost
        self.realized_pnl += (price - pos.avg_cost) * shares

        pos.shares -= shares
        if pos.shares < 1e-8:
            del self.positions[ticker]

        self.cash += net_proceeds
        logger.info(
            "SELL %-8s %.4f shares @ €%.4f | fee €%.2f | cash €%.2f",
            ticker, shares, price, fee, self.cash,
        )
        return net_proceeds

    # ── Initialization helper ─────────────────────────────────────────────────

    def initialize_positions(
        self,
        allocations: Dict[str, float],
        prices: Dict[str, float],
    ) -> None:
        """
        Set up the initial portfolio by buying assets at target weights.

        Allocations are normalized if they do not sum exactly to 1.
        No transaction costs are applied at inception (simulates T=0 setup).

        Args:
            allocations: Mapping of ticker → desired weight (e.g. 0.20 for 20%).
            prices:      Mapping of ticker → current price per share.
        """
        if not allocations:
            raise ValueError("allocations dict cannot be empty.")

        # Normalize weights
        total = sum(allocations.values())
        if abs(total - 1.0) > 1e-4:
            logger.warning("Allocations sum to %.4f — normalizing to 1.0.", total)
            allocations = {k: v / total for k, v in allocations.items()}

        self.target_weights = dict(allocations)

        for ticker, weight in allocations.items():
            if ticker not in prices:
                logger.warning("No price for %s at inception — skipping.", ticker)
                continue
            capital = self.initial_capital * weight
            shares = capital / prices[ticker]
            self.buy(ticker, shares, prices[ticker], apply_costs=False)

        invested = self.initial_capital - self.cash
        logger.info(
            "Portfolio initialized: €%.2f invested, €%.2f cash remaining.",
            invested, self.cash,
        )

    # ── Valuation ─────────────────────────────────────────────────────────────

    def compute_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value: sum of positions at market prices plus cash."""
        invested = sum(
            pos.market_value(prices[t])
            for t, pos in self.positions.items()
            if t in prices
        )
        return invested + self.cash

    def compute_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Current portfolio weights as fractions of total value.

        Returns a dict including a '_cash' key for uninvested cash weight.
        """
        total = self.compute_value(prices)
        if total <= 0:
            return {}
        weights = {
            t: pos.market_value(prices[t]) / total
            for t, pos in self.positions.items()
            if t in prices
        }
        weights["_cash"] = self.cash / total
        return weights

    def compute_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Aggregate unrealized P&L across all open positions."""
        return sum(
            pos.unrealized_pnl(prices[t])
            for t, pos in self.positions.items()
            if t in prices
        )

    def compute_drawdown(self, current_value: float) -> float:
        """
        Drawdown from peak as a negative fraction (e.g. -0.15 = -15%).

        Updates peak_value whenever a new high is reached.
        """
        if current_value > self.peak_value:
            self.peak_value = current_value
        if self.peak_value <= 0:
            return 0.0
        return (current_value - self.peak_value) / self.peak_value

    def best_worst_asset(
        self, prices: Dict[str, float]
    ) -> Dict[str, object]:
        """
        Return the best and worst performing assets by unrealized P&L %.

        Returns:
            Dict with keys 'best' and 'worst', each containing
            {'ticker', 'pnl_pct'}.
        """
        if not self.positions:
            return {}
        perf = {
            t: pos.unrealized_pnl_pct(prices[t])
            for t, pos in self.positions.items()
            if t in prices
        }
        best = max(perf, key=perf.get)
        worst = min(perf, key=perf.get)
        return {
            "best":  {"ticker": best,  "pnl_pct": perf[best]},
            "worst": {"ticker": worst, "pnl_pct": perf[worst]},
        }

    # ── History ───────────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        date: str,
        prices: Dict[str, float],
        transaction_costs: float = 0.0,
    ) -> MonthlySnapshot:
        """
        Append a snapshot of the current portfolio state to history.

        Args:
            date:              ISO date string for this period (e.g. '2024-01-01').
            prices:            Dict of current asset prices.
            transaction_costs: Total fees incurred during this period.

        Returns:
            The MonthlySnapshot that was appended.
        """
        value = self.compute_value(prices)
        positions_value = value - self.cash
        weights = self.compute_weights(prices)
        unrealized = self.compute_unrealized_pnl(prices)
        drawdown = self.compute_drawdown(value)

        prev_value = (
            self.history[-1].portfolio_value if self.history else self.initial_capital
        )
        monthly_ret_pct = (value / prev_value - 1.0) * 100 if prev_value > 0 else 0.0
        cumul_ret_pct = (value / self.initial_capital - 1.0) * 100

        snapshot = MonthlySnapshot(
            date=date,
            portfolio_value=value,
            cash=self.cash,
            positions_value=positions_value,
            weights=weights,
            monthly_return_pct=monthly_ret_pct,
            cumulative_return_pct=cumul_ret_pct,
            drawdown_pct=drawdown * 100,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            transaction_costs=transaction_costs,
            asset_prices=dict(prices),
        )
        self.history.append(snapshot)

        logger.info(
            "[%s] Value: €%,.2f | Monthly: %+.2f%% | Cumul: %+.2f%% | DD: %.2f%%",
            date, value, monthly_ret_pct, cumul_ret_pct, drawdown * 100,
        )
        return snapshot

    def history_to_dataframe(self) -> pd.DataFrame:
        """
        Convert snapshot history into a tidy DataFrame indexed by date.

        Returns:
            DataFrame with DatetimeIndex, one row per snapshot.
            'weights' and 'asset_prices' columns contain nested dicts.
        """
        if not self.history:
            return pd.DataFrame()
        records = [s.to_dict() for s in self.history]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def monthly_returns_series(self) -> pd.Series:
        """Return a pandas Series of monthly returns (%) indexed by date."""
        df = self.history_to_dataframe()
        if df.empty:
            return pd.Series(dtype=float)
        return df["monthly_return_pct"] / 100  # as decimals for risk metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize full portfolio state to a JSON-compatible dict."""
        return {
            "initial_capital": self.initial_capital,
            "cash":            self.cash,
            "inception_date":  self.inception_date,
            "realized_pnl":    self.realized_pnl,
            "peak_value":      self.peak_value,
            "target_weights":  self.target_weights,
            "positions":       {k: v.to_dict() for k, v in self.positions.items()},
            "history":         [s.to_dict() for s in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Portfolio":
        """Reconstruct a Portfolio from its serialized dict representation."""
        p = cls(
            initial_capital=data["initial_capital"],
            target_weights=data.get("target_weights", {}),
            inception_date=data.get("inception_date"),
        )
        p.cash = data["cash"]
        p.realized_pnl = data.get("realized_pnl", 0.0)
        p.peak_value = data.get("peak_value", data["initial_capital"])
        p.positions = {
            k: Position.from_dict(v)
            for k, v in data.get("positions", {}).items()
        }
        p.history = [
            MonthlySnapshot.from_dict(s) for s in data.get("history", [])
        ]
        return p

    def save(self, path: str = DEFAULT_STATE_FILE) -> None:
        """
        Persist the full portfolio state to a JSON file.

        Args:
            path: Destination file path (created if it does not exist).
        """
        dir_path = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Portfolio state saved → %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_STATE_FILE) -> "Portfolio":
        """
        Load a previously saved portfolio state from a JSON file.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Portfolio state file not found: {path}\n"
                "Run the simulator first to create one."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Portfolio state loaded ← %s", path)
        return cls.from_dict(data)

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self, prices: Optional[Dict[str, float]] = None) -> str:
        """
        Return a formatted one-line summary of the current portfolio state.

        Args:
            prices: If provided, computes live values; otherwise uses last snapshot.
        """
        if prices:
            value = self.compute_value(prices)
            pnl = value - self.initial_capital
            pnl_pct = pnl / self.initial_capital * 100
        elif self.history:
            snap = self.history[-1]
            value = snap.portfolio_value
            pnl = value - self.initial_capital
            pnl_pct = snap.cumulative_return_pct
        else:
            return f"Portfolio: €{self.initial_capital:,.2f} (not yet simulated)"

        sign = "+" if pnl >= 0 else ""
        return (
            f"Portfolio: €{value:,.2f} | "
            f"P&L: {sign}€{pnl:,.2f} ({sign}{pnl_pct:.2f}%) | "
            f"Cash: €{self.cash:,.2f} | "
            f"Positions: {len(self.positions)}"
        )

    def __repr__(self) -> str:
        return (
            f"Portfolio(capital=€{self.initial_capital:,.0f}, "
            f"cash=€{self.cash:,.0f}, "
            f"positions={list(self.positions.keys())}, "
            f"snapshots={len(self.history)})"
        )
