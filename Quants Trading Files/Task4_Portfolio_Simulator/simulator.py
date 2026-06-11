"""
Portfolio Monthly Simulation Engine
=====================================
Drives the month-by-month simulation loop: fetches adjusted close prices
for all tickers via yfinance, marks the portfolio to market at each period
end, triggers optional rebalancing, and persists the resulting history.

Supports two modes:
  • Historical  — step from a past start_date to today (or any end_date)
  • Paper-trade — fetch live prices and record a single forward-looking step

Usage (programmatic):
    portfolio = Portfolio(initial_capital=100_000)
    portfolio.initialize_positions({"AAPL": 0.3, "SPY": 0.5, "BND": 0.2}, prices)

    sim = PortfolioSimulator(portfolio, start_date="2022-01-01")
    history_df = sim.run()
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.config import (
    DEFAULT_TICKERS,
    DEFAULT_STATE_FILE,
    DEFAULT_OUTPUT_DIR,
    BENCHMARK_TICKER,
)

logger = get_logger(__name__)


class PortfolioSimulator:
    """
    Month-by-month portfolio simulation engine.

    Financial Context:
        The simulation snapshots portfolio value at the last available
        trading day of each calendar month. This mirrors standard
        performance reporting (end-of-month NAV). Using month-end
        prices also smooths intra-month noise for Sharpe/drawdown calculations.

    Attributes:
        portfolio:      The Portfolio instance being stepped through time.
        tickers:        Asset tickers to track (defaults to current positions).
        start_date:     First simulation month (YYYY-MM-DD).
        end_date:       Last simulation month (YYYY-MM-DD, default today).
        rebalance_freq: 'monthly' | 'quarterly' | 'never'.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: str = "quarterly",
    ):
        self.portfolio = portfolio
        self.tickers = tickers or list(portfolio.positions.keys()) or DEFAULT_TICKERS
        self.start_date = start_date or portfolio.inception_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.rebalance_freq = rebalance_freq

        if self.rebalance_freq not in ("monthly", "quarterly", "never"):
            raise ValueError(
                f"rebalance_freq must be 'monthly', 'quarterly', or 'never', "
                f"got '{self.rebalance_freq}'"
            )

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch_all_prices(
        self,
        extra_tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Download adjusted close prices for all simulation tickers in one request.

        Fetches from slightly before start_date to ensure the first month has
        a price even if start_date falls on a non-trading day.

        Args:
            extra_tickers: Additional tickers to fetch alongside the universe
                           (e.g. benchmark SPY).

        Returns:
            DataFrame with DatetimeIndex and one column per ticker.
        """
        all_tickers = list(dict.fromkeys(self.tickers + (extra_tickers or [])))
        buffer_start = (
            datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=15)
        ).strftime("%Y-%m-%d")

        logger.info(
            "Downloading price data for %d tickers [%s → %s]",
            len(all_tickers), buffer_start, self.end_date,
        )
        raw = yf.download(
            all_tickers,
            start=buffer_start,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for tickers: {all_tickers}"
            )

        # yfinance >= 0.2.31 returns MultiIndex (Price, Ticker) columns
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            # Single ticker — raw itself is the OHLCV frame
            closes = raw[["Close"]].copy()
            closes.columns = [all_tickers[0]]

        closes = closes.ffill().bfill()

        always_missing = closes.columns[closes.isnull().all()].tolist()
        if always_missing:
            logger.warning("Dropping tickers with no data at all: %s", always_missing)
            closes = closes.drop(columns=always_missing)

        logger.info(
            "Price matrix: %d rows × %d tickers", len(closes), len(closes.columns)
        )
        return closes

    def get_month_end_prices(
        self, prices: pd.DataFrame, year: int, month: int
    ) -> Dict[str, float]:
        """
        Return the last available closing price for each ticker in a given month.

        Falls back to the most recent price before month-end if no data exists
        within the requested month (e.g. during a gap or early in the series).

        Args:
            prices: Full price DataFrame from fetch_all_prices().
            year:   Calendar year.
            month:  Calendar month (1–12).

        Returns:
            Dict of ticker → closing price.
        """
        mask = (prices.index.year == year) & (prices.index.month == month)
        month_data = prices.loc[mask]

        if month_data.empty:
            eom = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
            month_data = prices.loc[prices.index <= eom].tail(1)

        if month_data.empty:
            return {}

        last_row = month_data.iloc[-1]
        return {
            col: float(last_row[col])
            for col in prices.columns
            if not pd.isna(last_row[col])
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        rebalancer: Optional[object] = None,
        state_file: str = DEFAULT_STATE_FILE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> pd.DataFrame:
        """
        Execute the full historical simulation month by month.

        At each step:
          1. Mark portfolio to month-end market prices.
          2. Optionally trigger rebalancer (if frequency condition is met).
          3. Record a MonthlySnapshot.

        After the loop, persist portfolio state to state_file.

        Args:
            rebalancer:  Optional Rebalancer instance for periodic rebalancing.
                         Must implement rebalancer.rebalance(portfolio, prices,
                         period_label) → float (total costs).
            state_file:  Where to save portfolio state after simulation.
            output_dir:  Directory for chart/report outputs (used by downstream).

        Returns:
            DataFrame of monthly portfolio history (one row per month).
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        logger.info(
            "Simulation starting | %s → %s | tickers: %s | rebalance: %s",
            self.start_date, self.end_date,
            ", ".join(self.tickers), self.rebalance_freq,
        )

        prices = self.fetch_all_prices(extra_tickers=[BENCHMARK_TICKER])

        # Build month sequence (first day of each calendar month)
        months = pd.date_range(
            start=start_dt.replace(day=1),
            end=end_dt,
            freq="MS",   # Month Start — pandas >= 2.0 uses 'MS'
        )

        for step, month_start in enumerate(months):
            year, month = month_start.year, month_start.month
            period_label = month_start.strftime("%Y-%m")

            all_prices = self.get_month_end_prices(prices, year, month)
            if not all_prices:
                logger.warning("[%s] No price data — skipping.", period_label)
                continue

            # Only pass prices relevant to the portfolio (exclude pure-benchmark tickers)
            portfolio_tickers = set(self.portfolio.positions.keys()) | set(
                self.portfolio.target_weights.keys()
            )
            port_prices = {k: v for k, v in all_prices.items() if k in portfolio_tickers}

            if not port_prices:
                logger.warning("[%s] No portfolio-ticker prices — skipping.", period_label)
                continue

            # ── Rebalancing ────────────────────────────────────────────────
            period_costs = 0.0
            if rebalancer is not None and self._should_rebalance(step):
                try:
                    period_costs = rebalancer.rebalance(
                        portfolio=self.portfolio,
                        prices=port_prices,
                        period_label=period_label,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] Rebalancer error: %s", period_label, exc)

            # ── Snapshot ───────────────────────────────────────────────────
            self.portfolio.record_snapshot(
                date=f"{period_label}-01",
                prices=port_prices,
                transaction_costs=period_costs,
            )

        n_months = len(self.portfolio.history)
        if n_months == 0:
            logger.warning("Simulation produced no snapshots — check date range and tickers.")
            return pd.DataFrame()

        final = self.portfolio.history[-1]
        logger.info(
            "Simulation complete | %d months | Final: €%,.2f | Cumul: %+.2f%%",
            n_months, final.portfolio_value, final.cumulative_return_pct,
        )

        self.portfolio.save(state_file)
        return self.portfolio.history_to_dataframe()

    def _should_rebalance(self, step: int) -> bool:
        """Return True if this step index should trigger a rebalance."""
        if self.rebalance_freq == "never":
            return False
        if self.rebalance_freq == "monthly":
            return True
        if self.rebalance_freq == "quarterly":
            return step % 3 == 0
        return False

    # ── Paper-trade (live price) step ─────────────────────────────────────────

    def step_live(self, state_file: str = DEFAULT_STATE_FILE) -> Optional[Dict]:
        """
        Record a single live-price snapshot (paper-trade mode).

        Fetches the latest available prices for all portfolio holdings,
        records a snapshot dated today, and saves state.

        Returns:
            Dict with today's snapshot data, or None if prices unavailable.
        """
        tickers = list(self.portfolio.positions.keys())
        if not tickers:
            logger.warning("No open positions — nothing to update.")
            return None

        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            logger.error("Could not fetch live prices.")
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw

        live_prices: Dict[str, float] = {}
        for t in tickers:
            if t in closes.columns:
                series = closes[t].dropna()
                if not series.empty:
                    live_prices[t] = float(series.iloc[-1])

        if not live_prices:
            logger.error("No valid live prices obtained.")
            return None

        today = datetime.now().strftime("%Y-%m-%d")
        snapshot = self.portfolio.record_snapshot(
            date=today,
            prices=live_prices,
            transaction_costs=0.0,
        )
        self.portfolio.save(state_file)
        return snapshot.to_dict()

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark_monthly_returns(self) -> pd.Series:
        """
        Compute monthly SPY returns over the simulation window.

        Useful for plotting portfolio performance vs. benchmark.

        Returns:
            Series of monthly simple returns, indexed by month-start date.
        """
        try:
            raw = yf.download(
                BENCHMARK_TICKER,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch benchmark data: %s", exc)
            return pd.Series(dtype=float)

        if raw.empty:
            return pd.Series(dtype=float)

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"][BENCHMARK_TICKER]
        else:
            closes = raw["Close"]

        monthly = closes.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.name = BENCHMARK_TICKER
        return returns

    # ── Utilities ─────────────────────────────────────────────────────────────

    def simulation_months(self) -> List[Tuple[int, int]]:
        """
        Return the list of (year, month) tuples covered by the simulation window.
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        months = pd.date_range(
            start=start_dt.replace(day=1),
            end=end_dt,
            freq="MS",
        )
        return [(d.year, d.month) for d in months]
