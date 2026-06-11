"""
Portfolio Simulator — Entry Point
===================================
Wires together all modules and provides two execution modes:

  1. simulate   — Run the full historical simulation (or resume from state)
  2. assistant  — Load saved state and launch the interactive CLI assistant

Usage examples:
    python -m quantforge.portfolio_simulator.run_simulator simulate \\
        --start 2022-01-01 \\
        --tickers AAPL MSFT SPY BND \\
        --weights 0.30 0.25 0.30 0.15 \\
        --rebalance quarterly

    python -m quantforge.portfolio_simulator.run_simulator assistant \\
        --state portfolio_state.json

    python -m quantforge.portfolio_simulator.run_simulator simulate \\
        --start 2020-01-01 --rebalance monthly --optimizer black_litterman
"""

import argparse
import os
import sys

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.simulator import PortfolioSimulator
from quantforge.portfolio_simulator.rebalancer import Rebalancer
from quantforge.portfolio_simulator.risk_dashboard import RiskDashboard
from quantforge.portfolio_simulator.monte_carlo import MonteCarloProjector
from quantforge.portfolio_simulator.reporter import ReportGenerator
from quantforge.portfolio_simulator.assistant import PortfolioAssistant
from quantforge.portfolio_simulator.config import (
    DEFAULT_TICKERS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_STATE_FILE,
    DEFAULT_OUTPUT_DIR,
)

logger = get_logger(__name__)


# ── CLI argument parsing ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_simulator",
        description="QuantForge Portfolio Simulator & Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── simulate ─────────────────────────────────────────────────────────────
    sim_p = sub.add_parser("simulate", help="Run the historical portfolio simulation.")
    sim_p.add_argument(
        "--start", default="2020-01-01",
        help="Simulation start date (YYYY-MM-DD). Default: 2020-01-01",
    )
    sim_p.add_argument(
        "--end", default=None,
        help="Simulation end date (YYYY-MM-DD). Default: today",
    )
    sim_p.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help=f"Asset tickers. Default: {' '.join(DEFAULT_TICKERS)}",
    )
    sim_p.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help=(
            "Target weights as decimals, must match --tickers order "
            "(e.g. 0.30 0.25 0.30 0.15). "
            "If omitted, equal weights are used."
        ),
    )
    sim_p.add_argument(
        "--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital in euros. Default: {DEFAULT_INITIAL_CAPITAL:,.0f}",
    )
    sim_p.add_argument(
        "--rebalance", choices=("monthly", "quarterly", "never"),
        default="quarterly",
        help="Rebalancing frequency. Default: quarterly",
    )
    sim_p.add_argument(
        "--optimizer", choices=("markowitz", "black_litterman"),
        default="markowitz",
        help="Optimizer for dynamic rebalancing. Default: markowitz",
    )
    sim_p.add_argument(
        "--rebalance-mode", choices=("fixed", "dynamic"),
        default="fixed", dest="rebalance_mode",
        help=(
            "Rebalancing mode. 'fixed' uses --weights as target; "
            "'dynamic' re-derives weights each period via the optimizer. Default: fixed"
        ),
    )
    sim_p.add_argument(
        "--state", default=DEFAULT_STATE_FILE,
        help=f"Portfolio state JSON file. Default: {DEFAULT_STATE_FILE}",
    )
    sim_p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir",
        help=f"Directory for charts and reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    sim_p.add_argument(
        "--no-report", action="store_true",
        help="Skip generating the HTML report after simulation.",
    )
    sim_p.add_argument(
        "--assistant", action="store_true",
        help="Launch the interactive assistant after the simulation completes.",
    )

    # ── assistant ─────────────────────────────────────────────────────────────
    asst_p = sub.add_parser(
        "assistant",
        help="Load saved portfolio state and launch the interactive assistant.",
    )
    asst_p.add_argument(
        "--state", default=DEFAULT_STATE_FILE,
        help=f"Path to portfolio_state.json. Default: {DEFAULT_STATE_FILE}",
    )
    asst_p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir",
        help=f"Directory for charts and reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    asst_p.add_argument(
        "--optimizer", choices=("markowitz", "black_litterman"),
        default="markowitz",
    )

    return parser


# ── Simulation pipeline ───────────────────────────────────────────────────────

def run_simulation(args: argparse.Namespace) -> Portfolio:
    """
    Execute the full simulation pipeline.

    1. Build allocations from tickers + weights.
    2. Fetch first-day prices to initialize positions.
    3. Create Portfolio and initialize positions.
    4. Run PortfolioSimulator month-by-month.
    5. Print the risk dashboard.
    6. (Optionally) generate the HTML report.

    Returns:
        The fully populated Portfolio instance.
    """
    tickers = [t.upper() for t in args.tickers]

    # ── Build target allocations ──────────────────────────────────────────
    if args.weights:
        if len(args.weights) != len(tickers):
            logger.error(
                "--weights count (%d) must match --tickers count (%d).",
                len(args.weights), len(tickers),
            )
            sys.exit(1)
        allocations = dict(zip(tickers, args.weights))
    else:
        eq_w = 1.0 / len(tickers)
        allocations = {t: eq_w for t in tickers}

    logger.info("Target allocations: %s", {k: f"{v:.2%}" for k, v in allocations.items()})

    # ── Fetch inception prices ────────────────────────────────────────────
    logger.info("Fetching inception prices as of %s …", args.start)
    import yfinance as yf
    import pandas as pd

    raw = yf.download(tickers, start=args.start, period="5d", auto_adjust=True, progress=False)

    if raw.empty:
        logger.error("Could not fetch prices for %s. Check tickers and date.", tickers)
        sys.exit(1)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw

    inception_prices: dict = {}
    for t in tickers:
        if t in closes.columns:
            s = closes[t].dropna()
            if not s.empty:
                inception_prices[t] = float(s.iloc[0])

    missing = [t for t in tickers if t not in inception_prices]
    if missing:
        logger.warning("No inception price for %s — removing from allocation.", missing)
        for t in missing:
            allocations.pop(t, None)
        tickers = [t for t in tickers if t not in missing]

    if not allocations:
        logger.error("No valid tickers with price data. Aborting.")
        sys.exit(1)

    # ── Build portfolio ───────────────────────────────────────────────────
    portfolio = Portfolio(
        initial_capital=args.capital,
        inception_date=args.start,
    )
    portfolio.initialize_positions(allocations, inception_prices)

    # ── Build rebalancer ──────────────────────────────────────────────────
    rebalancer = Rebalancer(
        mode=args.rebalance_mode,
        optimizer_mode=args.optimizer,
    )

    # ── Run simulation ────────────────────────────────────────────────────
    sim = PortfolioSimulator(
        portfolio=portfolio,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        rebalance_freq=args.rebalance,
    )

    history_df = sim.run(
        rebalancer=rebalancer if args.rebalance != "never" else None,
        state_file=args.state,
        output_dir=args.output_dir,
    )

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard = RiskDashboard(portfolio)
    dashboard.print_dashboard()

    # ── Save equity chart ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    bench = sim.benchmark_monthly_returns()
    chart_path = os.path.join(args.output_dir, "equity_chart.html")
    dashboard.plot_equity_and_drawdown(
        benchmark_returns=bench if not bench.empty else None,
        save_path=chart_path,
    )

    # ── HTML report ───────────────────────────────────────────────────────
    if not args.no_report:
        reporter = ReportGenerator(portfolio, output_dir=args.output_dir)
        report_path = reporter.generate(benchmark_returns=bench if not bench.empty else None)
        print(f"Report → {report_path}")

    return portfolio


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "simulate":
        portfolio = run_simulation(args)
        if args.assistant:
            rebalancer = Rebalancer(
                mode=args.rebalance_mode,
                optimizer_mode=args.optimizer,
            )
            asst = PortfolioAssistant(
                portfolio=portfolio,
                rebalancer=rebalancer,
                output_dir=args.output_dir,
                state_file=args.state,
            )
            asst.run()

    elif args.command == "assistant":
        try:
            portfolio = Portfolio.load(args.state)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        rebalancer = Rebalancer(
            mode="fixed",
            optimizer_mode=args.optimizer,
        )
        asst = PortfolioAssistant(
            portfolio=portfolio,
            rebalancer=rebalancer,
            output_dir=args.output_dir,
            state_file=args.state,
        )
        asst.run()


if __name__ == "__main__":
    main()
