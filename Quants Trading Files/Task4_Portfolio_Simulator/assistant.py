"""
Portfolio Assistant — Interactive CLI Chatbot
==============================================
A simple intent-routing CLI assistant that maps natural-language commands
to the appropriate module. No NLP library required: keyword matching is
sufficient for a structured financial domain.

Supported commands (case-insensitive, partial match):
  • "how is my portfolio"          → prints the risk dashboard
  • "worst month" / "best month"   → queries history
  • "rebalance" / "should i"       → shows rebalance recommendation
  • "drawdown chart" / "chart"     → opens equity & drawdown Plotly chart
  • "allocation" / "pie"           → shows allocation pie chart
  • "sector" / "exposure"          → prints sector breakdown
  • "sentiment"                    → runs FinBERT on current holdings
  • "monte carlo" / "simulate"     → runs GBM projection and shows chart
  • "report"                       → generates monthly HTML report
  • "help"                         → lists all commands
  • "quit" / "exit"                → exits the assistant
"""

import sys
from typing import Dict, List, Optional

from quantforge.utils.logger import get_logger
from quantforge.portfolio_simulator.portfolio import Portfolio
from quantforge.portfolio_simulator.risk_dashboard import RiskDashboard
from quantforge.portfolio_simulator.rebalancer import Rebalancer
from quantforge.portfolio_simulator.monte_carlo import MonteCarloProjector
from quantforge.portfolio_simulator.reporter import ReportGenerator
from quantforge.portfolio_simulator.config import (
    DEFAULT_STATE_FILE,
    DEFAULT_OUTPUT_DIR,
    MC_HORIZONS_MONTHS,
)

logger = get_logger(__name__)

# ── Intent routing table ──────────────────────────────────────────────────────
# Each entry: (keywords_tuple, handler_name)
_INTENTS = [
    (("how is",  "dashboard", "status", "summary", "overview"),  "dashboard"),
    (("worst month",),                                             "worst_month"),
    (("best month",),                                              "best_month"),
    (("rebalance", "should i", "suggest weight"),                  "rebalance"),
    (("drawdown chart", "equity chart", "performance chart",
      "show chart", "chart"),                                      "equity_chart"),
    (("allocation", "pie"),                                        "allocation_pie"),
    (("sector", "exposure", "tech"),                               "sector"),
    (("sentiment", "finbert", "news"),                             "sentiment"),
    (("monte carlo", "simulate next", "projection", "forward"),    "monte_carlo"),
    (("report",),                                                  "report"),
    (("help", "commands", "what can"),                             "help"),
    (("quit", "exit", "bye"),                                      "quit"),
]

_HELP_TEXT = """
Available commands:
  how is my portfolio          — current risk dashboard
  worst month / best month     — best and worst performing months
  should i rebalance?          — rebalancing recommendation
  show me my drawdown chart    — equity + drawdown Plotly chart
  show allocation / pie        — asset allocation pie chart
  what is my exposure to tech? — sector breakdown
  run sentiment analysis       — FinBERT sentiment on holdings
  simulate next 6 months       — Monte Carlo GBM projection
  generate report              — monthly HTML performance report
  help                         — show this list
  quit / exit                  — exit the assistant
"""


class PortfolioAssistant:
    """
    Interactive CLI assistant for the portfolio simulator.

    Attributes:
        portfolio:   The active Portfolio instance.
        dashboard:   RiskDashboard for metric and chart access.
        rebalancer:  Rebalancer for weight suggestions.
        output_dir:  Directory for saved charts and reports.
        state_file:  Path to the persistent portfolio state JSON.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        rebalancer: Optional[Rebalancer] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        state_file: str = DEFAULT_STATE_FILE,
    ):
        self.portfolio = portfolio
        self.dashboard = RiskDashboard(portfolio)
        self.rebalancer = rebalancer or Rebalancer(mode="fixed")
        self.output_dir = output_dir
        self.state_file = state_file

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive CLI loop."""
        print("\n" + "═" * 60)
        print("  QuantForge Portfolio Assistant")
        print("  Type 'help' for available commands, 'quit' to exit.")
        print("═" * 60 + "\n")
        print(self.portfolio.summary())

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue

            intent = self._classify(user_input)
            if intent is None:
                print("I didn't understand that. Type 'help' to see available commands.")
                continue

            if intent == "quit":
                print("Goodbye.")
                break

            self._dispatch(intent, user_input)

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify(self, text: str) -> Optional[str]:
        """Return the intent name for the user's input, or None."""
        lower = text.lower()
        for keywords, intent_name in _INTENTS:
            if any(kw in lower for kw in keywords):
                return intent_name
        return None

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self, intent: str, raw_input: str) -> None:
        """Call the handler for the resolved intent."""
        handlers = {
            "dashboard":     self._handle_dashboard,
            "worst_month":   self._handle_worst_month,
            "best_month":    self._handle_best_month,
            "rebalance":     self._handle_rebalance,
            "equity_chart":  self._handle_equity_chart,
            "allocation_pie": self._handle_allocation_pie,
            "sector":        self._handle_sector,
            "sentiment":     self._handle_sentiment,
            "monte_carlo":   self._handle_monte_carlo,
            "report":        self._handle_report,
            "help":          self._handle_help,
        }
        handler = handlers.get(intent)
        if handler:
            try:
                handler(raw_input)
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                logger.exception("Handler error for intent '%s'", intent)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_dashboard(self, _: str) -> None:
        """Print the full risk dashboard."""
        self.dashboard.print_dashboard()

    def _handle_worst_month(self, _: str) -> None:
        """Show the worst performing month."""
        snap = self.dashboard.worst_month()
        if snap is None:
            print("No simulation history found.")
            return
        print(f"\n  Worst Month: {snap['date'][:7]}")
        print(f"  Return      : {snap['monthly_return_pct']:+.2f}%")
        print(f"  Value       : €{snap['portfolio_value']:,.2f}")
        print(f"  Drawdown    : {snap['drawdown_pct']:.2f}%\n")

    def _handle_best_month(self, _: str) -> None:
        """Show the best performing month."""
        snap = self.dashboard.best_month()
        if snap is None:
            print("No simulation history found.")
            return
        print(f"\n  Best Month : {snap['date'][:7]}")
        print(f"  Return     : {snap['monthly_return_pct']:+.2f}%")
        print(f"  Value      : €{snap['portfolio_value']:,.2f}\n")

    def _handle_rebalance(self, _: str) -> None:
        """Show a rebalancing recommendation."""
        if not self.portfolio.history:
            print("Run the simulation first.")
            return
        prices = self.portfolio.history[-1].asset_prices
        if not prices:
            print("No price data in last snapshot — cannot suggest rebalance.")
            return
        df = self.rebalancer.suggest(self.portfolio, prices)
        print("\n  Rebalancing Recommendation:")
        print(df.to_string(index=False))
        print()

    def _handle_equity_chart(self, _: str) -> None:
        """Generate and open the equity + drawdown chart."""
        import os
        save_path = os.path.join(self.output_dir, "equity_chart.html")
        os.makedirs(self.output_dir, exist_ok=True)
        fig = self.dashboard.plot_equity_and_drawdown(save_path=save_path)
        fig.show()
        print(f"  Chart saved → {save_path}")

    def _handle_allocation_pie(self, _: str) -> None:
        """Generate and open the allocation pie chart."""
        import os
        save_path = os.path.join(self.output_dir, "allocation_pie.html")
        os.makedirs(self.output_dir, exist_ok=True)
        fig = self.dashboard.plot_allocation_pie(save_path=save_path)
        fig.show()
        print(f"  Chart saved → {save_path}")

    def _handle_sector(self, _: str) -> None:
        """Print sector exposure breakdown."""
        df = self.dashboard.sector_exposure()
        if df.empty:
            print("No sector data available.")
            return
        print("\n  Sector Exposure:")
        for _, row in df.iterrows():
            bar = "█" * int(row["weight_pct"] / 2)
            print(f"    {row['sector']:<30} {row['weight_pct']:>6.2f}%  {bar}")
        print()

    def _handle_sentiment(self, _: str) -> None:
        """Run FinBERT sentiment on all current holdings."""
        try:
            from quantforge.sentiment_analyzer.finbert_model import FinBERTAnalyzer
            from quantforge.sentiment_analyzer.headline_scraper import HeadlineLoader
        except ImportError:
            print("Sentiment module not available — check that Task 3 dependencies are installed.")
            return

        tickers = list(self.portfolio.positions.keys())
        if not tickers:
            print("No open positions to analyze.")
            return

        print(f"\n  Running FinBERT sentiment for: {', '.join(tickers)}")
        analyzer = FinBERTAnalyzer()

        for ticker in tickers:
            try:
                loader = HeadlineLoader(ticker=ticker)
                headlines_df = loader.scrape_yahoo_rss(ticker=ticker)
                if headlines_df.empty:
                    print(f"  {ticker}: no headlines found.")
                    continue
                texts = headlines_df["headline"].tolist()[:10]
                results = analyzer.analyze_batch(texts)
                avg_score = results["score"].mean()
                signal = analyzer.score_to_signal(avg_score)
                print(
                    f"  {ticker:<8} avg_score={avg_score:+.3f}  signal={signal}  "
                    f"({len(texts)} headlines)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {ticker}: error — {exc}")
        print()

    def _handle_monte_carlo(self, raw: str) -> None:
        """Run Monte Carlo projection and display the fan chart."""
        import re, os

        # Try to extract custom horizon from input (e.g. "next 6 months")
        match = re.search(r"(\d+)\s*month", raw.lower())
        horizons = [int(match.group(1))] if match else MC_HORIZONS_MONTHS

        if len(self.portfolio.history) < 3:
            print("Need at least 3 months of history for Monte Carlo projection.")
            return

        mc = MonteCarloProjector(self.portfolio)
        mc.print_summary(horizons=horizons)

        save_path = os.path.join(self.output_dir, "monte_carlo.html")
        os.makedirs(self.output_dir, exist_ok=True)
        fig = mc.plot(horizons=horizons, save_path=save_path)
        fig.show()
        print(f"  Chart saved → {save_path}")

    def _handle_report(self, _: str) -> None:
        """Generate the monthly HTML report."""
        gen = ReportGenerator(self.portfolio, output_dir=self.output_dir)
        path = gen.generate()
        print(f"  Report generated → {path}")
        # Attempt to open in the default browser
        try:
            import webbrowser
            webbrowser.open(f"file://{path}")
        except Exception:  # noqa: BLE001
            pass

    def _handle_help(self, _: str) -> None:
        """Print the help text."""
        print(_HELP_TEXT)
