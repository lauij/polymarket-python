"""
Polymarket Copy Trading Bot
============================
Monitors a target wallet's trades on Polymarket and copies them in real-time.
Polls every 1 second and handles simultaneous bets via thread pool execution.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import logging
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.constants import POLYGON

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_ADDRESS = "0x576b0696fd5a9225d66fd9500fd98f5be10b0cab"

SCALE_FACTOR = float(os.getenv("SCALE_FACTOR", "0.5"))
POLL_INTERVAL = 1  # seconds - check every second
MAX_PRICE = float(os.getenv("MAX_PRICE", "0.95"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "0.05"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

DATA_API = "https://data-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("copy_bot.log"),
    ],
)
log = logging.getLogger("copy_bot")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    trade_id: str
    market_id: str
    token_id: str
    side: str  # BUY or SELL
    price: float
    size_usd: float
    shares: float
    outcome: str  # Yes or No
    timestamp: int


# ---------------------------------------------------------------------------
# Trade Watcher - polls data API for new trades
# ---------------------------------------------------------------------------


class TradeWatcher:
    """Watches a target address for new trades on Polymarket."""

    def __init__(self, address: str):
        self.address = address.lower()
        self.seen_ids: set[str] = set()
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._bootstrap()

    def _bootstrap(self):
        """Load existing trades on startup so we only act on new ones."""
        log.info("Bootstrapping: loading existing trades for %s", self.address)
        trades = self._fetch_trades(limit=100)
        with self._lock:
            for t in trades:
                self.seen_ids.add(t.trade_id)
        log.info("Loaded %d existing trades - will ignore these.", len(self.seen_ids))

    def _fetch_trades(self, limit: int = 100) -> list[Trade]:
        """Fetch recent trades from the Polymarket activity API."""
        url = f"{DATA_API}/activity"
        params = {
            "user": self.address,
            "type": "TRADE",
            "limit": limit,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException as exc:
            log.error("Failed to fetch trades: %s", exc)
            return []

        trades: list[Trade] = []
        for item in raw:
            try:
                raw_size = float(item.get("size", 0))
                raw_price = float(item.get("price", 0))
                usdc_size = float(item.get("usdcSize", raw_price * raw_size))
                trade = Trade(
                    trade_id=item["transactionHash"] + "_" + item.get("asset", ""),
                    market_id=item.get("conditionId", ""),
                    token_id=item.get("asset", ""),
                    side=item.get("side", "BUY").upper(),
                    price=raw_price,
                    size_usd=round(usdc_size, 4),
                    shares=raw_size,
                    outcome=item.get("outcome", "Yes"),
                    timestamp=int(item.get("timestamp", 0)),
                )
                trades.append(trade)
            except (KeyError, ValueError) as exc:
                log.warning("Skipping malformed trade: %s", exc)

        return trades

    def get_new_trades(self) -> list[Trade]:
        """Return only trades we haven't seen before."""
        all_trades = self._fetch_trades()
        new_trades: list[Trade] = []
        with self._lock:
            for t in all_trades:
                if t.trade_id not in self.seen_ids:
                    self.seen_ids.add(t.trade_id)
                    new_trades.append(t)
        return new_trades


# ---------------------------------------------------------------------------
# Order Executor - places copy trades via CLOB
# ---------------------------------------------------------------------------


class OrderExecutor:
    """Places scaled copy-trades on Polymarket CLOB. Thread-safe."""

    def __init__(self, scale_factor: float, dry_run: bool = True):
        self.scale_factor = scale_factor
        self.dry_run = dry_run
        self._client: Optional[ClobClient] = None
        self._client_lock = threading.Lock()

        if not dry_run:
            self._init_client()

    def _init_client(self):
        key = os.getenv("POLY_API_KEY")
        secret = os.getenv("POLY_API_SECRET")
        passphrase = os.getenv("POLY_API_PASSPHRASE")
        pk = os.getenv("PRIVATE_KEY")

        if not all([key, secret, passphrase, pk]):
            raise ValueError(
                "Missing credentials. Set POLY_API_KEY, POLY_API_SECRET, "
                "POLY_API_PASSPHRASE, and PRIVATE_KEY in your .env file."
            )

        self._client = ClobClient(
            host=CLOB_HOST,
            chain_id=POLYGON,
            key=pk,
            signature_type=0,
            funder=os.getenv("FUNDER_ADDRESS"),
        )
        self._client.set_api_creds(self._client.derive_api_key())
        log.info("CLOB client initialized successfully.")

    def _passes_filters(self, trade: Trade) -> tuple[bool, str]:
        if trade.price > MAX_PRICE:
            return False, f"price {trade.price:.2f} > MAX_PRICE {MAX_PRICE}"
        if trade.price < MIN_PRICE:
            return False, f"price {trade.price:.2f} < MIN_PRICE {MIN_PRICE}"
        return True, "ok"

    def execute(self, trade: Trade):
        """Copy a single trade (called from thread pool for concurrency)."""
        ok, reason = self._passes_filters(trade)
        if not ok:
            log.info("  SKIP %s: %s", trade.trade_id[:16], reason)
            return

        copy_shares = round(trade.shares * self.scale_factor, 4)
        copy_usd = round(trade.size_usd * self.scale_factor, 4)

        log.info(
            "  COPY | %s %s @ %.3f | market %s | "
            "target: $%.2f / %.2f shares -> yours: $%.2f / %.2f shares",
            trade.side,
            trade.outcome,
            trade.price,
            trade.market_id[:12],
            trade.size_usd,
            trade.shares,
            copy_usd,
            copy_shares,
        )

        if self.dry_run:
            log.info("  [DRY RUN] Order not placed.")
            return

        try:
            order_args = OrderArgs(
                token_id=trade.token_id,
                price=trade.price,
                size=copy_shares,
                side=trade.side,
                order_type="GTC",
            )
            with self._client_lock:
                resp = self._client.create_and_post_order(order_args)
            log.info("  ORDER PLACED: %s", resp)
        except Exception as exc:
            log.error("  ORDER FAILED: %s", exc)


# ---------------------------------------------------------------------------
# Copy Bot - main loop
# ---------------------------------------------------------------------------


class CopyBot:
    """
    Main bot: polls every second and dispatches copy trades concurrently
    so simultaneous bets from the target are handled in parallel.
    """

    def __init__(self):
        self.watcher = TradeWatcher(TARGET_ADDRESS)
        self.executor = OrderExecutor(SCALE_FACTOR, dry_run=DRY_RUN)
        self.pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def _tick(self):
        new_trades = self.watcher.get_new_trades()
        if not new_trades:
            return

        log.info("Found %d new trade(s)!", len(new_trades))

        # Submit all new trades to the thread pool concurrently
        futures = []
        for trade in new_trades:
            future = self.pool.submit(self.executor.execute, trade)
            futures.append(future)

        # Wait for all copy orders to complete (non-blocking on main loop)
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception as exc:
                log.error("Trade execution error: %s", exc)

    def run(self):
        mode = "DRY RUN" if DRY_RUN else "LIVE"
        log.info("=" * 60)
        log.info("  Polymarket Copy Bot - %s", mode)
        log.info("  Target  : %s", TARGET_ADDRESS)
        log.info("  Scale   : %.0f%% of target position", SCALE_FACTOR * 100)
        log.info("  Polling : every %ds", POLL_INTERVAL)
        log.info("  Workers : %d (for simultaneous trades)", MAX_WORKERS)
        log.info("=" * 60)

        try:
            while True:
                self._tick()
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Shutting down...")
            self.pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = CopyBot()
    bot.run()
