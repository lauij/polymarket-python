import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import time
import logging
import requests
import schedule
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# py-clob-client: official Polymarket Python client
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TARGET_ADDRESS  = os.getenv("TARGET_ADDRESS", "0xYourTargetWalletHere")
SCALE_FACTOR    = float(os.getenv("SCALE_FACTOR", "0.5"))   # fraction of target's trade to copy (0.5 = 50%)
POLL_INTERVAL_S = int(os.getenv("POLL_INTERVAL", "30"))     # seconds between checks
MAX_PRICE       = float(os.getenv("MAX_PRICE", "0.95"))     # don't copy trades above this price (too little upside)
MIN_PRICE       = float(os.getenv("MIN_PRICE", "0.02"))     # don't copy trades below this price (too risky)
DRY_RUN         = os.getenv("DRY_RUN", "true").lower() == "true"  # Set to "false" to place real orders

# Polymarket API endpoints
DATA_API   = "https://data-api.polymarket.com"   # correct endpoint (gamma-api /trades returns 404)
CLOB_HOST  = "https://clob.polymarket.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("copy_bot.log"),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:  str      # transactionHash
    market_id: str      # conditionId
    token_id:  str      # asset (YES or NO token id)
    side:      str      # "BUY" or "SELL"
    price:     float    # 0–1 (implied probability)
    size_usd:  float    # original USD spent (price × shares)
    shares:    float    # original number of shares received
    outcome:   str      # "Yes" or "No"
    timestamp: int


# ─────────────────────────────────────────────
# Polymarket data fetcher
# ─────────────────────────────────────────────

class PolymarketWatcher:
    """Polls the data API for a user's recent trades."""

    def __init__(self, address: str):
        self.address = address.lower()
        self.seen_ids: set[str] = set()
        self._bootstrap()

    def _bootstrap(self):
        """On startup, load existing trade IDs so we don't replay history."""
        log.info(f"Bootstrapping — loading existing trades for {self.address}")
        trades = self._fetch_trades(limit=50)
        for t in trades:
            self.seen_ids.add(t.trade_id)
        log.info(f"Ignoring {len(self.seen_ids)} existing trades. Watching for new ones...")

    def _fetch_trades(self, limit: int = 50) -> list[Trade]:
        """Fetch recent trades from data API."""
        url = f"{DATA_API}/trades"
        params = {
            "user":  self.address,  # note: 'user' not 'maker'
            "limit": limit,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            log.error(f"Failed to fetch trades: {e}")
            return []

        trades = []
        for item in raw:
            try:
                raw_size  = float(item.get("size", 0))
                raw_price = float(item.get("price", 0))
                trade = Trade(
                    trade_id = item["transactionHash"] + "_" + item.get("asset", ""),          # unique id per trade
                    market_id = item.get("conditionId", ""),
                    token_id  = item.get("asset", ""),            # token id for the outcome
                    side      = item.get("side", "BUY").upper(),
                    price     = raw_price,
                    size_usd  = round(raw_price * raw_size, 4),
                    shares    = raw_size,
                    outcome   = item.get("outcome", "Yes"),
                    timestamp = int(item.get("timestamp", 0)),
                )
                trades.append(trade)
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping malformed trade entry: {e}")
        return trades

    def get_new_trades(self) -> list[Trade]:
        """Returns only trades not yet seen."""
        all_trades = self._fetch_trades()
        new = [t for t in all_trades if t.trade_id not in self.seen_ids]
        for t in new:
            self.seen_ids.add(t.trade_id)
        return new


# ─────────────────────────────────────────────
# Order executor
# ─────────────────────────────────────────────

class OrderExecutor:
    """Places scaled copy orders via the Polymarket CLOB."""

    def __init__(self, scale_factor: float, dry_run: bool = True):
        self.scale_factor = scale_factor
        self.dry_run      = dry_run
        self.client: Optional[ClobClient] = None

        if not dry_run:
            self._init_client()

    def _init_client(self):
        key        = os.getenv("POLY_API_KEY")
        secret     = os.getenv("POLY_API_SECRET")
        passphrase = os.getenv("POLY_API_PASSPHRASE")
        pk         = os.getenv("PRIVATE_KEY")

        if not all([key, secret, passphrase, pk]):
            raise ValueError(
                "Missing credentials in .env — set POLY_API_KEY, "
                "POLY_API_SECRET, POLY_API_PASSPHRASE, PRIVATE_KEY"
            )

        self.client = ClobClient(
            host           = CLOB_HOST,
            chain_id       = POLYGON,
            key            = pk,
            signature_type = 0,
            funder         = os.getenv("FUNDER_ADDRESS"),
        )
        self.client.set_api_creds(self.client.derive_api_key())
        log.info("CLOB client initialized ✓")

    def should_copy(self, trade: Trade) -> tuple[bool, str]:
        """Apply filters before copying a trade."""
        if trade.price > MAX_PRICE:
            return False, f"price {trade.price:.2f} above MAX_PRICE {MAX_PRICE}"
        if trade.price < MIN_PRICE:
            return False, f"price {trade.price:.2f} below MIN_PRICE {MIN_PRICE}"
        if trade.side == "SELL":
            return False, "skipping SELL (exit) trades — only copying buys"
        return True, "ok"

    def copy_trade(self, trade: Trade):
        ok, reason = self.should_copy(trade)
        if not ok:
            log.info(f"  ↳ Skipping trade {trade.trade_id[:10]}: {reason}")
            return

        # Scale both USD and shares proportionally — same price, same probability, smaller position
        your_shares = round(trade.shares   * self.scale_factor, 4)
        your_usd    = round(trade.size_usd * self.scale_factor, 4)

        log.info(
            f"  ↳ Copying trade | {trade.outcome} on {trade.market_id[:12]}... | "
            f"Side: {trade.side} | Price: {trade.price:.3f} | "
            f"Target: ${trade.size_usd:.2f} → {trade.shares} shares | "
            f"Yours ({self.scale_factor*100:.0f}%): ${your_usd:.2f} → {your_shares} shares"
        )

        if self.dry_run:
            log.info("  [DRY RUN] Order NOT placed. Set DRY_RUN=false in .env to go live.")
            return

        try:
            order_args = OrderArgs(
                token_id   = trade.token_id,
                price      = trade.price,
                size       = your_shares,
                side       = "BUY" if trade.side == "BUY" else "SELL",  # plain strings — Side class removed in newer versions
                order_type = "GTC",  # plain string — OrderType.GTC may not exist in newer versions
            )
            resp = self.client.create_and_post_order(order_args)
            log.info(f"  ✓ Order placed: {resp}")
        except Exception as e:
            log.error(f"  ✗ Order failed: {e}")


# ─────────────────────────────────────────────
# Bot runner
# ─────────────────────────────────────────────

class CopyBot:
    def __init__(self):
        self.watcher  = PolymarketWatcher(TARGET_ADDRESS)
        self.executor = OrderExecutor(SCALE_FACTOR, dry_run=DRY_RUN)

    def tick(self):
        log.info(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Checking for new trades...")
        new_trades = self.watcher.get_new_trades()

        if not new_trades:
            log.info("  No new trades found.")
            return

        log.info(f"  Found {len(new_trades)} new trade(s)!")
        for trade in new_trades:
            self.executor.copy_trade(trade)

    def run(self):
        mode = "DRY RUN 🧪" if DRY_RUN else "LIVE 🔴"
        log.info("=" * 55)
        log.info(f"  Polymarket Copy Bot — {mode}")
        log.info(f"  Target  : {TARGET_ADDRESS}")
        log.info(f"  Scale   : {SCALE_FACTOR*100:.0f}% of target's position")
        log.info(f"  Polling : every {POLL_INTERVAL_S}s")
        log.info("=" * 55)

        self.tick()
        schedule.every(POLL_INTERVAL_S).seconds.do(self.tick)

        while True:
            schedule.run_pending()
            time.sleep(1)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    bot = CopyBot()
    bot.run()



# ─────────────────────────────────────────────
# .env template (create this file next to the script)
# ─────────────────────────────────────────────
"""
TARGET_ADDRESS=0xABCDEF...          # wallet address to copy
SCALE_FACTOR=0.5                    # 0.5 = bet 50% of whatever they bet (0.1 = 10%, etc.)
POLL_INTERVAL=30                    # polling frequency in seconds
MAX_PRICE=0.95                      # skip trades above this probability
MIN_PRICE=0.02                      # skip trades below this probability
DRY_RUN=true                        # set to false when ready to go live

# Only needed when DRY_RUN=false:
PRIVATE_KEY=0x...                   # your Polygon wallet private key
FUNDER_ADDRESS=0x...                # your wallet address (usually same as PK)
POLY_API_KEY=...                    # from polymarket.com > Settings > API
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
"""
