"""
shared/sharp_detector.py

Sharp money detection via Kalshi order book analysis.

Sharp bettors move markets in detectable ways:
  — They collapse the bid/ask spread (they provide liquidity both ways)
  — They submit large orders relative to typical market volume
  — Price impact per unit is high (thin books get moved more per contract)

This module watches for these signatures and returns a SharpSignal that
the orchestrator uses to either:
  a) Follow the sharp (if aligned with our model direction)
  b) Stand down (if opposing our model — don't fight sharp money)

Usage:
    from shared.sharp_detector import SharpDetector
    detector = SharpDetector(kalshi_client)
    signal = detector.analyze(market)

    if signal.sharp_detected and not signal.aligned_with(our_direction):
        return  # don't fight sharp money
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SharpSignal:
    ticker:          str
    sharp_detected:  bool    = False
    sharp_direction: str     = "NONE"     # "YES", "NO", or "NONE"
    spread_cents:    float   = 0.0        # current bid/ask spread
    spread_pct:      float   = 0.0        # spread as % of yes price
    volume_spike:    bool    = False       # True if recent volume is elevated
    large_order:     bool    = False       # True if a large single order hit
    confidence:      float   = 0.0        # 0.0–1.0 composite sharp confidence
    notes:           str     = ""

    def aligned_with(self, our_direction: str) -> bool:
        """True if sharp money is betting the same way we are."""
        return self.sharp_direction == our_direction or self.sharp_direction == "NONE"


def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _db_execute(sql: str, params: tuple = (), fetch: bool = False) -> list:
    database_url = os.getenv("DATABASE_URL", "")
    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute(sql, params)
            result = []
            if fetch and cur.description:
                cols   = [d[0] for d in cur.description]
                result = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return result
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.row_factory = sqlite3.Row
            cur  = conn.execute(sql, params)
            result = [dict(r) for r in cur.fetchall()] if fetch else []
            conn.commit()
            conn.close()
            return result
    except Exception as e:
        logger.error("SharpDetector DB error: %s", e)
        return []


def init_sharp_tables() -> None:
    """Create sharp_events table. Called once on startup."""
    _db_execute("""
        CREATE TABLE IF NOT EXISTS sharp_events (
            id             SERIAL PRIMARY KEY,
            ticker         TEXT NOT NULL,
            spread_before  FLOAT8,
            spread_after   FLOAT8,
            large_order    BOOLEAN DEFAULT FALSE,
            direction      TEXT,
            confidence     FLOAT8,
            detected_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)


class SharpDetector:
    """
    Analyzes Kalshi order book data for sharp money signatures.

    Parameters
    ----------
    kalshi_client : KalshiClient
        Used to fetch order book snapshots.
    spread_threshold_pct : float
        Spread as % of midpoint below which we flag as sharp-tight.
        Default 0.04 = 4 cents on a 50¢ market.
    large_order_contracts : int
        Single order size above which we flag as institutional.
        Default 50 contracts ($50 at 100% resolution value).
    volume_spike_multiplier : float
        Recent volume vs 7-day average above which we flag a spike.
        Default 2.5x.
    """

    def __init__(
        self,
        kalshi_client,
        spread_threshold_pct:    float = 0.04,
        large_order_contracts:   int   = 50,
        volume_spike_multiplier: float = 2.5,
    ):
        self.client                  = kalshi_client
        self.spread_threshold_pct    = spread_threshold_pct
        self.large_order_contracts   = large_order_contracts
        self.volume_spike_multiplier = volume_spike_multiplier
        init_sharp_tables()

    def analyze(self, market: dict) -> SharpSignal:
        """
        Run full sharp detection pipeline on a single market.
        Returns a SharpSignal regardless of whether sharp money is detected.
        """
        ticker    = market.get("ticker", "UNKNOWN")
        yes_bid   = float(market.get("yes_bid_dollars")  or 0) * 100   # → cents
        yes_ask   = float(market.get("yes_ask_dollars")  or 0) * 100
        no_bid    = float(market.get("no_bid_dollars")   or 0) * 100
        no_ask    = float(market.get("no_ask_dollars")   or 0) * 100
        volume    = int(market.get("volume", 0)           or 0)
        volume_7d = int(market.get("volume_7d", 0)        or 0)
        open_int  = int(market.get("open_interest", 0)    or 0)

        if yes_bid <= 0 or yes_ask <= 0:
            return SharpSignal(ticker=ticker, notes="No order book data")

        midpoint     = (yes_bid + yes_ask) / 2
        spread_cents = yes_ask - yes_bid
        spread_pct   = spread_cents / midpoint if midpoint > 0 else 1.0

        # ── Spread tightness check ─────────────────────────────────────────────
        spread_tight = spread_pct < self.spread_threshold_pct

        # ── Volume spike check ────────────────────────────────────────────────
        avg_daily_vol  = volume_7d / 7 if volume_7d > 0 else volume
        volume_spike   = (
            avg_daily_vol > 0 and
            volume > avg_daily_vol * self.volume_spike_multiplier
        )

        # ── Large single order proxy ───────────────────────────────────────────
        # Kalshi doesn't expose individual order sizes directly in the market
        # snapshot. We proxy this using open_interest delta vs volume ratio.
        # High OI relative to volume = large positions being held, not flipped.
        large_order = (
            open_int > 0 and
            volume > 0 and
            open_int / max(volume, 1) > 0.4 and
            volume > self.large_order_contracts
        )

        # ── Direction inference ────────────────────────────────────────────────
        # If NO side is tighter than YES side, sharp money is on NO
        no_spread    = (no_ask - no_bid) if (no_ask > 0 and no_bid > 0) else spread_cents
        sharp_on_yes = yes_bid > (100 - no_ask)   # YES price has been bid up

        sharp_direction = "NONE"
        if spread_tight or volume_spike or large_order:
            sharp_direction = "YES" if sharp_on_yes else "NO"

        # ── Composite confidence ───────────────────────────────────────────────
        confidence = 0.0
        if spread_tight:   confidence += 0.40
        if volume_spike:   confidence += 0.35
        if large_order:    confidence += 0.25
        confidence = min(1.0, confidence)

        sharp_detected = confidence >= 0.35

        signal = SharpSignal(
            ticker          = ticker,
            sharp_detected  = sharp_detected,
            sharp_direction = sharp_direction,
            spread_cents    = spread_cents,
            spread_pct      = spread_pct,
            volume_spike    = volume_spike,
            large_order     = large_order,
            confidence      = confidence,
            notes           = (
                f"spread={spread_cents:.1f}¢ ({spread_pct*100:.1f}%) "
                f"vol_spike={volume_spike} large_order={large_order}"
            ),
        )

        if sharp_detected:
            logger.info(
                "[SHARP] %s | dir=%s conf=%.2f | %s",
                ticker, sharp_direction, confidence, signal.notes,
            )
            self._log_event(signal)

        return signal

    def _log_event(self, signal: SharpSignal) -> None:
        """Persist sharp event to DB for dashboard and post-analysis."""
        p = _ph()
        try:
            _db_execute(
                f"""
                INSERT INTO sharp_events
                    (ticker, spread_after, large_order, direction, confidence, detected_at)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p})
                """,
                (
                    signal.ticker,
                    signal.spread_cents,
                    signal.large_order,
                    signal.sharp_direction,
                    signal.confidence,
                    datetime.now(timezone.utc),
                ),
            )
        except Exception as e:
            logger.warning("Sharp event log failed: %s", e)