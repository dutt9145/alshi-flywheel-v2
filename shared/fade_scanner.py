"""
shared/fade_scanner.py

Closing price fade scanner.

Markets approaching expiry with heavy one-sided volume are systematically
mispriced because retail hammers the narrative. This scanner:

  1. Identifies markets closing within FADE_WINDOW_MINUTES (default 60)
  2. Flags markets where price is at an extreme (>FADE_THRESHOLD_CENTS)
     while our model disagrees by >FADE_MIN_DISAGREEMENT
  3. Checks sharp money alignment (don't fade sharp money)
  4. Emits FadeSignal objects that the orchestrator can execute on

This is the highest-alpha strategy available on Kalshi — retail
overcrowding in the final hour is a structural, repeatable inefficiency.

Usage:
    from shared.fade_scanner import FadeScanner
    scanner = FadeScanner(kalshi_client, sharp_detector)
    fades = scanner.scan(open_markets, bot_signals)
    for fade in fades:
        # execute fade.direction on fade.ticker
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FadeSignal:
    ticker:              str
    direction:           str        # "YES" (fade NO crowding) or "NO" (fade YES crowding)
    yes_price_cents:     int        # current market price
    our_prob:            float      # our model's estimated probability
    market_prob:         float      # market's implied probability
    disagreement:        float      # abs(our_prob - market_prob)
    minutes_to_close:    float      # how long until expiry
    sharp_aligned:       bool       # True if sharp money agrees with our fade
    confidence:          float      # composite fade confidence 0.0–1.0
    notes:               str = ""


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
        logger.error("FadeScanner DB error: %s", e)
        return []


def init_fade_tables() -> None:
    """Create fade_signals table. Called once on startup."""
    _db_execute("""
        CREATE TABLE IF NOT EXISTS fade_signals (
            id                 SERIAL PRIMARY KEY,
            ticker             TEXT NOT NULL,
            direction          TEXT NOT NULL,
            yes_price_cents    INT,
            our_prob           FLOAT8,
            market_prob        FLOAT8,
            disagreement       FLOAT8,
            minutes_to_close   FLOAT8,
            sharp_aligned      BOOLEAN DEFAULT FALSE,
            confidence         FLOAT8,
            executed           BOOLEAN DEFAULT FALSE,
            detected_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)


class FadeScanner:
    """
    Scans open markets for closing-time crowding opportunities.

    Parameters
    ----------
    kalshi_client : KalshiClient
        For fetching order book / market data if needed.
    sharp_detector : SharpDetector
        Used to check sharp money alignment before flagging a fade.
    fade_window_minutes : float
        Only look at markets closing within this many minutes. Default 60.
    fade_threshold_cents : int
        Only fade markets where YES price is above this (crowded YES)
        or below (100 - this) (crowded NO). Default 82.
    fade_min_disagreement : float
        Minimum abs difference between our_prob and market_prob to fade.
        Default 0.10 (10 percentage points).
    """

    def __init__(
        self,
        kalshi_client,
        sharp_detector,
        fade_window_minutes:    float = 60.0,
        fade_threshold_cents:   int   = 82,
        fade_min_disagreement:  float = 0.10,
    ):
        self.client                 = kalshi_client
        self.sharp                  = sharp_detector
        self.fade_window_minutes    = fade_window_minutes
        self.fade_threshold_cents   = fade_threshold_cents
        self.fade_min_disagreement  = fade_min_disagreement
        init_fade_tables()

    def _minutes_to_close(self, market: dict) -> Optional[float]:
        """Return minutes until close_time, or None if unparseable."""
        close_str = market.get("close_time") or market.get("expiration_time", "")
        if not close_str:
            return None
        try:
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            delta    = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60
            return delta
        except Exception:
            return None

    def _get_yes_price(self, market: dict) -> int:
        """Extract YES midpoint in cents."""
        try:
            bid = float(market.get("yes_bid_dollars") or 0) * 100
            ask = float(market.get("yes_ask_dollars") or 0) * 100
            if bid > 0 and ask > 0:
                return int(round((bid + ask) / 2))
        except (TypeError, ValueError):
            pass
        try:
            last = float(market.get("last_price_dollars") or 0) * 100
            if last > 0:
                return int(round(last))
        except (TypeError, ValueError):
            pass
        return 0

    def scan(
        self,
        open_markets: list[dict],
        bot_probs: dict[str, float],    # ticker → our estimated P(YES)
    ) -> list[FadeSignal]:
        """
        Main scan loop. Returns list of FadeSignal for markets that
        qualify for a fade trade.

        Parameters
        ----------
        open_markets : list[dict]
            Full list of open markets from Kalshi.
        bot_probs : dict[str, float]
            ticker → our_prob mapping, built from bot consensus outputs.
            Markets not in this dict are skipped (we need a model opinion).
        """
        fades = []
        now   = datetime.now(timezone.utc)

        for market in open_markets:
            ticker = market.get("ticker", "")

            # Must have a model opinion to fade
            if ticker not in bot_probs:
                continue

            # Must be closing soon
            mins = self._minutes_to_close(market)
            if mins is None or mins <= 0 or mins > self.fade_window_minutes:
                continue

            yes_price   = self._get_yes_price(market)
            if yes_price == 0:
                continue

            market_prob = yes_price / 100.0
            our_prob    = bot_probs[ticker]
            disagreement = abs(our_prob - market_prob)

            # Must be at an extreme price (crowded)
            crowded_yes = yes_price >= self.fade_threshold_cents
            crowded_no  = yes_price <= (100 - self.fade_threshold_cents)

            if not (crowded_yes or crowded_no):
                continue

            # Must disagree enough with the market
            if disagreement < self.fade_min_disagreement:
                continue

            # Determine fade direction
            # Crowded YES (price too high) → bet NO
            # Crowded NO (price too low) → bet YES
            if crowded_yes and our_prob < market_prob:
                fade_direction = "NO"
            elif crowded_no and our_prob > market_prob:
                fade_direction = "YES"
            else:
                # Market is crowded but our model agrees — don't fade
                continue

            # Sharp money check — don't fade into sharp money
            sharp_signal = self.sharp.analyze(market)
            sharp_aligned = (
                not sharp_signal.sharp_detected or
                sharp_signal.aligned_with(fade_direction)
            )

            # Composite confidence
            # Base: disagreement quality (scaled to 0.4 max at 30% disagreement)
            conf_disagreement = min(0.40, disagreement / 0.30 * 0.40)
            # Urgency: closer to close = more certain the crowding won't unwind
            conf_urgency      = max(0.0, (self.fade_window_minutes - mins) / self.fade_window_minutes * 0.35)
            # Sharp alignment bonus
            conf_sharp        = 0.25 if sharp_aligned and sharp_signal.sharp_detected else 0.0
            confidence        = min(1.0, conf_disagreement + conf_urgency + conf_sharp)

            # Minimum confidence threshold
            if confidence < 0.35:
                continue

            signal = FadeSignal(
                ticker           = ticker,
                direction        = fade_direction,
                yes_price_cents  = yes_price,
                our_prob         = our_prob,
                market_prob      = market_prob,
                disagreement     = disagreement,
                minutes_to_close = mins,
                sharp_aligned    = sharp_aligned,
                confidence       = confidence,
                notes            = (
                    f"crowded={'YES' if crowded_yes else 'NO'} "
                    f"mins={mins:.1f} disagree={disagreement:.2%} "
                    f"sharp_aligned={sharp_aligned}"
                ),
            )

            logger.info(
                "[FADE] %s → %s | price=%d¢ our_p=%.2f mkt_p=%.2f "
                "mins=%.1f conf=%.2f",
                ticker, fade_direction, yes_price, our_prob, market_prob,
                mins, confidence,
            )

            self._log_fade(signal)
            fades.append(signal)

        if fades:
            logger.info("FadeScanner found %d fade opportunities", len(fades))

        return fades

    def _log_fade(self, signal: FadeSignal) -> None:
        p = _ph()
        try:
            _db_execute(
                f"""
                INSERT INTO fade_signals
                    (ticker, direction, yes_price_cents, our_prob, market_prob,
                     disagreement, minutes_to_close, sharp_aligned, confidence, detected_at)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
                """,
                (
                    signal.ticker, signal.direction, signal.yes_price_cents,
                    signal.our_prob, signal.market_prob, signal.disagreement,
                    signal.minutes_to_close, signal.sharp_aligned, signal.confidence,
                    datetime.now(timezone.utc),
                ),
            )
        except Exception as e:
            logger.warning("Fade signal log failed: %s", e)

    def mark_executed(self, ticker: str) -> None:
        """Mark a fade signal as executed (called from orchestrator after order)."""
        p = _ph()
        _db_execute(
            f"UPDATE fade_signals SET executed = TRUE WHERE ticker = {p} AND executed = FALSE",
            (ticker,),
        )