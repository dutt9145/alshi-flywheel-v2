"""
shared/resolution_timer.py

Resolution timing arbitrage.

Some Kalshi markets resolve hours before their official close_time.
If you know a weather market resolves at 9am EST regardless of its
listed close_time, you can position into markets where the outcome
is near-certain and the price hasn't caught up yet.

How it works:
  1. Builds a resolution_patterns table from your outcomes history
     — calculates avg offset between close_time and actual resolution
  2. At scan time, flags markets that are past their expected resolution
     time but not yet settled — these often have stale prices
  3. Combines with model probability to identify near-certain positions

Usage:
    from shared.resolution_timer import ResolutionTimer
    rt = ResolutionTimer()
    rt.rebuild_patterns()       # call during nightly retrain

    near_certain = rt.scan(open_markets, bot_probs)
    for signal in near_certain:
        # execute signal.direction on signal.ticker at signal.price_cents
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolutionTimingSignal:
    ticker:             str
    direction:          str        # "YES" or "NO"
    yes_price_cents:    int
    our_prob:           float
    expected_resolution: datetime   # when we think it should have resolved
    minutes_overdue:    float       # how far past expected resolution
    sector:             str
    confidence:         float
    notes:              str = ""


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
        logger.error("ResolutionTimer DB error: %s", e)
        return []


def init_resolution_timer_tables() -> None:
    _db_execute("""
        CREATE TABLE IF NOT EXISTS resolution_patterns (
            ticker_prefix              TEXT PRIMARY KEY,
            sector                     TEXT,
            avg_resolution_offset_min  FLOAT8,
            stddev_offset_min          FLOAT8 DEFAULT 30.0,
            sample_count               INT DEFAULT 0,
            last_updated               TIMESTAMPTZ DEFAULT NOW()
        )
    """)


# ── Known sector resolution patterns (hard-coded priors) ─────────────────────
# These are empirical Kalshi resolution timing patterns.
# The builder will override these with actual data as outcomes accumulate.

SECTOR_PRIORS: dict[str, dict] = {
    "weather": {
        "avg_offset_min": -240.0,   # resolves ~4 hours BEFORE close_time
        "stddev_min":      60.0,
        "notes":           "NOAA data posts at 6am EST — weather markets resolve then",
    },
    "sports": {
        "avg_offset_min": -20.0,    # resolves ~20 min after game ends, which is before close_time
        "stddev_min":      30.0,
        "notes":           "Sports markets resolve when the game final is official",
    },
    "economics": {
        "avg_offset_min": -5.0,     # resolves quickly after data release
        "stddev_min":      15.0,
        "notes":           "BLS/Fed data releases are instant — markets resolve within minutes",
    },
    "crypto": {
        "avg_offset_min":  0.0,     # resolves at close_time (price is continuous)
        "stddev_min":      10.0,
        "notes":           "Crypto price markets resolve exactly at close_time",
    },
    "politics": {
        "avg_offset_min": -30.0,    # resolves when vote count / announcement hits
        "stddev_min":     120.0,    # high variance — could be hours after polls close
        "notes":           "Politics markets highly variable — AP call timing",
    },
    "tech": {
        "avg_offset_min":  -10.0,   # earnings post-market, resolve when confirmed
        "stddev_min":       20.0,
        "notes":           "Tech/earnings markets resolve after official filing",
    },
}


class ResolutionTimer:
    """
    Builds and queries resolution timing patterns.

    Parameters
    ----------
    min_overdue_minutes : float
        How many minutes past expected resolution before flagging.
        Default 15 — gives buffer for data ingestion lag.
    min_prob_to_trade : float
        Minimum model probability to enter a timing arb position.
        Default 0.80 — only near-certain outcomes.
    min_sample_count : int
        Minimum historical samples before trusting the learned pattern
        over the hard-coded prior. Default 20.
    """

    def __init__(
        self,
        min_overdue_minutes: float = 15.0,
        min_prob_to_trade:   float = 0.80,
        min_sample_count:    int   = 20,
    ):
        self.min_overdue_minutes = min_overdue_minutes
        self.min_prob_to_trade   = min_prob_to_trade
        self.min_sample_count    = min_sample_count
        init_resolution_timer_tables()
        self._seed_priors()

    def _seed_priors(self) -> None:
        """Insert hard-coded priors for sectors without enough data yet."""
        for sector, prior in SECTOR_PRIORS.items():
            p = _ph()
            _db_execute(
                f"""
                INSERT INTO resolution_patterns
                    (ticker_prefix, sector, avg_resolution_offset_min,
                     stddev_offset_min, sample_count)
                VALUES ({p}, {p}, {p}, {p}, 0)
                ON CONFLICT (ticker_prefix) DO NOTHING
                """,
                (
                    f"prior_{sector}",
                    sector,
                    prior["avg_offset_min"],
                    prior["stddev_min"],
                ),
            )

    def rebuild_patterns(self) -> int:
        """
        Rebuild resolution timing patterns from outcomes history.
        Call during nightly retrain.

        Calculates the average offset between close_time and outcome_at
        for each ticker prefix, then upserts into resolution_patterns.

        Returns number of patterns updated.
        """
        logger.info("ResolutionTimer: rebuilding patterns from outcomes...")

        # Pull outcomes that have both a close_time and an outcome_at
        rows = _db_execute(
            """
            SELECT
                o.ticker,
                s.sector,
                o.resolved_at,
                s.close_time
            FROM outcomes o
            JOIN signals s ON s.ticker = o.ticker
            WHERE o.resolved_at IS NOT NULL
              AND s.close_time   IS NOT NULL
              AND s.sector       IS NOT NULL
            ORDER BY o.resolved_at DESC
            LIMIT 5000
            """,
            fetch=True,
        )

        if not rows:
            logger.info("ResolutionTimer: no outcomes with timing data yet")
            return 0

        # Group by sector and compute offsets
        from collections import defaultdict
        sector_offsets: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            try:
                resolved_at = row.get("resolved_at")
                close_time  = row.get("close_time")
                sector      = row.get("sector", "")

                if not resolved_at or not close_time or not sector:
                    continue

                if isinstance(resolved_at, str):
                    resolved_at = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
                if isinstance(close_time, str):
                    close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))

                offset_min = (resolved_at - close_time).total_seconds() / 60
                sector_offsets[sector].append(offset_min)

            except Exception as e:
                logger.debug("ResolutionTimer row error: %s", e)

        updated = 0
        for sector, offsets in sector_offsets.items():
            if len(offsets) < 5:
                continue

            import statistics
            avg    = statistics.mean(offsets)
            stddev = statistics.stdev(offsets) if len(offsets) > 1 else 30.0
            count  = len(offsets)

            p = _ph()
            _db_execute(
                f"""
                INSERT INTO resolution_patterns
                    (ticker_prefix, sector, avg_resolution_offset_min,
                     stddev_offset_min, sample_count, last_updated)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p})
                ON CONFLICT (ticker_prefix) DO UPDATE
                  SET avg_resolution_offset_min = EXCLUDED.avg_resolution_offset_min,
                      stddev_offset_min         = EXCLUDED.stddev_offset_min,
                      sample_count              = EXCLUDED.sample_count,
                      last_updated              = EXCLUDED.last_updated
                """,
                (
                    f"learned_{sector}", sector, avg, stddev, count,
                    datetime.now(timezone.utc),
                ),
            )
            logger.info(
                "ResolutionTimer pattern: sector=%s avg_offset=%.1fmin "
                "stddev=%.1fmin n=%d",
                sector, avg, stddev, count,
            )
            updated += 1

        return updated

    def _get_pattern(self, sector: str) -> dict:
        """
        Get the best resolution pattern for a sector.
        Prefers learned patterns (if enough samples) over priors.
        """
        p = _ph()

        # Try learned pattern first
        learned = _db_execute(
            f"""
            SELECT avg_resolution_offset_min, stddev_offset_min, sample_count
            FROM resolution_patterns
            WHERE ticker_prefix = {p}
            """,
            (f"learned_{sector}",),
            fetch=True,
        )
        if learned and learned[0]["sample_count"] >= self.min_sample_count:
            return learned[0]

        # Fall back to prior
        prior = _db_execute(
            f"""
            SELECT avg_resolution_offset_min, stddev_offset_min, sample_count
            FROM resolution_patterns
            WHERE ticker_prefix = {p}
            """,
            (f"prior_{sector}",),
            fetch=True,
        )
        if prior:
            return prior[0]

        # Default: no pattern known
        return {"avg_resolution_offset_min": 0.0, "stddev_offset_min": 60.0, "sample_count": 0}

    def scan(
        self,
        open_markets: list[dict],
        bot_probs: dict[str, float],       # ticker → our estimated P(YES)
        bot_sectors: dict[str, str],       # ticker → sector
    ) -> list[ResolutionTimingSignal]:
        """
        Identify open markets that should have resolved by now.

        Returns list of ResolutionTimingSignal sorted by confidence desc.
        """
        signals = []
        now     = datetime.now(timezone.utc)

        for market in open_markets:
            ticker = market.get("ticker", "")

            if ticker not in bot_probs:
                continue

            our_prob = bot_probs[ticker]
            if our_prob < self.min_prob_to_trade and our_prob > (1 - self.min_prob_to_trade):
                continue

            sector = bot_sectors.get(ticker, "")
            if not sector:
                continue

            close_str = market.get("close_time") or market.get("expiration_time", "")
            if not close_str:
                continue

            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except Exception:
                continue

            pattern = self._get_pattern(sector)
            avg_offset_min = pattern["avg_resolution_offset_min"]
            stddev_min     = pattern["stddev_offset_min"]

            # Expected resolution time = close_time + avg_offset
            expected_resolution = close_dt + timedelta(minutes=avg_offset_min)
            minutes_overdue     = (now - expected_resolution).total_seconds() / 60

            if minutes_overdue < self.min_overdue_minutes:
                continue

            # Get YES price
            yes_price = 0
            try:
                bid = float(market.get("yes_bid_dollars") or 0) * 100
                ask = float(market.get("yes_ask_dollars") or 0) * 100
                if bid > 0 and ask > 0:
                    yes_price = int(round((bid + ask) / 2))
            except (TypeError, ValueError):
                pass

            if yes_price == 0:
                continue

            direction = "YES" if our_prob >= self.min_prob_to_trade else "NO"

            # Confidence: decays with stddev, grows with overdue time and model conviction
            overdue_conf  = min(0.40, minutes_overdue / (stddev_min * 2) * 0.40)
            prob_conf     = max(our_prob, 1 - our_prob) * 0.40   # conviction score
            sample_conf   = min(0.20, pattern["sample_count"] / 100 * 0.20)
            confidence    = min(1.0, overdue_conf + prob_conf + sample_conf)

            if confidence < 0.50:
                continue

            sig = ResolutionTimingSignal(
                ticker              = ticker,
                direction           = direction,
                yes_price_cents     = yes_price,
                our_prob            = our_prob,
                expected_resolution = expected_resolution,
                minutes_overdue     = minutes_overdue,
                sector              = sector,
                confidence          = confidence,
                notes               = (
                    f"Overdue by {minutes_overdue:.0f}min "
                    f"(pattern: {avg_offset_min:.0f}±{stddev_min:.0f}min) "
                    f"our_p={our_prob:.2f}"
                ),
            )

            logger.info(
                "[RESTIME] %s | %s | overdue=%.0fmin our_p=%.2f conf=%.2f",
                ticker, direction, minutes_overdue, our_prob, confidence,
            )

            signals.append(sig)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals