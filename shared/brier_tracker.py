"""
shared/brier_tracker.py

Tracks Brier score per prop type (HIT, TB, HR, KS) in real-time.
Returns a confidence multiplier that the model can use to self-calibrate.

When the bot is nailing a prop type (Brier < 0.15 over 20+ samples),
confidence gets a 1.2x boost → bigger Kelly sizes.

When it's struggling (Brier > 0.30), confidence drops to 0.8x → smaller
sizes until accuracy improves.

The tracker can persist to Supabase so state survives Railway restarts.

Usage in sector_bots.py:
    from shared.brier_tracker import BrierTracker

    tracker = BrierTracker.from_supabase(database_url)

    # After computing a prediction:
    multiplier = tracker.confidence_multiplier(prop_code)
    adjusted_confidence = prediction.confidence * multiplier

    # After a market resolves:
    tracker.record(prop_code, our_prob, outcome)
    tracker.save_to_supabase(database_url)
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PropStats:
    """Running statistics for a single prop type."""
    count:       int   = 0
    sum_brier:   float = 0.0
    sum_correct: int   = 0      # directional accuracy (our_p > 0.5 matches outcome)
    recent:      list  = field(default_factory=list)  # last N brier scores for trend

    MAX_RECENT = 50  # rolling window for trend detection

    @property
    def brier(self) -> Optional[float]:
        return self.sum_brier / self.count if self.count > 0 else None

    @property
    def accuracy(self) -> Optional[float]:
        return self.sum_correct / self.count if self.count > 0 else None

    @property
    def recent_brier(self) -> Optional[float]:
        """Brier over the last N resolved predictions."""
        if len(self.recent) < 5:
            return None
        return sum(self.recent) / len(self.recent)

    @property
    def trend(self) -> str:
        """Is the model getting better or worse on this prop?"""
        if len(self.recent) < 10:
            return "insufficient_data"
        half = len(self.recent) // 2
        first_half = sum(self.recent[:half]) / half
        second_half = sum(self.recent[half:]) / (len(self.recent) - half)
        if second_half < first_half - 0.02:
            return "improving"
        if second_half > first_half + 0.02:
            return "degrading"
        return "stable"


class BrierTracker:
    """Track Brier scores per prop type with confidence multipliers."""

    # Brier thresholds for confidence adjustment
    EXCELLENT = 0.15   # → 1.2x confidence
    GOOD      = 0.20   # → 1.1x
    AVERAGE   = 0.25   # → 1.0x (no adjustment)
    POOR      = 0.30   # → 0.9x
    TERRIBLE  = 0.35   # → 0.8x

    MIN_SAMPLES = 20   # need this many before adjusting confidence

    def __init__(self):
        self._props: dict[str, PropStats] = defaultdict(PropStats)

    def record(self, prop_code: str, our_prob: float, outcome: int):
        """Record a resolved prediction."""
        brier = (our_prob - float(outcome)) ** 2
        correct = int((our_prob > 0.5) == bool(outcome))

        stats = self._props[prop_code]
        stats.count += 1
        stats.sum_brier += brier
        stats.sum_correct += correct
        stats.recent.append(brier)

        # Trim rolling window
        if len(stats.recent) > PropStats.MAX_RECENT:
            stats.recent = stats.recent[-PropStats.MAX_RECENT:]

        logger.debug(
            "[brier] %s: recorded brier=%.4f (avg=%.4f over %d samples)",
            prop_code, brier, stats.brier, stats.count,
        )

    def confidence_multiplier(self, prop_code: str) -> float:
        """Return a multiplier for prediction confidence based on track record.

        < 1.0 = model is underperforming on this prop, reduce confidence
        = 1.0 = average or insufficient data
        > 1.0 = model is outperforming, boost confidence
        """
        stats = self._props.get(prop_code)
        if not stats or stats.count < self.MIN_SAMPLES:
            return 1.0  # not enough data to judge

        # Use recent Brier if available, otherwise lifetime
        brier = stats.recent_brier or stats.brier
        if brier is None:
            return 1.0

        if brier <= self.EXCELLENT:
            mult = 1.2
        elif brier <= self.GOOD:
            mult = 1.1
        elif brier <= self.AVERAGE:
            mult = 1.0
        elif brier <= self.POOR:
            mult = 0.9
        else:
            mult = 0.8

        # Log when multiplier is non-neutral
        if mult != 1.0:
            logger.info(
                "[brier] %s: Brier=%.4f (%d samples) → confidence ×%.1f",
                prop_code, brier, stats.count, mult,
            )

        return mult

    def get_stats(self, prop_code: str) -> Optional[dict]:
        """Get detailed stats for a prop type."""
        stats = self._props.get(prop_code)
        if not stats:
            return None
        return {
            "prop_code":    prop_code,
            "count":        stats.count,
            "brier":        round(stats.brier, 4) if stats.brier else None,
            "accuracy":     round(stats.accuracy, 3) if stats.accuracy else None,
            "recent_brier": round(stats.recent_brier, 4) if stats.recent_brier else None,
            "trend":        stats.trend,
            "multiplier":   self.confidence_multiplier(prop_code),
        }

    def all_stats(self) -> list[dict]:
        """Get stats for all tracked prop types."""
        return [self.get_stats(code) for code in sorted(self._props.keys())]

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            code: {
                "count": s.count,
                "sum_brier": s.sum_brier,
                "sum_correct": s.sum_correct,
                "recent": s.recent[-PropStats.MAX_RECENT:],
            }
            for code, s in self._props.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BrierTracker":
        """Deserialize from dict."""
        tracker = cls()
        for code, d in data.items():
            stats = PropStats(
                count=d["count"],
                sum_brier=d["sum_brier"],
                sum_correct=d["sum_correct"],
                recent=d.get("recent", []),
            )
            tracker._props[code] = stats
        return tracker

    def save_to_supabase(self, database_url: str):
        """Persist tracker state to a Supabase table."""
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS brier_tracker (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                INSERT INTO brier_tracker (id, data, updated_at)
                VALUES (1, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET data = %s, updated_at = NOW()
            """, (json.dumps(self.to_dict()), json.dumps(self.to_dict())))
            conn.commit()
            conn.close()
            logger.debug("[brier] Saved tracker state to Supabase")
        except Exception as e:
            logger.warning("[brier] Failed to save to Supabase: %s", e)

    @classmethod
    def from_supabase(cls, database_url: str) -> "BrierTracker":
        """Load tracker state from Supabase."""
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            cur.execute("SELECT data FROM brier_tracker WHERE id = 1")
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                tracker = cls.from_dict(data)
                logger.info(
                    "[brier] Loaded tracker: %d prop types, %d total samples",
                    len(tracker._props),
                    sum(s.count for s in tracker._props.values()),
                )
                return tracker
        except Exception as e:
            logger.info("[brier] No saved state found (%s), starting fresh", e)
        return cls()

    @classmethod
    def bootstrap_from_signals(cls, database_url: str) -> "BrierTracker":
        """Bootstrap tracker from resolved signals in Supabase.

        Call once to seed the tracker with historical data.
        """
        tracker = cls()
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            cur.execute("""
                SELECT ticker, our_prob, outcome
                FROM signals
                WHERE outcome IS NOT NULL
                  AND sector = 'sports'
                  AND ticker LIKE 'KXMLB%'
            """)
            for ticker, our_prob, outcome in cur.fetchall():
                # Extract prop code from ticker: KXMLBHIT → HIT, KXMLBTB → TB
                prop_code = None
                for code in ("HIT", "TB", "HRR", "HR", "KS"):
                    if f"KXMLB{code}" in ticker.upper():
                        prop_code = code
                        break
                if prop_code:
                    tracker.record(prop_code, float(our_prob), int(outcome))

            conn.close()
            logger.info(
                "[brier] Bootstrapped from %d resolved signals",
                sum(s.count for s in tracker._props.values()),
            )
        except Exception as e:
            logger.warning("[brier] Bootstrap failed: %s", e)
        return tracker