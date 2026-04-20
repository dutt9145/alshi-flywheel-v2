"""
shared/consensus_engine.py  (v5 — sector calibration offsets)

Changes vs v4:
  1. Added SECTOR_CALIBRATION dict with probability offsets based on
     historical prediction vs actual analysis (2026-04-19):
       Sports: predicted 33% YES, actual 46% → +13% offset
       Crypto: predicted 25% YES, actual 38% → +13% offset
       Weather: predicted 34% YES, actual 25% → -9% offset
  2. Gate 2 (direction agreement) now uses calibrated probabilities
     to determine direction, fixing the all-NO-trades bug.
  3. Weighted average probs are calibrated before direction/edge calc.
  4. All other gates unchanged from v4.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import (
    CONSENSUS_EDGE_PCT, CONSENSUS_CONFIDENCE, SECTORS,
    MAX_EDGE_PCT, DIRECTION_FILTER,
    sector_max_edge,   # v4: per-sector edge ceilings
)

logger = logging.getLogger(__name__)

# ── Sector calibration offsets (v5) ────────────────────────────────────────────
# Based on historical prediction vs actual analysis (2026-04-19):
#   Sector        | Predicted | Actual | Offset
#   sports        | 33% YES   | 46%    | +0.13 (under-predicting)
#   crypto        | 25% YES   | 38%    | +0.13 (under-predicting)
#   weather       | 34% YES   | 25%    | -0.09 (over-predicting)
#   financial_mkts| (tbd)     | (tbd)  | 0.00  (no data yet)
#
# These offsets shift raw model probabilities before direction is determined.
# This fixes the bug where 971/971 trades were NO because models consistently
# predicted low YES probabilities.
SECTOR_CALIBRATION = {
    'sports':            0.13,
    'crypto':            0.13,
    'weather':          -0.09,
    'financial_markets': 0.00,
    'economics':         0.00,
    'politics':          0.00,
    'global_events':     0.00,
}


def _calibrate_prob(prob: float, sector: str) -> float:
    """Apply sector calibration offset to probability, clamped to [0.05, 0.95]."""
    offset = SECTOR_CALIBRATION.get(sector, 0.0)
    return max(0.05, min(0.95, prob + offset))


@dataclass
class BotSignal:
    """
    Output from a single sector bot for a given contract.
    """
    sector:          str
    prob:            float          # our P(YES) — raw, before calibration
    confidence:      float          # model confidence 0–1
    market_prob:     float          # market's implied P(YES) from yes_price/100
    brier_score:     Optional[float] = None   # rolling Brier — lower = better

    @property
    def calibrated_prob(self) -> float:
        """Probability after applying sector calibration offset."""
        return _calibrate_prob(self.prob, self.sector)

    @property
    def edge(self) -> float:
        """Signed edge: positive = we think YES is underpriced (uses calibrated prob)."""
        return self.calibrated_prob - self.market_prob

    @property
    def direction(self) -> str:
        """YES if we think it should resolve YES, NO otherwise (uses calibrated prob)."""
        return "YES" if self.calibrated_prob > self.market_prob else "NO"


@dataclass
class ConsensusResult:
    """
    Outcome of the consensus gate for a single contract.
    """
    ticker:          str
    execute:         bool
    direction:       Optional[str]
    avg_prob:        float
    avg_edge:        float
    avg_confidence:  float
    signals:         list[BotSignal]
    reject_reason:   Optional[str] = None

    def summary(self) -> str:
        status = "EXECUTE" if self.execute else f"PASS ({self.reject_reason})"
        return (
            f"{self.ticker} → {status} | dir={self.direction} "
            f"avg_prob={self.avg_prob:.2%} edge={self.avg_edge:+.2%} "
            f"conf={self.avg_confidence:.2%}"
        )


class ConsensusEngine:
    """
    Aggregates signals from all registered sector bots and decides
    whether to execute a trade.

    Six gates (all must pass):
      1. At least one bot responded
      2. All responding bots agree on direction
      3. Weighted-average edge >= CONSENSUS_EDGE_PCT (floor)
      4. Weighted-average confidence >= CONSENSUS_CONFIDENCE
      5. Weighted-average edge <= sector edge ceiling (per-sector, v4)
      6. Direction matches DIRECTION_FILTER (NO-only mode)
    """

    def __init__(self):
        self._brier_baseline = {s: None for s in SECTORS}

    def _bot_weight(self, signal: BotSignal) -> float:
        """
        Inverse-Brier weighting: better-calibrated bots get more say.
        Falls back to equal weight if no Brier score is available.
        """
        bs = signal.brier_score
        if bs is None or bs <= 0:
            return 1.0
        return min(1.0 / bs, 10.0)

    def evaluate(
        self,
        ticker:          str,
        yes_price_cents: int,
        signals:         list[BotSignal],
    ) -> ConsensusResult:
        """
        Run the six-gate consensus check.

        Parameters
        ----------
        ticker          : Kalshi market ticker
        yes_price_cents : current market YES price in cents
        signals         : list of BotSignal — one per sector bot
        """
        market_prob = yes_price_cents / 100.0

        # ── Gate 1: at least one bot must respond ──────────────────────────────
        if len(signals) < 1:
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=None, avg_prob=0.5,
                avg_edge=0.0, avg_confidence=0.0,
                signals=signals,
                reject_reason="No bots responded",
            )

        # ── Gate 2: direction agreement (v5: uses calibrated probs) ────────────
        # Direction is now determined from calibrated_prob, not raw prob.
        # This fixes the all-NO-trades bug caused by systematic under-prediction.
        directions = [s.direction for s in signals]  # Uses calibrated_prob via property
        if len(set(directions)) > 1:
            disagreements = ", ".join(f"{s.sector}={s.direction}" for s in signals)
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=None, avg_prob=0.5,
                avg_edge=0.0, avg_confidence=0.0,
                signals=signals,
                reject_reason=f"Direction disagreement: {disagreements}",
            )

        direction = directions[0]

        # ── Weighted averages (v5: uses calibrated probs) ──────────────────────
        weights    = np.array([self._bot_weight(s) for s in signals])
        weights   /= weights.sum()

        # v5: Use calibrated probabilities for averaging
        probs      = np.array([s.calibrated_prob for s in signals])
        confs      = np.array([s.confidence for s in signals])

        avg_prob  = float(np.dot(weights, probs))
        avg_conf  = float(np.dot(weights, confs))
        avg_edge  = avg_prob - market_prob
        if direction == "NO":
            avg_edge = market_prob - avg_prob

        # Log calibration effect for debugging
        raw_probs = [s.prob for s in signals]
        cal_probs = [s.calibrated_prob for s in signals]
        if raw_probs != cal_probs:
            logger.debug(
                "[Consensus] %s calibration: raw=%s cal=%s dir=%s",
                ticker, 
                [f"{p:.2f}" for p in raw_probs],
                [f"{p:.2f}" for p in cal_probs],
                direction,
            )

        # ── Gate 3: edge floor ─────────────────────────────────────────────────
        if avg_edge < CONSENSUS_EDGE_PCT:
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=direction, avg_prob=avg_prob,
                avg_edge=avg_edge, avg_confidence=avg_conf,
                signals=signals,
                reject_reason=(
                    f"Edge {avg_edge:.2%} < floor {CONSENSUS_EDGE_PCT:.2%}"
                ),
            )

        # ── Gate 4: confidence floor ───────────────────────────────────────────
        if avg_conf < CONSENSUS_CONFIDENCE:
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=direction, avg_prob=avg_prob,
                avg_edge=avg_edge, avg_confidence=avg_conf,
                signals=signals,
                reject_reason=(
                    f"Confidence {avg_conf:.2%} < floor {CONSENSUS_CONFIDENCE:.2%}"
                ),
            )

        # ── Gate 5: edge ceiling (v4: per-sector) ─────────────────────────────
        # v3 used global MAX_EDGE_PCT (25%) for all sectors. This rejected
        # genuine weather signals (NOAA 96% conf, 58% edge on Houston temp)
        # and extreme MLB player props (≥3 hits for a .200 hitter).
        # v4 uses sector_max_edge() which gives weather 60%, sports 40%.
        lead_sector = signals[0].sector
        edge_ceiling = sector_max_edge(lead_sector)

        if avg_edge > edge_ceiling:
            logger.info(
                "[Consensus] %s REJECTED: edge %.2f%% exceeds ceiling %.2f%% (sector=%s)",
                ticker, avg_edge * 100, edge_ceiling * 100, lead_sector,
            )
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=direction, avg_prob=avg_prob,
                avg_edge=avg_edge, avg_confidence=avg_conf,
                signals=signals,
                reject_reason=(
                    f"Edge {avg_edge:.2%} > ceiling {edge_ceiling:.2%}"
                ),
            )

        # ── Gate 6: direction filter (audit: YES 22% WR, NO 65% WR) ───────────
        if DIRECTION_FILTER != "BOTH" and direction != DIRECTION_FILTER:
            logger.info(
                "[Consensus] %s REJECTED: direction %s blocked (%s only)",
                ticker, direction, DIRECTION_FILTER,
            )
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=direction, avg_prob=avg_prob,
                avg_edge=avg_edge, avg_confidence=avg_conf,
                signals=signals,
                reject_reason=(
                    f"Direction {direction} blocked by filter ({DIRECTION_FILTER} only)"
                ),
            )

        # ── All gates passed → EXECUTE ─────────────────────────────────────────
        result = ConsensusResult(
            ticker=ticker, execute=True,
            direction=direction, avg_prob=avg_prob,
            avg_edge=avg_edge, avg_confidence=avg_conf,
            signals=signals,
        )
        logger.info("✅ Consensus reached: %s", result.summary())
        return result