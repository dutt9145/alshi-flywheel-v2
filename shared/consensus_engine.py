"""
shared/consensus_engine.py  (v3 — calibration audit gates)

Changes vs v2:
  1. Gate 5 added: edge ceiling (MAX_EDGE_PCT). 30%+ edge trades went
     0/4 for -$600 — the market was right, the model was fooled by
     stale pricing or efficient markets. Rejects anything above 25%.
  2. Gate 6 added: direction filter (DIRECTION_FILTER). YES went 2/9
     (22% WR), NO went 11/17 (65% WR). Default NO-only until YES
     recalibrates. Override via Railway env var without redeploying.
  3. Import updated to pull MAX_EDGE_PCT, DIRECTION_FILTER from settings.
  4. Gates renumbered: 1=bot count, 2=direction agreement, 3=edge floor,
     4=confidence floor, 5=edge ceiling, 6=direction filter.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import (
    CONSENSUS_EDGE_PCT, CONSENSUS_CONFIDENCE, SECTORS,
    MAX_EDGE_PCT, DIRECTION_FILTER,
)

logger = logging.getLogger(__name__)


@dataclass
class BotSignal:
    """
    Output from a single sector bot for a given contract.
    """
    sector:          str
    prob:            float          # our P(YES)
    confidence:      float          # model confidence 0–1
    market_prob:     float          # market's implied P(YES) from yes_price/100
    brier_score:     Optional[float] = None   # rolling Brier — lower = better

    @property
    def edge(self) -> float:
        """Signed edge: positive = we think YES is underpriced."""
        return self.prob - self.market_prob

    @property
    def direction(self) -> str:
        """YES if we think it should resolve YES, NO otherwise."""
        return "YES" if self.prob > self.market_prob else "NO"


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
      5. Weighted-average edge <= MAX_EDGE_PCT (ceiling — traps)
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

        # ── Gate 2: direction agreement ────────────────────────────────────────
        directions = [s.direction for s in signals]
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

        # ── Weighted averages ──────────────────────────────────────────────────
        weights    = np.array([self._bot_weight(s) for s in signals])
        weights   /= weights.sum()

        probs      = np.array([s.prob       for s in signals])
        confs      = np.array([s.confidence for s in signals])

        avg_prob  = float(np.dot(weights, probs))
        avg_conf  = float(np.dot(weights, confs))
        avg_edge  = avg_prob - market_prob
        if direction == "NO":
            avg_edge = market_prob - avg_prob

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

        # ── Gate 5: edge ceiling (calibration audit: 30%+ = 0/4, -$600) ───────
        if avg_edge > MAX_EDGE_PCT:
            logger.info(
                "[Consensus] %s REJECTED: edge %.2f%% exceeds ceiling %.2f%%",
                ticker, avg_edge * 100, MAX_EDGE_PCT * 100,
            )
            return ConsensusResult(
                ticker=ticker, execute=False,
                direction=direction, avg_prob=avg_prob,
                avg_edge=avg_edge, avg_confidence=avg_conf,
                signals=signals,
                reject_reason=(
                    f"Edge {avg_edge:.2%} > ceiling {MAX_EDGE_PCT:.2%}"
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