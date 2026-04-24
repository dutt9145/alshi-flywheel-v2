"""
shared/consensus_engine.py  (v6 — sports calibration reset after v14 HRR fix)

Changes vs v5:
  1. SECTOR_CALIBRATION['sports'] changed from +0.13 → 0.00.

     Context: The +0.13 sports offset was added in v5 to compensate for the
     "971/971 trades were NO" bug, which was rooted in poisoned Bayesian
     priors in the pre-v12.7 sports path. Since v12.7, sports routes
     exclusively through the MLB/NBA player-prop models (bypassing
     Bayesian), and since v14 of mlb_hit_model.py, HRR has a dedicated
     predictor that outputs real threshold-sensitive probabilities.

     With those fixes in place, the +0.13 offset is no longer correcting
     a systematic under-prediction — it is introducing bias. Specifically:
     it was pinning HRR raw probs (~0.001) to 0.13 regardless of threshold,
     masking the underlying model and generating trades that looked like
     fades but were ghost signals.

  2. SECTOR_CALIBRATION['weather'] left at -0.15 — weather continues to
     show predicted vs actual divergence per the 2026-04-20 audit.

  3. NEW: Per-signal calibration logging. When calibration shifts a
     probability, we log raw/calibrated/direction so future Brier-score
     analysis can distinguish model error from calibration error.

  All other gates unchanged from v5.

  FOLLOW-UP (recommended):
    - Add `raw_prob` and `calibrated_prob` columns to the `trades` and
      `main_signals` tables. Without this, we cannot Brier-score the
      underlying model independently of the calibration layer.
    - Re-audit weather calibration after 30 days on the v14 HRR-aware
      sports baseline, to confirm -0.15 is still the right weather offset
      relative to a fairly-calibrated sports sector.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import (
    CONSENSUS_EDGE_PCT, CONSENSUS_CONFIDENCE, SECTORS,
    MAX_EDGE_PCT, DIRECTION_FILTER,
    sector_max_edge,
)

logger = logging.getLogger(__name__)

# ── Sector calibration offsets (v6) ────────────────────────────────────────────
# v6 (2026-04-23): Set sports to 0.00 after v14 HRR predictor fix.
#
# The +0.13 sports offset was introduced in v5 when all sports markets ran
# through a shared poisoned Bayesian model that systematically under-predicted
# YES. That Bayesian path no longer exists — SportsBot v12.7 only routes
# through player-prop models (MLB hit, MLB HR, MLB HRR via v14, MLB KS,
# NBA props). These models produce per-game, threshold-sensitive probabilities
# with their own calibration. Adding +0.13 on top of them pushes everything
# uniformly toward YES, which floors near-zero probabilities into the 0.13
# range and distorts edge calculations.
#
# Weather offset remains at -0.15 per 2026-04-20 audit (282 resolved signals,
# predicted 35% YES, actual 20%). Revisit after 30 days.
SECTOR_CALIBRATION = {
    'sports':            0.00,   # v6: was +0.13, now zero — player-prop models are self-calibrating
    'crypto':            0.13,
    'weather':           0.00,
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
    """Output from a single sector bot for a given contract."""
    sector:          str
    prob:            float          # our P(YES) — raw, before calibration
    confidence:      float          # model confidence 0–1
    market_prob:     float          # market's implied P(YES) from yes_price/100
    brier_score:     Optional[float] = None

    @property
    def calibrated_prob(self) -> float:
        return _calibrate_prob(self.prob, self.sector)

    @property
    def edge(self) -> float:
        return self.calibrated_prob - self.market_prob

    @property
    def direction(self) -> str:
        return "YES" if self.calibrated_prob > self.market_prob else "NO"


@dataclass
class ConsensusResult:
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
    """Aggregates signals from all registered sector bots and decides
    whether to execute a trade."""

    def __init__(self):
        self._brier_baseline = {s: None for s in SECTORS}

    def _bot_weight(self, signal: BotSignal) -> float:
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

        # ── Gate 2: direction agreement (uses calibrated probs) ───────────────
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

        # ── Weighted averages (uses calibrated probs) ──────────────────────────
        weights    = np.array([self._bot_weight(s) for s in signals])
        weights   /= weights.sum()

        probs      = np.array([s.calibrated_prob for s in signals])
        confs      = np.array([s.confidence for s in signals])

        avg_prob  = float(np.dot(weights, probs))
        avg_conf  = float(np.dot(weights, confs))
        avg_edge  = avg_prob - market_prob
        if direction == "NO":
            avg_edge = market_prob - avg_prob

        # v6: Log calibration effect when it shifted any signal's direction
        # or materially moved a probability. Helps distinguish model error
        # from calibration error in future Brier analysis.
        raw_probs = [s.prob for s in signals]
        cal_probs = [s.calibrated_prob for s in signals]
        any_shifted = any(abs(r - c) > 0.001 for r, c in zip(raw_probs, cal_probs))
        if any_shifted:
            logger.info(
                "[Consensus-calib] %s raw=%s cal=%s sector=%s dir=%s",
                ticker,
                [f"{p:.3f}" for p in raw_probs],
                [f"{p:.3f}" for p in cal_probs],
                [s.sector for s in signals],
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

        # ── Gate 5: edge ceiling (per-sector) ─────────────────────────────────
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

        # ── Gate 6: direction filter ─────────────────────────────────────────
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