"""
bots/base_bot.py  (v3 — double-logging fix)

Changes vs v2:
  1. Removed log_signal() call from evaluate()
     — orchestrator.py already calls log_signal() after consensus with
       better data (consensus prob, edge, confidence vs raw bot output)
     — calling it in both places was doubling every signal in Supabase
       and inflating dashboard counts by 2x
  2. calibration_logger import removed (no longer needed here)
  3. Everything else unchanged
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import MIN_EDGE_PCT
from shared.bayesian_poly_model import BayesianPolyModel
from shared.consensus_engine import BotSignal
from shared.calibration_logger import compute_calibration

logger = logging.getLogger(__name__)


def _market_prob(market: dict) -> float:
    """
    Extract yes probability (0.0-1.0) from a Kalshi market object.
    Kalshi returns prices as dollar strings in 0.0-1.0 range.
    """
    try:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
    except (TypeError, ValueError):
        pass

    try:
        last = float(market.get("last_price_dollars") or 0)
        if last > 0:
            return last
    except (TypeError, ValueError):
        pass

    return 0.50  # neutral fallback if no price data


class BaseBot(ABC):
    """
    Sector bot base class.

    Subclasses must implement:
        sector_name (property)  → str
        fetch_features(market)  → (np.ndarray, dict)
        is_relevant(market)     → bool
    """

    def __init__(self):
        self.model:        BayesianPolyModel = BayesianPolyModel.load(self.sector_name)
        self._calibration: Optional[dict]   = None

    # ── Must override ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def sector_name(self) -> str:
        """Human-readable sector identifier, e.g. 'economics'."""
        ...

    @abstractmethod
    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        """
        Pull live data for a given Kalshi market and return:
            features  → np.ndarray fed to BayesianPolyModel
            context   → dict for logging / debugging
        """
        ...

    @abstractmethod
    def is_relevant(self, market: dict) -> bool:
        """
        Return True if this sector bot should evaluate the given market.
        """
        ...

    # ── Core pipeline (do not override) ───────────────────────────────────────

    def evaluate(self, market: dict) -> Optional[BotSignal]:
        """
        Full pipeline for one market:
            fetch → predict → pre-filter on edge → return BotSignal

        NOTE: log_signal() is intentionally NOT called here.
        The orchestrator calls it once after consensus with better data.
        Calling it here AND in the orchestrator was doubling every signal.
        """
        ticker = market.get("ticker", "UNKNOWN")

        if not self.is_relevant(market):
            return None

        market_prob = _market_prob(market)

        try:
            features, context = self.fetch_features(market)
        except Exception as e:
            logger.error("[%s] Feature fetch failed for %s: %s",
                         self.sector_name, ticker, e)
            return None

        prediction = self.model.predict(features)
        our_prob   = prediction["prob"]
        confidence = prediction["confidence"]
        edge       = abs(our_prob - market_prob)

        # Pre-filter: skip sub-threshold opportunities
        if edge < MIN_EDGE_PCT:
            logger.debug("[%s] %s edge %.2f%% too small, skip",
                         self.sector_name, ticker, edge * 100)
            return None

        direction = "YES" if our_prob > market_prob else "NO"

        signal = BotSignal(
            sector      = self.sector_name,
            prob        = our_prob,
            confidence  = confidence,
            market_prob = market_prob,
            brier_score = self.model.brier_score(),
        )
        logger.info(
            "[%s] %s → our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
            self.sector_name, ticker, our_prob, market_prob,
            our_prob - market_prob, direction, confidence,
        )
        return signal

    def record_outcome(self, features: np.ndarray, resolved_yes: bool) -> None:
        """Feed a resolved contract back into the model for online learning."""
        self.model.add_observation(features, resolved_yes)

    def refresh_calibration(self) -> dict:
        self._calibration = compute_calibration(self.sector_name)
        return self._calibration

    def save_model(self) -> None:
        self.model.save()