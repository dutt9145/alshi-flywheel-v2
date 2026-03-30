"""
bots/base_bot.py  (v4 — news velocity feature injection)

Changes vs v3:
  1. evaluate() now accepts an optional news_signal param
     — if provided, injects velocity features into the feature vector
       before prediction (velocity_score, is_spiking)
     — if not provided, pads [0.0, 0.0] to maintain consistent shape
  2. fetch_features() subclass implementations should NOT add news features —
     base class handles injection uniformly after fetch_features() returns
  3. Everything else unchanged from v3
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

from config.settings import MIN_EDGE_PCT
from shared.bayesian_poly_model import BayesianPolyModel
from shared.consensus_engine import BotSignal
from shared.calibration_logger import compute_calibration

if TYPE_CHECKING:
    from shared.news_signal import NewsSignal

logger = logging.getLogger(__name__)


def _market_prob(market: dict) -> float:
    """Extract yes probability (0.0–1.0) from a Kalshi market object."""
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
    return 0.50  # neutral fallback


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

    # ── Must override ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def sector_name(self) -> str:
        """Human-readable sector identifier, e.g. 'economics'."""
        ...

    @abstractmethod
    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        """
        Pull live data for a given Kalshi market and return:
            features → np.ndarray fed to BayesianPolyModel
            context  → dict for logging / debugging
        Do NOT add news velocity features here — base class handles injection.
        """
        ...

    @abstractmethod
    def is_relevant(self, market: dict) -> bool:
        """Return True if this sector bot should evaluate the given market."""
        ...

    # ── Core pipeline (do not override) ────────────────────────────────────────

    def evaluate(
        self,
        market:      dict,
        news_signal: Optional["NewsSignal"] = None,
    ) -> Optional[BotSignal]:
        """
        Full pipeline for one market:
            fetch → inject news velocity → predict → pre-filter on edge → BotSignal

        Parameters
        ----------
        market : dict
            Kalshi market object.
        news_signal : NewsSignal, optional
            If provided, velocity features are injected into the feature vector.
            The orchestrator passes this in; bots never import NewsSignal directly.

        NOTE: log_signal() is intentionally NOT called here.
        The orchestrator calls it once after consensus with better data.
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

        # ── News velocity injection ────────────────────────────────────────────
        # Appended AFTER sector features so subclass feature shapes are stable.
        # Bots trained without news features will have [0.0, 0.0] appended,
        # which is backward-compatible with existing model checkpoints as long
        # as retrain is run after this deploy (nightly at 2am).
        title = market.get("title", "")
        if news_signal is not None:
            velocity, is_spiking = news_signal.get_velocity_features(title)
        else:
            velocity, is_spiking = 0.0, 0.0

        features = np.append(features, [velocity, is_spiking])
        context["news_velocity"]  = velocity
        context["news_is_spiking"] = bool(is_spiking)

        if is_spiking:
            logger.info(
                "[%s] NEWS SPIKE on %s — velocity=%.2f",
                self.sector_name, ticker, velocity,
            )

        # ── Prediction ────────────────────────────────────────────────────────
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

        log_msg = (
            "[%s] %s → our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f"
        )
        if is_spiking:
            log_msg += " ⚡NEWS"
        logger.info(
            log_msg,
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