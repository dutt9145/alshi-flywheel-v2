"""
bots/base_bot.py

Abstract base class every sector bot must extend.
Enforces a consistent interface:  evaluate(ticker, market) → BotSignal
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import MIN_EDGE_PCT
from shared.bayesian_poly_model import BayesianPolyModel
from shared.consensus_engine import BotSignal
from shared.calibration_logger import log_signal, compute_calibration

logger = logging.getLogger(__name__)


class BaseBot(ABC):
    """
    Sector bot base class.

    Subclasses must implement:
        sector_name (property)  → str
        fetch_features(market)  → (np.ndarray, dict)
        is_relevant(market)     → bool
    """

    def __init__(self):
        self.model:       BayesianPolyModel = BayesianPolyModel.load(self.sector_name)
        self._calibration: Optional[dict]  = None

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
        Use market title keywords, series ticker prefix, etc.
        """
        ...

    # ── Core pipeline (do not override) ───────────────────────────────────────

    def evaluate(self, market: dict) -> Optional[BotSignal]:
        """
        Full pipeline for one market:
            fetch → predict → pre-filter on edge → log → return BotSignal
        Returns None if this bot is not relevant for the market.
        """
        ticker = market.get("ticker", "UNKNOWN")

        if not self.is_relevant(market):
            return None

        # Market's implied probability
        yes_bid = market.get("yes_bid", 50)
        yes_ask = market.get("yes_ask", 50)
        yes_price = (yes_bid + yes_ask) / 2
        market_prob = yes_price / 100.0

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

        # Pre-filter: don't even log sub-threshold opportunities
        if edge < MIN_EDGE_PCT:
            logger.debug("[%s] %s edge %.2%% too small, skip",
                         self.sector_name, ticker, edge)
            return None

        # Log signal
        direction = "YES" if our_prob > market_prob else "NO"
        brier     = self.model.brier_score()
        log_signal(
            ticker=ticker, sector=self.sector_name,
            our_prob=our_prob, market_prob=market_prob,
            edge=our_prob - market_prob, confidence=confidence,
            direction=direction, brier_score=brier,
        )

        signal = BotSignal(
            sector=self.sector_name, prob=our_prob, confidence=confidence,
            market_prob=market_prob, brier_score=brier,
        )
        logger.info("[%s] %s → our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
                    self.sector_name, ticker, our_prob, market_prob,
                    our_prob - market_prob, direction, confidence)
        return signal

    def record_outcome(self, features: np.ndarray, resolved_yes: bool) -> None:
        """Feed a resolved contract back into the model for online learning."""
        self.model.add_observation(features, resolved_yes)

    def refresh_calibration(self) -> dict:
        self._calibration = compute_calibration(self.sector_name)
        return self._calibration

    def save_model(self) -> None:
        self.model.save()
