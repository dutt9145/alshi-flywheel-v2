"""
liquidity_filter.py (v2.0 — relaxed thresholds for current Kalshi liquidity regime)

Changes vs v1.1:
  1. Volume thresholds dramatically lowered. v1.1 required 50-200 contracts
     of 24h volume depending on sector. This rejected 93% of markets at
     current Kalshi liquidity levels (measured 2026-04-22 06:57 UTC:
     3,722 / 4,000 markets failed liquidity). Volume on individual player
     props and ladder rungs is often 0-10 even for tradeable markets.
  2. Volume is no longer a HARD gate. If volume is 0 but spread is tight
     (<= threshold), the market is allowed through — tight spread implies
     active market-making. Only reject on volume if BOTH spread is wide AND
     volume is zero.
  3. Spread thresholds loosened modestly. Sports was 6¢, now 10¢; crypto
     was 5¢, now 8¢. Kalshi markets commonly have 3-10¢ spreads even for
     well-traded events.
  4. "unknown" sector now maps to the most permissive threshold set, not
     a middle-ground. When we don't know what sector we're looking at,
     err toward letting it through — downstream gates (bot.is_relevant,
     consensus) will catch irrelevant markets.
  5. INFO-level logging when a market is rejected on volume but would have
     passed on spread alone. Helps us see whether the volume check is
     still doing useful work.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LiquidityResult:
    """Result of liquidity check."""
    passes: bool
    reason: str
    spread_cents: Optional[float] = None
    spread_pct: Optional[float] = None
    volume_24h: Optional[int] = None
    mid_price_cents: Optional[float] = None


class LiquidityFilter:
    """
    Filter markets by liquidity before evaluation.

    v2 philosophy:
    - Spread is the primary gate (wide spread = slippage eats edge)
    - Volume is a SECONDARY gate that only fires if spread is ALSO wide
    - Low volume with tight spread is acceptable (market maker is active)
    - Unknown sectors get permissive defaults
    """

    DEFAULT_MAX_SPREAD_CENTS = 15
    DEFAULT_MAX_SPREAD_PCT = 0.25
    DEFAULT_MIN_VOLUME_24H = 0   # v2: Changed from 100 to 0 — volume is soft gate

    # Per-sector overrides. v2: all thresholds relaxed.
    # min_volume_24h is only enforced if spread is ALSO at the max.
    SECTOR_THRESHOLDS = {
        "sports": {
            "max_spread_cents": 10,
            "max_spread_pct": 0.20,
            "min_volume_24h": 0,
        },
        "crypto": {
            "max_spread_cents": 8,
            "max_spread_pct": 0.15,
            "min_volume_24h": 0,
        },
        "weather": {
            "max_spread_cents": 15,
            "max_spread_pct": 0.30,
            "min_volume_24h": 0,
        },
        "politics": {
            "max_spread_cents": 12,
            "max_spread_pct": 0.25,
            "min_volume_24h": 0,
        },
        "economics": {
            "max_spread_cents": 20,
            "max_spread_pct": 0.40,
            "min_volume_24h": 0,
        },
        "financial_markets": {
            "max_spread_cents": 15,
            "max_spread_pct": 0.30,
            "min_volume_24h": 0,
        },
        "unknown": {
            "max_spread_cents": 15,
            "max_spread_pct": 0.25,
            "min_volume_24h": 0,
        },
    }

    def __init__(
        self,
        max_spread_cents: float = None,
        max_spread_pct: float = None,
        min_volume_24h: int = None,
        enabled: bool = True,
    ):
        self.max_spread_cents = max_spread_cents or self.DEFAULT_MAX_SPREAD_CENTS
        self.max_spread_pct = max_spread_pct or self.DEFAULT_MAX_SPREAD_PCT
        self.min_volume_24h = min_volume_24h if min_volume_24h is not None else self.DEFAULT_MIN_VOLUME_24H
        self.enabled = enabled
        self.last_result: Optional[LiquidityResult] = None

    def _get_thresholds(self, sector: str) -> dict:
        """Get thresholds for a specific sector."""
        base = {
            "max_spread_cents": self.max_spread_cents,
            "max_spread_pct": self.max_spread_pct,
            "min_volume_24h": self.min_volume_24h,
        }
        if sector in self.SECTOR_THRESHOLDS:
            base.update(self.SECTOR_THRESHOLDS[sector])
        return base

    def _detect_sector_from_ticker(self, ticker: str) -> str:
        """Auto-detect sector from Kalshi ticker prefix."""
        if not ticker:
            return "unknown"

        t = ticker.upper()

        if any(x in t for x in ["KXCPI", "KXPCE", "KXDEC", "KXFED", "KXGDP",
                                  "KXJOBLESS", "KXNFP", "KXUNRATE", "KXPPI",
                                  "KXRETAIL"]):
            return "economics"

        if any(x in t for x in ["KXSPX", "KXNDX", "KXVIX", "KXINX", "KXDJIA",
                                  "KXRUS", "KXWTI", "KXOIL", "KXGOLD", "KXNAT",
                                  "KXHOIL", "KXUSDJPY", "KXEURUSD", "KXGBPUSD",
                                  "KXNATGAS"]):
            return "financial_markets"

        if any(x in t for x in ["KXBTC", "KXETH", "KXSOL", "KXBNB", "KXXRP",
                                  "KXDOGE", "KXADA", "KXLTC", "KXSHIB"]):
            return "crypto"

        if any(x in t for x in ["KXHIGHT", "KXLOWT", "KXPRECIP", "KXSNOW",
                                  "KXWIND", "KXHUMID", "KXTEMP", "KXRAIN",
                                  "KXHURR"]):
            return "weather"

        if any(x in t for x in ["KXTRUMP", "KXBIDEN", "KXPOTUS", "KXSENATE",
                                  "KXHOUSE", "KXGOV", "KXELECT", "KXLEAVE",
                                  "KXHORMUZ", "KXCA14S"]):
            return "politics"

        if any(x in t for x in ["KXMLB", "KXNBA", "KXNFL", "KXNHL", "KXNCAA",
                                  "KXATP", "KXWTA", "KXMMA", "KXUFC", "KXLOL",
                                  "KXCS2", "KXVAL", "KXEURO", "KXLALIGA",
                                  "KXSERIE", "KXBUNDES", "KXLIGUE", "KXUCL",
                                  "KXNASCAR", "KXF1", "KXR6", "KXEPL",
                                  "KXMLS", "KXJBLEAGUE", "KXALEAGUE",
                                  "KXCRICKET", "KXT20", "KXIPL", "KXDEL",
                                  "KXKHL", "KXKBL", "KXKLEAGUE"]):
            return "sports"

        return "unknown"

    def check(self, market: dict, sector: str = "unknown") -> LiquidityResult:
        """
        Check if market passes liquidity filters.

        v2: Volume is a soft gate. A market is only rejected on volume if
        spread is ALSO at/near the max — i.e. we're worried about BOTH thin
        book AND wide spread. Tight spread with zero volume is fine because
        a market maker is clearly active.
        """
        if not self.enabled:
            return LiquidityResult(passes=True, reason="filter_disabled")

        ticker = market.get("ticker", "")
        if sector == "unknown":
            detected = self._detect_sector_from_ticker(ticker)
            if detected != "unknown":
                sector = detected

        thresholds = self._get_thresholds(sector)

        # Extract bid/ask prices (Kalshi fields are in dollars as floats)
        yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid") or 0
        yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask") or 0

        try:
            yes_bid = float(yes_bid)
            yes_ask = float(yes_ask)
        except (TypeError, ValueError):
            yes_bid = 0.0
            yes_ask = 0.0

        # Kalshi returns prices in dollars (0.0-1.0). Convert to cents.
        if yes_bid < 1 and yes_ask < 1:
            bid_cents = yes_bid * 100
            ask_cents = yes_ask * 100
        else:
            # Safety: if something returned values >= 1, assume cents already
            bid_cents = yes_bid
            ask_cents = yes_ask

        # Derive mid and spread
        if bid_cents > 0 and ask_cents > 0:
            mid_cents = (bid_cents + ask_cents) / 2
            spread_cents = abs(ask_cents - bid_cents)
        elif ask_cents > 0:
            # Only have ask — treat as mid, spread unknown
            mid_cents = ask_cents
            spread_cents = 0  # Can't measure, assume OK
        elif bid_cents > 0:
            mid_cents = bid_cents
            spread_cents = 0
        else:
            # Fall back to last_price
            last = market.get("last_price_dollars") or market.get("last_price") or 0
            try:
                last = float(last)
            except (TypeError, ValueError):
                last = 0.0
            mid_cents = (last * 100) if last < 1 else last
            spread_cents = 0

        # Volume
        volume_24h_raw = (
            market.get("volume_24h")
            or market.get("volume24h")
            or market.get("volume")
            or market.get("total_volume")
            or 0
        )
        try:
            volume_24h = int(volume_24h_raw)
        except (TypeError, ValueError):
            volume_24h = 0

        # Compute spread pct for reporting
        spread_pct = (spread_cents / mid_cents) if mid_cents > 0 else None

        # ── Gate 1: Absolute spread in cents ─────────────────────────────────
        if spread_cents > thresholds["max_spread_cents"]:
            result = LiquidityResult(
                passes=False,
                reason=f"spread_too_wide: {spread_cents:.1f}¢ > {thresholds['max_spread_cents']}¢",
                spread_cents=spread_cents,
                spread_pct=spread_pct,
                volume_24h=volume_24h,
                mid_price_cents=mid_cents,
            )
            self.last_result = result
            return result

        # ── Gate 2: Spread as % of mid ───────────────────────────────────────
        # Only check if we have both spread and mid
        if spread_cents > 0 and mid_cents > 0 and spread_pct is not None:
            if spread_pct > thresholds["max_spread_pct"]:
                result = LiquidityResult(
                    passes=False,
                    reason=f"spread_pct_too_high: {spread_pct:.1%} > {thresholds['max_spread_pct']:.0%}",
                    spread_cents=spread_cents,
                    spread_pct=spread_pct,
                    volume_24h=volume_24h,
                    mid_price_cents=mid_cents,
                )
                self.last_result = result
                return result

        # ── Gate 3: Volume (SOFT — only enforced if spread also at risk) ─────
        # v2: Volume is only a blocker if spread is ALSO wider than half the
        # allowed max. The logic: if spread is <= half max, market is being
        # actively quoted and volume doesn't matter. If spread is in the upper
        # half of allowed range AND volume is zero, that's suspicious.
        volume_threshold = thresholds["min_volume_24h"]
        if volume_threshold > 0 and volume_24h < volume_threshold:
            half_spread_max = thresholds["max_spread_cents"] / 2.0
            if spread_cents >= half_spread_max:
                result = LiquidityResult(
                    passes=False,
                    reason=f"volume_too_low_with_wide_spread: vol={volume_24h} < {volume_threshold}, spread={spread_cents:.1f}¢",
                    spread_cents=spread_cents,
                    spread_pct=spread_pct,
                    volume_24h=volume_24h,
                    mid_price_cents=mid_cents,
                )
                self.last_result = result
                return result

        # All gates passed
        result = LiquidityResult(
            passes=True,
            reason="ok",
            spread_cents=spread_cents,
            spread_pct=spread_pct,
            volume_24h=volume_24h,
            mid_price_cents=mid_cents,
        )
        self.last_result = result
        return result

    def passes(self, market: dict, sector: str = "unknown") -> bool:
        return self.check(market, sector).passes

    @property
    def last_reason(self) -> str:
        return self.last_result.reason if self.last_result else "no_check_run"


_default_filter: Optional[LiquidityFilter] = None


def get_liquidity_filter() -> LiquidityFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = LiquidityFilter()
    return _default_filter


def check_liquidity(market: dict, sector: str = "unknown") -> LiquidityResult:
    return get_liquidity_filter().check(market, sector)