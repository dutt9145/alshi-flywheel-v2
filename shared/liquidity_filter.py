"""
liquidity_filter.py (v1.0)

Liquidity filter for market selection. Skip thin/illiquid markets where:
  1. Spread is too wide (execution slippage eats edge)
  2. Volume is too low (hard to exit, stale prices)

Usage:
    from shared.liquidity_filter import LiquidityFilter
    
    liq = LiquidityFilter()
    if not liq.passes(market):
        logger.info("LIQUIDITY SKIP %s: %s", ticker, liq.last_reason)
        continue
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
    
    Thresholds (configurable by sector):
      - max_spread_cents: Skip if bid-ask spread exceeds this
      - max_spread_pct: Skip if spread as % of mid exceeds this
      - min_volume_24h: Skip if 24h volume below this
      - min_volume_for_size: Skip if volume < N × intended position size
    """
    
    # Default thresholds (override per sector if needed)
    DEFAULT_MAX_SPREAD_CENTS = 8       # 8¢ spread max
    DEFAULT_MAX_SPREAD_PCT = 0.15      # 15% of mid price
    DEFAULT_MIN_VOLUME_24H = 100       # At least 100 contracts in 24h
    
    # Sector-specific overrides (tighter for high-volume sectors)
    SECTOR_THRESHOLDS = {
        "sports": {
            "max_spread_cents": 6,
            "max_spread_pct": 0.12,
            "min_volume_24h": 50,
        },
        "crypto": {
            "max_spread_cents": 5,
            "max_spread_pct": 0.10,
            "min_volume_24h": 200,
        },
        "weather": {
            "max_spread_cents": 10,    # Weather markets less liquid
            "max_spread_pct": 0.20,
            "min_volume_24h": 20,
        },
        "politics": {
            "max_spread_cents": 8,
            "max_spread_pct": 0.15,
            "min_volume_24h": 50,
        },
        # v12.5: Economics/financial markets typically have wider spreads
        "economics": {
            "max_spread_cents": 12,
            "max_spread_pct": 0.25,
            "min_volume_24h": 25,
        },
        "financial_markets": {
            "max_spread_cents": 10,
            "max_spread_pct": 0.20,
            "min_volume_24h": 30,
        },
        # Default for unknown sectors — more permissive to avoid false skips
        "unknown": {
            "max_spread_cents": 12,
            "max_spread_pct": 0.25,
            "min_volume_24h": 20,
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
        self.min_volume_24h = min_volume_24h or self.DEFAULT_MIN_VOLUME_24H
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
    
    def check(self, market: dict, sector: str = "unknown") -> LiquidityResult:
        """
        Check if market passes liquidity filters.
        
        Args:
            market: Kalshi market dict with price/volume fields
            sector: Sector name for threshold lookup
            
        Returns:
            LiquidityResult with pass/fail and diagnostics
        """
        if not self.enabled:
            return LiquidityResult(passes=True, reason="filter_disabled")
        
        thresholds = self._get_thresholds(sector)
        
        # Extract bid/ask prices (Kalshi uses dollars, convert to cents)
        yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid") or 0
        yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask") or 0
        
        # Handle both dollar and cent formats
        if isinstance(yes_bid, (int, float)) and yes_bid < 1:
            # Already in dollars, convert to cents
            bid_cents = yes_bid * 100
            ask_cents = yes_ask * 100
        else:
            # Might be in cents already
            bid_cents = float(yes_bid)
            ask_cents = float(yes_ask)
        
        # If no bid/ask, try last_price as mid
        if bid_cents == 0 and ask_cents == 0:
            last = market.get("last_price_dollars") or market.get("last_price") or 0
            if isinstance(last, (int, float)) and last < 1:
                mid_cents = last * 100
            else:
                mid_cents = float(last)
            spread_cents = 0
        else:
            mid_cents = (bid_cents + ask_cents) / 2 if bid_cents > 0 and ask_cents > 0 else max(bid_cents, ask_cents)
            spread_cents = abs(ask_cents - bid_cents) if bid_cents > 0 and ask_cents > 0 else 0
        
        # Extract volume (try various field names)
        volume_24h = (
            market.get("volume_24h") or 
            market.get("volume24h") or 
            market.get("volume") or 
            market.get("total_volume") or
            0
        )
        
        # --- Check 1: Spread in cents ---
        if spread_cents > 0 and spread_cents > thresholds["max_spread_cents"]:
            result = LiquidityResult(
                passes=False,
                reason=f"spread_too_wide: {spread_cents:.1f}¢ > {thresholds['max_spread_cents']}¢",
                spread_cents=spread_cents,
                spread_pct=spread_cents / mid_cents if mid_cents > 0 else None,
                volume_24h=volume_24h,
                mid_price_cents=mid_cents,
            )
            self.last_result = result
            return result
        
        # --- Check 2: Spread as % of mid ---
        if mid_cents > 0 and spread_cents > 0:
            spread_pct = spread_cents / mid_cents
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
        
        # --- Check 3: Minimum volume ---
        if volume_24h < thresholds["min_volume_24h"]:
            result = LiquidityResult(
                passes=False,
                reason=f"volume_too_low: {volume_24h} < {thresholds['min_volume_24h']}",
                spread_cents=spread_cents,
                spread_pct=spread_cents / mid_cents if mid_cents > 0 else None,
                volume_24h=volume_24h,
                mid_price_cents=mid_cents,
            )
            self.last_result = result
            return result
        
        # All checks passed
        result = LiquidityResult(
            passes=True,
            reason="ok",
            spread_cents=spread_cents,
            spread_pct=spread_cents / mid_cents if mid_cents > 0 else None,
            volume_24h=volume_24h,
            mid_price_cents=mid_cents,
        )
        self.last_result = result
        return result
    
    def passes(self, market: dict, sector: str = "unknown") -> bool:
        """Convenience method: returns True if market passes liquidity filter."""
        return self.check(market, sector).passes
    
    @property
    def last_reason(self) -> str:
        """Get reason from last check."""
        return self.last_result.reason if self.last_result else "no_check_run"


# ── Singleton for global access ───────────────────────────────────────────────
_default_filter: Optional[LiquidityFilter] = None


def get_liquidity_filter() -> LiquidityFilter:
    """Get the default liquidity filter instance."""
    global _default_filter
    if _default_filter is None:
        _default_filter = LiquidityFilter()
    return _default_filter


def check_liquidity(market: dict, sector: str = "unknown") -> LiquidityResult:
    """Convenience function to check liquidity with default filter."""
    return get_liquidity_filter().check(market, sector)