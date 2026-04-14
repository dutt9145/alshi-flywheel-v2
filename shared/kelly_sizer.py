"""
shared/kelly_sizer.py  (v3 — time-decay Kelly)

Changes vs v2:
  1. Added time_decay_factor() — scales Kelly down as expiry approaches
     - 24+ hours out: 100% Kelly
     - 12 hours out: ~75%
     - 6 hours out: ~55%  
     - 1 hour out: 25%
     - <15 min out: 0% (no bet)
  2. kelly_stake() and no_kelly_stake() now accept optional expiry_time
     or market dict to auto-extract expiry
  3. Returns include "time_decay" field showing the multiplier applied

Changes vs v1:
  1. kelly_stake() and no_kelly_stake() now accept live_bankroll param
     — use this for real-time sizing rather than the module-level BANKROLL
     — falls back to module BANKROLL if live_bankroll not provided
  2. Added drawdown_factor param (0.0–1.0) — scales Kelly fraction down
     when the bot has been losing recently, to protect capital during
     drawdown periods. Supplied by CircuitBreaker.get_drawdown_factor().
  3. MAX_SINGLE_TRADE_USD cap now respected in both YES and NO sizers
  4. Everything else unchanged from v1
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    KELLY_FRACTION, MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD,
    MAX_SECTOR_EXPOSURE, BANKROLL,
)

logger = logging.getLogger(__name__)


# ── Time Decay Configuration ──────────────────────────────────────────────────

TIME_DECAY_FULL_HOURS = 24.0      # 100% Kelly at 24+ hours out
TIME_DECAY_MIN_HOURS = 1.0        # Minimum hours where we still bet
TIME_DECAY_MIN_SCALE = 0.25       # Scale at 1 hour out (25% of Kelly)
TIME_DECAY_CUTOFF_HOURS = 0.25    # Below 15 min, don't bet at all


def time_decay_factor(expiry_time: Optional[datetime], now: Optional[datetime] = None) -> float:
    """
    Calculate time-decay multiplier for Kelly sizing.
    
    The idea: Less time until expiry = less time to be right = smaller bet.
    
    Decay curve:
      - 24+ hours out: 1.0 (full Kelly)
      - 12 hours out: ~0.75
      - 6 hours out: ~0.55
      - 1 hour out: 0.25 (minimum)
      - <15 min out: 0.0 (no bet)
    
    Args:
        expiry_time: When the market closes/settles (timezone-aware)
        now: Current time (defaults to utcnow)
        
    Returns:
        Multiplier between 0.0 and 1.0
    """
    if expiry_time is None:
        # No expiry info — assume plenty of time
        return 1.0
    
    if now is None:
        now = datetime.now(timezone.utc)
    
    # Make sure both are timezone-aware
    if expiry_time.tzinfo is None:
        expiry_time = expiry_time.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    
    hours_until = (expiry_time - now).total_seconds() / 3600
    
    if hours_until < TIME_DECAY_CUTOFF_HOURS:
        # Too close to expiry — don't bet
        return 0.0
    
    if hours_until >= TIME_DECAY_FULL_HOURS:
        # Plenty of time — full Kelly
        return 1.0
    
    if hours_until <= TIME_DECAY_MIN_HOURS:
        # At or below minimum threshold — use minimum scale
        return TIME_DECAY_MIN_SCALE
    
    # Linear decay between min and full
    # hours_until is between 1 and 24
    progress = (hours_until - TIME_DECAY_MIN_HOURS) / (TIME_DECAY_FULL_HOURS - TIME_DECAY_MIN_HOURS)
    scale = TIME_DECAY_MIN_SCALE + progress * (1.0 - TIME_DECAY_MIN_SCALE)
    
    return round(scale, 3)


def parse_expiry_time(market: dict) -> Optional[datetime]:
    """
    Extract expiry/close time from a Kalshi market dict.
    
    Tries various field names that Kalshi might use.
    """
    if market is None:
        return None
        
    for field in ["close_time", "expiration_time", "end_date_time", "settle_time", "end_time"]:
        raw = market.get(field)
        if raw:
            try:
                if isinstance(raw, datetime):
                    return raw
                # Try ISO format
                if isinstance(raw, str):
                    # Handle various ISO formats
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            dt = datetime.strptime(raw, fmt)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt
                        except ValueError:
                            continue
            except Exception:
                pass
    return None


def kelly_stake(
    prob:             float,
    yes_price_cents:  int,
    bankroll:         float,
    sector:           str   = "",
    sector_exposure:  float = 0.0,
    live_bankroll:    float = 0.0,     # Real-time balance from Kalshi API
    drawdown_factor:  float = 1.0,     # Scale down during losing streaks
    expiry_time:      Optional[datetime] = None,  # v3: For time decay
    market:           Optional[dict] = None,      # v3: Auto-extract expiry
) -> dict:
    """
    Compute the recommended dollar stake for a YES position.

    Parameters
    ----------
    prob : float
        Our estimated P(YES), 0.01–0.99.
    yes_price_cents : int
        Kalshi YES midpoint in cents (1–99).
    bankroll : float
        Stored bankroll (from settings or last sync).
    sector : str
        Sector name for exposure tracking.
    sector_exposure : float
        Current dollars at risk in this sector.
    live_bankroll : float
        Real-time balance from Kalshi API (overrides bankroll if > 0).
    drawdown_factor : float
        Multiplier on Kelly fraction. 1.0 = full Kelly. 0.5 = half Kelly
        during a drawdown. Supplied by CircuitBreaker.
    expiry_time : datetime, optional
        When market expires (for time decay). If not provided, extracted from market.
    market : dict, optional
        Full market dict to auto-extract expiry_time if not provided.

    Returns
    -------
    {
        "dollars"    : float,   recommended stake
        "contracts"  : int,     contracts to buy
        "fraction"   : float,   full Kelly f before safety caps
        "rationale"  : str,
        "time_decay" : float,   v3: time decay multiplier applied (1.0 = no decay)
    }
    """
    # Use real-time bankroll if provided and sane
    effective_bankroll = live_bankroll if live_bankroll > 0 else bankroll
    if effective_bankroll <= 0:
        return {
            "dollars": 0, "contracts": 0, "fraction": 0.0,
            "rationale": "Bankroll is 0 — check BANKROLL env var",
            "time_decay": 1.0,
        }

    # v3: Calculate time decay
    if expiry_time is None and market is not None:
        expiry_time = parse_expiry_time(market)
    td_factor = time_decay_factor(expiry_time)
    
    if td_factor == 0.0:
        return {
            "dollars": 0, "contracts": 0, "fraction": 0.0,
            "rationale": "Too close to expiry (time_decay=0)",
            "time_decay": 0.0,
        }

    p     = max(0.01, min(0.99, prob))
    price = yes_price_cents / 100.0

    # Net odds: gain (1 - price) per dollar staked if YES resolves
    b = (1.0 - price) / price

    # Full Kelly fraction
    f_full = (p * (b + 1) - 1) / b

    if f_full <= 0:
        return {
            "dollars": 0, "contracts": 0, "fraction": f_full,
            "rationale": "Negative Kelly — no edge, skip",
            "time_decay": td_factor,
        }

    # Apply quarter-Kelly × drawdown factor × time decay
    effective_kelly = KELLY_FRACTION * max(0.1, min(1.0, drawdown_factor)) * td_factor
    f_safe = f_full * effective_kelly

    # Dollar amount
    raw_dollars = f_safe * effective_bankroll

    # Hard caps
    max_single  = min(effective_bankroll * MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD)
    max_sector  = max(0.0, effective_bankroll * MAX_SECTOR_EXPOSURE - sector_exposure)
    dollars     = min(raw_dollars, max_single, max_sector)

    if dollars < 1.0:
        return {
            "dollars": 0, "contracts": 0, "fraction": f_full,
            "rationale": "Sector exposure or single-trade cap reached",
            "time_decay": td_factor,
        }

    # Kalshi: each contract costs price dollars, pays $1 if YES
    contracts = max(1, int(dollars / price))

    # v3: Include time decay in logging
    td_note = f" td={td_factor:.2f}" if td_factor < 1.0 else ""
    
    if drawdown_factor < 1.0 or td_factor < 1.0:
        logger.info(
            "[%s] Kelly f=%.3f dd=%.2f%s → adj f=%.3f → $%.2f → %d contracts @ %dc",
            sector or "?", f_full, drawdown_factor, td_note, f_safe, dollars, contracts, yes_price_cents,
        )
    else:
        logger.info(
            "[%s] Kelly f=%.3f → ¼K f=%.3f → $%.2f → %d contracts @ %dc",
            sector or "?", f_full, f_safe, dollars, contracts, yes_price_cents,
        )

    return {
        "dollars":   round(dollars, 2),
        "contracts": contracts,
        "fraction":  round(f_full, 4),
        "rationale": (
            f"Full Kelly={f_full:.2%}, adj Kelly=${dollars:.2f}, "
            f"{contracts} contracts @ {yes_price_cents}¢ "
            f"(drawdown={drawdown_factor:.2f}, time_decay={td_factor:.2f})"
        ),
        "time_decay": td_factor,
    }


def no_kelly_stake(
    prob:             float,
    yes_price_cents:  int,
    bankroll:         float,
    sector:           str   = "",
    sector_exposure:  float = 0.0,
    live_bankroll:    float = 0.0,
    drawdown_factor:  float = 1.0,
    expiry_time:      Optional[datetime] = None,  # v3: For time decay
    market:           Optional[dict] = None,      # v3: Auto-extract expiry
) -> dict:
    """
    Kelly sizing for a NO position.
    A NO at yes_price p is equivalent to a YES at (1-p).
    All parameters identical to kelly_stake().
    """
    no_price_cents = 100 - yes_price_cents
    no_prob        = 1.0 - prob
    return kelly_stake(
        prob            = no_prob,
        yes_price_cents = no_price_cents,
        bankroll        = bankroll,
        sector          = sector,
        sector_exposure = sector_exposure,
        live_bankroll   = live_bankroll,
        drawdown_factor = drawdown_factor,
        expiry_time     = expiry_time,
        market          = market,
    )