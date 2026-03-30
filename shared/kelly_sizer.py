"""
shared/kelly_sizer.py  (v2 — real-time bankroll + drawdown-adjusted fraction)

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
from config.settings import (
    KELLY_FRACTION, MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD,
    MAX_SECTOR_EXPOSURE, BANKROLL,
)

logger = logging.getLogger(__name__)


def kelly_stake(
    prob:             float,
    yes_price_cents:  int,
    bankroll:         float,
    sector:           str   = "",
    sector_exposure:  float = 0.0,
    live_bankroll:    float = 0.0,     # NEW: real-time balance from Kalshi API
    drawdown_factor:  float = 1.0,     # NEW: scale down during losing streaks
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

    Returns
    -------
    {
        "dollars"   : float,   recommended stake
        "contracts" : int,     contracts to buy
        "fraction"  : float,   full Kelly f before safety caps
        "rationale" : str,
    }
    """
    # Use real-time bankroll if provided and sane
    effective_bankroll = live_bankroll if live_bankroll > 0 else bankroll
    if effective_bankroll <= 0:
        return {
            "dollars": 0, "contracts": 0, "fraction": 0.0,
            "rationale": "Bankroll is 0 — check BANKROLL env var",
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
        }

    # Apply quarter-Kelly × drawdown factor
    effective_kelly = KELLY_FRACTION * max(0.1, min(1.0, drawdown_factor))
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
        }

    # Kalshi: each contract costs price dollars, pays $1 if YES
    contracts = max(1, int(dollars / price))

    if drawdown_factor < 1.0:
        logger.info(
            "[%s] Kelly f=%.3f drawdown=%.2f → adj f=%.3f → $%.2f → %d contracts @ %dc",
            sector or "?", f_full, drawdown_factor, f_safe, dollars, contracts, yes_price_cents,
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
            f"(drawdown_factor={drawdown_factor:.2f})"
        ),
    }


def no_kelly_stake(
    prob:             float,
    yes_price_cents:  int,
    bankroll:         float,
    sector:           str   = "",
    sector_exposure:  float = 0.0,
    live_bankroll:    float = 0.0,
    drawdown_factor:  float = 1.0,
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
    )