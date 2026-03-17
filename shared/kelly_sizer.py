"""
shared/kelly_sizer.py
Quarter-Kelly position sizing with correlation guard and bankroll protection.
"""

import logging
from config.settings import (
    KELLY_FRACTION, MAX_SINGLE_TRADE_PCT, MAX_SECTOR_EXPOSURE, BANKROLL
)

logger = logging.getLogger(__name__)


def kelly_stake(
    prob:            float,   # our estimated P(YES)
    yes_price_cents: int,     # market's YES price in cents (1–99)
    bankroll:        float,   # current available bankroll in USD
    sector:          str = "",
    sector_exposure: float = 0.0,  # currently at risk in this sector (USD)
) -> dict:
    """
    Compute the recommended dollar stake for a YES position.

    Returns
    -------
    {
        "dollars"   : float,  # recommended stake
        "contracts" : int,    # number of contracts to buy
        "fraction"  : float,  # Kelly f before safety caps
        "rationale" : str,
    }
    """
    p = max(0.01, min(0.99, prob))
    price = yes_price_cents / 100.0

    # Payout odds: if YES resolves, you gain (1 - price) per dollar staked
    # If NO resolves, you lose your stake
    b = (1.0 - price) / price   # net odds (payout per unit risked)

    # Full Kelly fraction
    f_full = (p * (b + 1) - 1) / b
    if f_full <= 0:
        return {"dollars": 0, "contracts": 0, "fraction": f_full,
                "rationale": "Negative Kelly — no edge, skip"}

    # Apply quarter-Kelly safety cap
    f_safe = f_full * KELLY_FRACTION

    # Dollar amount
    raw_dollars = f_safe * bankroll

    # Hard caps
    max_single = bankroll * MAX_SINGLE_TRADE_PCT
    max_sector = max(0.0, bankroll * MAX_SECTOR_EXPOSURE - sector_exposure)
    dollars     = min(raw_dollars, max_single, max_sector)

    if dollars < 1.0:
        return {"dollars": 0, "contracts": 0, "fraction": f_full,
                "rationale": "Sector exposure or single-trade cap reached"}

    # Kalshi contracts: each contract costs yes_price cents, pays $1 if YES
    # dollars ÷ price_per_contract (in $)
    contracts = max(1, int(dollars / price))

    logger.info(
        "[%s] Kelly f=%.3f → ¼K f=%.3f → $%.2f → %d contracts @ %dc",
        sector or "?", f_full, f_safe, dollars, contracts, yes_price_cents
    )

    return {
        "dollars":   round(dollars, 2),
        "contracts": contracts,
        "fraction":  round(f_full, 4),
        "rationale": (
            f"Full Kelly={f_full:.2%}, ¼ Kelly=${dollars:.2f}, "
            f"{contracts} contracts @ {yes_price_cents}¢"
        ),
    }


def no_kelly_stake(
    prob:            float,
    yes_price_cents: int,
    bankroll:        float,
    sector:          str = "",
    sector_exposure: float = 0.0,
) -> dict:
    """
    Kelly sizing for a NO position.
    A NO at yes_price p is equivalent to a YES at (1-p).
    """
    no_price_cents = 100 - yes_price_cents
    no_prob        = 1.0 - prob
    return kelly_stake(no_prob, no_price_cents, bankroll, sector, sector_exposure)
