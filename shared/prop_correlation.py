"""
shared/prop_correlation.py

Prevents overexposure on correlated same-game player props.

Problem:
  Machado ≥1 hit, ≥2 hits, ≥1 TB, ≥2 TB are all from the same at-bats.
  If the bot bets all 4, it's 4x exposed to one player's performance.
  P(≥2 hits) is a strict subset of P(≥1 hit) — maximally correlated.

Solution:
  Group pending signals by (game, player). For groups with n > 1
  correlated bets, downscale the Kelly fraction by 1/sqrt(n).

  Why sqrt(n)? Two perfectly correlated bets should behave like ~1.4x
  the risk of one bet, not 2x. sqrt captures this partial correlation
  better than 1/n (too aggressive) or 1 (ignoring it).

Usage in orchestrator.py:
    from shared.prop_correlation import apply_correlation_downscale

    # After collecting all signals for this scan cycle:
    signals = apply_correlation_downscale(signals)
    # Each signal's .kelly_fraction is now adjusted
"""

import logging
import math
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _extract_game_player_key(ticker: str) -> str | None:
    """Extract a (game, player) key from an MLB player prop ticker.

    Ticker format: KXMLBHIT-26APR092140COLSD-SDFTATIS23-1
    Game+Player:   26APR092140COLSD-SDFTATIS23

    Returns None for non-MLB or unparseable tickers.
    """
    if not ticker.upper().startswith("KXMLB"):
        return None

    parts = ticker.split("-")
    if len(parts) < 3:
        return None

    # parts[1] = game identifier (date+time+teams)
    # parts[2] = player identifier (team+name+jersey)
    return f"{parts[1]}-{parts[2]}"


def apply_correlation_downscale(signals: list[Any]) -> list[Any]:
    """Group signals by (game, player) and downscale correlated bets.

    Each signal must have:
      .ticker (str) — the Kalshi ticker
      .kelly_fraction (float) — the raw Kelly fraction to downscale

    Returns the same list with kelly_fraction adjusted in-place.
    """
    # Group by game+player
    groups: dict[str, list] = defaultdict(list)
    ungrouped = []

    for sig in signals:
        ticker = getattr(sig, "ticker", "") or ""
        key = _extract_game_player_key(ticker)
        if key:
            groups[key].append(sig)
        else:
            ungrouped.append(sig)

    # Downscale correlated groups
    total_downscaled = 0
    for key, group in groups.items():
        n = len(group)
        if n <= 1:
            continue

        scale = 1.0 / math.sqrt(n)
        for sig in group:
            old = getattr(sig, "kelly_fraction", 1.0)
            sig.kelly_fraction = old * scale
            total_downscaled += 1

        # Extract player name for logging (last part before jersey number)
        player_part = key.split("-")[-1] if "-" in key else key
        logger.info(
            "[correlation] %s: %d correlated props → scale %.2f (1/√%d)",
            player_part[:15], n, scale, n,
        )

    if total_downscaled > 0:
        logger.info(
            "[correlation] Downscaled %d signals across %d correlated groups",
            total_downscaled,
            sum(1 for g in groups.values() if len(g) > 1),
        )

    return signals


def get_correlation_report(signals: list[Any]) -> dict:
    """Generate a report of correlated groups without modifying signals.

    Useful for dashboard/logging.
    """
    groups: dict[str, list] = defaultdict(list)
    for sig in signals:
        ticker = getattr(sig, "ticker", "") or ""
        key = _extract_game_player_key(ticker)
        if key:
            groups[key].append(sig)

    correlated = {k: v for k, v in groups.items() if len(v) > 1}

    return {
        "total_signals":      len(signals),
        "correlated_groups":  len(correlated),
        "correlated_signals": sum(len(v) for v in correlated.values()),
        "max_group_size":     max((len(v) for v in correlated.values()), default=0),
        "groups": {
            k: {
                "count": len(v),
                "scale": round(1.0 / math.sqrt(len(v)), 3),
                "tickers": [getattr(s, "ticker", "") for s in v],
            }
            for k, v in correlated.items()
        },
    }