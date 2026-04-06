"""
kalshi-flywheel / config/settings.py  (v4 — fix sector_kelly_fraction docstring)

Changes vs v3:
  1. sector_kelly_fraction() docstring corrected. The previous docstring
     showed a usage example in kelly_sizer.py that was never implemented —
     the function is used in orchestrator._execute_trade() as a POST-SIZING
     multiplier, not as an input to the Kelly formula itself.

  2. Clarified that sector_kelly_fraction() returns the TARGET effective
     Kelly fraction (already including KELLY_FRACTION). Callers should
     apply it as a multiplier AFTER kelly_stake() computes the base stake,
     using: exploration_scale = kf / KELLY_FRACTION.

  3. EXPLORATION_KELLY_FRACTION comment updated to match actual behavior:
     exploration trades size at 25% of full Kelly (not 6.25%). The previous
     comment said "0.25 * 0.25 = 6.25%" which described the (now-fixed)
     double-penalty bug, not the intended behavior.

API TIERS
---------
Tier 1  Free    — get these day 1, zero cost
Tier 2  $9–$50  — high ROI, add after first profitable week
Tier 3  $29–$150— premium alpha, add after model is calibrated
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Kalshi ─────────────────────────────────────────────────────────────────────
KALSHI_API_BASE    = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_KEY_ID      = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")

# ── Demo mode ──────────────────────────────────────────────────────────────────
# Default is TRUE — you must explicitly set DEMO_MODE=false in Railway to go live.
# This prevents accidental live trading if the env var is missing.
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# ── Bankroll & risk ────────────────────────────────────────────────────────────
# BANKROLL defaults to 0 — bot will refuse to place trades if env var is unset.
# This prevents sizing real trades against a phantom $10k if Railway drops the var.
BANKROLL = float(os.getenv("BANKROLL") or 0)

# Kelly fraction: 0.25 = quarter-Kelly, standard for professional bettors.
# Do NOT raise above 0.25 until you have 500+ resolved trades with verified edge.
KELLY_FRACTION = 0.25

# Minimum model edge required to place a trade.
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.02"))

# Consensus thresholds (used when multiple sector signals must agree)
CONSENSUS_EDGE_PCT   = 0.02
CONSENSUS_CONFIDENCE = 0.65

# Per-trade size cap as % of bankroll.
MAX_SINGLE_TRADE_PCT = 0.04

# Hard dollar cap per trade regardless of bankroll size.
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "150"))

# Maximum exposure in one sector as % of bankroll.
MAX_SECTOR_EXPOSURE = 0.15

# Maximum exposure in one individual market as % of bankroll.
MAX_MARKET_EXPOSURE = float(os.getenv("MAX_MARKET_EXPOSURE", "0.05"))

# ── Per-sector daily loss caps ─────────────────────────────────────────────────
# Hard stop per sector per day in dollars.
# Once a sector hits its cap, no new trades fire in that sector until midnight.
#
# Calibration guide:
#   - Sports: highest cap — most data, most resolved, proven edge
#   - Politics: low cap — long-dated markets, slow feedback loop
#   - Weather: low cap — NOAA integration is new, model still learning
#   - Economics/Crypto/Tech: medium cap — markets haven't resolved yet,
#     models are unvalidated. Raise these after 30+ resolved trades each.
#
# Set any sector to 0.0 to completely disable trading in that sector.
# These are intentionally conservative — raise them as Brier scores improve.
SECTOR_MAX_DAILY_LOSS: dict[str, float] = {
    "sports":    float(os.getenv("SECTOR_MAX_LOSS_SPORTS",    "200.0")),
    "politics":  float(os.getenv("SECTOR_MAX_LOSS_POLITICS",   "25.0")),
    "weather":   float(os.getenv("SECTOR_MAX_LOSS_WEATHER",    "30.0")),
    "economics": float(os.getenv("SECTOR_MAX_LOSS_ECONOMICS",  "40.0")),
    "crypto":    float(os.getenv("SECTOR_MAX_LOSS_CRYPTO",     "40.0")),
    "tech":      float(os.getenv("SECTOR_MAX_LOSS_TECH",       "40.0")),
}

# ── Per-sector minimum resolved trades before full Kelly sizing ────────────────
# Below this threshold the bot trades at EXPLORATION_KELLY_FRACTION of normal
# Kelly (25% of normal = cautious exploration while the model learns).
#
# Sports is set low (20) because it already has existing resolved markets.
# All others are set to 30 — raise to 50 after the model stabilizes.
SECTOR_MIN_RESOLVED: dict[str, int] = {
    "sports":    int(os.getenv("SECTOR_MIN_RESOLVED_SPORTS",    "20")),
    "politics":  int(os.getenv("SECTOR_MIN_RESOLVED_POLITICS",  "30")),
    "weather":   int(os.getenv("SECTOR_MIN_RESOLVED_WEATHER",   "30")),
    "economics": int(os.getenv("SECTOR_MIN_RESOLVED_ECONOMICS", "30")),
    "crypto":    int(os.getenv("SECTOR_MIN_RESOLVED_CRYPTO",    "30")),
    "tech":      int(os.getenv("SECTOR_MIN_RESOLVED_TECH",      "30")),
}

# Exploration Kelly multiplier — applied POST-sizing when resolved count is
# below SECTOR_MIN_RESOLVED. Trades at 25% of normal Kelly during exploration.
# Example: normal Kelly produces $40 stake → exploration stake = $40 * 0.25 = $10.
# Do NOT set below 0.1 — stakes will fall below the $1 floor and size to zero.
EXPLORATION_KELLY_FRACTION = 0.25

# ── Circuit breaker ────────────────────────────────────────────────────────────
# Halt ALL trading if total daily loss exceeds this fraction of bankroll.
# 5% = $500 on a $10k bankroll.
CIRCUIT_BREAKER_PCT = float(os.getenv("CIRCUIT_BREAKER_PCT", "0.05"))

# ── Sharp detector ─────────────────────────────────────────────────────────────
SHARP_SPREAD_THRESHOLD_PCT    = float(os.getenv("SHARP_SPREAD_THRESHOLD_PCT",    "0.04"))
SHARP_LARGE_ORDER_CONTRACTS   = int(os.getenv("SHARP_LARGE_ORDER_CONTRACTS",     "50"))
SHARP_VOLUME_SPIKE_MULTIPLIER = float(os.getenv("SHARP_VOLUME_SPIKE_MULTIPLIER", "2.5"))

# ── Fade scanner ───────────────────────────────────────────────────────────────
FADE_WINDOW_MINUTES   = float(os.getenv("FADE_WINDOW_MINUTES",   "60"))
FADE_THRESHOLD_CENTS  = int(os.getenv("FADE_THRESHOLD_CENTS",    "82"))
FADE_MIN_DISAGREEMENT = float(os.getenv("FADE_MIN_DISAGREEMENT", "0.10"))

# ── Correlation engine ─────────────────────────────────────────────────────────
CORR_MIN_DIVERGENCE_CENTS = float(os.getenv("CORR_MIN_DIVERGENCE_CENTS", "8.0"))
CORR_MAX_GROUP_SIZE       = int(os.getenv("CORR_MAX_GROUP_SIZE",         "10"))

# ── Resolution timer ───────────────────────────────────────────────────────────
RESTIME_MIN_OVERDUE_MIN = float(os.getenv("RESTIME_MIN_OVERDUE_MIN", "15"))
RESTIME_MIN_PROB        = float(os.getenv("RESTIME_MIN_PROB",        "0.80"))
RESTIME_MIN_SAMPLES     = int(os.getenv("RESTIME_MIN_SAMPLES",       "20"))

# ── News signal ────────────────────────────────────────────────────────────────
NEWS_POLL_INTERVAL_SEC        = int(os.getenv("NEWS_POLL_INTERVAL_SEC",        "300"))
NEWS_VELOCITY_WINDOW_MIN      = int(os.getenv("NEWS_VELOCITY_WINDOW_MIN",      "30"))
NEWS_VELOCITY_SPIKE_THRESHOLD = float(os.getenv("NEWS_VELOCITY_SPIKE_THRESHOLD", "2.0"))

# ── Model ──────────────────────────────────────────────────────────────────────
POLY_DEGREE          = 3
BETA_ALPHA_PRIOR     = 2
BETA_BETA_PRIOR      = 2
MIN_TRAINING_SAMPLES = 50
CALIBRATION_WINDOW   = 30

# ── TIER 1: Free APIs ──────────────────────────────────────────────────────────
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY     = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── TIER 2: $9–$50/mo ─────────────────────────────────────────────────────────
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")
ODDSPAPI_KEY      = os.getenv("ODDSPAPI_KEY", "")
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
MYSPORTSFEEDS_KEY = os.getenv("MYSPORTSFEEDS_KEY", "")
FMP_API_KEY       = os.getenv("FMP_API_KEY", "")

# ── TIER 3: $29–$150/mo ───────────────────────────────────────────────────────
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
TOMORROW_IO_KEY   = os.getenv("TOMORROW_IO_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
COINGLASS_KEY     = os.getenv("COINGLASS_KEY", "")
LUNARCRUSH_KEY    = os.getenv("LUNARCRUSH_KEY", "")

# ── Cross-venue arbitrage ──────────────────────────────────────────────────────
ARB_MIN_SPREAD_PCT   = 0.03
POLYMARKET_CLOB_URL  = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# ── Scheduling ─────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 300
RETRAIN_HOUR      = 2

# ── Storage ────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "flywheel.db")

# ── Sectors ────────────────────────────────────────────────────────────────────
SECTORS = ["economics", "crypto", "politics", "weather", "tech", "sports"]

# ── Bet sizing helpers ─────────────────────────────────────────────────────────
def max_bet_size() -> float:
    """
    Returns the maximum allowable bet in dollars, respecting both the
    percentage cap and the hard dollar cap.
    """
    if BANKROLL <= 0:
        raise ValueError(
            "BANKROLL is 0 or unset — set BANKROLL in your Railway environment variables."
        )
    return min(BANKROLL * MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD)


def sector_kelly_fraction(sector: str, resolved_count: int) -> float:
    """
    Returns the TARGET effective Kelly fraction for a given sector.

    This value already encodes KELLY_FRACTION:
      - Exploration (resolved < SECTOR_MIN_RESOLVED):
            returns KELLY_FRACTION * EXPLORATION_KELLY_FRACTION
            e.g. 0.25 * 0.25 = 0.0625
      - Full Kelly (resolved >= SECTOR_MIN_RESOLVED):
            returns KELLY_FRACTION
            e.g. 0.25

    IMPORTANT — how to use this in orchestrator._execute_trade():
    ──────────────────────────────────────────────────────────────
    Do NOT pass kf into kelly_stake() — that function applies KELLY_FRACTION
    internally. Passing kf in would double-apply KELLY_FRACTION.

    Instead, call kelly_stake() with the raw drawdown_factor, then apply
    exploration scaling as a POST-SIZING multiplier:

        sizing = kelly_stake(..., drawdown_factor=drawdown_factor)

        if in_exploration and sizing["contracts"] > 0:
            exploration_scale = kf / KELLY_FRACTION   # e.g. 0.0625 / 0.25 = 0.25
            sizing["dollars"]   = sizing["dollars"] * exploration_scale
            sizing["contracts"] = max(1, int(sizing["dollars"] / price_frac))

    This ensures KELLY_FRACTION is applied exactly once (inside kelly_sizer),
    and exploration scaling is a clean 0.25x multiplier on the already-sized stake.
    """
    min_resolved = SECTOR_MIN_RESOLVED.get(sector, 30)
    if resolved_count < min_resolved:
        return KELLY_FRACTION * EXPLORATION_KELLY_FRACTION
    return KELLY_FRACTION


def sector_loss_cap(sector: str) -> float:
    """
    Returns the daily loss cap in dollars for a given sector.
    Returns 0.0 if the sector is unknown (disables trading).

    Usage in orchestrator._execute_trade():
        cap = sector_loss_cap(lead_sector)
        if sector_daily_loss > cap:
            return  # skip trade
    """
    return SECTOR_MAX_DAILY_LOSS.get(sector, 0.0)