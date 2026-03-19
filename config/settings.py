"""
kalshi-flywheel / config/settings.py

Global configuration. All secrets come from .env — never hardcode.

CHANGES vs v1:
  - DEMO_MODE default flipped to False (opt IN to demo, not out of it)
  - BANKROLL default set to 0 — bot refuses to trade if env var is missing
  - MIN_EDGE_PCT lowered to 2% — 5% was filtering out real sharp edges on Kalshi
  - MIN_TRAINING_SAMPLES raised to 50 — 20 is too noisy for polynomial model
  - MAX_SINGLE_TRADE_USD hard cap added — 4% of $1M = $40k, moves the market
  - MAX_MARKET_EXPOSURE added — per-market cap for when bankroll scales
  - CONSENSUS_CONFIDENCE kept at 0.65 (lower to 0.60 after 200+ resolved trades)
  - KELLY_FRACTION kept at 0.25 (standard quarter-Kelly for professionals)

API TIERS
---------
Tier 1  Free    — get these day 1, zero cost
Tier 2  $9-50   — high ROI, add after first profitable week
Tier 3  $29-150 — premium alpha, add after model is calibrated
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Kalshi ────────────────────────────────────────────────────────────────────
KALSHI_API_BASE    = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_KEY_ID      = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")

# ── Demo mode ─────────────────────────────────────────────────────────────────
# Default is FALSE — you must explicitly set DEMO_MODE=true in .env to simulate.
# This prevents silent demo runs if the env var is missing on Railway.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ── Bankroll & risk ───────────────────────────────────────────────────────────
# BANKROLL defaults to 0 — bot will refuse to place trades if env var is unset.
# This prevents sizing real trades against a phantom $10k if Railway drops the var.
BANKROLL = float(os.getenv("BANKROLL") or 0)

# Kelly fraction: 0.25 = quarter-Kelly, standard for professional bettors.
# Do NOT raise above 0.25 until you have 500+ resolved trades with verified edge.
KELLY_FRACTION = 0.25

# Minimum model edge required to place a trade.
# 5% was too aggressive — most sharp Kalshi edges are 2–4% on liquid markets.
# Lower to 2% to capture real opportunities without sacrificing much precision.
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.02"))

# Consensus thresholds (used when multiple sector signals must agree)
CONSENSUS_EDGE_PCT   = 0.02   # keep in sync with MIN_EDGE_PCT
CONSENSUS_CONFIDENCE = 0.65   # lower to 0.60 after 200+ resolved trades

# Per-trade size cap as % of bankroll.
# At $10k this is $400/trade — fine.
# At $1M this is $40k/trade — will move Kalshi's market and get you limited.
MAX_SINGLE_TRADE_PCT = 0.04

# Hard dollar cap per trade regardless of bankroll size.
# Prevents the % cap from scaling to market-moving sizes.
# Raise this deliberately as you verify edge, not automatically.
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "2000"))

# Maximum exposure in one sector as % of bankroll.
MAX_SECTOR_EXPOSURE  = 0.15

# Maximum exposure in one individual market as % of bankroll.
# Kalshi enforces position limits per market — this keeps you under them.
MAX_MARKET_EXPOSURE  = float(os.getenv("MAX_MARKET_EXPOSURE", "0.05"))

# ── Model ─────────────────────────────────────────────────────────────────────
POLY_DEGREE = 3              # degree 4+ overfits badly on small samples

# Beta prior: (2, 2) = mild uniform prior, good for cold start.
# Shift toward (3, 2) or (2, 3) once you have sector-level win rate history.
BETA_ALPHA_PRIOR = 2
BETA_BETA_PRIOR  = 2

# Minimum resolved trades before the model is trusted for live sizing.
# 20 was too low — polynomial regression needs at least 50 clean samples.
MIN_TRAINING_SAMPLES = 50

# Rolling window (days) for model calibration.
# 30 days is correct for slow sectors (economics, politics).
# TODO: make this per-sector (crypto/sports benefit from 14-day windows).
CALIBRATION_WINDOW = 30

# ── TIER 1: Free APIs ─────────────────────────────────────────────────────────
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY     = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── TIER 2: $9–$50/mo ────────────────────────────────────────────────────────
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")
ODDSPAPI_KEY      = os.getenv("ODDSPAPI_KEY", "")
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
MYSPORTSFEEDS_KEY = os.getenv("MYSPORTSFEEDS_KEY", "")
FMP_API_KEY       = os.getenv("FMP_API_KEY", "")

# ── TIER 3: $29–$150/mo ──────────────────────────────────────────────────────
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
TOMORROW_IO_KEY   = os.getenv("TOMORROW_IO_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
COINGLASS_KEY     = os.getenv("COINGLASS_KEY", "")
LUNARCRUSH_KEY    = os.getenv("LUNARCRUSH_KEY", "")

# ── Cross-venue arbitrage ─────────────────────────────────────────────────────
# 3% arb threshold is realistic after vig on both sides.
ARB_MIN_SPREAD_PCT   = 0.03
POLYMARKET_CLOB_URL  = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# ── Scheduling ────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 300   # 5 min — appropriate, not hammering the API
RETRAIN_HOUR      = 2     # 2 AM retrain — low market activity window

# ── Storage ───────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "flywheel.db")

# ── Sectors ───────────────────────────────────────────────────────────────────
SECTORS = ["economics", "crypto", "politics", "weather", "tech", "sports"]

# ── Bet sizing helper ─────────────────────────────────────────────────────────
def max_bet_size() -> float:
    """
    Returns the maximum allowable bet in dollars, respecting both the
    percentage cap and the hard dollar cap. Use this everywhere a trade
    is sized — never reference MAX_SINGLE_TRADE_PCT directly.

    Example:
        bankroll=$10k  → min(400, 2000) = $400
        bankroll=$100k → min(4000, 2000) = $2000   ← cap kicks in
        bankroll=$1M   → min(40000, 2000) = $2000  ← cap protects the market
    """
    if BANKROLL <= 0:
        raise ValueError(
            "BANKROLL is 0 or unset — set BANKROLL in your Railway environment variables."
        )
    pct_cap    = BANKROLL * MAX_SINGLE_TRADE_PCT
    dollar_cap = MAX_SINGLE_TRADE_USD
    return min(pct_cap, dollar_cap)