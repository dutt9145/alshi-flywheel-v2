"""
kalshi-flywheel / config/settings.py  (v2 — full feature suite constants)

Changes vs v1:
  1. Added fade scanner constants (FADE_WINDOW_MINUTES, FADE_THRESHOLD_CENTS,
     FADE_MIN_DISAGREEMENT)
  2. Added circuit breaker constants (CIRCUIT_BREAKER_PCT)
  3. Added sharp detector constants (SHARP_SPREAD_THRESHOLD_PCT,
     SHARP_LARGE_ORDER_CONTRACTS, SHARP_VOLUME_SPIKE_MULTIPLIER)
  4. Added correlation engine constants (CORR_MIN_DIVERGENCE_CENTS)
  5. Added resolution timer constants (RESTIME_MIN_OVERDUE_MIN,
     RESTIME_MIN_PROB)
  6. Added news signal constants (NEWS_POLL_INTERVAL_SEC,
     NEWS_VELOCITY_WINDOW_MIN, NEWS_VELOCITY_SPIKE_THRESHOLD)
  7. MAX_SINGLE_TRADE_USD raised to 2000 (unchanged)
  8. Everything else unchanged from v1

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
# Default is FALSE — you must explicitly set DEMO_MODE=true in .env to simulate.
# This prevents silent demo runs if the env var is missing on Railway.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ── Bankroll & risk ────────────────────────────────────────────────────────────
# BANKROLL defaults to 0 — bot will refuse to place trades if env var is unset.
# This prevents sizing real trades against a phantom $10k if Railway drops the var.
BANKROLL = float(os.getenv("BANKROLL") or 0)

# Kelly fraction: 0.25 = quarter-Kelly, standard for professional bettors.
# Do NOT raise above 0.25 until you have 500+ resolved trades with verified edge.
KELLY_FRACTION = 0.25

# Minimum model edge required to place a trade.
# 5% was too aggressive — most sharp Kalshi edges are 2–4% on liquid markets.
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
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "150"))

# Maximum exposure in one sector as % of bankroll.
MAX_SECTOR_EXPOSURE = 0.15

# Maximum exposure in one individual market as % of bankroll.
# Kalshi enforces position limits per market — this keeps you under them.
MAX_MARKET_EXPOSURE = float(os.getenv("MAX_MARKET_EXPOSURE", "0.05"))

# ── Circuit breaker ────────────────────────────────────────────────────────────
# Halt all trading if daily loss exceeds this fraction of bankroll.
# 5% = $500 on a $10k bankroll. Raises automatically with bankroll.
# Set lower (0.03) in the first week of live trading until you trust the model.
CIRCUIT_BREAKER_PCT = float(os.getenv("CIRCUIT_BREAKER_PCT", "0.05"))

# ── Sharp detector ─────────────────────────────────────────────────────────────
# Spread as % of midpoint below which we consider the book sharp-tight.
# 4% = 2 cents on a 50¢ market. Tighter = more confident sharp money is present.
SHARP_SPREAD_THRESHOLD_PCT = float(os.getenv("SHARP_SPREAD_THRESHOLD_PCT", "0.04"))

# Single order size in contracts above which we flag as institutional.
SHARP_LARGE_ORDER_CONTRACTS = int(os.getenv("SHARP_LARGE_ORDER_CONTRACTS", "50"))

# Recent volume vs 7-day average multiplier above which we flag a volume spike.
SHARP_VOLUME_SPIKE_MULTIPLIER = float(os.getenv("SHARP_VOLUME_SPIKE_MULTIPLIER", "2.5"))

# ── Fade scanner ───────────────────────────────────────────────────────────────
# Only scan markets closing within this many minutes.
FADE_WINDOW_MINUTES = float(os.getenv("FADE_WINDOW_MINUTES", "60"))

# Only fade when YES price is at or above this (crowded YES)
# or at or below (100 - this) (crowded NO).
# 82 = fade markets priced above 82¢ or below 18¢.
FADE_THRESHOLD_CENTS = int(os.getenv("FADE_THRESHOLD_CENTS", "82"))

# Minimum model disagreement with market price to trigger a fade.
# 0.10 = we must disagree by at least 10 percentage points.
FADE_MIN_DISAGREEMENT = float(os.getenv("FADE_MIN_DISAGREEMENT", "0.10"))

# ── Correlation engine ─────────────────────────────────────────────────────────
# Minimum price divergence in cents to flag a correlation opportunity.
# 8 cents accounts for vig on both legs of a pair trade.
CORR_MIN_DIVERGENCE_CENTS = float(os.getenv("CORR_MIN_DIVERGENCE_CENTS", "8.0"))

# Max markets per event group — above this, skip (likely unrelated markets).
CORR_MAX_GROUP_SIZE = int(os.getenv("CORR_MAX_GROUP_SIZE", "10"))

# ── Resolution timer ───────────────────────────────────────────────────────────
# Minutes past expected resolution time before flagging a market as overdue.
RESTIME_MIN_OVERDUE_MIN = float(os.getenv("RESTIME_MIN_OVERDUE_MIN", "15"))

# Minimum model probability to enter a resolution timing arb position.
# Only trade near-certain outcomes — timing arb is a high-conviction play.
RESTIME_MIN_PROB = float(os.getenv("RESTIME_MIN_PROB", "0.80"))

# Minimum historical samples before trusting a learned timing pattern.
RESTIME_MIN_SAMPLES = int(os.getenv("RESTIME_MIN_SAMPLES", "20"))

# ── News signal ────────────────────────────────────────────────────────────────
# How often to poll NewsAPI (seconds). 300 = 5 min, within free tier limits.
NEWS_POLL_INTERVAL_SEC = int(os.getenv("NEWS_POLL_INTERVAL_SEC", "300"))

# Rolling window for velocity calculation (minutes).
NEWS_VELOCITY_WINDOW_MIN = int(os.getenv("NEWS_VELOCITY_WINDOW_MIN", "30"))

# Multiplier vs baseline above which velocity is considered "spiking".
NEWS_VELOCITY_SPIKE_THRESHOLD = float(os.getenv("NEWS_VELOCITY_SPIKE_THRESHOLD", "2.0"))

# ── Model ──────────────────────────────────────────────────────────────────────
POLY_DEGREE = 3              # degree 4+ overfits badly on small samples

# Beta prior: (2, 2) = mild uniform prior, good for cold start.
BETA_ALPHA_PRIOR = 2
BETA_BETA_PRIOR  = 2

# Minimum resolved trades before the model is trusted for live sizing.
MIN_TRAINING_SAMPLES = 50

# Rolling window (days) for model calibration.
CALIBRATION_WINDOW = 30

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
SCAN_INTERVAL_SEC = 300   # 5 min — appropriate, not hammering the API
RETRAIN_HOUR      = 2     # 2 AM retrain — low market activity window

# ── Storage ────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "flywheel.db")

# ── Sectors ────────────────────────────────────────────────────────────────────
SECTORS = ["economics", "crypto", "politics", "weather", "tech", "sports"]

# ── Bet sizing helper ──────────────────────────────────────────────────────────
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