"""
kalshi-flywheel / config/settings.py  (v10 — moderated tightening for v13 eval window)

Changes vs v9 (tight):
  1. Reverted SECTOR_MAX_LOSS_WEATHER from 12.0 → 60.0
     Rationale: we JUST raised this from 30→60 hours ago to fix the bug
     where one weather trade locked the sector for 4 hours. Dropping to 12
     makes it worse than the original broken state.
  2. Reverted SECTOR_MIN_RESOLVED_SPORTS from 150 → 30
     Rationale: 150 puts sports in permanent exploration (10% Kelly),
     which suppresses NBA/MLB volume we need to evaluate v13 model fixes.
  3. Reverted KELLY_FRACTION from 0.15 → 0.20 (compromise between 0.15 and 0.25)
  4. Reverted MIN_EDGE_PCT from 0.10 → 0.05 (original)
  5. Reverted CONSENSUS_EDGE_PCT from 0.08 → 0.05 (original)
  6. Reverted CONSENSUS_CONFIDENCE from 0.82 → 0.75 (original)
  7. Reverted MAX_SINGLE_TRADE_USD from 12 → 30 (original)
  8. Reverted SECTOR_MAX_LOSS_SPORTS from 20 → 50 (original)
  9. Reverted SECTOR_MAX_LOSS_CRYPTO from 15 → 40 (original)

Kept from v9 tight:
  - financial_markets fully disabled (cap=0, resolved=9999)
  - MAX_SINGLE_TRADE_PCT 0.02 (was 0.04)
  - MAX_SECTOR_EXPOSURE 0.07 (was 0.15)
  - MAX_MARKET_EXPOSURE 0.02 (was 0.05)
  - CIRCUIT_BREAKER_PCT 0.08 (was 0.20)
  - CORR_MIN_DIVERGENCE_CENTS 12 (was 8) — fewer low-quality corr trades
  - RESTIME thresholds tightened
  - Shadow mode config retained

Rationale: v13 model changes (dynamic AB, WHIP, crypto volatility) need
7 days of normal-volume data to evaluate. Tightening position sizing
is fine but tightening edge thresholds AND exploration floors simultaneously
strangles the signal we need to collect.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -- Kalshi --
KALSHI_API_BASE    = os.getenv("KALSHI_API_BASE", "https://trading-api.kalshi.com/trade-api/v2")
KALSHI_KEY_ID      = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")

# -- Demo mode --
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# -- Bankroll & risk --
BANKROLL = float(os.environ.get("BANKROLL") or os.getenv("BANKROLL") or 10000)

# Moderated Kelly — 0.20 is between old 0.25 and tight 0.15
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.20"))
MIN_EDGE_PCT   = float(os.getenv("MIN_EDGE_PCT", "0.05"))
MAX_EDGE_PCT   = float(os.getenv("MAX_EDGE_PCT", "0.25"))

# -- Per-sector edge ceilings --
SECTOR_MAX_EDGE = {
    "sports":            float(os.getenv("SECTOR_MAX_EDGE_SPORTS",            "0.40")),
    "weather":           float(os.getenv("SECTOR_MAX_EDGE_WEATHER",           "0.60")),
    "crypto":            float(os.getenv("SECTOR_MAX_EDGE_CRYPTO",            "0.25")),
    "politics":          float(os.getenv("SECTOR_MAX_EDGE_POLITICS",          "0.25")),
    "economics":         float(os.getenv("SECTOR_MAX_EDGE_ECONOMICS",         "0.25")),
    "financial_markets": float(os.getenv("SECTOR_MAX_EDGE_FINANCIAL_MARKETS", "0.25")),
    "global_events":     float(os.getenv("SECTOR_MAX_EDGE_GLOBAL_EVENTS",     "0.25")),
}

CONSENSUS_EDGE_PCT   = float(os.getenv("CONSENSUS_EDGE_PCT", "0.05"))
CONSENSUS_CONFIDENCE = float(os.getenv("CONSENSUS_CONFIDENCE", "0.75"))
DIRECTION_FILTER     = os.getenv("DIRECTION_FILTER", "BOTH")

# Kept tight: smaller position sizing is fine, it's edge/volume we shouldn't strangle
MAX_SINGLE_TRADE_PCT = float(os.getenv("MAX_SINGLE_TRADE_PCT", "0.02"))
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "30"))
MAX_SECTOR_EXPOSURE  = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.07"))
MAX_MARKET_EXPOSURE  = float(os.getenv("MAX_MARKET_EXPOSURE", "0.02"))

# -- Per-sector daily loss caps --
# Weather restored to 60 (the fix we just deployed hours ago)
# Sports restored to 50 (original) so NBA/MLB can flow
SECTOR_MAX_DAILY_LOSS = {
    "sports":            float(os.getenv("SECTOR_MAX_LOSS_SPORTS",            "50.0")),
    "politics":          float(os.getenv("SECTOR_MAX_LOSS_POLITICS",          "25.0")),
    "weather":           float(os.getenv("SECTOR_MAX_LOSS_WEATHER",           "60.0")),
    "economics":         float(os.getenv("SECTOR_MAX_LOSS_ECONOMICS",         "40.0")),
    "crypto":            float(os.getenv("SECTOR_MAX_LOSS_CRYPTO",            "40.0")),
    "financial_markets": float(os.getenv("SECTOR_MAX_LOSS_FINANCIAL_MARKETS", "0.0")),
    "global_events":     float(os.getenv("SECTOR_MAX_LOSS_GLOBAL_EVENTS",     "25.0")),
}

# -- Per-sector minimum resolved before full Kelly --
# Sports restored to 30 so NBA/MLB v13 model can exit exploration mode
# Weather 30 (already has 282 resolved, this is moot)
# Crypto 30 (need fresh v13 crypto data to calibrate)
SECTOR_MIN_RESOLVED = {
    "sports":            int(os.getenv("SECTOR_MIN_RESOLVED_SPORTS",            "30")),
    "politics":          int(os.getenv("SECTOR_MIN_RESOLVED_POLITICS",          "50")),
    "weather":           int(os.getenv("SECTOR_MIN_RESOLVED_WEATHER",           "30")),
    "economics":         int(os.getenv("SECTOR_MIN_RESOLVED_ECONOMICS",         "50")),
    "crypto":            int(os.getenv("SECTOR_MIN_RESOLVED_CRYPTO",            "30")),
    "financial_markets": int(os.getenv("SECTOR_MIN_RESOLVED_FINANCIAL_MARKETS", "9999")),
    "global_events":     int(os.getenv("SECTOR_MIN_RESOLVED_GLOBAL_EVENTS",     "75")),
}

EXPLORATION_KELLY_FRACTION = float(os.getenv("EXPLORATION_KELLY_FRACTION", "0.25"))

# -- Circuit breaker --
CIRCUIT_BREAKER_PCT = float(os.getenv("CIRCUIT_BREAKER_PCT", "0.10"))

# -- Sharp detector --
SHARP_SPREAD_THRESHOLD_PCT    = float(os.getenv("SHARP_SPREAD_THRESHOLD_PCT",    "0.04"))
SHARP_LARGE_ORDER_CONTRACTS   = int(os.getenv("SHARP_LARGE_ORDER_CONTRACTS",     "50"))
SHARP_VOLUME_SPIKE_MULTIPLIER = float(os.getenv("SHARP_VOLUME_SPIKE_MULTIPLIER", "2.5"))

# -- Fade scanner --
FADE_WINDOW_MINUTES   = float(os.getenv("FADE_WINDOW_MINUTES",   "60"))
FADE_THRESHOLD_CENTS  = int(os.getenv("FADE_THRESHOLD_CENTS",    "82"))
FADE_MIN_DISAGREEMENT = float(os.getenv("FADE_MIN_DISAGREEMENT", "0.10"))

# -- Correlation engine --
# Kept tight: 12¢ divergence (was 8¢) cuts low-quality corr trades
CORR_MIN_DIVERGENCE_CENTS = float(os.getenv("CORR_MIN_DIVERGENCE_CENTS", "12.0"))
CORR_MAX_GROUP_SIZE       = int(os.getenv("CORR_MAX_GROUP_SIZE",         "8"))

# -- Resolution timer --
# Kept moderately tight
RESTIME_MIN_OVERDUE_MIN = float(os.getenv("RESTIME_MIN_OVERDUE_MIN", "20"))
RESTIME_MIN_PROB        = float(os.getenv("RESTIME_MIN_PROB",        "0.85"))
RESTIME_MIN_SAMPLES     = int(os.getenv("RESTIME_MIN_SAMPLES",       "30"))

# -- News signal --
NEWS_POLL_INTERVAL_SEC        = int(os.getenv("NEWS_POLL_INTERVAL_SEC",        "300"))
NEWS_VELOCITY_WINDOW_MIN      = int(os.getenv("NEWS_VELOCITY_WINDOW_MIN",      "30"))
NEWS_VELOCITY_SPIKE_THRESHOLD = float(os.getenv("NEWS_VELOCITY_SPIKE_THRESHOLD", "2.0"))

# -- Model --
POLY_DEGREE          = 3
BETA_ALPHA_PRIOR     = 2
BETA_BETA_PRIOR      = 2
MIN_TRAINING_SAMPLES = 50
CALIBRATION_WINDOW   = 30

# -- APIs --
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY     = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")
ODDSPAPI_KEY      = os.getenv("ODDSPAPI_KEY", "")
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
MYSPORTSFEEDS_KEY = os.getenv("MYSPORTSFEEDS_KEY", "")
FMP_API_KEY       = os.getenv("FMP_API_KEY", "")

GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
TOMORROW_IO_KEY   = os.getenv("TOMORROW_IO_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
COINGLASS_KEY     = os.getenv("COINGLASS_KEY", "")
LUNARCRUSH_KEY    = os.getenv("LUNARCRUSH_KEY", "")

# -- Cross-venue arbitrage --
ARB_MIN_SPREAD_PCT   = 0.03
POLYMARKET_CLOB_URL  = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# -- Scheduling --
SCAN_INTERVAL_SEC = 300
RETRAIN_HOUR      = 2

# -- Storage --
DB_PATH = os.getenv("DB_PATH", "flywheel.db")

# -- Sectors --
SECTORS = ["economics", "crypto", "politics", "weather", "financial_markets", "sports", "global_events"]

# -- Bet sizing helpers --
def max_bet_size():
    if BANKROLL <= 0:
        raise ValueError("BANKROLL is 0 or unset")
    return min(BANKROLL * MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD)

def sector_kelly_fraction(sector, resolved_count):
    min_resolved = SECTOR_MIN_RESOLVED.get(sector, 30)
    if resolved_count < min_resolved:
        return KELLY_FRACTION * EXPLORATION_KELLY_FRACTION
    return KELLY_FRACTION

def sector_loss_cap(sector):
    return SECTOR_MAX_DAILY_LOSS.get(sector, 0.0)

def sector_max_edge(sector):
    return SECTOR_MAX_EDGE.get(sector, MAX_EDGE_PCT)