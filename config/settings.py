"""
kalshi-flywheel / config/settings.py
Global configuration. All secrets come from .env — never hardcode.

API TIERS
---------
Tier 1  Free — get these day 1, zero cost
Tier 2  $9-50/mo — high ROI, add after first profitable week
Tier 3  $29-150/mo — premium alpha, add after model is calibrated
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Kalshi
KALSHI_API_BASE      = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_KEY_ID        = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY   = os.getenv("KALSHI_PRIVATE_KEY", "")

# Bankroll & risk
BANKROLL             = float(os.getenv("BANKROLL", "10000"))
KELLY_FRACTION       = 0.25
MIN_EDGE_PCT         = 0.05
CONSENSUS_EDGE_PCT   = 0.05
CONSENSUS_CONFIDENCE = 0.65
MAX_SINGLE_TRADE_PCT = 0.04
MAX_SECTOR_EXPOSURE  = 0.15

# Model
POLY_DEGREE          = 3
BETA_ALPHA_PRIOR     = 2
BETA_BETA_PRIOR      = 2
MIN_TRAINING_SAMPLES = 20
CALIBRATION_WINDOW   = 30

# TIER 1: Free APIs
FRED_API_KEY         = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY          = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY      = os.getenv("FINNHUB_API_KEY", "")

# TIER 2: $9-$50/mo
ODDS_API_KEY         = os.getenv("ODDS_API_KEY", "")
ODDSPAPI_KEY         = os.getenv("ODDSPAPI_KEY", "")
POLYGON_API_KEY      = os.getenv("POLYGON_API_KEY", "")
MYSPORTSFEEDS_KEY    = os.getenv("MYSPORTSFEEDS_KEY", "")
FMP_API_KEY          = os.getenv("FMP_API_KEY", "")

# TIER 3: $29-$150/mo
GLASSNODE_API_KEY    = os.getenv("GLASSNODE_API_KEY", "")
TOMORROW_IO_KEY      = os.getenv("TOMORROW_IO_KEY", "")
ALPHA_VANTAGE_KEY    = os.getenv("ALPHA_VANTAGE_KEY", "")
COINGLASS_KEY        = os.getenv("COINGLASS_KEY", "")
LUNARCRUSH_KEY       = os.getenv("LUNARCRUSH_KEY", "")

# Cross-venue arbitrage
ARB_MIN_SPREAD_PCT   = 0.03
POLYMARKET_CLOB_URL  = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# Scheduling
SCAN_INTERVAL_SEC    = 300
RETRAIN_HOUR         = 2

# Storage
DB_PATH              = os.getenv("DB_PATH", "flywheel.db")

# Demo mode
DEMO_MODE            = os.getenv("DEMO_MODE", "true").lower() == "true"

# Sectors
SECTORS = ["economics", "crypto", "politics", "weather", "tech", "sports"]
