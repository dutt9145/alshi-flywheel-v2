"""
kalshi-flywheel / config/settings.py  (v8 — sector edge ceilings + entertainment)

Changes vs v7:
  1. SECTOR_MAX_EDGE added — per-sector edge ceilings. Weather gets 60%
     (NOAA-backed), sports gets 40% (MLB Poisson model), others stay 25%.
     Previously the global 25% ceiling was rejecting 58%+ weather edges
     that were genuine NOAA signal, not traps.
  2. sector_max_edge() helper added for consensus_engine to call.
  3. "entertainment" added to SECTORS, SECTOR_MAX_DAILY_LOSS, SECTOR_MIN_RESOLVED.
  4. All other settings unchanged from v7.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Kalshi ─────────────────────────────────────────────────────────────────────
KALSHI_API_BASE    = os.getenv("KALSHI_API_BASE", "https://trading-api.kalshi.com/trade-api/v2")
KALSHI_KEY_ID      = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")

# ── Demo mode ──────────────────────────────────────────────────────────────────
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# ── Bankroll & risk ────────────────────────────────────────────────────────────
BANKROLL = float(os.environ.get("BANKROLL") or os.getenv("BANKROLL") or 10000)

# Kelly fraction: 0.25 = quarter-Kelly, standard for professional bettors.
KELLY_FRACTION = 0.25

# Minimum model edge required to place a trade.
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.05"))

# Maximum model edge — edges above this are traps, not opportunities.
# Calibration audit: 30%+ edge went 0/4 (-$600). Market was right.
MAX_EDGE_PCT = float(os.getenv("MAX_EDGE_PCT", "0.25"))

# ── Per-sector edge ceilings (v8) ─────────────────────────────────────────────
# Override the global MAX_EDGE_PCT for sectors where high edges are real signal.
# Weather: NOAA-backed predictions hit 30-60% edges with 96% confidence.
#          These are genuine edges — the market is wrong, not the model.
# Sports:  MLB Poisson model can find 25-40% edges on extreme thresholds
#          (e.g. ≥3 hits for a .200 hitter — model correctly says 2.7%).
# Uncalibrated sectors stay at the conservative global default.
SECTOR_MAX_EDGE: dict[str, float] = {
    "sports":        float(os.getenv("SECTOR_MAX_EDGE_SPORTS",        "0.40")),
    "weather":       float(os.getenv("SECTOR_MAX_EDGE_WEATHER",       "0.60")),
    "crypto":        float(os.getenv("SECTOR_MAX_EDGE_CRYPTO",        "0.25")),
    "politics":      float(os.getenv("SECTOR_MAX_EDGE_POLITICS",      "0.25")),
    "economics":     float(os.getenv("SECTOR_MAX_EDGE_ECONOMICS",     "0.25")),
    "tech":          float(os.getenv("SECTOR_MAX_EDGE_TECH",          "0.25")),
    "entertainment": float(os.getenv("SECTOR_MAX_EDGE_ENTERTAINMENT", "0.25")),
}

# Consensus thresholds
CONSENSUS_EDGE_PCT   = 0.05
CONSENSUS_CONFIDENCE = 0.75

# Direction filter: "NO" = only NO trades, "YES" = only YES, "BOTH" = no filter.
# Calibration audit: YES 2/9 (22% WR), NO 11/17 (65% WR).
DIRECTION_FILTER = os.getenv("DIRECTION_FILTER", "NO")

# Per-trade size cap as % of bankroll.
MAX_SINGLE_TRADE_PCT = 0.04

# Hard dollar cap per trade regardless of bankroll size.
MAX_SINGLE_TRADE_USD = float(os.getenv("MAX_SINGLE_TRADE_USD", "30"))

# Maximum exposure in one sector as % of bankroll.
MAX_SECTOR_EXPOSURE = 0.15

# Maximum exposure in one individual market as % of bankroll.
MAX_MARKET_EXPOSURE = float(os.getenv("MAX_MARKET_EXPOSURE", "0.05"))

# ── Per-sector daily loss caps ─────────────────────────────────────────────────
SECTOR_MAX_DAILY_LOSS: dict[str, float] = {
    "sports":        float(os.getenv("SECTOR_MAX_LOSS_SPORTS",        "50.0")),
    "politics":      float(os.getenv("SECTOR_MAX_LOSS_POLITICS",      "25.0")),
    "weather":       float(os.getenv("SECTOR_MAX_LOSS_WEATHER",       "30.0")),
    "economics":     float(os.getenv("SECTOR_MAX_LOSS_ECONOMICS",     "40.0")),
    "crypto":        float(os.getenv("SECTOR_MAX_LOSS_CRYPTO",        "40.0")),
    "tech":          float(os.getenv("SECTOR_MAX_LOSS_TECH",          "40.0")),
    "entertainment": float(os.getenv("SECTOR_MAX_LOSS_ENTERTAINMENT", "25.0")),
}

# ── Per-sector minimum resolved trades before full Kelly sizing ────────────────
SECTOR_MIN_RESOLVED: dict[str, int] = {
    "sports":        int(os.getenv("SECTOR_MIN_RESOLVED_SPORTS",        "20")),
    "politics":      int(os.getenv("SECTOR_MIN_RESOLVED_POLITICS",      "30")),
    "weather":       int(os.getenv("SECTOR_MIN_RESOLVED_WEATHER",       "30")),
    "economics":     int(os.getenv("SECTOR_MIN_RESOLVED_ECONOMICS",     "30")),
    "crypto":        int(os.getenv("SECTOR_MIN_RESOLVED_CRYPTO",        "30")),
    "tech":          int(os.getenv("SECTOR_MIN_RESOLVED_TECH",          "30")),
    "entertainment": int(os.getenv("SECTOR_MIN_RESOLVED_ENTERTAINMENT", "50")),
}

# Exploration Kelly multiplier — trades at 25% of normal Kelly during exploration.
EXPLORATION_KELLY_FRACTION = 0.25

# ── Circuit breaker ────────────────────────────────────────────────────────────
CIRCUIT_BREAKER_PCT = float(os.getenv("CIRCUIT_BREAKER_PCT", "0.20"))

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
SECTORS = ["economics", "crypto", "politics", "weather", "tech", "sports", "entertainment"]

# ── Bet sizing helpers ─────────────────────────────────────────────────────────
def max_bet_size() -> float:
    if BANKROLL <= 0:
        raise ValueError(
            "BANKROLL is 0 or unset — set BANKROLL in your Railway environment variables."
        )
    return min(BANKROLL * MAX_SINGLE_TRADE_PCT, MAX_SINGLE_TRADE_USD)


def sector_kelly_fraction(sector: str, resolved_count: int) -> float:
    min_resolved = SECTOR_MIN_RESOLVED.get(sector, 30)
    if resolved_count < min_resolved:
        return KELLY_FRACTION * EXPLORATION_KELLY_FRACTION
    return KELLY_FRACTION


def sector_loss_cap(sector: str) -> float:
    return SECTOR_MAX_DAILY_LOSS.get(sector, 0.0)


def sector_max_edge(sector: str) -> float:
    """Return the edge ceiling for a given sector.

    Weather/sports have higher ceilings because their models produce
    genuine high-edge predictions (NOAA temperature data, MLB Poisson).
    Uncalibrated sectors use the conservative global default.
    """
    return SECTOR_MAX_EDGE.get(sector, MAX_EDGE_PCT)