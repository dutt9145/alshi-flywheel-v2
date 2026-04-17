"""
bots/sector_bots.py  (v12.6 — LLMBot for qualitative sectors)

Changes vs v12.5:
  - all_bots(): Replaced EconomicsBot, PoliticsBot, FinancialMarketsBot, 
    GlobalEventsBot with a single LLMBot that uses Claude API + web search.
    These qualitative sectors benefit from LLM reasoning + current news context.
  - Quantitative bots (SportsBot, WeatherBot, CryptoBot) unchanged — they use
    specialized models with better calibration than LLM guessing.
  - Old bots kept in file for reference but not instantiated.

Changes vs v12.4:
  - WeatherBot: Lowered NOAA confidence threshold from 0.60 to 0.40.
    This captures more independent weather signals while correlation
    engine continues finding high-edge divergences (+29% avg edge).
    The 60% threshold was too strict — confidence formula is
    `0.4 + 0.05 * sample_count`, so needed 4+ samples to pass.

Changes vs v12.3:
  - EconomicsBot: RESURRECTED from disabled state. Now uses FRED API for
    real economic signals (CPI, unemployment, Fed funds, treasury yields).
    Handles rate decision markets, inflation markets, employment markets.
    Confidence scaling based on data freshness and market type.
  - FinancialMarketsBot: RESURRECTED from disabled state. Now uses Yahoo
    Finance for real market data (stock prices, VIX, momentum).
    Handles earnings markets, stock price threshold markets.
    VIX-adjusted confidence based on volatility regime.
  - New imports: fred_client, market_data_client

Changes vs v12.2:
  - WeatherBot: Added evaluate() override that uses NOAA directly when
    NOAA confidence >= 0.60, bypassing the poisoned Bayesian model.
    The Bayesian prior was pushed to ~3% by duplicate signals and
    misattributed outcomes, causing our_p=0.069 when NOAA said P(YES)=0.93.
    Now NOAA-confident markets get our_p=noaa_prior directly.
  - WeatherBot: When NOAA is unavailable or low confidence, returns None
    instead of falling back to poisoned Bayesian model.

Changes vs v12.1:
  - SportsBot: Added SPORT_CONFIDENCE_SCALE dict for per-sport calibration.
    Based on resolved Brier scores:
      NCAAMB 0.1159 → 1.00 (keep full)
      LALIGA 0.0343 → 1.00
      CS2    0.2141 → 0.85
      MLB    0.2267 → 0.75 (was 0.918 avg_conf, now ~0.69)
      NBA    0.2960 → 0.55 (worst major category)
      VALORANT 0.2495 → 0.65
      LOL    0.2521 → 0.65
      TENNIS 0.2607 → 0.60
  - SportsBot: Added SPORT_BLOCKLIST for leagues with catastrophic Brier:
      KXBELGIA  0.5662
      KXJBLEAG  0.4262
      KXACBGAM  0.6045
      KXMVENBA  0.3243
      KXMVECBC  4530 signals, 0 resolved, 0.32 conf — futures noise
  - SportsBot: Added _detect_sport_code() helper.
  - SportsBot: Modified evaluate() to apply scaling and blocklist.
  - Fixed KXNEXTNBACOACH-MIL26 misclassified as WEATHER (matched "milwaukee").
    Added "kxnext" to SPORTS_PREFIXES, WeatherBot._NON_WEATHER_BLOCKLIST,
    and SportsBot.KEYWORDS.
  - Fixed KXHIGHTDAL/KXHIGHTDC leaking to ECONOMICS (matched "dallas"/"dc").
    Added weather prefix guard to EconomicsBot.is_relevant().
  - Fixed KXLPGATOUR leaking to WEATHER and KXUECLGAME leaking to WEATHER.
    Added "kxlpga" and "kxuecl" to SPORTS_PREFIXES, WeatherBot._NON_WEATHER_BLOCKLIST,
    SportsBot.KEYWORDS, and SportsBot.LEAGUE_MAP.
  - MLB platoon splits: Updated logging to show batter hand vs pitcher hand
    for debugging platoon adjustments (actual model changes in mlb_hit_model.py).

Changes vs v12:
  - REMOVED "kxartiststream" from SPORTS_PREFIXES
    It was blocking GlobalEventsBot from claiming Spotify artist streaming
    markets. Now only in ENTERTAINMENT_PREFIXES where it belongs.
    Flow: all bots reject via _is_entertainment_market() → GlobalEventsBot
    claims via _is_entertainment_market() check → evaluate() returns None.
  - Added "kxartiststream" to GlobalEventsBot.KEYWORDS for redundancy.

Changes vs v11.8 (carried forward from v12):
  - Added NBA feature logging in _try_nba_player_prop() (mirrors MLB)
  - Both MLB and NBA now log features for Phase 2 logreg training

Changes vs v11.5:
  Bug fixes:
    - WeatherBot: added kxt20, kxhighinfl, kxfestival, kxthevoice to
      _NON_WEATHER_BLOCKLIST. KXHIGHINFLATION was matching "kxhigh" weather
      prefix. T20 cricket (SWE vs IDN) leaked via city keywords. The Voice
      and festival events matched city keywords ("nyc").
    - SPORTS_PREFIXES: added kxt20 (T20 cricket was leaking to weather/politics)
    - _is_entertainment_market: added kxthevoice, kxfestival
    - SportsBot: added kxt20 to KEYWORDS and LEAGUE_MAP

  Carried forward from v11.5:
    - CryptoBot: kxspot → kxspotstream (was matching KXSPOTIFY incorrectly)
    - SPORTS_PREFIXES: added kxufl (United Football League leaking to weather)
    - CryptoBot KEYWORDS: added kxshiba (SHIB token explicit)
    - _is_unmodelable_market(): returns True for MENTION markets and Survivor.
    - _is_entertainment_market(): routes to GlobalEventsBot.
    - EntertainmentBot → GlobalEventsBot (v11.7)

Changes vs v11.3:
  SportsBot:
    - Added evaluate() override that routes MLB player prop markets through
      the new hit-rate model (shared/mlb_hit_model.py) instead of the broken
      flat-prior BayesianPolyModel path.
    - All non-MLB markets and non-player-prop MLB markets (TOTAL, SPREAD,
      F5, futures) continue through the base class evaluate() unchanged.
    - New _try_mlb_player_prop() helper handles the full pipeline:
      parse ticker → lookup player → fetch stats → run model → return BotSignal.
    - Requires shared/kalshi_ticker_parser.py, shared/mlb_stats_fetcher.py,
      and shared/mlb_hit_model.py to be deployed alongside this file.

Changes vs v11.2 (unchanged from v11.3):
  _has_sports_prefix:
    - Added ~35 missing international sports prefixes that were causing
      hundreds of signals to bleed into wrong sectors:
        crypto:    ~400 of 529 signals were actually sports (Argentine soccer,
                   PGA golf, La Liga, cricket, ATP tennis, etc.)
        weather:   ~700 of 1,624 signals were sports (A-League via "melbourne"
                   city match, Bundesliga, Serie A, Diamond League, AHL, etc.)
                   Plus 429 Survivor TV show signals matched via city keywords.
                   6 "resolved" weather outcomes were all cricket matches with
                   0.92 confidence NOAA priors → Brier 0.4269 (catastrophic).
        economics: 6 of 16 signals were golf (KXOWGA)
    - After cleanup: all 185 resolved outcomes are sports, Brier 0.2535.
      All other sectors have 0 resolved outcomes.

  SportsBot:
    - Added all new prefixes to KEYWORDS and LEAGUE_MAP.

  PoliticsBot:
    - Removed "kxintl" from KEYWORDS (international sports, not politics).

  WeatherBot:
    - Added kxsurv, kxalea, kxdima, kxcba to _ESPORTS_BLOCKLIST (renamed
      to _NON_WEATHER_BLOCKLIST for clarity).
"""

import asyncio
import logging
import re
from typing import Optional

import numpy as np

from bots.base_bot import BaseBot, _market_prob
from shared.data_fetchers import (
    fetch_economics_features, fetch_crypto_features, fetch_politics_features,
    fetch_weather_features, fetch_tech_features, fetch_sports_features,
)
from shared.noaa_client import (
    fetch_weather_package,
    city_to_coords,
    extract_temp_signals,
    extract_precip_signals,
    extract_alert_signals,
    parse_market_date,
    parse_temp_threshold,
    classify_weather_market,
)

# ── v11.4: MLB player prop integration imports ──────────────────────────────
from shared.kalshi_ticker_parser import (
    parse_mlb_ticker,
    is_mlb_non_player_prop_market,
    MLB_TEAMS,
)
from shared.mlb_stats_fetcher import (
    lookup_player,
    fetch_batter_stats,
    fetch_batter_rolling,
    fetch_pitcher_stats,
    fetch_opposing_pitcher,
)
from shared.mlb_hit_model import predict_mlb_prop
from shared.mlb_logistic_model import (
    extract_features as mlb_extract_features,
    log_features as mlb_log_features,
)
from shared.consensus_engine import BotSignal
from config.settings import MIN_EDGE_PCT

# ── v11.6: NBA player prop integration imports ──────────────────────────────
from shared.nba_ticker_parser import (
    parse_nba_ticker,
    is_nba_non_player_prop_market,
    NBA_TEAMS as NBA_TEAMS_MAP,
)
from shared.nba_stats_fetcher import (
    lookup_player as nba_lookup_player,
    fetch_player_stats as nba_fetch_player_stats,
    fetch_player_rolling as nba_fetch_player_rolling,
)
from shared.nba_props_model import predict_nba_prop

# ── v12: NBA feature logging for logreg training ────────────────────────────
from shared.nba_logistic_model import (
    extract_features as nba_extract_features,
    log_features as nba_log_features,
)

# ── v12.4: Economic and market data clients ─────────────────────────────────
from shared.fred_client import get_fred_client, FredClient
from shared.market_data_client import get_market_data_client, MarketDataClient

logger = logging.getLogger(__name__)

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _extract_mlb_game_date(ticker: str) -> Optional[str]:
    """Extract game date from MLB ticker as YYYY-MM-DD.

    Ticker format: KXMLBHIT-26APR092140COLSD-...
    Date portion:  YY MON DD after the first hyphen.
    """
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})", parts[1])
    if not m:
        return None
    year = 2000 + int(m.group(1))
    month = _MONTH_MAP.get(m.group(2), 0)
    day = int(m.group(3))
    if month == 0:
        return None
    return f"{year}-{month:02d}-{day:02d}"


def _search_fields(market: dict, keywords: list[str]) -> bool:
    haystack = " ".join([
        market.get("event_ticker", ""),
        market.get("ticker", ""),
        market.get("title", ""),
        market.get("subtitle", ""),
    ]).lower()
    return any(kw.lower() in haystack for kw in keywords)


def _has_sports_prefix(market: dict) -> bool:
    """Gate that prevents non-sports bots from claiming sports markets.

    v11.3: Added ~35 missing prefixes. Every prefix here blocks weather/crypto/
    economics/politics/tech bots from claiming the market. If a prefix is here
    but NOT in any bot's KEYWORDS, the market is simply skipped (correct
    behavior for unmodelable markets like Survivor TV show).

    v12.1: REMOVED kxartiststream — it belongs in ENTERTAINMENT_PREFIXES only.
    Having it here blocked GlobalEventsBot from claiming Spotify streaming.
    """
    SPORTS_PREFIXES = (
        # ── KXMVE family ──────────────────────────────────────────────────
        "kxmve",

        # ── US major sports ───────────────────────────────────────────────
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
        "kxufc", "kxncaa", "kxcbb", "kxcfb",

        # ── Tennis ────────────────────────────────────────────────────────
        "kxatp", "kxwta", "kxtennis", "kxabagame",

        # ── Golf ──────────────────────────────────────────────────────────
        # v11.3: kxpga was missing → bled into crypto ("pga" no match but
        # ticker substring issues) and weather (golfer city names).
        # kxowga was missing → bled into economics.
        "kxgolf", "kxpga", "kxowga",
        "kxlpga",      # v12.2: LPGA Tour (was leaking to weather)

        # ── International soccer ──────────────────────────────────────────
        # v11.3: ALL of these were missing. Argentine soccer (kxarg) bled
        # into crypto (120 signals). La Liga, Bundesliga, Serie A, etc.
        # bled into weather via city name keyword matching.
        "kxepl", "kxsoccer",
        "kxarg",       # Argentine Primera División (kxargp, kxargl)
        "kxlali",      # La Liga
        "kxbund",      # Bundesliga
        "kxseri",      # Serie A
        "kxliga",      # Liga generic
        "kxligu",      # Ligue 1
        "kxbras",      # Brasileirão
        "kxswis",      # Swiss Super League
        "kxbelg",      # Belgian Pro League
        "kxecul",      # Ecuadorian Liga Pro
        "kxpslg",      # Pakistan/South Africa Premier League
        "kxsaud",      # Saudi Pro League
        "kxjlea",      # J-League (Japan)
        "kxuclt",      # UEFA Champions League
        "kxuecl",      # v12.2: UEFA Europa Conference League (was leaking to weather)
        "kxfifa",      # FIFA (World Cup)  
        "kxdensuperliga",  # Danish Superliga         

        # ── Cricket ───────────────────────────────────────────────────────
        # v11.3: kxcba was missing → cricket bled into weather AND crypto.
        # KXCBAGAME matched WeatherBot city keywords, got 0.92 NOAA priors
        # applied to cricket matches. All 6 "resolved" weather outcomes
        # were cricket → Brier 0.4269.
        "kxcba", "kxcricket",
        "kxt20",             # v11.8: T20 cricket (was leaking to weather/politics)

        # ── International basketball ──────────────────────────────────────
        # v11.3: all missing → bled into crypto and weather.
        "kxfiba",      # FIBA
        "kxacbg",      # ACB (Spanish basketball)
        "kxvtbg",      # VTB League (Russian basketball)
        "kxbalg",      # Baltic basketball league
        "kxbbse",      # BBL / basketball
        "kxnpbg",      # NBP / basketball

        # ── Hockey ────────────────────────────────────────────────────────
        # v11.3: kxahlg missing → AHL bled into weather.
        "kxahlg",      # AHL (American Hockey League)

        # ── Track & field ─────────────────────────────────────────────────
        # v11.3: kxdima missing → Diamond League bled into weather and economics.
        "kxdima",      # Diamond League

        # ── A-League (Australian soccer) ──────────────────────────────────
        # v11.3: kxalea missing → A-League bled into weather via "melbourne"
        # city keyword match (Melbourne City FC, Melbourne Victory).
        "kxalea",
        "kxafl",       # AFL (Australian Football League)

        # ── UFL (United Football League) — v11.5 ─────────────────────────
        # Was leaking into weather via city name keyword matching.
        "kxufl",

        # ── Combat sports ─────────────────────────────────────────────────
        "kxboxing", "kxwwe",

        # ── Racing ────────────────────────────────────────────────────────
        "kxf1", "kxnascar",

        # ── Rugby / Olympic / other ───────────────────────────────────────
        "kxrugby", "kxolympic", "kxthail", "kxsl",

        # ── Esports ───────────────────────────────────────────────────────
        "kxow", "kxvalorant", "kxlol", "kxleague",
        "kxrl", "kxrocketleague",
        "kxcsgo", "kxcs2", "kxdota", "kxintlf",
        "kxapex", "kxfort", "kxhalo", "kxsc2",
        "kxesport", "kxegypt", "kxvenf",
        "kxr6g",       # Rainbow Six — v11.3

        # ── International sports (generic) ────────────────────────────────
        # v11.3: kxintl was missing → bled into politics via "kxintl" keyword.
        "kxintl",

        # ── v11.7: Prefixes found by classifier_audit.py ────────────────
        # CONMEBOL (98 signals → weather/crypto)
        "kxconmebol",
        # Chilean soccer (22 signals → weather via "kxchll" chill keyword)
        "kxchll",
        # Euroleague basketball (6 signals → politics via "kxeu" prefix)
        "kxeuroleague",
        # Ekstraklasa Polish soccer (3 signals → politics/economics)
        "kxekstraklasa",
        # Swedish Hockey League
        "kxshl",
        # Uruguayan Primera División
        "kxurypd",
        # UCL spread (kxuclt was too specific — missed kxuclspread)
        "kxucl",
        # Turkish Super Lig
        "kxsuperlig",
        # Danish Superliga soccer
        "kxdensuperliga",
        # Egyptian Premier League
        "kxegypl",
        # ITF tennis (women + men)
        "kxitf",
        # IPL cricket (team totals, fours, sixes)
        "kxipl",
        # Scottish Premiership
        "kxscottishprem",
        # UEFA Europa League
        "kxuel",
        # Eredivisie (Dutch soccer)
        "kxeredivisie",
        # MotoGP racing
        "kxmotogp",
        # BBL (Big Bash League)
        "kxbbl",
        # KF tour
        "kxkf",
        # Allsvenskan (Swedish soccer)
        "kxallsvenskan",
        # Argentine Premier Division (more specific than kxarg)
        "kxargpremdiv",
        # NextAG
        "kxnextag",

        # v11.8d: New leagues found in 2026-04-12 logs
        # DEL (Deutsche Eishockey Liga — German hockey)
        "kxdel",
        # KHL (Kontinental Hockey League — Russian hockey)
        "kxkhl",
        # J.B.League (Japanese basketball)
        "kxjbleague",
        # KBL (Korean basketball)
        "kxkbl",
        # K League (Korean soccer)
        "kxkleague",

        # v12.2: "Next X" prediction markets (coaching hires, etc.)
        # KXNEXTNBACOACH-MIL26 was matching WeatherBot "milwaukee" keyword
        "kxnext",

        # ── Entertainment / unmodelable (block from all sector bots) ──────
        # v11.3: Survivor TV show (429 signals) matched WeatherBot via city
        # keywords in episode titles. No bot can model this, so blocking it
        # here causes it to be skipped entirely (correct behavior).
        "kxsurv",      # Survivor TV show mentions
        # v12.1: REMOVED kxartiststream — now ONLY in ENTERTAINMENT_PREFIXES
        # so GlobalEventsBot can claim it. Was causing deadlock where no bot
        # could claim KXARTISTSTREAMS markets.
    )
    et = market.get("event_ticker", "").lower()
    tk = market.get("ticker", "").lower()
    return any(et.startswith(p) or tk.startswith(p) for p in SPORTS_PREFIXES)


def _is_unmodelable_market(market: dict) -> bool:
    """v11.5: Markets that no current bot can meaningfully predict.

    These markets either:
    - Require live transcript / video analysis (MENTION markets)
    - Require domain knowledge no API exposes (Survivor TV outcomes)
    - Are pure entertainment trivia (awards show predictions)

    Every bot's is_relevant() checks this FIRST and returns False.
    The market is then skipped entirely — no signal is logged, no garbage
    predictions pollute the calibration data.

    This eliminates ~460+ noise signals/day in 26-04 audit:
      - 440 KXSURVIVORMENT
      - 13  KXMLBMENTION
      - 13  KXNBAMENTION
      - 13  KXPOLITICSMENT
      - 13  KXGOVERNORMENT
      - 1   KXFOXNEWSMENTI
      - 1   KXFURYMENTION
    """
    et = market.get("event_ticker", "").lower()
    tk = market.get("ticker", "").lower()

    # MENTION markets — require live transcript analysis
    if "mention" in et or "mention" in tk:
        return True

    # Survivor TV show — requires watching the episode
    if et.startswith("kxsurv") or tk.startswith("kxsurv"):
        return True

    return False


def _is_entertainment_market(market: dict) -> bool:
    """v11.5: Entertainment / streaming / awards markets.

    These get claimed by GlobalEventsBot. Listed separately from
    _is_unmodelable_market because GlobalEventsBot WILL eventually
    have a real model — for now it just claims the markets to keep
    them out of crypto/weather/economics.

    v11.8: Added kxthevoice, kxfestival.
    v12.1: kxartiststream is now ONLY here (removed from SPORTS_PREFIXES).
    """
    ENTERTAINMENT_PREFIXES = (
        "kxspotify",     # Spotify chart positions
        "kxspotstream",  # Spotify streaming markets (was incorrectly in crypto)
        "kxbox",         # Box office (if exists)
        "kxnetflix",     # Netflix subscriber/show predictions
        "kxhbo",         # HBO show predictions
        "kxgrammy",      # Grammy awards
        "kxoscar",       # Oscar awards
        "kxemmy",        # Emmy awards
        "kxgoldenglobe", # Golden Globes
        "kxbillboard",   # Billboard chart
        "kxthevoice",    # The Voice TV show
        "kxfestival",    # Festival events (KXFESTIVALEVENTPACHNYC etc.)
        "kxartiststream",  # v11.8d: Spotify artist streaming (KXARTISTSTREAMSU, KXARTISTSTREAMS)
    )
    et = market.get("event_ticker", "").lower()
    tk = market.get("ticker", "").lower()
    return any(et.startswith(p) or tk.startswith(p) for p in ENTERTAINMENT_PREFIXES)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ECONOMICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class EconomicsBot(BaseBot):
    """
    Macro-economic markets — US + global central banks, commodities, indices.
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxcpi", "kxfed", "kxfomc", "kxgdp", "kxjobs", "kxunrate",
        "kxpce", "kxnfp", "kxyield", "kxrate", "kxinfl",
        "kxhousing", "kxretail", "kxdebt", "kxtrade", "kxconsumer",
        "kxoil", "kxgold", "kxcommodity", "kxdow", "kxspx", "kxvix",
        "kxwti", "kxjble", "kxekst", "kxapfd",

        # ── US economic indicators ─────────────────────────────────────────
        "cpi", "inflation", "unemployment", "nonfarm", "payroll",
        "gdp", "recession", "rate hike", "rate cut", "pce", "fomc",
        "interest rate", "treasury", "federal reserve", "basis points",
        "retail sales", "housing starts", "consumer confidence",
        "debt ceiling", "deficit", "tariff", "trade balance",
        "consumer sentiment", "jobless claims", "trade war",
        "durable goods", "factory orders", "industrial production",
        "capacity utilization", "trade deficit", "current account",
        "balance of payments", "personal income", "personal spending",
        "core inflation", "producer price", "ppi", "import price",
        "export price", "ism manufacturing", "ism services", "pmi",
        "purchasing managers", "new home sales", "existing home sales",
        "pending home sales", "case shiller", "building permits",
        "construction spending", "beige book",

        # ── Global central banks ───────────────────────────────────────────
        "ecb", "european central bank", "bank of japan", "boj",
        "bank of england", "boe", "reserve bank of australia", "rba",
        "peoples bank of china", "pboc", "swiss national bank", "snb",
        "bank of canada", "boc", "riksbank", "norges bank",
        "reserve bank of india", "rbi", "central bank",
        "rate decision", "monetary policy", "quantitative easing",
        "tapering", "yield curve control", "negative rates",

        # ── International economic terms ───────────────────────────────────
        "eurozone", "euro area", "g7", "g20", "imf", "world bank",
        "opec", "opec+", "wto", "oecd", "brics", "davos",
        "sovereign debt", "austerity", "stimulus", "bailout",
        "currency intervention", "devaluation", "revaluation",
        "forex", "exchange rate", "dollar index", "dxy",

        # ── Commodities ────────────────────────────────────────────────────
        "crude oil", "brent crude", "wti", "natural gas", "gasoline",
        "gold price", "silver price", "copper price", "iron ore",
        "wheat price", "corn price", "soybean", "cotton price",
        "lumber price", "uranium", "lithium price", "palladium",
        "platinum price", "cocoa price", "coffee price", "sugar price",

        # ── Market indices ─────────────────────────────────────────────────
        "dow jones", "s&p 500", "sp500", "nasdaq composite",
        "russell 2000", "vix", "volatility index", "ftse", "dax",
        "cac 40", "nikkei", "hang seng", "shanghai composite",
        "sensex", "kospi", "asx 200", "tsx",

        # ── Financial sector earnings ──────────────────────────────────────
        "bank earnings", "tech earnings", "energy sector",
        "financial sector", "big banks", "jpmorgan", "goldman sachs",
        "wells fargo", "bank of america", "citigroup", "morgan stanley",
        "blackrock", "berkshire", "hedge fund", "private equity",
        "venture capital", "ipo filing", "spac",
    ]

    @property
    def sector_name(self) -> str:
        return "economics"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False
        if _has_sports_prefix(market):
            return False
        
        # v12.2: Reject weather temperature markets (KXHIGHTDAL matched "dallas")
        et = market.get("event_ticker", "").lower()
        tk = market.get("ticker", "").lower()
        weather_prefixes = ("kxhight", "kxlowt", "kxtemp", "kxchll", "kxdens", "kxwthr", "kxhurr", "kxsnow", "kxrain")
        if any(et.startswith(p) or tk.startswith(p) for p in weather_prefixes):
            return False
        
        return _search_fields(market, self.KEYWORDS)

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        features, context = fetch_economics_features()
        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        target  = float(numbers[0]) if numbers else 0.0
        features = np.append(features, target)
        context["contract_target"] = target
        return features, context

    def evaluate(self, market, news_signal=None):
        """v12.4: RESURRECTED with real FRED-based model.
        
        Handles:
          - Fed rate decision markets (FOMC, rate hike/cut)
          - Inflation markets (CPI, PCE thresholds)
          - Employment markets (unemployment, jobless claims)
          - Treasury yield markets
        """
        ticker = market.get("ticker", "")
        title = market.get("title", "").lower()
        
        try:
            fred = get_fred_client()
            snapshot = fred.get_economic_snapshot()
        except Exception as e:
            logger.warning("[economics] FRED unavailable: %s", e)
            return None
        
        our_prob = None
        confidence = 0.55  # Base confidence for economics
        rationale = ""
        
        # ── Rate Decision Markets ────────────────────────────────────────────
        if any(kw in title for kw in ["rate hike", "rate cut", "fomc", "federal reserve", "fed funds", "basis points"]):
            # Detect decision type
            if "hike" in title or "raise" in title or "increase" in title:
                decision = "hike"
            elif "cut" in title or "lower" in title or "decrease" in title:
                decision = "cut"
            else:
                decision = "hold"
            
            # Get current rate
            current_rate = snapshot.fed_funds_upper or 5.25
            
            # Extract target rate from title if present
            rate_match = re.search(r"(\d+\.?\d*)\s*%", title)
            target_rate = float(rate_match.group(1)) if rate_match else current_rate
            
            our_prob = fred.predict_rate_decision(current_rate, target_rate, decision)
            
            # Higher confidence when inflation data is recent
            if snapshot.cpi_yoy is not None:
                confidence = 0.65
            
            rationale = f"fed_{decision}_cpi={snapshot.cpi_yoy:.1f}%_unemp={snapshot.unemployment_rate:.1f}%"
            logger.info(
                "[economics] %s: %s decision → P=%.1f%% (CPI=%.1f%% UNEMP=%.1f%%)",
                ticker[:35], decision, our_prob * 100,
                snapshot.cpi_yoy or 0, snapshot.unemployment_rate or 0,
            )
        
        # ── Inflation Markets ────────────────────────────────────────────────
        elif any(kw in title for kw in ["cpi", "inflation", "pce"]):
            # Extract threshold
            threshold_match = re.search(r"(\d+\.?\d*)\s*%", title)
            threshold = float(threshold_match.group(1)) if threshold_match else 3.0
            
            # Determine comparison
            above = any(kw in title for kw in ["above", "exceed", "over", "higher"])
            below = any(kw in title for kw in ["below", "under", "lower"])
            
            current_cpi = snapshot.cpi_yoy or snapshot.core_cpi_yoy or 3.0
            
            # Simple probability based on current vs threshold
            distance = current_cpi - threshold
            
            if above or (not below):  # Default to "above"
                # P(CPI > threshold)
                if distance > 0.5:
                    our_prob = min(0.90, 0.70 + distance * 0.05)
                elif distance > 0:
                    our_prob = 0.55 + distance * 0.10
                elif distance > -0.5:
                    our_prob = 0.45 + distance * 0.10
                else:
                    our_prob = max(0.10, 0.40 + distance * 0.05)
            else:
                # P(CPI < threshold) = 1 - P(CPI > threshold)
                if distance < -0.5:
                    our_prob = min(0.90, 0.70 + abs(distance) * 0.05)
                elif distance < 0:
                    our_prob = 0.55 + abs(distance) * 0.10
                else:
                    our_prob = max(0.10, 0.45 - distance * 0.10)
            
            confidence = 0.60
            rationale = f"cpi_current={current_cpi:.1f}%_vs_threshold={threshold:.1f}%"
            logger.info(
                "[economics] %s: CPI=%.1f%% vs threshold=%.1f%% → P=%.1f%%",
                ticker[:35], current_cpi, threshold, our_prob * 100,
            )
        
        # ── Employment Markets ───────────────────────────────────────────────
        elif any(kw in title for kw in ["unemployment", "jobless", "nonfarm", "payroll"]):
            # Extract threshold
            threshold_match = re.search(r"(\d+\.?\d*)", title)
            
            if "unemployment" in title:
                current = snapshot.unemployment_rate or 4.0
                threshold = float(threshold_match.group(1)) if threshold_match else 4.5
                
                above = any(kw in title for kw in ["above", "exceed", "rise"])
                distance = current - threshold
                
                if above:
                    our_prob = 0.50 + min(0.35, max(-0.35, distance * 0.10))
                else:
                    our_prob = 0.50 - min(0.35, max(-0.35, distance * 0.10))
                
                confidence = 0.58
                rationale = f"unemp={current:.1f}%_vs_{threshold:.1f}%"
                
            elif "jobless" in title:
                current = snapshot.jobless_claims or 220000
                # Threshold usually in thousands
                threshold = float(threshold_match.group(1)) * 1000 if threshold_match else 250000
                
                above = any(kw in title for kw in ["above", "exceed"])
                distance_pct = (current - threshold) / threshold
                
                if above:
                    our_prob = 0.50 + min(0.35, max(-0.35, distance_pct))
                else:
                    our_prob = 0.50 - min(0.35, max(-0.35, distance_pct))
                
                confidence = 0.55
                rationale = f"claims={current/1000:.0f}K_vs_{threshold/1000:.0f}K"
            
            else:
                # Generic employment — need more specific parsing
                return None
            
            logger.info(
                "[economics] %s: %s → P=%.1f%%",
                ticker[:35], rationale, our_prob * 100,
            )
        
        # ── Treasury Yield Markets ───────────────────────────────────────────
        elif any(kw in title for kw in ["treasury", "yield", "10-year", "10 year", "2-year", "2 year"]):
            if "10" in title:
                current = snapshot.treasury_10y or 4.5
            elif "2" in title:
                current = snapshot.treasury_2y or 4.8
            else:
                current = snapshot.treasury_10y or 4.5
            
            # Extract threshold
            threshold_match = re.search(r"(\d+\.?\d*)\s*%", title)
            threshold = float(threshold_match.group(1)) if threshold_match else current
            
            above = any(kw in title for kw in ["above", "exceed", "rise"])
            distance = current - threshold
            
            if above:
                our_prob = 0.50 + min(0.30, max(-0.30, distance * 0.15))
            else:
                our_prob = 0.50 - min(0.30, max(-0.30, distance * 0.15))
            
            confidence = 0.52  # Yields are harder to predict
            rationale = f"yield={current:.2f}%_vs_{threshold:.2f}%"
            logger.info(
                "[economics] %s: %s → P=%.1f%%",
                ticker[:35], rationale, our_prob * 100,
            )
        
        else:
            # Unhandled economics market type — skip for now
            logger.debug("[economics] %s: no handler for this market type", ticker[:35])
            return None
        
        if our_prob is None:
            return None
        
        # Clamp probability
        our_prob = max(0.05, min(0.95, our_prob))
        
        # Build BotSignal
        market_price = _market_prob(market)
        edge = our_prob - market_price
        direction = "yes" if our_prob > market_price else "no"
        
        return BotSignal(
            sector="economics",
            ticker=ticker,
            our_prob=our_prob,
            market_prob=market_price,
            edge=edge,
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CRYPTO BOT
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoBot(BaseBot):
    """
    Crypto price, dominance, ETF, DeFi, Layer 2, and exchange markets.

    v11.2: Removed bare short keywords "op", "near", "base", "sei" that were
    substring-matching inside unrelated words ("operation", "nearby",
    "baseball", "series"). Replaced with more specific multi-word variants.
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
        "kxbnb", "kxavax", "kxlink", "kxcoin", "kxmatic", "kxada",
        "kxdot", "kxatom", "kxnear", "kxfil", "kxapt", "kxsui",
        # v11.7: removed kxspotstream — it was Spotify streaming, not crypto.
        # Spot BTC ETF tickers use kxspotbtc/kxnetf, not kxspotstream.
        "kxshib", "kxnetf",
        "kxshiba",  # SHIB token explicit

        # ── Major coins ────────────────────────────────────────────────────
        "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "ripple",
        "dogecoin", "doge", "cardano", "ada", "polkadot", "dot",
        "avalanche", "avax", "chainlink", "link", "polygon", "matic",
        "shiba inu", "shib", "litecoin", "ltc", "bitcoin cash", "bch",
        "stellar", "xlm", "cosmos", "atom", "algorand", "algo",
        "tron", "trx", "filecoin", "fil", "near protocol", "near protocol",
        "aptos", "apt", "sui network", "sei network", "injective", "inj",
        "arbitrum", "arb", "optimism op", "base l2",
        "pepe", "floki", "bonk", "wif",

        # ── DeFi protocols ─────────────────────────────────────────────────
        "uniswap", "uni", "aave", "compound", "curve", "crv",
        "makerdao", "dai", "synthetix", "snx", "yearn", "yfi",
        "pancakeswap", "cake", "sushi", "balancer", "1inch",
        "lido", "steth", "rocketpool", "reth",

        # ── Exchanges & infrastructure ─────────────────────────────────────
        "coinbase", "binance", "kraken", "bybit", "okx", "bitfinex",
        "gemini", "ftx", "celsius", "crypto.com",

        # ── Market concepts ────────────────────────────────────────────────
        "crypto", "defi", "nft", "web3", "blockchain",
        "bitcoin etf", "crypto etf", "spot etf", "bitcoin halving",
        "stablecoin", "usdt", "usdc", "tether", "altcoin",
        "crypto dominance", "total market cap", "crypto regulation",
        "sec crypto", "bitcoin spot", "ethereum etf",
        "layer 2", "layer2", "rollup", "zk proof",
        "crypto mining", "hash rate", "mempool", "gas fees",
    ]

    TICKER_COIN_MAP = {
        "kxbtc": "bitcoin",  "kxeth":    "ethereum",
        "kxsol": "solana",   "kxxrp":    "ripple",
        "kxdoge":"dogecoin", "kxavax":   "avalanche-2",
        "kxbnb": "binancecoin", "kxmatic": "matic-network",
    }

    # v12.2: Blocklist for crypto markets we can't model
    # KXBTCD = BTC daily price targets — 0.488 Brier on 7 resolved, catastrophic
    CRYPTO_BLOCKLIST = (
        "kxbtcd",      # BTC daily targets (not 15M, different dynamics)
        "kxethd",      # ETH daily targets
        "kxsold",      # SOL daily targets
        # v12.2: 15M markets for volatile altcoins — no edge
        "kxsol15m",    # 8 resolved, 0.2375 Brier
        "kxxrp15m",    # 5 resolved, 0.2688 Brier
        # v12.2: Random sports leaking to crypto
        "kxtabletennis",  # Table tennis
        "kxballerleague", # Baller League soccer
        # v12.2: Politics leaking via NEWS SPIKE
        "kxleavecherfilus",  # Congress leave (politics)
        "kxleavegonzales",   # Congress leave (politics)
        "kxleavemills",      # Congress leave (politics)
        "kxtrumpact",        # Trump actions (politics)
        "kxtrumptime",       # Trump timing (politics)
        "kxmamdanieo",       # Mamdani EO (politics)
        "kxhormuznorm",      # Hormuz blockade (politics)
        "kxca14swinner",     # CA-14 special election (politics)
    )

    TITLE_COIN_MAP = {
        "btc": "bitcoin",  "bitcoin":  "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "solana": "solana", "sol": "solana",
        "xrp": "ripple",   "ripple":   "ripple",
        "doge": "dogecoin", "dogecoin": "dogecoin",
        "avax": "avalanche-2",
        "bnb":  "binancecoin",
    }

    @property
    def sector_name(self) -> str:
        return "crypto"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False
        if _has_sports_prefix(market):
            return False
        
        # v12.2: Blocklist daily target markets (KXBTCD, etc.)
        et = market.get("event_ticker", "").lower()
        tk = market.get("ticker", "").lower()
        if any(et.startswith(p) or tk.startswith(p) for p in self.CRYPTO_BLOCKLIST):
            return False
        
        return _search_fields(market, self.KEYWORDS)

    def _detect_coin(self, market: dict) -> str:
        et = market.get("event_ticker", "").lower()
        for prefix, coin_id in self.TICKER_COIN_MAP.items():
            if et.startswith(prefix):
                return coin_id
        title = market.get("title", "").lower()
        for kw, coin_id in self.TITLE_COIN_MAP.items():
            if kw in title:
                return coin_id
        return "bitcoin"

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        coin_id  = self._detect_coin(market)
        features, context = fetch_crypto_features(coin_id=coin_id)
        title   = market.get("title", "")
        numbers = re.findall(r"[\d,]+\.?\d*", title.replace(",", ""))
        target  = float(numbers[0]) if numbers else 0.0
        current_price = context.get("price", 1.0) or 1.0
        ratio   = current_price / target if target > 0 else 1.0
        features = np.append(features, [target, ratio])
        context["contract_target"] = target
        context["price_vs_target"]  = ratio
        return features, context

    # v11.8d: Per-coin confidence scaling based on historical Brier scores
    COIN_CONFIDENCE_SCALE = {
        "bitcoin":      0.85,   # Brier 0.113 — strong
        "binancecoin":  0.85,   # Brier 0.127 — strong
        "ethereum":     0.70,   # Brier 0.187 — decent
        "solana":       0.50,   # Brier 0.276 — weak
        "ripple":       0.50,   # Brier 0.269 — weak
        "dogecoin":     0.0,    # Brier 0.395 — excluded
    }

    def evaluate(self, market, news_signal=None):
        """v11.8d: Scale confidence per coin based on historical Brier scores.

        BTC/BNB keep near-full confidence (Brier <= 0.13).
        ETH gets 70% (Brier 0.187).
        SOL/XRP get 50% (Brier ~0.27).
        DOGE 15M markets excluded entirely (Brier 0.395).
        """
        ticker  = market.get("ticker", "").lower()
        coin_id = self._detect_coin(market)

        # DOGE 15M: exclude (worst performer, Brier 0.395)
        if coin_id == "dogecoin" and "15m" in ticker:
            logger.debug("[crypto] DOGE 15M excluded: %s", ticker)
            return None

        signal = super().evaluate(market, news_signal)
        if signal is None:
            return None

        scale = self.COIN_CONFIDENCE_SCALE.get(coin_id, 0.60)
        if scale <= 0.0:
            logger.debug("[crypto] %s excluded (scale=0): %s", coin_id, ticker)
            return None

        return BotSignal(
            sector      = signal.sector,
            prob        = signal.prob,
            confidence  = signal.confidence * scale,
            market_prob = signal.market_prob,
            brier_score = signal.brier_score,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. POLITICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class PoliticsBot(BaseBot):
    """
    US + international elections, legislation, geopolitics, appointments.

    v11.3: Removed "kxintl" from KEYWORDS — those are international sports
    (KXINTLFIGHT etc.), not politics. Politics retains "kxun", "kxnato",
    "kxeu" for international political markets.
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxpres", "kxsenate", "kxhouse", "kxgov", "kxelect",
        "kxpol", "kxcong", "kxsupct", "kxadmin",
        # v11.3: removed "kxintl" — international sports, not politics
        "kxun", "kxnato", "kxeu",

        # ── US politics ────────────────────────────────────────────────────
        "election", "vote", "congress", "senate", "president",
        "governor", "democrat", "republican", "white house",
        "approval", "impeach", "nominee", "ballot", "primary",
        "caucus", "supreme court", "executive order", "policy",
        "trump", "harris", "biden", "pelosi", "mcconnell",
        "filibuster", "reconciliation", "continuing resolution",
        "government shutdown", "debt limit", "midterm",
        "electoral college", "popular vote", "swing state",
        "attorney general", "secretary of state", "cabinet",
        "veto", "pardon", "indictment", "conviction",

        # ── International politics ─────────────────────────────────────────
        "united nations", "un security council", "nato",
        "european union", "european parliament", "g7", "g20",
        "brics", "imf", "world bank", "wto",
        "prime minister", "chancellor", "parliament",
        "referendum", "coup", "sanctions", "treaty",
        "ceasefire", "peace talks", "diplomatic",
        "macron", "scholz", "sunak", "modi", "xi jinping",
        "putin", "zelensky", "netanyahu", "erdogan",
        "boris johnson", "trudeau", "albanese",

        # ── Geopolitical events ────────────────────────────────────────────
        "ukraine", "russia", "israel", "gaza", "taiwan",
        "north korea", "iran nuclear", "south china sea",
        "nato expansion", "un resolution", "armed conflict",
        "trade sanctions", "export controls",

        # ── Supreme Court ──────────────────────────────────────────────────
        "supreme court ruling", "scotus", "oral arguments",
        "majority opinion", "dissent", "overturned", "landmark ruling",
        "constitutional", "first amendment", "second amendment",
        "roe v wade", "affirmative action", "title ix",
    ]

    @property
    def sector_name(self) -> str:
        return "politics"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False
        if _has_sports_prefix(market):
            return False
        return _search_fields(market, self.KEYWORDS)

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        poly_slug  = market.get("polymarket_slug")
        features, context = fetch_politics_features(poly_slug)
        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        target  = float(numbers[0]) if numbers else 50.0
        features = np.append(features, target)
        close_ts = market.get("close_time", "")
        try:
            from datetime import datetime, timezone
            close_dt      = datetime.fromisoformat(close_ts.replace("Z", "+00:00"))
            days_to_close = (close_dt - datetime.now(timezone.utc)).days
        except Exception:
            days_to_close = 30
        features = np.append(features, days_to_close)
        context["days_to_close"] = days_to_close
        return features, context


# ═══════════════════════════════════════════════════════════════════════════════
#  4. WEATHER BOT
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherBot(BaseBot):
    """
    Temperature, precipitation, storms — worldwide city coverage.

    v11.3: Renamed _ESPORTS_BLOCKLIST → _NON_WEATHER_BLOCKLIST and expanded
    with cricket (kxcba), track (kxdima), A-League (kxalea), Survivor (kxsurv),
    and all international soccer/basketball prefixes that were matching city
    keywords and getting NOAA priors applied to sports markets.

    v11.8: Added kxt20, kxhighinfl, kxfestival, kxthevoice.
    
    v12.3: Added evaluate() override that uses NOAA directly when NOAA
    confidence >= 0.60, bypassing the poisoned Bayesian model.
    """

    # ── Non-weather prefix blocklist — defense in depth ────────────────────────
    # Primary guard is _has_sports_prefix(). This secondary check ensures no
    # non-weather market ever gets a NOAA prior even if prefix normalization
    # or UUID stripping produces an unexpected form.
    _NON_WEATHER_BLOCKLIST = (
        # Esports (from v11.2)
        "kxow", "kxvalorant", "kxlol", "kxleague",
        "kxrl", "kxrocketleague",
        "kxcsgo", "kxcs2", "kxdota",
        "kxapex", "kxfort", "kxhalo", "kxsc2",
        "kxesport", "kxintlf",
        # v11.3 additions — these all matched city keywords
        "kxcba",       # Cricket (matched city names → 0.92 NOAA priors)
        "kxdima",      # Diamond League
        "kxalea",      # A-League ("melbourne" match)
        "kxsurv",      # Survivor TV show
        "kxarg",       # Argentine soccer ("buenos aires")
        "kxbras",      # Brazilian soccer ("sao paulo", "rio")
        "kxlali",      # La Liga ("madrid", "barcelona")
        "kxbund",      # Bundesliga ("munich", "berlin")
        "kxseri",      # Serie A ("rome", "milan")
        "kxbelg",      # Belgian ("brussels")
        "kxswis",      # Swiss ("zurich", "geneva")
        "kxjlea",      # J-League ("tokyo", "osaka")
        "kxpga",       # PGA golf
        "kxowga",      # Golf
        "kxahlg",      # AHL hockey
        "kxfiba",      # FIBA basketball
        "kxacbg",      # ACB basketball
        "kxvtbg",      # VTB basketball
        "kxuclt",      # Champions League
        "kxr6g",       # Rainbow Six
        "kxintl",      # International sports
        "kxfifa",      # FIFA
        # v11.8 additions
        "kxt20",       # T20 cricket (SWE/IDN matched city keywords)
        "kxhighinfl",  # KXHIGHINFLATION matched "kxhigh" weather prefix
        "kxfestival",  # Festival events (matched "nyc" city keyword)
        "kxthevoice",  # The Voice TV show
        "kxafl",       # AFL (Australian Football League — "sydney" city match)
        # v11.8d: news-spike bypass leaks — these tickers fire on ALL bots
        # via BaseBot.evaluate() news-velocity path, bypassing is_relevant()
        "kxswalwell",  # Swalwell dropout (politics)
        "kxtrumppardons",  # Trump pardons (politics)
        "kxtrumpendorse",  # Trump endorsements (politics)
        "kxdenmarkpm",     # Denmark PM (politics)
        "kxisraelpm",      # Israel PM (politics)
        "kxswencounters",  # Star Wars encounters (entertainment)
        "kxabagame",       # WTA tennis (ABA → sports, not weather)
        "kxpayrolls",      # Payrolls (economics)
        "kxlcpi",          # CPI (economics)
        "kxcpi",           # CPI (economics)
        "kxwti",           # WTI crude oil (economics)
        # v11.8d: new leagues found 2026-04-12
        "kxdel",           # DEL German hockey
        "kxkhl",           # KHL Russian hockey
        "kxjbleague",      # J.B.League Japanese basketball
        "kxkbl",           # KBL Korean basketball
        "kxkleague",       # K League Korean soccer
        "kxartiststream",  # Spotify streaming (entertainment)
        # v12.2: "Next X" prediction markets (coaching hires, etc.)
        "kxnext",          # KXNEXTNBACOACH matched "milwaukee" city keyword
        # v12.2: Golf and UEFA leaks
        "kxlpga",          # LPGA Tour golf
        "kxuecl",          # UEFA Europa Conference League
        # v12.2: Random leagues leaking to weather
        "kxtabletennis",   # Table tennis
        "kxkbogame",       # Korean baseball
        "kxlnbelite",      # French basketball (LNB Elite)
        "kxchnsl",         # Chinese Super League
        "kxapfddh",        # Unknown league
        "kxballerleague",  # Baller League soccer
        "kxitfwmatch",     # ITF women's tennis
        "kxitfmmatch",     # ITF men's tennis
        # v12.2: Politics tickers leaking via NEWS SPIKE
        "kxtrumpact",      # Trump actions (politics)
        "kxtrumptime",     # Trump timing (politics)
        "kxmamdanieo",     # Mamdani EO (politics)
        "kxleavecherfilus",# Congress leave (politics)
        "kxleavegonzales", # Congress leave (politics)
        "kxleavemills",    # Congress leave (politics)
        "kxhormuznorm",    # Hormuz blockade (politics)
        "kxca14swinner",   # CA-14 special election (politics)
    )

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxwthr", "kxhurr", "kxsnow", "kxtmp", "kxrain",
        "kxstorm", "kxtornado", "kxblizz", "kxfrost",
        "kxcyclone", "kxtyphoon", "kxmonsoon",
        "kxlowt", "kxhigh", "kxtemp", "kxchll", "kxdens",

        # ── Core weather terms ─────────────────────────────────────────────
        "temperature", "fahrenheit", "celsius",
        "hurricane", "tornado", "flood", "drought",
        "blizzard", "precipitation", "weather",
        "freezing", "snowfall", "snowstorm",
        "cyclone", "typhoon", "monsoon", "derecho",
        "nor'easter", "noreaster", "polar vortex",
        "dust storm", "sandstorm", "ice storm",
        "wildfire smoke", "air quality index",
        "heat wave", "wind chill", "heat index",
        "high temperature", "low temperature",
        "degrees fahrenheit", "degrees celsius",
        "weather forecast", "rain forecast",
        "snow accumulation", "storm surge",
        "tropical storm", "severe weather",
        "feels like temperature", "record high", "record low",
        "above average temperature", "below average temperature",
        "atmospheric river", "el nino", "la nina",
        "lake effect snow", "freezing rain", "sleet",

        # ── North America — US Cities ──────────────────────────────────────
        "new york", "nyc", "los angeles", "chicago", "houston",
        "dallas", "miami", "atlanta", "boston", "seattle",
        "phoenix", "denver", "las vegas", "detroit", "minneapolis",
        "portland", "san francisco", "san diego", "nashville",
        "charlotte", "orlando", "tampa", "memphis", "baltimore",
        "washington dc", "philadelphia", "kansas city", "st louis",
        "new orleans", "pittsburgh", "cleveland", "cincinnati",
        "salt lake city", "boise", "albuquerque", "tucson",
        "anchorage", "honolulu", "fargo", "billings", "omaha",
        "louisville", "indianapolis", "columbus", "milwaukee",
        "hartford", "providence", "richmond", "raleigh",
        "oklahoma city", "wichita", "sioux falls",

        # ── Canada ─────────────────────────────────────────────────────────
        "toronto", "montreal", "vancouver", "calgary", "edmonton",
        "ottawa", "winnipeg", "halifax", "quebec city",

        # ── Europe ─────────────────────────────────────────────────────────
        "london", "paris", "berlin", "madrid", "rome", "amsterdam",
        "brussels", "vienna", "zurich", "geneva", "stockholm",
        "oslo", "copenhagen", "helsinki", "dublin", "lisbon",
        "barcelona", "milan", "munich", "frankfurt", "hamburg",
        "warsaw", "prague", "budapest", "bucharest", "athens",
        "istanbul", "moscow", "st petersburg", "kyiv", "kiev",
        "reykjavik", "valletta", "nicosia",

        # ── Asia ───────────────────────────────────────────────────────────
        "tokyo", "osaka", "beijing", "shanghai", "hong kong",
        "singapore", "seoul", "bangkok", "mumbai", "delhi",
        "kolkata", "chennai", "bangalore", "karachi", "lahore",
        "dhaka", "kathmandu", "colombo", "male", "manila",
        "jakarta", "kuala lumpur", "ho chi minh city", "hanoi",
        "yangon", "taipei", "macau", "ulaanbaatar", "almaty",
        "tashkent", "baku", "tbilisi", "yerevan", "tehran",
        "baghdad", "riyadh", "dubai", "abu dhabi", "doha",
        "kuwait city", "muscat", "beirut", "amman", "tel aviv",

        # ── Africa ─────────────────────────────────────────────────────────
        "cairo", "lagos", "nairobi", "johannesburg", "cape town",
        "casablanca", "tunis", "algiers", "accra", "dakar",
        "addis ababa", "dar es salaam", "lusaka", "harare",
        "maputo", "antananarivo", "khartoum", "tripoli",

        # ── South America ──────────────────────────────────────────────────
        "sao paulo", "rio de janeiro", "buenos aires", "bogota",
        "lima", "santiago", "caracas", "quito", "montevideo",
        "asuncion", "la paz", "santa cruz", "guayaquil",
        "medellin", "cali", "belo horizonte", "brasilia",
        "manaus", "fortaleza", "recife", "porto alegre",

        # ── Australia / Oceania ────────────────────────────────────────────
        "sydney", "melbourne", "brisbane", "perth", "adelaide",
        "canberra", "darwin", "auckland", "wellington",
        "christchurch", "suva", "port moresby",

        # ── US airport codes (NWS uses these in market titles) ────────────
        "jfk", "lax", "ord", "dfw", "atl", "sfo", "mia", "sea",
        "den", "bos", "iah", "phx", "las", "msp", "dtw", "ewr",
        "phl", "slc", "pdx", "mco", "tpa", "mdw", "bwi", "iad",
    ]

    CITY_COORDS = {
        # US
        "new york": (40.71, -74.01), "nyc": (40.71, -74.01),
        "los angeles": (34.05, -118.24), "chicago": (41.85, -87.65),
        "houston": (29.76, -95.37), "dallas": (32.78, -96.80),
        "miami": (25.77, -80.19), "atlanta": (33.75, -84.39),
        "boston": (42.36, -71.06), "seattle": (47.61, -122.33),
        "phoenix": (33.45, -112.07), "denver": (39.74, -104.98),
        "las vegas": (36.17, -115.14), "detroit": (42.33, -83.05),
        "minneapolis": (44.98, -93.27), "portland": (45.52, -122.68),
        "san francisco": (37.77, -122.42), "san diego": (32.72, -117.16),
        "nashville": (36.17, -86.78), "charlotte": (35.23, -80.84),
        "orlando": (28.54, -81.38), "tampa": (27.95, -82.46),
        "washington dc": (38.91, -77.04), "philadelphia": (39.95, -75.17),
        "new orleans": (29.95, -90.07), "pittsburgh": (40.44, -79.99),
        "salt lake city": (40.76, -111.89), "albuquerque": (35.08, -106.65),
        "anchorage": (61.22, -149.90), "honolulu": (21.31, -157.86),
        # Canada
        "toronto": (43.65, -79.38), "montreal": (45.50, -73.57),
        "vancouver": (49.25, -123.12), "calgary": (51.05, -114.06),
        # Europe
        "london": (51.51, -0.13), "paris": (48.85, 2.35),
        "berlin": (52.52, 13.40), "madrid": (40.42, -3.70),
        "rome": (41.90, 12.50), "amsterdam": (52.37, 4.90),
        "moscow": (55.75, 37.62), "istanbul": (41.01, 28.95),
        "stockholm": (59.33, 18.07), "oslo": (59.91, 10.75),
        "zurich": (47.38, 8.54), "vienna": (48.21, 16.37),
        # Asia
        "tokyo": (35.69, 139.69), "beijing": (39.91, 116.39),
        "shanghai": (31.23, 121.47), "hong kong": (22.32, 114.17),
        "singapore": (1.35, 103.82), "seoul": (37.57, 126.98),
        "bangkok": (13.75, 100.52), "mumbai": (19.08, 72.88),
        "delhi": (28.61, 77.21), "dubai": (25.20, 55.27),
        "riyadh": (24.69, 46.72), "tehran": (35.69, 51.42),
        # Australia
        "sydney": (-33.87, 151.21), "melbourne": (-37.81, 144.96),
        "brisbane": (-27.47, 153.02), "perth": (-31.95, 115.86),
        # South America
        "sao paulo": (-23.55, -46.63), "buenos aires": (-34.60, -58.38),
        "bogota": (4.71, -74.07), "lima": (-12.05, -77.04),
        "santiago": (-33.45, -70.67),
        # Africa
        "cairo": (30.06, 31.25), "lagos": (6.46, 3.38),
        "nairobi": (-1.29, 36.82), "johannesburg": (-26.20, 28.04),
        "cape town": (-33.93, 18.42),
    }

    @property
    def sector_name(self) -> str:
        return "weather"

    def is_relevant(self, market: dict) -> bool:
        # ── v11.5 guards ──────────────────────────────────────────────────
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False

        # ── Primary guard: sports prefix check ────────────────────────────
        if _has_sports_prefix(market):
            return False

        # ── Secondary guard: explicit non-weather blocklist ────────────────
        # Defense in depth — catches non-weather markets even if
        # _has_sports_prefix misses them due to UUID suffix or normalization.
        et = market.get("event_ticker", "").lower()
        tk = market.get("ticker", "").lower()
        if any(et.startswith(p) or tk.startswith(p) for p in self._NON_WEATHER_BLOCKLIST):
            return False

        return _search_fields(market, self.KEYWORDS)

    def _detect_city(self, market: dict) -> tuple[float, float, str]:
        title = market.get("title", "").lower()
        for city, coords in self.CITY_COORDS.items():
            if city in title:
                return coords[0], coords[1], city
        return 40.71, -74.01, "New York"

    def _get_noaa_prior(self, market: dict) -> Optional[dict]:
        title      = market.get("title", "")
        close_time = market.get("close_time")

        city_str = self._parse_city_from_title(title)
        if not city_str:
            return None
        coords = city_to_coords(city_str)
        if not coords:
            return None

        lat, lon    = coords
        target_date = parse_market_date(title, close_time)
        market_type = classify_weather_market(title)

        try:
            pkg = asyncio.run(fetch_weather_package(lat, lon))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            pkg  = loop.run_until_complete(fetch_weather_package(lat, lon))

        if not pkg:
            return None

        prior_yes  = 0.5
        confidence = 0.30
        summary    = "NOAA: no signal matched"

        if market_type in ("temp_high", "temp_low"):
            temps     = extract_temp_signals(pkg["hourly"], target_date)
            threshold = parse_temp_threshold(title)
            if temps and threshold is not None:
                observed = temps["high_f"] if market_type == "temp_high" else temps["low_f"]
                spread   = max((temps["high_f"] - temps["low_f"]) / 2, 1.0)
                z        = (observed - threshold) / spread
                prior_yes  = round(1 / (1 + pow(2.718, -1.7 * z)), 4)
                confidence = min(0.85, 0.4 + 0.05 * temps["sample_count"])
                summary    = (
                    f"NOAA temp: observed={observed:.1f}°F "
                    f"threshold={threshold:.1f}°F P(YES)={prior_yes:.2f}"
                )

        elif market_type == "precipitation":
            precip = extract_precip_signals(pkg["forecast"], target_date)
            if precip:
                prior_yes  = round(precip["precip_pct"] / 100.0, 4)
                confidence = 0.70
                summary    = f"NOAA precip: PoP={precip['precip_pct']:.0f}%"

        elif market_type == "snow":
            precip     = extract_precip_signals(pkg["forecast"], target_date)
            alerts     = extract_alert_signals(pkg["alerts"])
            snow_alert = any("snow" in (a or "").lower() for a in alerts["alert_types"])
            base       = precip["precip_pct"] / 100.0 * 0.6
            if snow_alert:
                base = min(1.0, base + 0.25)
            prior_yes  = round(base, 4)
            confidence = 0.65
            summary    = f"NOAA snow: precip={precip['precip_pct']:.0f}% snow_alert={snow_alert}"

        elif market_type == "severe_weather":
            alerts = extract_alert_signals(pkg["alerts"])
            if alerts["has_severe_alert"]:
                prior_yes, confidence = 0.82, 0.80
            else:
                prior_yes, confidence = 0.12, 0.60
            summary = f"NOAA alerts: severity={alerts['highest_severity']}"

        logger.info(f"[NOAA] {city_str} | {target_date} | {market_type} | {summary}")

        return {
            "prior_yes":    prior_yes,
            "confidence":   confidence,
            "market_type":  market_type,
            "noaa_summary": summary,
            "city":         city_str,
            "target_date":  target_date,
        }

    @staticmethod
    def _parse_city_from_title(title: str) -> Optional[str]:
        patterns = [
            r"\bin\s+([A-Za-z\s]+?)(?:\s+on|\s+exceed|\s+be|\s+reach|\?|$)",
            r"^Will\s+([A-Za-z\s]+?)\s+(?:high|low|temp)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:temperature|temp|high|low|weather)",
        ]
        for pattern in patterns:
            m = re.search(pattern, title, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        lat, lon, city = self._detect_city(market)
        features, context = fetch_weather_features(lat, lon, city)

        title    = market.get("title", "")
        numbers  = re.findall(r"\d+\.?\d*", title)
        target_f = float(numbers[0]) if numbers else 75.0
        target_c = (target_f - 32) * 5 / 9
        delta    = context.get("temp_max_c", 0.0) - target_c
        features = np.append(features, [target_c, delta])
        context["target_c"]          = target_c
        context["delta_from_target"] = delta

        if not skip_noaa:
            noaa = self._get_noaa_prior(market)
            if noaa:
                existing_prob = context.get("model_probability", 0.5)
                w_noaa   = noaa["confidence"]
                w_model  = 1.0 - w_noaa
                blended  = round(w_noaa * noaa["prior_yes"] + w_model * existing_prob, 4)
                features = np.append(features, [noaa["prior_yes"], noaa["confidence"]])
                context["noaa_prior"]        = noaa["prior_yes"]
                context["noaa_confidence"]   = noaa["confidence"]
                context["noaa_blended_prob"] = blended
                context["noaa_summary"]      = noaa["noaa_summary"]
                context["noaa_city"]         = noaa["city"]
                context["noaa_date"]         = noaa["target_date"]
                context["noaa_market_type"]  = noaa["market_type"]
            else:
                features = np.append(features, [0.5, 0.0])
                context["noaa_prior"]      = None
                context["noaa_confidence"] = 0.0
                context["noaa_summary"]    = "NOAA unavailable"
        else:
            features = np.append(features, [0.5, 0.0])
            context["noaa_prior"]      = None
            context["noaa_confidence"] = 0.0
            context["noaa_summary"]    = "skipped during ingestion"

        return features, context

    # ─────────────────────────────────────────────────────────────────────────
    # v12.3: NOAA-FIRST EVALUATION — bypasses poisoned Bayesian model
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, market, news_signal=None):
        """v12.3: NOAA-first evaluation — bypass Bayesian model when NOAA is confident.
        
        The Bayesian model prior was poisoned to ~3% by duplicate signals and
        misattributed outcomes. Use NOAA directly when NOAA confidence >= 0.60
        (covers temp_high, temp_low, precip, snow markets).
        
        This ensures P(YES)=0.93 from NOAA becomes our_p=0.93, not our_p=0.069.
        """
        if not self.is_relevant(market):
            return None
        
        ticker = market.get("ticker", "")
        
        # Try NOAA first
        noaa = self._get_noaa_prior(market)
        
        # v12.5: Lowered threshold from 0.60 to 0.40 to capture more weather signals
        # Correlation engine finds high-edge trades, but we want independent signals too
        if noaa and noaa.get("confidence", 0) >= 0.40:
            # NOAA has signal — use it directly, bypass Bayesian model
            our_prob    = noaa["prior_yes"]
            market_prob = _market_prob(market)
            confidence  = noaa["confidence"]
            
            logger.info(
                "[weather/NOAA] %s | %s | our_p=%.3f mkt_p=%.3f conf=%.2f | %s",
                ticker, noaa.get("city", "?"), our_prob, market_prob,
                confidence, noaa.get("noaa_summary", ""),
            )
            
            return BotSignal(
                sector      = self.sector_name,
                prob        = our_prob,
                confidence  = confidence,
                market_prob = market_prob,
                brier_score = 0.15,  # Conservative until calibrated
            )
        
        # NOAA unavailable or low confidence — skip instead of using poisoned Bayesian
        logger.debug("[weather] %s — NOAA unavailable/low conf (<40%%), skipping", ticker)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  5. FINANCIAL MARKETS BOT (replaces TechBot v11.7)
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialMarketsBot(BaseBot):
    """
    Individual stocks, earnings, M&A, IPOs across ALL sectors — not just tech.

    v11.7: Replaces TechBot. Broader scope covers banks, healthcare, energy,
    retail alongside tech. Economics handles macro (CPI, GDP, FOMC);
    this bot handles company-level financial events.
    """

    # v12.2: Blocklist for random sports/games leaking to financial_markets
    FINMARKETS_BLOCKLIST = (
        "kxtabletennis",  # Table tennis
        "kxballerleague", # Baller League soccer
        "kxkbogame",      # Korean baseball
        "kxlnbelite",     # French basketball
        "kxchnsl",        # Chinese Super League
        "kxapfddh",       # Unknown league
        "kxitfwmatch",    # ITF women's tennis
        "kxitfmmatch",    # ITF men's tennis
    )

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
        "kxtsla", "kxearnings", "kxipo",
        "kxintc", "kxamd", "kxqcom", "kxtsm", "kxsamsng",
        "kxnasdaq", "kxsemiconductor",

        # ── US Big Tech ─────────────────────────────────────────────────────
        "apple", "google", "microsoft", "amazon", "nvidia", "tesla",
        "meta", "adobe", "salesforce", "oracle", "ibm",
        "intel", "amd", "qualcomm", "broadcom", "texas instruments",
        "micron", "western digital", "hp", "dell",
        "paypal", "shopify", "uber", "lyft",
        "airbnb", "doordash", "snap",

        # ── Global Tech ─────────────────────────────────────────────────────
        "samsung", "tsmc", "taiwan semiconductor", "asml",
        "arm holdings", "softbank", "sony",
        "baidu", "alibaba", "tencent", "bytedance",
        "infosys", "accenture", "sap",

        # ── Banks & financial services ──────────────────────────────────────
        "jpmorgan", "goldman sachs", "morgan stanley",
        "bank of america", "citigroup", "wells fargo",
        "blackrock", "berkshire", "charles schwab",
        "visa", "mastercard", "american express",

        # ── Healthcare / pharma ─────────────────────────────────────────────
        "pfizer", "moderna", "johnson & johnson", "unitedhealth",
        "eli lilly", "novo nordisk", "abbvie", "merck",

        # ── Energy ──────────────────────────────────────────────────────────
        "exxon", "chevron", "conocophillips",

        # ── Retail / consumer ───────────────────────────────────────────────
        "walmart", "costco", "target", "home depot",
        "starbucks", "mcdonalds", "nike", "coca cola", "pepsi",
        "procter gamble", "disney",

        # ── Semiconductors ──────────────────────────────────────────────────
        "semiconductor", "chip shortage", "chip act",
        "foundry", "fab", "wafer",
        "gpu", "cpu", "h100", "blackwell",

        # ── Business events ─────────────────────────────────────────────────
        "earnings", "revenue", "eps", "ipo", "merger",
        "acquisition", "nasdaq", "antitrust",
        "stock split", "buyback", "dividend",
    ]

    TICKER_PREFIX_MAP = {
        "kxaapl": "AAPL", "kxgoog": "GOOGL", "kxmsft": "MSFT",
        "kxamzn": "AMZN", "kxmeta": "META",  "kxnvda": "NVDA",
        "kxtsla": "TSLA", "kxintc": "INTC",  "kxamd":  "AMD",
        "kxqcom": "QCOM", "kxtsm":  "TSM",
    }

    TITLE_TICKER_MAP = {
        "apple": "AAPL",    "google": "GOOGL",   "microsoft": "MSFT",
        "amazon": "AMZN",   "nvidia": "NVDA",    "tesla": "TSLA",
        "intel": "INTC",    "amd": "AMD",         "qualcomm": "QCOM",
        "tsmc": "TSM",
        # v11.8d: removed "samsung": "005930.KS" — Finnhub 403s on Korean
        # tickers every scan. FinancialMarketsBot disabled anyway.
        "jpmorgan": "JPM",  "goldman sachs": "GS", "bank of america": "BAC",
        "pfizer": "PFE",    "exxon": "XOM",       "walmart": "WMT",
    }

    @property
    def sector_name(self) -> str:
        return "financial_markets"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False
        if _has_sports_prefix(market):
            return False
        # v12.2: Check blocklist for random sports leaking
        ticker = market.get("ticker", "").lower()
        if any(ticker.startswith(bl) for bl in self.FINMARKETS_BLOCKLIST):
            return False
        return _search_fields(market, self.KEYWORDS)

    def _detect_company(self, market: dict) -> str:
        et = market.get("event_ticker", "").lower()
        for prefix, symbol in self.TICKER_PREFIX_MAP.items():
            if et.startswith(prefix):
                return symbol
        title = market.get("title", "").lower()
        for name, symbol in self.TITLE_TICKER_MAP.items():
            if name in title:
                return symbol
        return "AAPL"

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        company  = self._detect_company(market)
        features, context = fetch_tech_features(company)
        title   = market.get("title", "")
        numbers = re.findall(r"[\d,]+\.?\d*", title.replace(",", ""))
        target  = float(numbers[0]) if numbers else 0.0
        features = np.append(features, target)
        context["contract_target"] = target
        return features, context

    def evaluate(self, market, news_signal=None):
        """v12.4: RESURRECTED with real Yahoo Finance-based model.
        
        Handles:
          - Stock price threshold markets (AAPL above $200, etc.)
          - Earnings beat/miss markets
          - Index level markets (S&P 500 above X, etc.)
          - VIX level markets
        """
        ticker = market.get("ticker", "")
        title = market.get("title", "").lower()
        
        try:
            mkt = get_market_data_client()
            snapshot = mkt.get_market_snapshot()
        except Exception as e:
            logger.warning("[financial_markets] Market data unavailable: %s", e)
            return None
        
        our_prob = None
        confidence = 0.55  # Base confidence
        rationale = ""
        
        # ── Stock Price Threshold Markets ────────────────────────────────────
        # Detect company
        company_symbol = self._detect_company(market)
        
        # Check if it's a price threshold market
        price_match = re.search(r"\$?([\d,]+(?:\.\d+)?)", title.replace(",", ""))
        has_above_below = any(kw in title for kw in ["above", "below", "exceed", "under", "over"])
        
        if price_match and has_above_below and company_symbol:
            threshold = float(price_match.group(1))
            above = any(kw in title for kw in ["above", "exceed", "over"])
            
            # Get current stock price
            quote = mkt.get_quote(company_symbol)
            if quote and quote.price > 0:
                current_price = quote.price
                
                # Determine timeframe from title
                if any(kw in title for kw in ["week", "7 day", "next week"]):
                    timeframe = "week"
                elif any(kw in title for kw in ["month", "30 day"]):
                    timeframe = "month"
                else:
                    timeframe = "day"
                
                comparison = "above" if above else "below"
                our_prob = mkt.predict_price_threshold(
                    company_symbol, threshold, comparison, timeframe
                )
                
                # VIX-adjusted confidence
                if snapshot.vix and snapshot.vix > 25:
                    confidence = 0.45  # High VIX = more uncertainty
                elif snapshot.vix and snapshot.vix < 15:
                    confidence = 0.62  # Low VIX = more predictable
                else:
                    confidence = 0.55
                
                # Boost confidence if near the threshold
                distance_pct = abs(current_price - threshold) / current_price * 100
                if distance_pct < 2:
                    confidence = min(0.70, confidence + 0.10)  # Near threshold = more edge
                
                rationale = f"{company_symbol}={current_price:.2f}_vs_{threshold:.2f}_{timeframe}_vix={snapshot.vix:.1f}"
                logger.info(
                    "[financial_markets] %s: %s $%.2f vs threshold $%.2f → P=%.1f%% (VIX=%.1f)",
                    ticker[:30], company_symbol, current_price, threshold,
                    our_prob * 100, snapshot.vix or 0,
                )
        
        # ── Earnings Markets ─────────────────────────────────────────────────
        elif any(kw in title for kw in ["earnings", "eps", "revenue", "beat", "miss"]):
            company_symbol = self._detect_company(market)
            
            if "beat" in title:
                direction = "beat"
            elif "miss" in title:
                direction = "miss"
            else:
                direction = "beat"  # Default to asking about beat
            
            our_prob = mkt.predict_earnings_move(company_symbol, 0, direction)
            
            # Earnings are hard to predict — lower confidence
            confidence = 0.48
            rationale = f"earnings_{direction}_{company_symbol}"
            logger.info(
                "[financial_markets] %s: earnings %s → P=%.1f%%",
                ticker[:35], direction, our_prob * 100,
            )
        
        # ── Index Level Markets ──────────────────────────────────────────────
        elif any(kw in title for kw in ["s&p", "sp500", "s&p 500", "dow", "nasdaq"]):
            # Detect index
            if "nasdaq" in title:
                index_symbol = "QQQ"
                current = snapshot.qqq_price or 0
            else:
                index_symbol = "SPY"
                current = snapshot.spy_price or 0
            
            # Extract threshold
            level_match = re.search(r"([\d,]+)", title.replace(",", ""))
            if level_match and current > 0:
                threshold = float(level_match.group(1))
                
                # Index levels are often quoted differently (SPY vs S&P 500)
                # S&P 500 ~5000 → SPY ~500
                if threshold > current * 5:
                    threshold = threshold / 10  # Likely S&P points, convert to ETF
                
                above = any(kw in title for kw in ["above", "exceed", "over"])
                comparison = "above" if above else "below"
                
                our_prob = mkt.predict_price_threshold(
                    index_symbol, threshold, comparison, "day"
                )
                
                # VIX adjustment
                if snapshot.vix and snapshot.vix > 25:
                    confidence = 0.42
                else:
                    confidence = 0.52
                
                rationale = f"{index_symbol}={current:.2f}_vs_{threshold:.2f}"
                logger.info(
                    "[financial_markets] %s: index %s vs %.2f → P=%.1f%%",
                    ticker[:35], index_symbol, threshold, our_prob * 100,
                )
            else:
                return None
        
        # ── VIX Level Markets ────────────────────────────────────────────────
        elif "vix" in title:
            vix_data = mkt.get_vix()
            if not vix_data:
                return None
            
            # Extract threshold
            level_match = re.search(r"(\d+\.?\d*)", title)
            if level_match:
                threshold = float(level_match.group(1))
                current = vix_data.current
                
                above = any(kw in title for kw in ["above", "exceed", "spike"])
                distance = current - threshold
                
                # VIX is mean-reverting, so adjust probability
                if above:
                    if distance > 5:
                        our_prob = 0.75  # Already well above
                    elif distance > 0:
                        our_prob = 0.55 + distance * 0.03
                    else:
                        # Mean reversion: VIX tends to spike
                        our_prob = 0.35 + abs(distance) * 0.01  
                else:
                    if distance < -5:
                        our_prob = 0.75
                    elif distance < 0:
                        our_prob = 0.55 + abs(distance) * 0.03
                    else:
                        our_prob = 0.40 - distance * 0.02
                
                confidence = 0.50  # VIX is volatile
                rationale = f"vix={current:.1f}_vs_{threshold:.1f}"
                logger.info(
                    "[financial_markets] %s: VIX %.1f vs %.1f → P=%.1f%%",
                    ticker[:35], current, threshold, our_prob * 100,
                )
            else:
                return None
        
        else:
            # Unhandled market type
            logger.debug("[financial_markets] %s: no handler", ticker[:35])
            return None
        
        if our_prob is None:
            return None
        
        # Clamp probability
        our_prob = max(0.05, min(0.95, our_prob))
        
        # Build BotSignal
        market_price = _market_prob(market)
        edge = our_prob - market_price
        direction = "yes" if our_prob > market_price else "no"
        
        return BotSignal(
            sector="financial_markets",
            ticker=ticker,
            our_prob=our_prob,
            market_prob=market_price,
            edge=edge,
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. SPORTS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class SportsBot(BaseBot):
    """
    US + international sports, esports, combat sports, Olympic events.

    v11.3: Added all international soccer, cricket, basketball, golf, hockey,
    track & field, and A-League prefixes to KEYWORDS and LEAGUE_MAP.

    v12.2: Added per-sport confidence scaling based on resolved Brier scores.
    """

    # ── Structural market blocklist ────────────────────────────────────────────
    _SINGLE_GAME_BLOCKLIST = (
        "kxmvecrosscategory",
        "kxmvecrosscat",
        "kxmvecross",
        "kxmvesportsmultigameextended",
        "kxmvesportsmultigame",
        "kxmvesportsmulti",
    )

    # ── v12.2: Per-sport confidence scaling based on historical Brier scores ───
    # Derived from resolved signals query 2026-04-13:
    #   NCAAMB: 0.1159 (7 resolved) → keep full confidence
    #   LALIGA: 0.0343 (2 resolved) → keep full
    #   CS2:    0.2141 (4 resolved) → minor scale
    #   MLB:    0.2267 (86 resolved) → was 0.918 avg_conf, scale to ~0.69
    #   NBA:    0.2960 (12 resolved) → worst major, significant haircut
    #   VALORANT: 0.2495 (16 resolved)
    #   LOL:    0.2521 (6 resolved)
    #   TENNIS: 0.2607 (7 resolved)
    SPORT_CONFIDENCE_SCALE = {
        # ── Strong performers (Brier < 0.15) ──────────────────────────────
        "NCAAMB": 1.00,     # 0.1159 Brier — excellent
        "LALIGA": 1.00,     # 0.0343 Brier (small sample)
        "ECUL":   1.00,     # 0.0362 Brier (small sample)
        "SOCCER": 0.95,     # 0.1525 Brier (1 resolved)

        # ── Decent performers (Brier 0.15-0.22) ───────────────────────────
        "ALEAGU": 0.90,     # 0.2160 Brier
        "CS2":    0.85,     # 0.2141 Brier

        # ── Volume drivers needing calibration (Brier 0.22-0.30) ──────────
        "MLB":      0.75,   # 0.2267 Brier, 86 resolved — highest volume
        "VALORANT": 0.65,   # 0.2495 Brier
        "LOL":      0.65,   # 0.2521 Brier
        "NCAABB":   0.65,   # 0.2559 Brier (KXNCAABB)
        "TENNIS":   0.60,   # 0.2607 Brier
        "BUNDES":   0.60,   # 0.2654 Brier
        "NCAAWB":   0.60,   # 0.2667 Brier (KXNCAAWB)
        "NBA":      0.55,   # 0.2960 Brier — worst major category

        # ── Unproven / low sample (conservative) ──────────────────────────
        "NHL":      0.60,
        "NFL":      0.60,
        "MMA":      0.50,
        "GOLF":     0.50,
        "PGA":      0.50,
    }

    # ── v12.2: Blocklist for catastrophic Brier (> 0.40) or noise markets ──────
    # These markets should not generate signals at all.
    SPORT_BLOCKLIST = (
        "kxbelgia",   # 0.5662 Brier — Belgian league garbage
        "kxjbleag",   # 0.4262 Brier — J.B.League
        "kxacbgam",   # 0.6045 Brier — ACB basketball
        "kxmvenba",   # 0.3243 Brier — MVE NBA (structural issue)
        "kxmvecbc",   # 4530 signals, 0 resolved, 0.32 conf — futures noise
        # v12.2: Game-level MLB markets — no real model, flat prior pollution
        "kxmlbtot",   # 31 resolved, 0.2360 Brier — game totals
        "kxmlbspr",   # 53 resolved, 0.2210 Brier — game spreads
        # v12.2: Game-level NBA markets — no real model, flat prior pollution
        "kxnbatotal",     # 707 signals, 5 resolved, 0.31 Brier
        "kxnbateamtotal", # 230 signals, 2 resolved
        "kxnbaspread",    # 695 signals
        "kxnba1hwinner",  # 29 signals — 1st half winner
        "kxnba2hwinner",  # 28 signals — 2nd half winner
        "kxnbagame",      # 134 signals — game props
        "kxnbamention",   # 222 signals — broadcast mentions (unmodelable)
        # v12.2: Esports — no real model, flat prior
        "kxvalorant",     # 19 resolved, 0.2404 Brier
        "kxlol",          # 9 resolved, 0.2272 Brier
        "kxcs2",          # 4 resolved, 0.21 Brier
        # v12.2: Bad NCAA — women's and generic college basketball
        "kxncaawb",       # 10 resolved, 0.2667 Brier — women's
        "kxncaabb",       # 6 resolved, 0.2559 Brier — generic CBB
        # v12.2: Tennis and soccer without real models
        "kxatpchall",     # 5 resolved, 0.2648 Brier — ATP challenger
        "kxatpsetwi",     # 2 resolved, 0.2503 Brier — ATP set winner
        "kxbundesli",     # 4 resolved, 0.2654 Brier — Bundesliga
        "kxaleagueg",     # 3 resolved, 0.2862 Brier — A-League games
        # v12.2: Random leagues leaking to wrong sectors — blocklist all
        "kxtabletennis",  # Table tennis — leaking to crypto/weather
        "kxkbogame",      # Korean baseball — leaking to weather
        "kxlnbelite",     # French basketball — leaking to weather
        "kxchnsl",        # Chinese Super League — leaking to weather
        "kxapfddh",       # Unknown league — leaking to weather
        "kxballerleague", # Baller League — leaking to crypto
        "kxitfwmatch",    # ITF women's tennis
        "kxitfmmatch",    # ITF men's tennis
    )

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxmvesports",
        "kxmvecbchampionship",
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        # ── Tennis ─────────────────────────────────────────────────────────
        "kxatp", "kxwta", "kxtennis", "kxabagame",        "kxf1", "kxolympic", "kxepl", "kxsoccer",
        "kxboxing", "kxwwe", "kxcricket", "kxrugby", "kxesport",
        "kxthail", "kxsl",
        # ── Esports ────────────────────────────────────────────────────────
        "kxow", "kxvalorant", "kxlol", "kxleague",
        "kxrl", "kxrocketleague",
        "kxdota", "kxintlf", "kxcs2", "kxcsgo",
        "kxapex", "kxfort",
        "kxegypt", "kxvenf",
        "kxr6g",           # Rainbow Six — v11.3

        # ── International soccer — v11.3 ──────────────────────────────────
        "kxarg",           # Argentine Primera División
        "kxlali",          # La Liga
        "kxbund",          # Bundesliga
        "kxseri",          # Serie A
        "kxliga",          # Liga generic
        "kxligu",          # Ligue 1
        "kxbras",          # Brasileirão
        "kxswis",          # Swiss Super League
        "kxbelg",          # Belgian Pro League
        "kxecul",          # Ecuadorian Liga Pro
        "kxpslg",          # PSL
        "kxsaud",          # Saudi Pro League
        "kxjlea",          # J-League
        "kxuclt",          # UEFA Champions League
        "kxuecl",          # v12.2: UEFA Europa Conference League
        "kxfifa",          # FIFA
        "kxintl",          # International sports
        "kxalea",          # A-League (Australia)
        "kxafl",           # AFL (Australian Football League)

        # ── Golf — v11.3 ──────────────────────────────────────────────────
        "kxpga",           # PGA Tour
        "kxlpga",          # v12.2: LPGA Tour
        "kxowga",          # Golf (was leaking into economics)

        # ── Cricket — v11.3 ───────────────────────────────────────────────
        "kxcba",           # CBA cricket

        # ── International basketball — v11.3 ──────────────────────────────
        "kxfiba",          # FIBA
        "kxacbg",          # ACB (Spanish)
        "kxvtbg",          # VTB League (Russian)
        "kxbalg",          # Baltic basketball
        "kxbbse",          # BBL / basketball
        "kxnpbg",          # NBP / basketball

        # ── Hockey — v11.3 ────────────────────────────────────────────────
        "kxahlg",          # AHL

        # ── Track & field — v11.3 ─────────────────────────────────────────
        "kxdima",          # Diamond League

        # ── US major leagues ───────────────────────────────────────────────
        "nba", "nfl", "mlb", "nhl", "mls", "ufc", "mma",
        "basketball", "football", "baseball", "hockey",
        "super bowl", "world series", "nba finals", "stanley cup",
        "nfl championship", "ncaa championship", "march madness",
        "nfl draft", "nba draft", "mlb trade deadline",

        # ── US teams ────────────────────────────────────────────────────────
        "lakers", "celtics", "warriors", "bulls", "knicks", "nets",
        "heat", "bucks", "suns", "nuggets", "clippers", "sixers",
        "patriots", "chiefs", "cowboys", "eagles", "packers",
        "49ers", "ravens", "steelers", "bills", "bengals",
        "yankees", "dodgers", "red sox", "cubs", "mets",
        "astros", "braves", "cardinals", "giants baseball",
        "rangers hockey", "bruins", "maple leafs", "canadiens",

        # ── Soccer / Football ──────────────────────────────────────────────
        "premier league", "la liga", "serie a", "bundesliga",
        "ligue 1", "champions league", "europa league",
        "conference league", "copa america", "euros", "euro 2024",
        "world cup", "fifa", "uefa", "mls cup",
        "arsenal", "chelsea", "liverpool", "manchester city",
        "manchester united", "tottenham", "real madrid", "barcelona",
        "atletico madrid", "bayern munich", "borussia dortmund",
        "juventus", "inter milan", "ac milan", "napoli",
        "psg", "paris saint germain", "ajax", "benfica",
        "porto", "celtic", "rangers",
        # v11.3 additions
        "argentine primera", "brasileirao", "j-league",
        "saudi pro league", "belgian pro league", "swiss super league",
        "a-league",

        # ── Tennis ─────────────────────────────────────────────────────────
        "wimbledon", "us open tennis", "french open", "roland garros",
        "australian open", "atp", "wta", "grand slam",
        "djokovic", "alcaraz", "sinner", "nadal", "federer",
        "swiatek", "sabalenka", "gauff",
        "atp set", "wta set", "set winner", "match winner tennis",

        # ── Golf ───────────────────────────────────────────────────────────
        "masters golf", "pga championship", "us open golf",
        "the open championship", "ryder cup", "pga tour", "liv golf",
        "tiger woods", "rory mcilroy", "scottie scheffler",

        # ── Combat sports ──────────────────────────────────────────────────
        "boxing", "heavyweight championship", "world title fight",
        "fury", "usyk", "canelo", "crawford", "spence",
        "wrestling", "wwe", "aew", "royal rumble", "wrestlemania",
        "bellator", "one championship",

        # ── Racing ─────────────────────────────────────────────────────────
        "formula 1", "f1", "grand prix", "monaco gp", "daytona 500",
        "indy 500", "nascar cup", "verstappen", "hamilton", "leclerc",

        # ── Esports ────────────────────────────────────────────────────────
        "league of legends", "valorant", "cs2", "counter strike",
        "dota 2", "overwatch", "overwatch league", "owl match",
        "rocket league", "apex legends",
        "fortnite tournament", "esports championship", "worlds lol",
        "majors cs2", "vlr valorant", "vct", "valorant champions",
        "rainbow six", "r6 siege",

        # ── Cricket — v11.3 ───────────────────────────────────────────────
        "cricket", "test match", "odi cricket", "ipl", "ashes",

        # ── Other sports ───────────────────────────────────────────────────
        "rugby world cup", "six nations", "rugby union",
        "olympic games", "summer olympics", "winter olympics",
        "tour de france", "cycling grand tour",
        "diamond league", "ahl hockey", "fiba basketball",

        # ── Generic sports terms ────────────────────────────────────────────
        "wins by", "points scored", "spread", "mvp",
        "playoff", "championship series",
        "nba finals", "stanley cup", "nfl championship",
        "world series finals", "ncaa championship",

        # ── v11.7: classifier audit additions ─────────────────────────────
        "kxconmebol",      # CONMEBOL (South American soccer)
        "kxchll",          # Chilean soccer
        "kxeuroleague",    # Euroleague basketball
        "kxekstraklasa",   # Polish Ekstraklasa
        "kxshl",           # Swedish Hockey League
        "kxurypd",         # Uruguayan Primera División
        "kxucl",           # UCL (broader than kxuclt)
        "kxsuperlig",      # Turkish Super Lig
        "kxegypl",         # Egyptian Premier League
        "kxitf",           # ITF tennis
        "kxipl",           # IPL cricket
        "kxt20",           # v11.8: T20 cricket internationals
        "kxscottishprem",  # Scottish Premiership
        "kxuel",           # UEFA Europa League
        "kxeredivisie",    # Eredivisie (Dutch)
        "kxmotogp",        # MotoGP
        "kxbbl",           # Big Bash League
        "kxkf",            # KF tour
        "kxallsvenskan",   # Allsvenskan (Swedish soccer)
        "kxargpremdiv",    # Argentine Premier Division
        "kxnextag",        # NextAG
        # v11.8d: new leagues 2026-04-12
        "kxdel",           # DEL (Deutsche Eishockey Liga)
        "kxkhl",           # KHL (Kontinental Hockey League)
        "kxjbleague",      # J.B.League (Japanese basketball)
        "kxkbl",           # KBL (Korean basketball)
        "kxkleague",       # K League (Korean soccer)
        # v12.2: "Next X" prediction markets
        "kxnext",          # KXNEXTNBACOACH, etc.
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        "kxmvesports": "NBA", "kxmvecbchampionship": "NCAAB",
        "kxepl": "EPL", "kxsoccer": "MLS",
        # Tennis
        "kxatp": "ATP", "kxwta": "WTA", "kxtennis": "ATP", "kxabagame": "ABA",
        "kxf1": "F1",
        "kxboxing": "BOXING", "kxgolf": "PGA",
        # Esports
        "kxow": "OWL", "kxvalorant": "VCT", "kxlol": "LOL",
        "kxleague": "LOL", "kxrl": "RL", "kxrocketleague": "RL",
        "kxdota": "DOTA2", "kxcs2": "CS2", "kxcsgo": "CS2",
        "kxr6g": "R6",
        # ── v11.3 additions ────────────────────────────────────────────────
        # International soccer
        "kxarg":  "ARGPD",
        "kxlali": "LALIGA",
        "kxbund": "BUNDESLIGA",
        "kxseri": "SERIEA",
        "kxliga": "LIGA",
        "kxligu": "LIGUE1",
        "kxbras": "BRASILEIRAO",
        "kxswis": "SWISS",
        "kxbelg": "BELGIAN",
        "kxecul": "ECUADORIAN",
        "kxpslg": "PSL",
        "kxsaud": "SAUDI",
        "kxjlea": "JLEAGUE",
        "kxuclt": "UCL",
        "kxuecl": "UECL",  # v12.2: UEFA Europa Conference League
        "kxfifa": "FIFA",
        "kxintl": "INTL",
        "kxalea": "ALEAGUE",
        "kxafl":  "AFL",
        # Golf
        "kxpga":  "PGA",
        "kxlpga": "LPGA",  # v12.2
        "kxowga": "PGA",
        # Cricket
        "kxcba":  "CRICKET",
        "kxt20":  "T20",       # v11.8
        # Basketball
        "kxfiba": "FIBA",
        "kxacbg": "ACB",
        "kxvtbg": "VTB",
        "kxbalg": "BALTIC",
        "kxbbse": "BBL",
        "kxnpbg": "NBP",
        # Hockey
        "kxahlg": "AHL",
        # Track
        "kxdima": "DIAMOND",
        # v11.8d: new leagues 2026-04-12
        "kxdel":  "DEL",
        "kxkhl":  "KHL",
        "kxjbleague": "JBLEAGUE",
        "kxkbl":  "KBL",
        "kxkleague": "KLEAGUE",
    }

    @property
    def sector_name(self) -> str:
        return "sports"

    # ─────────────────────────────────────────────────────────────────────
    # v12.2: SPORT DETECTION AND CONFIDENCE SCALING
    # ─────────────────────────────────────────────────────────────────────

    def _detect_sport_code(self, ticker: str) -> str:
        """Extract sport code from ticker for confidence scaling."""
        tk = ticker.upper()

        prefix_map = [
            ("KXMLB", "MLB"),
            ("KXNBA", "NBA"),
            ("KXNFL", "NFL"),
            ("KXNHL", "NHL"),
            ("KXNCAAMB", "NCAAMB"),
            ("KXNCAAWB", "NCAAWB"),
            ("KXNCAABB", "NCAABB"),
            ("KXNCAA", "NCAAMB"),
            ("KXVALORANT", "VALORANT"),
            ("KXLOL", "LOL"),
            ("KXCS2", "CS2"),
            ("KXCSGO", "CS2"),
            ("KXATP", "TENNIS"),
            ("KXWTA", "TENNIS"),
            ("KXTENNIS", "TENNIS"),
            ("KXLALI", "LALIGA"),
            ("KXLALIGA", "LALIGA"),
            ("KXBUND", "BUNDES"),
            ("KXBUNDES", "BUNDES"),
            ("KXSERI", "SERIEA"),
            ("KXEPL", "EPL"),
            ("KXSOCCER", "SOCCER"),
            ("KXECUL", "ECUL"),
            ("KXALEA", "ALEAGU"),
            ("KXPGA", "PGA"),
            ("KXGOLF", "GOLF"),
            ("KXUFC", "MMA"),
            ("KXMMA", "MMA"),
            ("KXMVENBA", "MVENBA"),
            ("KXMVECBC", "MVECBC"),
        ]

        for prefix, code in prefix_map:
            if tk.startswith(prefix):
                return code

        return "OTHER"

    def _is_blocklisted_sport(self, ticker: str) -> bool:
        """Check if ticker matches any blocklisted sport prefix."""
        tk = ticker.lower()
        return any(tk.startswith(prefix) for prefix in self.SPORT_BLOCKLIST)

    def _scale_confidence(self, ticker: str, raw_conf: float) -> float:
        """Apply per-sport confidence scaling based on historical Brier."""
        if self._is_blocklisted_sport(ticker):
            return 0.0

        sport_code = self._detect_sport_code(ticker)
        scale = self.SPORT_CONFIDENCE_SCALE.get(sport_code, 0.50)
        return raw_conf * scale

    # ─────────────────────────────────────────────────────────────────────
    # v11.4: MLB PLAYER PROP ROUTING
    # ─────────────────────────────────────────────────────────────────────

    def _try_mlb_player_prop(self, market: dict):
        """Route MLB player prop markets through the new hit-rate model."""
        ticker = market.get("ticker", "")
        if not ticker.startswith("KXMLB"):
            return False, None

        if is_mlb_non_player_prop_market(ticker):
            return False, None

        parsed = parse_mlb_ticker(ticker)
        if parsed is None:
            return False, None

        team_info = MLB_TEAMS.get(parsed.player_team_code)
        if not team_info:
            logger.debug("[sports/mlb] unknown team code %s", parsed.player_team_code)
            return False, None
        team_id = team_info[0]

        player = lookup_player(
            team_id, parsed.player_first, parsed.player_last, parsed.player_jersey,
        )
        if not player:
            logger.debug("[sports/mlb] player not found: %s %s #%s on %s",
                         parsed.player_first, parsed.player_last,
                         parsed.player_jersey, parsed.player_team_code)
            return False, None

        home_team_info = MLB_TEAMS.get(parsed.home_team_code)
        parsed.home_team_id = home_team_info[0] if home_team_info else None

        season = rolling = pitcher = opp_pitcher_stats = None

        if parsed.prop_code == "KS":
            pitcher = fetch_pitcher_stats(player.player_id)
            if not pitcher or pitcher.innings <= 0:
                return False, None
            prediction = predict_mlb_prop(parsed, None, None, pitcher)
        else:
            season = fetch_batter_stats(player.player_id)
            if not season or season.plate_apps <= 0:
                return False, None
            rolling = fetch_batter_rolling(player.player_id, n_games=10)

            try:
                opp_code = (parsed.away_team_code
                            if parsed.player_team_code == parsed.home_team_code
                            else parsed.home_team_code)
                opp_team_info = MLB_TEAMS.get(opp_code)
                if opp_team_info:
                    game_date = _extract_mlb_game_date(ticker)
                    if game_date:
                        opp_pitcher_stats = fetch_opposing_pitcher(game_date, opp_team_info[0])
                        if opp_pitcher_stats:
                            logger.debug(
                                "[sports/mlb] %s (%s) facing pitcher (%s) K/9=%.1f",
                                player.full_name,
                                getattr(season, 'bat_hand', '?'),
                                getattr(opp_pitcher_stats, 'hand', '?'),
                                opp_pitcher_stats.k_per_9,
                            )
            except Exception as e:
                logger.debug("[sports/mlb] probable pitcher lookup failed: %s", e)

            prediction = predict_mlb_prop(parsed, season, rolling, opp_pitcher_stats)

        if not prediction:
            return False, None

        market_prob = _market_prob(market)
        our_prob    = prediction.prob_yes
        edge        = abs(our_prob - market_prob)
        direction   = "YES" if our_prob > market_prob else "NO"

        try:
            from shared.mlb_stats_fetcher import PARK_FACTORS
            _park = PARK_FACTORS.get(parsed.home_team_id, 1.0) if parsed.home_team_id else 1.0
            _fv = mlb_extract_features(
                parsed, season, rolling,
                opp_pitcher=(pitcher if parsed.prop_code == "KS" else opp_pitcher_stats),
                park_factor=_park,
                market_prob=market_prob,
            )
            if _fv:
                mlb_log_features(ticker, player.full_name, _fv, our_prob)
        except Exception:
            pass

        logger.debug(
            "[sports/mlb] %s | %s | our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
            ticker, player.full_name, our_prob, market_prob,
            our_prob - market_prob, direction, prediction.confidence,
        )

        if edge < MIN_EDGE_PCT:
            return True, None

        scaled_conf = self._scale_confidence(ticker, prediction.confidence)
        if scaled_conf <= 0.0:
            logger.debug("[sports] %s blocklisted", ticker)
            return True, None

        return True, BotSignal(
            sector      = self.sector_name,
            prob        = our_prob,
            confidence  = scaled_conf,
            market_prob = market_prob,
            brier_score = 0.10,
        )

    def evaluate(self, market, news_signal=None):
        """Override BaseBot.evaluate() to try player prop models first."""
        ticker = market.get("ticker", "")

        if self._is_blocklisted_sport(ticker):
            logger.debug("[sports] %s blocklisted (catastrophic Brier)", ticker)
            return None

        handled, signal = self._try_mlb_player_prop(market)
        if handled:
            return signal

        handled, signal = self._try_nba_player_prop(market)
        if handled:
            return signal

        return None

    # ─────────────────────────────────────────────────────────────────────
    # v11.6 + v12: NBA PLAYER PROP ROUTING + FEATURE LOGGING
    # ─────────────────────────────────────────────────────────────────────

    def _try_nba_player_prop(self, market: dict):
        """Route NBA player prop markets through the Poisson/Binomial model."""
        ticker = market.get("ticker", "")
        if not ticker.upper().startswith("KXNBA"):
            return False, None

        if is_nba_non_player_prop_market(ticker):
            return False, None

        parsed = parse_nba_ticker(ticker)
        if parsed is None:
            logger.debug("[sports/nba] PARSE FAIL: %s", ticker)
            return False, None

        team_info = NBA_TEAMS_MAP.get(parsed.player_team_code)
        if not team_info:
            logger.debug("[sports/nba] unknown team: %s in %s", parsed.player_team_code, ticker)
            return False, None
        team_id = team_info[0]

        player = nba_lookup_player(
            team_id, parsed.player_first, parsed.player_last, parsed.player_jersey,
            team_code=parsed.player_team_code,
        )
        if not player:
            logger.info("[sports/nba] PLAYER NOT FOUND: %s.%s #%s (%s) — %s",
                         parsed.player_first, parsed.player_last,
                         parsed.player_jersey, parsed.player_team_code, ticker)
            return False, None

        season = nba_fetch_player_stats(player.player_id)
        if not season or season.games_played <= 0:
            logger.info("[sports/nba] NO STATS for %s (ESPN ID %d)",
                         player.full_name, player.player_id)
            return False, None

        rolling = nba_fetch_player_rolling(player.player_id, n_games=10)

        prediction = predict_nba_prop(parsed, season, rolling)
        if not prediction:
            logger.debug("[sports/nba] MODEL FAIL for %s %s ≥%d",
                         player.full_name, parsed.prop_code, parsed.threshold)
            return False, None

        market_prob = _market_prob(market)
        our_prob    = prediction.prob_yes
        edge        = abs(our_prob - market_prob)
        direction   = "YES" if our_prob > market_prob else "NO"

        try:
            nba_features = nba_extract_features(
                parsed=parsed,
                season_stats=season,
                rolling_stats=rolling,
                poisson_prob=our_prob,
                market_prob=market_prob,
            )
            nba_log_features(nba_features)
        except Exception as e:
            logger.debug("[sports/nba] feature logging failed: %s", e)

        logger.debug(
            "[sports/nba] %s | %s | our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
            ticker, player.full_name, our_prob, market_prob,
            our_prob - market_prob, direction, prediction.confidence,
        )

        if edge < MIN_EDGE_PCT:
            return True, None

        scaled_conf = self._scale_confidence(ticker, prediction.confidence)
        if scaled_conf <= 0.0:
            logger.debug("[sports] %s blocklisted", ticker)
            return True, None

        return True, BotSignal(
            sector      = self.sector_name,
            prob        = our_prob,
            confidence  = scaled_conf,
            market_prob = market_prob,
            brier_score = 0.10,
        )

    # ─────────────────────────────────────────────────────────────────────

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False

        et = market.get("event_ticker", "").lower()
        tk = market.get("ticker", "").lower()

        if any(et.startswith(p) or tk.startswith(p) for p in self._SINGLE_GAME_BLOCKLIST):
            return False

        return _search_fields(market, self.KEYWORDS)

    def _extract_teams_and_league(self, market: dict) -> tuple[str, str, str]:
        et     = market.get("event_ticker", "").lower()
        league = "NBA"

        for prefix, lg in self.LEAGUE_MAP.items():
            if et.startswith(prefix):
                league = lg
                break

        title = market.get("title", "")
        title_lower = title.lower()

        if "nfl" in title_lower or "touchdown" in title_lower:
            league = "NFL"
        elif "mlb" in title_lower or "home run" in title_lower:
            league = "MLB"
        elif "nhl" in title_lower or " goal" in title_lower:
            league = "NHL"
        elif "ncaa" in title_lower or "college" in title_lower:
            league = "NCAAB"
        elif any(t in title_lower for t in ["premier league", "champions league", "la liga", "serie a", "bundesliga"]):
            league = "EPL"
        elif "f1" in title_lower or "formula 1" in title_lower:
            league = "F1"
        elif "tennis" in title_lower or "wimbledon" in title_lower or "atp" in title_lower:
            league = "ATP"
        elif "valorant" in title_lower or "vct" in title_lower:
            league = "VCT"
        elif "overwatch" in title_lower or "owl" in title_lower:
            league = "OWL"
        elif "rocket league" in title_lower:
            league = "RL"

        team_a, team_b = "Team A", "Team B"
        if " vs " in title:
            parts  = title.split(" vs ")
            team_a = parts[0].strip().split()[-1]
            team_b = parts[1].strip().split()[0]
        elif " beat " in title.lower():
            idx    = title.lower().index(" beat ")
            team_a = title[:idx].split()[-1]
            team_b = title[idx + 6:].split()[0]
        elif "wins" in title_lower:
            clean  = title.replace("yes ", "").replace("no ", "")
            words  = clean.split()
            team_a = words[0] if words else "Team A"

        return team_a, team_b, league

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        team_a, team_b, league = self._extract_teams_and_league(market)
        features, context = fetch_sports_features(team_a, team_b, league)
        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        ou_line = float(numbers[0]) if numbers else 0.0
        features = np.append(features, ou_line)
        context["ou_line"] = ou_line
        return features, context


# ═══════════════════════════════════════════════════════════════════════════════
#  7. GLOBAL EVENTS BOT  (v11.7 — replaces EntertainmentBot)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalEventsBot(BaseBot):
    """
    Alpha bucket — entertainment, AI milestones, product launches, misc events.

    v11.7: Replaces EntertainmentBot. Absorbs entertainment markets AND catches
    everything that doesn't fit the core 6 sectors. This is the "everything
    else" bucket where new market types land until they get a dedicated model.

    Subsectors:
      - Entertainment: Spotify streaming, Netflix, box office, awards
      - AI / Tech launches: OpenAI releases, product launches, WWDC, CES
      - Miscellaneous: anything not sports/crypto/weather/econ/politics/finance

    No model yet — evaluate() returns None. The value is CLAIMING these
    markets so they don't bleed into other sectors and corrupt calibration.
    """

    KEYWORDS = [
        # ── Entertainment / streaming ───────────────────────────────────────
        "kxspotify",     # Spotify chart positions
        "kxspotstream",  # Spotify streaming
        "kxbox",         # Box office
        "kxnetflix",     # Netflix
        "kxhbo",         # HBO
        "kxgrammy",
        "kxoscar",
        "kxemmy",
        "kxgoldenglobe",
        "kxbillboard",
        "kxartiststream",  # v12.1: Spotify artist streaming

        # ── AI / tech launches ──────────────────────────────────────────────
        "kxtech",        # General tech events
        "kxai",          # AI milestones
        "kxopenai",      # OpenAI releases
        "kxanthropic",   # Anthropic releases
        "openai", "anthropic", "chatgpt", "gpt-4", "gpt-5",
        "claude", "gemini", "grok", "llama", "mistral",
        "deepmind", "stability ai", "midjourney", "dall-e", "sora",
        "artificial intelligence", "large language model", "llm",
        "agi", "ai regulation", "ai safety", "foundation model",
        "product launch", "wwdc", "google io", "microsoft build",
        "ces", "mwc", "tech layoffs",
    ]

    @property
    def sector_name(self) -> str:
        return "global_events"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _has_sports_prefix(market):
            return False
        # Claim entertainment markets
        if _is_entertainment_market(market):
            return True
        return _search_fields(market, self.KEYWORDS)

    def evaluate(self, market, news_signal=None):
        """v11.7: scaffold — claim the market, return no signal.

        Prevents global event markets from polluting other sectors
        without producing flat-prior garbage signals.
        """
        if not self.is_relevant(market):
            return None
        ticker = market.get("ticker", "")
        logger.debug(
            "[global_events] %s claimed (no model yet)",
            ticker,
        )
        return None

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        return np.array([0.0]), {}


# ── Convenience factory ───────────────────────────────────────────────────────

def all_bots() -> list[BaseBot]:
    """
    Return all active bots.
    
    v12.5: Qualitative sectors (politics, economics, financial_markets, 
    global_events) now use LLMBot with Claude API + web search.
    
    Quantitative sectors (sports, weather, crypto) keep their specialized models.
    """
    from shared.llm_bot import LLMBot
    
    return [
        # Quantitative bots — keep specialized models
        CryptoBot(),
        WeatherBot(),
        SportsBot(),
        # Qualitative bot — LLM with web search handles:
        # politics, economics, financial_markets, global_events
        LLMBot(),
    ]