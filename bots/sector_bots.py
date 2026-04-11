"""
bots/sector_bots.py  (v11.5 — MENTION skip + EntertainmentBot scaffold + leak fixes)

Changes vs v11.4:
  Bug fixes:
    - CryptoBot: kxspot → kxspotstream (was matching KXSPOTIFY incorrectly)
    - SPORTS_PREFIXES: added kxufl (United Football League leaking to weather)
    - CryptoBot KEYWORDS: added kxshiba (SHIB token explicit)

  New helpers:
    - _is_unmodelable_market(): returns True for MENTION markets and Survivor.
      Catches ~460 noise signals/day. Every bot now checks this first.
    - _is_entertainment_market(): returns True for KXSPOTIFY/KXNETFLIX/etc.
      Routes them to the new EntertainmentBot.

  New bot:
    - EntertainmentBot: scaffold only. Claims Spotify/Netflix/awards markets
      so they stop polluting other sectors. evaluate() returns None until
      a real model is built (next session).

  All bots now check _is_unmodelable_market() and _is_entertainment_market()
  at the top of is_relevant() so noise markets are skipped before any
  feature fetching or model evaluation happens.

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

from bots.base_bot import BaseBot
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
    """
    SPORTS_PREFIXES = (
        # ── KXMVE family ──────────────────────────────────────────────────
        "kxmve",

        # ── US major sports ───────────────────────────────────────────────
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
        "kxufc", "kxncaa", "kxcbb", "kxcfb",

        # ── Tennis ────────────────────────────────────────────────────────
        "kxatp", "kxwta", "kxtennis",

        # ── Golf ──────────────────────────────────────────────────────────
        # v11.3: kxpga was missing → bled into crypto ("pga" no match but
        # ticker substring issues) and weather (golfer city names).
        # kxowga was missing → bled into economics.
        "kxgolf", "kxpga", "kxowga",

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
        "kxfifa",      # FIFA

        # ── Cricket ───────────────────────────────────────────────────────
        # v11.3: kxcba was missing → cricket bled into weather AND crypto.
        # KXCBAGAME matched WeatherBot city keywords, got 0.92 NOAA priors
        # applied to cricket matches. All 6 "resolved" weather outcomes
        # were cricket → Brier 0.4269.
        "kxcba", "kxcricket",

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

        # ── Entertainment / unmodelable (block from all sector bots) ──────
        # v11.3: Survivor TV show (429 signals) matched WeatherBot via city
        # keywords in episode titles. No bot can model this, so blocking it
        # here causes it to be skipped entirely (correct behavior).
        "kxsurv",      # Survivor TV show mentions
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

    These get claimed by EntertainmentBot. Listed separately from
    _is_unmodelable_market because EntertainmentBot WILL eventually
    have a real model — for now it just claims the markets to keep
    them out of crypto/weather/economics.
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
        return _search_fields(market, self.KEYWORDS)

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        features, context = fetch_economics_features()
        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        target  = float(numbers[0]) if numbers else 0.0
        features = np.append(features, target)
        context["contract_target"] = target
        return features, context


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


# ═══════════════════════════════════════════════════════════════════════════════
#  5. TECH BOT
# ═══════════════════════════════════════════════════════════════════════════════

class TechBot(BaseBot):
    """
    Earnings, M&A, AI milestones, product launches — worldwide tech companies.
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
        "kxtsla", "kxearnings", "kxtech", "kxai", "kxipo",
        "kxopenai", "kxanthropic", "kxnasdaq", "kxsemiconductor",
        "kxintc", "kxamd", "kxqcom", "kxtsm", "kxsamsng",

        # ── US Big Tech ─────────────────────────────────────────────────────
        "apple", "google", "microsoft", "amazon", "nvidia", "tesla",
        "meta", "netflix", "adobe", "salesforce", "oracle", "ibm",
        "intel", "amd", "qualcomm", "broadcom", "texas instruments",
        "micron", "western digital", "seagate", "hp", "dell",
        "paypal", "square", "stripe", "shopify", "uber", "lyft",
        "airbnb", "doordash", "instacart", "twitter", "snap",
        "pinterest", "reddit", "discord",

        # ── Global Tech ─────────────────────────────────────────────────────
        "samsung", "tsmc", "taiwan semiconductor", "asml",
        "arm holdings", "arm", "softbank", "sony", "panasonic",
        "toshiba", "hitachi", "fujitsu", "nec", "nintendo",
        "baidu", "alibaba", "tencent", "bytedance", "tiktok",
        "huawei", "xiaomi", "oppo", "dji", "lenovo",
        "infosys", "wipro", "tata consultancy", "accenture",
        "capgemini", "sap", "siemens digital", "ericsson", "nokia",

        # ── AI models and companies ─────────────────────────────────────────
        "openai", "anthropic", "gpt", "chatgpt", "gpt-4", "gpt-5",
        "claude", "gemini", "grok", "llama", "mistral",
        "deepmind", "google deepmind", "stability ai",
        "midjourney", "dall-e", "sora", "artificial intelligence",
        "large language model", "llm", "agi", "ai regulation",
        "ai safety", "foundation model",

        # ── Semiconductors ─────────────────────────────────────────────────
        "semiconductor", "chip shortage", "chip act",
        "foundry", "fab", "wafer", "moore's law",
        "gpu", "cpu", "tpu", "npu", "h100", "blackwell",

        # ── Business events ─────────────────────────────────────────────────
        "earnings", "revenue", "eps", "ipo", "merger",
        "acquisition", "nasdaq", "antitrust", "regulation",
        "product launch", "wwdc", "google io", "microsoft build",
        "ces", "mwc", "tech layoffs",
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
        "tsmc": "TSM",      "samsung": "005930.KS",
    }

    @property
    def sector_name(self) -> str:
        return "tech"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        if _is_entertainment_market(market):
            return False
        if _has_sports_prefix(market):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  6. SPORTS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class SportsBot(BaseBot):
    """
    US + international sports, esports, combat sports, Olympic events.

    v11.3: Added all international soccer, cricket, basketball, golf, hockey,
    track & field, and A-League prefixes to KEYWORDS and LEAGUE_MAP.
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

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxmvesports",
        "kxmvecbchampionship",
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        # ── Tennis ─────────────────────────────────────────────────────────
        "kxatp", "kxwta", "kxtennis",
        "kxf1", "kxolympic", "kxepl", "kxsoccer",
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
        "kxfifa",          # FIFA
        "kxintl",          # International sports
        "kxalea",          # A-League (Australia)

        # ── Golf — v11.3 ──────────────────────────────────────────────────
        "kxpga",           # PGA Tour
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
        "kxscottishprem",  # Scottish Premiership
        "kxuel",           # UEFA Europa League
        "kxeredivisie",    # Eredivisie (Dutch)
        "kxmotogp",        # MotoGP
        "kxbbl",           # Big Bash League
        "kxkf",            # KF tour
        "kxallsvenskan",   # Allsvenskan (Swedish soccer)
        "kxargpremdiv",    # Argentine Premier Division
        "kxnextag",        # NextAG
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        "kxmvesports": "NBA", "kxmvecbchampionship": "NCAAB",
        "kxepl": "EPL", "kxsoccer": "MLS",
        # Tennis
        "kxatp": "ATP", "kxwta": "WTA", "kxtennis": "ATP",
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
        "kxfifa": "FIFA",
        "kxintl": "INTL",
        "kxalea": "ALEAGUE",
        # Golf
        "kxpga":  "PGA",
        "kxowga": "PGA",
        # Cricket
        "kxcba":  "CRICKET",
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
    }

    @property
    def sector_name(self) -> str:
        return "sports"

    # ─────────────────────────────────────────────────────────────────────
    # v11.4: MLB PLAYER PROP ROUTING
    # ─────────────────────────────────────────────────────────────────────

    def _try_mlb_player_prop(self, market: dict):
        """Route MLB player prop markets through the new hit-rate model.

        Returns a tuple (handled, signal):
          handled=False → couldn't process, caller should fall through to base class
          handled=True, signal=None → processed successfully but edge too small, SKIP
          handled=True, signal=BotSignal → processed successfully, emit signal

        The critical invariant: once we successfully compute a prediction, we
        NEVER fall through to the flat-prior base class path. A "no edge"
        answer from a real model is still correct; substituting a flat prior
        for it would corrupt the signal.
        """
        ticker = market.get("ticker", "")
        if not ticker.startswith("KXMLB"):
            return False, None

        # Game-level MLB markets (TOTAL, SPREAD, F5, futures) → base class path
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

        # Fetch stats — different path for batters vs pitchers
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

            # v11.7: Wire opposing pitcher context into batter predictions
            opp_pitcher_stats = None
            try:
                # Determine opponent team
                opp_code = (parsed.away_team_code
                            if parsed.player_team_code == parsed.home_team_code
                            else parsed.home_team_code)
                opp_team_info = MLB_TEAMS.get(opp_code)
                if opp_team_info:
                    # Extract game date from ticker: KXMLB...-26APR092140COLSD-...
                    game_date = _extract_mlb_game_date(ticker)
                    if game_date:
                        opp_pitcher_stats = fetch_opposing_pitcher(game_date, opp_team_info[0])
                        if opp_pitcher_stats:
                            logger.info(
                                "[sports/mlb] %s facing pitcher K/9=%.1f",
                                player.full_name, opp_pitcher_stats.k_per_9,
                            )
            except Exception as e:
                logger.debug("[sports/mlb] probable pitcher lookup failed: %s", e)

            prediction = predict_mlb_prop(parsed, season, rolling, opp_pitcher_stats)

        if not prediction:
            return False, None

        # ── We have a valid model prediction — COMMIT to it ──────────────
        from bots.base_bot import _market_prob
        market_prob = _market_prob(market)
        our_prob    = prediction.prob_yes
        edge        = abs(our_prob - market_prob)
        direction   = "YES" if our_prob > market_prob else "NO"

        # Log EVERY prediction so we can see the model running on every market,
        # not just ones that cross the edge threshold.
        logger.info(
            "[sports/mlb] %s | %s | our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
            ticker, player.full_name, our_prob, market_prob,
            our_prob - market_prob, direction, prediction.confidence,
        )

        # Below edge threshold: return (handled, None) so we skip cleanly
        # without falling through to the flat-prior path.
        if edge < MIN_EDGE_PCT:
            return True, None

        return True, BotSignal(
            sector      = self.sector_name,
            prob        = our_prob,
            confidence  = prediction.confidence,
            market_prob = market_prob,
            brier_score = 0.10,
        )

    def evaluate(self, market, news_signal=None):
        """Override BaseBot.evaluate() to try player prop models first.

        Order: MLB → NBA → silence.
        If neither model can handle the market, return None instead of
        falling through to the flat-prior BayesianPolyModel which produces
        garbage our_p=0.452 on every unmodeled sport.
        """
        handled, signal = self._try_mlb_player_prop(market)
        if handled:
            return signal

        handled, signal = self._try_nba_player_prop(market)
        if handled:
            return signal

        # v11.7: No model for this market → stay silent.
        # Don't fall through to base class flat prior.
        return None

    # ─────────────────────────────────────────────────────────────────────
    # v11.6: NBA PLAYER PROP ROUTING
    # ─────────────────────────────────────────────────────────────────────

    def _try_nba_player_prop(self, market: dict):
        """Route NBA player prop markets through the Poisson/Binomial model.

        Same (handled, signal) tuple pattern as MLB:
          handled=False → couldn't process, fall through to base class
          handled=True, signal=None → processed but edge too small, skip
          handled=True, signal=BotSignal → processed, emit signal
        """
        ticker = market.get("ticker", "")
        if not ticker.upper().startswith("KXNBA"):
            return False, None

        # Game-level NBA markets → base class path
        if is_nba_non_player_prop_market(ticker):
            return False, None

        parsed = parse_nba_ticker(ticker)
        if parsed is None:
            logger.info("[sports/nba] PARSE FAIL: %s", ticker)
            return False, None

        team_info = NBA_TEAMS_MAP.get(parsed.player_team_code)
        if not team_info:
            logger.info("[sports/nba] unknown team: %s in %s", parsed.player_team_code, ticker)
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
            logger.info("[sports/nba] MODEL FAIL for %s %s ≥%d",
                         player.full_name, parsed.prop_code, parsed.threshold)
            return False, None

        # ── We have a valid model prediction — COMMIT to it ──────────────
        from bots.base_bot import _market_prob
        market_prob = _market_prob(market)
        our_prob    = prediction.prob_yes
        edge        = abs(our_prob - market_prob)
        direction   = "YES" if our_prob > market_prob else "NO"

        logger.info(
            "[sports/nba] %s | %s | our_p=%.3f mkt_p=%.3f edge=%+.3f dir=%s conf=%.2f",
            ticker, player.full_name, our_prob, market_prob,
            our_prob - market_prob, direction, prediction.confidence,
        )

        if edge < MIN_EDGE_PCT:
            return True, None

        return True, BotSignal(
            sector      = self.sector_name,
            prob        = our_prob,
            confidence  = prediction.confidence,
            market_prob = market_prob,
            brier_score = 0.10,
        )

    # ─────────────────────────────────────────────────────────────────────

    def is_relevant(self, market: dict) -> bool:
        # ── v11.5 guards ──────────────────────────────────────────────────
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
#  7. ENTERTAINMENT BOT  (v11.5 — scaffold, no model yet)
# ═══════════════════════════════════════════════════════════════════════════════

class EntertainmentBot(BaseBot):
    """
    Entertainment / streaming / awards markets — Spotify, Netflix, awards.

    v11.5: SCAFFOLD ONLY. This bot CLAIMS entertainment markets to keep them
    out of crypto (KXSPOTIFY was matching crypto's kxspot keyword). It does
    NOT yet produce predictions — evaluate() returns None until a real model
    is built.

    Why this exists despite having no model:
      - Prevents KXSPOTIFY → crypto misclassification (7 signals/day)
      - Reserves the "entertainment" sector slot in the calibration table
      - Documents which prefixes belong here for future model work
      - Logs claimed markets so we can size the opportunity

    Future model ideas:
      - Spotify chart positions: predict via current rank + 7-day velocity
        (Spotify Charts API is free, no auth)
      - Box office: opening weekend predictions from BoxOfficeMojo trends
      - Awards shows: nominee count + Vegas odds aggregator
    """

    KEYWORDS = [
        "kxspotify",     # Spotify chart positions
        "kxspotstream",  # Spotify streaming markets (v11.7: moved from crypto)
        "kxbox",         # Box office
        "kxnetflix",     # Netflix
        "kxhbo",         # HBO
        "kxgrammy",
        "kxoscar",
        "kxemmy",
        "kxgoldenglobe",
        "kxbillboard",
    ]

    @property
    def sector_name(self) -> str:
        return "entertainment"

    def is_relevant(self, market: dict) -> bool:
        if _is_unmodelable_market(market):
            return False
        return _is_entertainment_market(market)

    def evaluate(self, market, news_signal=None):
        """v11.5: scaffold only — log the claim, return no signal.

        This prevents entertainment markets from polluting other sectors
        without producing flat-prior garbage signals.
        """
        if not self.is_relevant(market):
            return None
        ticker = market.get("ticker", "")
        logger.info(
            "[entertainment] %s claimed (no model yet — scaffold)",
            ticker,
        )
        return None

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
        # Stub — never called because evaluate() returns early
        return np.array([0.0]), {}


# ── Convenience factory ───────────────────────────────────────────────────────

def all_bots() -> list[BaseBot]:
    return [
        EconomicsBot(),
        CryptoBot(),
        PoliticsBot(),
        WeatherBot(),
        TechBot(),
        SportsBot(),
        EntertainmentBot(),  # v11.5
    ]