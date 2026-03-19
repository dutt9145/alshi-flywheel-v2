"""
bots/sector_bots.py  (v3 — KXMVE multivariate market matching)

Fixes vs v2:
  1. SportsBot now matches KXMVESPORTSMULTIGAMEEXTENDED and KXMVECROSSCATEGORY
     — these are the dominant liquid market types on Kalshi right now
       (March Madness + NBA season = ~2300 of the 2302 priced markets)
     — previously none of the bots matched any KXMVE prefix so all 2302
       liquid markets were invisible to every bot
  2. Added kxmvesports, kxmvesportsmulti, kxmvecross to SportsBot keywords
  3. All other bots unchanged — they'll activate when their markets
     get liquidity (economics events, crypto price markets etc.)

Kalshi KXMVE format:
  KXMVESPORTSMULTIGAMEEXTENDED  → multi-game sports parlay
  KXMVECROSSCATEGORY            → cross-team spread combos (still sports)
  KXMVECROSSCAT                 → same family, shorter prefix variant
"""

import logging
import re
from typing import Optional

import numpy as np

from bots.base_bot import BaseBot
from shared.data_fetchers import (
    fetch_economics_features, fetch_crypto_features, fetch_politics_features,
    fetch_weather_features, fetch_tech_features, fetch_sports_features,
)

logger = logging.getLogger(__name__)


def _search_fields(market: dict, keywords: list[str]) -> bool:
    """
    Search across all useful Kalshi market fields for any keyword match.
    Checks: event_ticker, ticker, title, subtitle.
    """
    haystack = " ".join([
        market.get("event_ticker", ""),
        market.get("ticker", ""),
        market.get("title", ""),
        market.get("subtitle", ""),
    ]).lower()
    return any(kw.lower() in haystack for kw in keywords)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ECONOMICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class EconomicsBot(BaseBot):
    """
    Focuses on macro-economic resolution markets:
    - "Will CPI be above X% this month?"
    - "Will the Fed raise rates at the next meeting?"
    - "Will unemployment exceed X%?"
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes
        "kxcpi", "kxfed", "kxfomc", "kxgdp", "kxjobs", "kxunrate",
        "kxpce", "kxnfp", "kxyield", "kxrate", "kxinfl",
        # Human-readable fallback
        "cpi", "inflation", "fed", "unemployment", "nonfarm",
        "payroll", "gdp", "recession", "rate hike", "rate cut",
        "pce", "fomc", "interest rate", "treasury", "yield",
        "federal reserve", "basis points",
    ]

    @property
    def sector_name(self) -> str:
        return "economics"

    def is_relevant(self, market: dict) -> bool:
        return _search_fields(market, self.KEYWORDS)

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
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
    Markets resolved by crypto price thresholds, dominance, ETF flows.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes
        "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
        "kxbnb", "kxavax", "kxlink", "kxcoin",
        # Human-readable fallback
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
        "sol", "xrp", "ripple", "altcoin", "defi", "nft", "etf",
        "blockchain", "coinbase", "binance", "stablecoin",
    ]

    TICKER_COIN_MAP = {
        "kxbtc":  "bitcoin",
        "kxeth":  "ethereum",
        "kxsol":  "solana",
        "kxxrp":  "ripple",
        "kxdoge": "dogecoin",
    }

    TITLE_COIN_MAP = {
        "btc": "bitcoin",  "bitcoin":  "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "sol": "solana",   "solana":   "solana",
        "xrp": "ripple",   "ripple":   "ripple",
    }

    @property
    def sector_name(self) -> str:
        return "crypto"

    def is_relevant(self, market: dict) -> bool:
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

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
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
    Markets on elections, legislation, approval ratings, appointments.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes
        "kxpres", "kxsenate", "kxhouse", "kxgov", "kxelect",
        "kxpol", "kxcong", "kxsupct", "kxadmin",
        # Human-readable fallback
        "election", "vote", "congress", "senate", "president",
        "governor", "democrat", "republican", "bill", "law",
        "approval", "impeach", "nominee", "ballot", "primary",
        "caucus", "supreme court", "executive order", "policy",
        "trump", "harris", "biden", "white house",
    ]

    @property
    def sector_name(self) -> str:
        return "politics"

    def is_relevant(self, market: dict) -> bool:
        return _search_fields(market, self.KEYWORDS)

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
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
    Markets on temperature records, precipitation, storm landfalls.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes
        "kxwthr", "kxhurr", "kxsnow", "kxtmp", "kxrain",
        "kxstorm", "kxtornado", "kxblizz",
        # Human-readable fallback
        "temperature", "temp", "fahrenheit", "celsius", "snow",
        "rain", "hurricane", "storm", "tornado", "flood", "drought",
        "heat", "cold", "blizzard", "wind", "precipitation", "weather",
    ]

    CITY_COORDS = {
        "new york": (40.71, -74.01), "nyc": (40.71, -74.01),
        "los angeles": (34.05, -118.24), "la": (34.05, -118.24),
        "chicago": (41.85, -87.65), "miami": (25.77, -80.19),
        "houston": (29.76, -95.37), "dallas": (32.78, -96.80),
        "seattle": (47.61, -122.33), "boston": (42.36, -71.06),
        "phoenix": (33.45, -112.07), "denver": (39.74, -104.98),
    }

    @property
    def sector_name(self) -> str:
        return "weather"

    def is_relevant(self, market: dict) -> bool:
        return _search_fields(market, self.KEYWORDS)

    def _detect_city(self, market: dict) -> tuple[float, float, str]:
        title = market.get("title", "").lower()
        for city, coords in self.CITY_COORDS.items():
            if city in title:
                return coords[0], coords[1], city
        return 40.71, -74.01, "New York"

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
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
        return features, context


# ═══════════════════════════════════════════════════════════════════════════════
#  5. TECH BOT
# ═══════════════════════════════════════════════════════════════════════════════

class TechBot(BaseBot):
    """
    Markets on earnings, product launches, M&A, AI milestones.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes
        "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
        "kxtsla", "kxearnings", "kxtech", "kxai", "kxipo",
        "kxopenai", "kxanthropic", "kxnasdaq",
        # Human-readable fallback
        "earnings", "revenue", "eps", "profit", "ipo", "merger",
        "acquisition", "apple", "google", "microsoft", "amazon",
        "meta", "nvidia", "tesla", "openai", "anthropic", "ai",
        "artificial intelligence", "model", "launch", "release",
        "stock", "share", "market cap", "nasdaq", "tech",
    ]

    TICKER_PREFIX_MAP = {
        "kxaapl": "AAPL", "kxgoog": "GOOGL", "kxmsft": "MSFT",
        "kxamzn": "AMZN", "kxmeta": "META",  "kxnvda": "NVDA",
        "kxtsla": "TSLA",
    }

    TITLE_TICKER_MAP = {
        "apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT",
        "amazon": "AMZN", "meta": "META",   "nvidia": "NVDA",
        "tesla": "TSLA",
    }

    @property
    def sector_name(self) -> str:
        return "tech"

    def is_relevant(self, market: dict) -> bool:
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

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
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
    Markets on game outcomes, over/unders, season props, parlays.

    Covers all liquid Kalshi sports markets including:
      KXMVESPORTSMULTIGAMEEXTENDED — multi-game parlay markets (dominant)
      KXMVECROSSCATEGORY           — cross-team spread combos (dominant)
      KXNBA, KXNFL, KXMLB etc.    — single-game markets (less liquid)
    """

    KEYWORDS = [
        # ── KXMVE multivariate sports (these are the 2300 liquid markets) ──
        "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
        "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
        "kxmve",  # broad catch-all for any multivariate market

        # ── Standard single-game Kalshi prefixes ──
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic",

        # ── Title / outcome label keywords ──
        # (title contains team names + outcome descriptions)
        "nba", "nfl", "mlb", "nhl", "mls", "ufc", "mma",
        "basketball", "football", "baseball", "hockey", "tennis",
        "golf", "f1", "formula 1", "champion", "playoff",
        "super bowl", "world series", "finals", "march madness",
        "wins by", "points scored", "over", "under", "spread", "mvp",
        "lakers", "celtics", "warriors", "bulls", "heat", "knicks",
        "patriots", "chiefs", "cowboys", "eagles",
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        # KXMVE markets are mostly NBA/NCAAB this time of year
        "kxmvesports":  "NBA",
        "kxmvecross":   "NBA",
        "kxmve":        "NBA",
    }

    @property
    def sector_name(self) -> str:
        return "sports"

    def is_relevant(self, market: dict) -> bool:
        return _search_fields(market, self.KEYWORDS)

    def _extract_teams_and_league(self, market: dict) -> tuple[str, str, str]:
        et     = market.get("event_ticker", "").lower()
        league = "NBA"

        # Check event_ticker prefix for league
        for prefix, lg in self.LEAGUE_MAP.items():
            if et.startswith(prefix):
                league = lg
                break

        # Refine league from title if we only have generic KXMVE
        title = market.get("title", "")
        title_lower = title.lower()
        if "nfl" in title_lower or "touchdown" in title_lower:
            league = "NFL"
        elif "mlb" in title_lower or "home run" in title_lower:
            league = "MLB"
        elif "nhl" in title_lower or "goal" in title_lower:
            league = "NHL"
        elif "ncaa" in title_lower or "college" in title_lower:
            league = "NCAAB"

        # Extract teams from title (title has real team names)
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
            # Format: "yes Detroit wins by over 16.5 Points,..."
            # Extract first team name after "yes "
            clean = title.replace("yes ", "").replace("no ", "")
            words = clean.split()
            team_a = words[0] if words else "Team A"

        return team_a, team_b, league

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        team_a, team_b, league = self._extract_teams_and_league(market)
        features, context = fetch_sports_features(team_a, team_b, league)
        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        ou_line = float(numbers[0]) if numbers else 0.0
        features = np.append(features, ou_line)
        context["ou_line"] = ou_line
        return features, context


# ── Convenience factory ───────────────────────────────────────────────────────

def all_bots() -> list[BaseBot]:
    """Instantiate one bot per sector."""
    return [
        EconomicsBot(),
        CryptoBot(),
        PoliticsBot(),
        WeatherBot(),
        TechBot(),
        SportsBot(),
    ]