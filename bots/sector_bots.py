"""
bots/sector_bots.py  (v6 — full keyword bleed audit)

Changes vs v5:
  EconomicsBot:
    - REMOVED "yield"   → appeared in sports spread/stats context
  CryptoBot:
    - REMOVED "sol"     → substring of "resolved", "console", many common words
    - REMOVED "link"    → too generic; kxlink prefix retained
  WeatherBot:
    - REMOVED "heat"    → Miami Heat is an NBA team (huge bleed)
    - REMOVED "cold"    → appeared in sports cold-weather game titles
    - REMOVED "wind"    → appeared in sports stadium/conditions titles
  PoliticsBot:
    - REMOVED "law"     → matched player names and general sports content
    - REMOVED "bill"    → matched player names (Bill X) in sports titles
  TechBot:
    - Unchanged from v5 (already cleaned)
  SportsBot:
    - Unchanged from v4 (already cleaned)

Rule applied: if a keyword could plausibly appear in a KXMVE sports market
title without being the primary topic, it should be removed. kx-prefixes
are always safe (they're Kalshi's own taxonomy), bare English words are not.
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
    Macro-economic resolution markets: CPI, Fed rate decisions, GDP, jobs.
    Keywords restricted to kx-prefixes + terms that ONLY appear in financial
    market titles — "yield" removed (sports spread context).
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe — Kalshi's own taxonomy)
        "kxcpi", "kxfed", "kxfomc", "kxgdp", "kxjobs", "kxunrate",
        "kxpce", "kxnfp", "kxyield", "kxrate", "kxinfl",
        # Human-readable — only terms that cannot appear in sports titles
        "cpi", "inflation", "unemployment", "nonfarm",
        "payroll", "gdp", "recession", "rate hike", "rate cut",
        "pce", "fomc", "interest rate", "treasury",
        "federal reserve", "basis points",
        # "yield" REMOVED — appears in sports spread/stats context
        # "fed" REMOVED — too short, risky substring match
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
    Crypto price threshold, dominance, and ETF flow markets.
    "sol" removed — substring of "resolved", "console", etc.
    "link" removed — too generic; kxlink prefix retained.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe)
        "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
        "kxbnb", "kxavax", "kxlink", "kxcoin",
        # Human-readable — specific enough to not bleed
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
        "xrp", "ripple", "altcoin", "defi", "nft", "etf",
        "blockchain", "coinbase", "binance", "stablecoin",
        # "sol" REMOVED — substring of "resolved", "console", many words
        # "link" REMOVED — too generic
    ]

    TICKER_COIN_MAP = {
        "kxbtc":  "bitcoin", "kxeth":  "ethereum",
        "kxsol":  "solana",  "kxxrp":  "ripple",
        "kxdoge": "dogecoin",
    }

    TITLE_COIN_MAP = {
        "btc": "bitcoin",  "bitcoin":  "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "solana": "solana",
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
    Elections, legislation, approval ratings, appointments.
    "law" and "bill" removed — match player names in sports titles.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe)
        "kxpres", "kxsenate", "kxhouse", "kxgov", "kxelect",
        "kxpol", "kxcong", "kxsupct", "kxadmin",
        # Human-readable — specific enough
        "election", "vote", "congress", "senate", "president",
        "governor", "democrat", "republican",
        "approval", "impeach", "nominee", "ballot", "primary",
        "caucus", "supreme court", "executive order", "policy",
        "trump", "harris", "biden", "white house",
        # "law" REMOVED — matches player names (e.g. "Ty Law")
        # "bill" REMOVED — matches player names (e.g. "Bill Belichick")
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
    Temperature records, precipitation, storm landfalls.
    "heat" removed — Miami Heat is an NBA team (massive bleed into sports).
    "cold" and "wind" removed — appear in sports game condition titles.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe)
        "kxwthr", "kxhurr", "kxsnow", "kxtmp", "kxrain",
        "kxstorm", "kxtornado", "kxblizz",
        # Human-readable — specific enough
        "temperature", "fahrenheit", "celsius", "snow",
        "hurricane", "tornado", "flood", "drought",
        "blizzard", "precipitation", "weather",
        # "heat"  REMOVED — Miami Heat (NBA team)
        # "cold"  REMOVED — appears in sports cold-weather game titles
        # "wind"  REMOVED — appears in sports stadium/conditions titles
        # "rain"  REMOVED — can appear in player names
        # "storm" REMOVED — appears in team names / sports contexts
        # "temp"  REMOVED — too short, risky substring
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
    Earnings, M&A, AI milestones markets.
    Generic words removed in v5: "tech", "model", "launch", "release",
    "stock", "share", "market cap", "ai" — all appeared in KXMVE sports titles.
    Only specific company names, tickers, and kx-prefixes remain.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe)
        "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
        "kxtsla", "kxearnings", "kxtech", "kxai", "kxipo",
        "kxopenai", "kxanthropic", "kxnasdaq",
        # Specific company names only — cannot appear in sports titles
        "earnings", "revenue", "eps", "profit", "ipo", "merger",
        "acquisition", "apple", "google", "microsoft", "amazon",
        "nvidia", "tesla", "openai", "anthropic",
        "artificial intelligence", "nasdaq",
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
    Game outcomes, over/unders, season props, parlays.
    KXMVESPORTSMULTIGAMEEXTENDED and KXMVECROSSCATEGORY are the dominant
    liquid market types on Kalshi (March Madness + NBA season).
    Bare "kxmve" catch-all excluded — was causing ~83% signal skew.
    """

    KEYWORDS = [
        # KXMVE multivariate sports (specific prefixes only)
        "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
        "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
        # Standard single-game Kalshi prefixes
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic",
        # Title keywords — sports-specific enough to be safe
        "nba", "nfl", "mlb", "nhl", "mls", "ufc", "mma",
        "basketball", "football", "baseball", "hockey", "tennis",
        "golf", "f1", "formula 1", "champion", "playoff",
        "super bowl", "world series", "finals", "march madness",
        "wins by", "points scored", "over", "under", "spread", "mvp",
        "lakers", "celtics", "warriors", "bulls", "knicks",
        "patriots", "chiefs", "cowboys", "eagles",
        # "heat" moved here from WeatherBot — Miami Heat is sports
        "miami heat",
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        "kxmvesports": "NBA",
        "kxmvecross":  "NBA",
    }

    @property
    def sector_name(self) -> str:
        return "sports"

    def is_relevant(self, market: dict) -> bool:
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
        elif "nhl" in title_lower or "goal" in title_lower:
            league = "NHL"
        elif "ncaa" in title_lower or "college" in title_lower:
            league = "NCAAB"

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