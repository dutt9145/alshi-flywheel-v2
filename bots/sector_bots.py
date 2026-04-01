"""
bots/sector_bots.py  (v9 — keyword audit + sports bleed fixes)

Changes vs v8:
  EconomicsBot:
    - Added: retail sales, housing starts, consumer confidence, debt ceiling,
      deficit, tariff, trade balance + kx-prefixes for each
  CryptoBot:
    - "etf" replaced with "bitcoin etf", "crypto etf", "spot etf" (too generic)
  PoliticsBot:
    - No changes
  WeatherBot:
    - No changes
  TechBot:
    - "profit" removed — appears in sports box scores
    - "meta" removed from KEYWORDS (kept in TITLE_TICKER_MAP for detection only)
    - "acquisition" removed — risky substring match
    - Added SPORTS_EXCLUDE list — hard blocks any market with sports ticker prefix
    - is_relevant() now checks SPORTS_EXCLUDE before keyword match
  SportsBot:
    - "over" and "under" removed — appear in economics market titles constantly
    - "champion" and "finals" replaced with specific sport variants
    - "champion" REMOVED — "champion the bill" in politics titles
    - "finals" REMOVED — "Fed finals decision" false positive
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


def _has_sports_prefix(market: dict) -> bool:
    """
    Returns True if the market ticker/event_ticker starts with a known
    sports prefix. Used to hard-exclude sports markets from non-sports bots.
    """
    SPORTS_PREFIXES = (
        "kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
        "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic", "kxthail", "kxsl",
        "kxdota", "kxintlf", "kxcs2", "kxegypt", "kxvenf",
    )
    et = market.get("event_ticker", "").lower()
    tk = market.get("ticker", "").lower()
    return any(et.startswith(p) or tk.startswith(p) for p in SPORTS_PREFIXES)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ECONOMICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class EconomicsBot(BaseBot):
    """
    Macro-economic resolution markets: CPI, Fed rate decisions, GDP, jobs,
    housing, retail, trade, consumer sentiment.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe — Kalshi's own taxonomy)
        "kxcpi", "kxfed", "kxfomc", "kxgdp", "kxjobs", "kxunrate",
        "kxpce", "kxnfp", "kxyield", "kxrate", "kxinfl",
        "kxhousing", "kxretail", "kxdebt", "kxtrade", "kxconsumer",
        # Human-readable — only terms that cannot appear in sports titles
        "cpi", "inflation", "unemployment", "nonfarm",
        "payroll", "gdp", "recession", "rate hike", "rate cut",
        "pce", "fomc", "interest rate", "treasury",
        "federal reserve", "basis points",
        "retail sales", "housing starts", "consumer confidence",
        "debt ceiling", "deficit", "tariff", "trade balance",
        "consumer sentiment", "jobless claims", "trade war",
        # "yield" REMOVED — appears in sports spread/stats context
        # "fed" REMOVED — too short, risky substring match
    ]

    @property
    def sector_name(self) -> str:
        return "economics"

    def is_relevant(self, market: dict) -> bool:
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
    Crypto price threshold, dominance, and ETF flow markets.
    "sol" removed — substring of "resolved", "console", etc.
    "link" removed — too generic; kxlink prefix retained.
    "etf" replaced with specific "bitcoin etf", "crypto etf" — plain "etf"
    matches S&P ETF and bond ETF markets which belong in economics.
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe)
        "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
        "kxbnb", "kxavax", "kxlink", "kxcoin",
        # Human-readable — specific enough to not bleed
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
        "xrp", "ripple", "altcoin", "defi", "nft",
        "bitcoin etf", "crypto etf", "spot etf",
        "blockchain", "coinbase", "binance", "stablecoin",
        # "etf" REMOVED — matches S&P/bond ETF markets (economics territory)
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
#  4. WEATHER BOT  (v8 — skip_noaa flag, unchanged in v9)
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherBot(BaseBot):
    """
    Temperature records, precipitation, storm landfalls.
    NOAA API integrated (v7). skip_noaa flag added (v8).
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
        if _has_sports_prefix(market):
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

        lat, lon      = coords
        target_date   = parse_market_date(title, close_time)
        market_type   = classify_weather_market(title)

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
#  5. TECH BOT  (v9 — sports bleed fix)
# ═══════════════════════════════════════════════════════════════════════════════

class TechBot(BaseBot):
    """
    Earnings, M&A, AI milestones markets.

    v9 changes:
    - _has_sports_prefix() guard added to is_relevant() — hard blocks the
      79k+ KXMVE/KXNCAA/KXNBA markets that were bleeding in via generic keywords
    - "profit" REMOVED — appears in sports box score headlines
    - "meta" REMOVED from KEYWORDS — too short, matched KXMVE* prefixes.
      Still used in TITLE_TICKER_MAP for company detection after relevance check.
    - "acquisition" REMOVED — risky substring match in broader market titles
    """

    KEYWORDS = [
        # Kalshi event_ticker prefixes (safe — Kalshi's own taxonomy)
        "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
        "kxtsla", "kxearnings", "kxtech", "kxai", "kxipo",
        "kxopenai", "kxanthropic", "kxnasdaq",
        # Specific company/tech terms — safe from sports bleed
        "earnings", "revenue", "eps", "ipo", "merger",
        "apple", "google", "microsoft", "amazon",
        "nvidia", "tesla", "openai", "anthropic",
        "artificial intelligence", "nasdaq",
        # "profit"      REMOVED — appears in sports headlines
        # "meta"        REMOVED — too short, matched KXMVE* sports tickers
        # "acquisition" REMOVED — risky substring match
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
        # Hard block — sports prefixes cannot be tech markets
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
#  6. SPORTS BOT  (v9 — removed generic bleed keywords)
# ═══════════════════════════════════════════════════════════════════════════════

class SportsBot(BaseBot):
    """
    Game outcomes, over/unders, season props, parlays.

    v9 changes:
    - "over" and "under" REMOVED — appear constantly in economics market titles
      ("will CPI come in over 3%", "will unemployment go under 4%")
    - "champion" REMOVED — matches "champion the bill" in politics titles
    - "finals" REMOVED — matches "Fed finals decision" in economics titles
    - Replaced with specific sport variants: "nba finals", "stanley cup", etc.
    """

    KEYWORDS = [
        # KXMVE multivariate sports (specific prefixes only)
        "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
        "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
        "kxmvecbchampionship",
        # Standard single-game Kalshi prefixes
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic",
        "kxthail", "kxsl", "kxdota", "kxintlf", "kxcs2",
        # Title keywords — sports-specific enough to be safe
        "nba", "nfl", "mlb", "nhl", "mls", "ufc", "mma",
        "basketball", "football", "baseball", "hockey", "tennis",
        "golf", "f1", "formula 1", "playoff",
        "super bowl", "world series", "march madness",
        "nba finals", "stanley cup", "nfl championship",
        "world series finals", "ncaa championship",
        "wins by", "points scored", "spread", "mvp",
        "lakers", "celtics", "warriors", "bulls", "knicks",
        "patriots", "chiefs", "cowboys", "eagles",
        "miami heat",
        # "over"     REMOVED — "will CPI come in over 3%"
        # "under"    REMOVED — "will unemployment go under 4%"
        # "champion" REMOVED — "champion the bill" in politics
        # "finals"   REMOVED — "Fed finals decision" in economics
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        "kxmvesports": "NBA",
        "kxmvecross":  "NBA",
        "kxmvecbchampionship": "NCAAB",
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

    def fetch_features(self, market: dict, skip_noaa: bool = False) -> tuple[np.ndarray, dict]:
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