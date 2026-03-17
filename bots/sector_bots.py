"""
bots/sector_bots.py

All six concrete sector bots in one file.
Each inherits BaseBot and implements the three required methods.

Sectors
-------
1. EconomicsBot  — CPI, NFP, Fed policy markets
2. CryptoBot     — BTC/ETH price and dominance markets
3. PoliticsBot   — Election, approval, legislative markets
4. WeatherBot    — Temperature, precipitation, storm markets
5. TechBot       — Earnings surprises, product launch markets
6. SportsBot     — Game outcome, over/under, MVP markets
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


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ECONOMICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class EconomicsBot(BaseBot):
    """
    Focuses on macro-economic resolution markets:
    - "Will CPI be above X% this month?"
    - "Will the Fed raise rates at the next meeting?"
    - "Will unemployment exceed X%?"

    Features: CPI, PCE, unemployment, fed funds, NFP, 10y yield, GDP growth
    """

    KEYWORDS = ["cpi", "inflation", "fed", "unemployment", "nonfarm",
                "payroll", "gdp", "recession", "rate hike", "rate cut",
                "pce", "fomc", "interest rate", "treasury", "yield"]

    @property
    def sector_name(self) -> str:
        return "economics"

    def is_relevant(self, market: dict) -> bool:
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        features, context = fetch_economics_features()

        # Inject contract-specific numeric target if parseable
        # e.g. "Will CPI be above 3.2%?" → target = 3.2
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
    Markets resolved by crypto price thresholds, dominance, ETF flows, etc.
    - "Will BTC close above $X on date Y?"
    - "Will ETH dominance exceed X%?"
    """

    KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
                "sol", "xrp", "ripple", "altcoin", "defi", "nft", "etf",
                "blockchain", "coinbase", "binance", "stablecoin"]

    COIN_MAP = {"btc": "bitcoin", "bitcoin": "bitcoin",
                "eth": "ethereum", "ethereum": "ethereum",
                "sol": "solana", "xrp": "ripple"}

    @property
    def sector_name(self) -> str:
        return "crypto"

    def is_relevant(self, market: dict) -> bool:
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def _detect_coin(self, market: dict) -> str:
        title = market.get("title", "").lower()
        for kw, coin_id in self.COIN_MAP.items():
            if kw in title:
                return coin_id
        return "bitcoin"

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        coin_id  = self._detect_coin(market)
        features, context = fetch_crypto_features(coin_id=coin_id)

        # Extract price target from title
        title   = market.get("title", "")
        numbers = re.findall(r"[\d,]+\.?\d*", title.replace(",", ""))
        target  = float(numbers[0]) if numbers else 0.0

        # Add ratio of current price vs target as a feature
        current_price = context.get("price", 1.0) or 1.0
        ratio = current_price / target if target > 0 else 1.0
        features = np.append(features, [target, ratio])

        context["contract_target"] = target
        context["price_vs_target"] = ratio
        return features, context


# ═══════════════════════════════════════════════════════════════════════════════
#  3. POLITICS BOT
# ═══════════════════════════════════════════════════════════════════════════════

class PoliticsBot(BaseBot):
    """
    Markets on elections, legislation, approval ratings, appointments.
    - "Will X win the primary?"
    - "Will Congress pass Y bill?"
    - "Will approval rating exceed X%?"
    """

    KEYWORDS = ["election", "vote", "congress", "senate", "president",
                "governor", "democrat", "republican", "bill", "law",
                "approval", "impeach", "nominee", "ballot", "primary",
                "caucus", "supreme court", "executive order", "policy"]

    @property
    def sector_name(self) -> str:
        return "politics"

    def is_relevant(self, market: dict) -> bool:
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        # Try to extract Polymarket slug from market metadata
        poly_slug  = market.get("polymarket_slug")
        features, context = fetch_politics_features(poly_slug)

        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        target  = float(numbers[0]) if numbers else 50.0
        features = np.append(features, target)

        # Days to market close
        close_ts = market.get("close_time", "")
        try:
            from datetime import datetime, timezone
            close_dt = datetime.fromisoformat(close_ts.replace("Z", "+00:00"))
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
    - "Will NYC temperature exceed 90°F on date X?"
    - "Will hurricane landfall in Florida this week?"
    """

    KEYWORDS = ["temperature", "temp", "fahrenheit", "celsius", "snow",
                "rain", "hurricane", "storm", "tornado", "flood", "drought",
                "heat", "cold", "blizzard", "wind", "precipitation", "weather"]

    # Known city coordinates — extend as needed
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
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def _detect_city(self, market: dict) -> tuple[float, float, str]:
        title = market.get("title", "").lower()
        for city, coords in self.CITY_COORDS.items():
            if city in title:
                return coords[0], coords[1], city
        return 40.71, -74.01, "New York"  # default

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        lat, lon, city = self._detect_city(market)
        features, context = fetch_weather_features(lat, lon, city)

        title   = market.get("title", "")
        numbers = re.findall(r"\d+\.?\d*", title)
        target_f = float(numbers[0]) if numbers else 75.0
        target_c = (target_f - 32) * 5 / 9

        # Feature: how close current max temp is to the threshold
        delta = context.get("temp_max_c", 0.0) - target_c
        features = np.append(features, [target_c, delta])
        context["target_c"] = target_c
        context["delta_from_target"] = delta
        return features, context


# ═══════════════════════════════════════════════════════════════════════════════
#  5. TECH BOT
# ═══════════════════════════════════════════════════════════════════════════════

class TechBot(BaseBot):
    """
    Markets on earnings, product launches, M&A, AI milestones.
    - "Will Apple beat Q3 EPS estimates?"
    - "Will GPT-5 launch before date X?"
    - "Will AAPL close above $X?"
    """

    KEYWORDS = ["earnings", "revenue", "eps", "profit", "ipo", "merger",
                "acquisition", "apple", "google", "microsoft", "amazon",
                "meta", "nvidia", "tesla", "openai", "anthropic", "ai",
                "artificial intelligence", "model", "launch", "release",
                "stock", "share", "market cap", "nasdaq", "tech"]

    TICKER_MAP = {"apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT",
                  "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
                  "tesla": "TSLA"}

    @property
    def sector_name(self) -> str:
        return "tech"

    def is_relevant(self, market: dict) -> bool:
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def _detect_company(self, market: dict) -> str:
        title = market.get("title", "").lower()
        for name, ticker in self.TICKER_MAP.items():
            if name in title:
                return ticker
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
    Markets on game outcomes, over/unders, season props.
    - "Will the Lakers beat the Celtics on X date?"
    - "Will the NFL game go over 48.5 points?"
    - "Will X player score 25+ points?"
    """

    KEYWORDS = ["nba", "nfl", "mlb", "nhl", "soccer", "mls", "ufc", "mma",
                "basketball", "football", "baseball", "hockey", "tennis",
                "golf", "f1", "formula 1", "champion", "playoff", "super bowl",
                "world series", "finals", "game", "match", "season", "win",
                "score", "points", "over", "under", "spread", "mvp"]

    @property
    def sector_name(self) -> str:
        return "sports"

    def is_relevant(self, market: dict) -> bool:
        title = (market.get("title", "") + " " +
                 market.get("ticker", "")).lower()
        return any(kw in title for kw in self.KEYWORDS)

    def _extract_teams(self, market: dict) -> tuple[str, str, str]:
        title  = market.get("title", "")
        league = "NBA"
        for lg in ["NBA", "NFL", "MLB", "NHL", "MLS", "UFC"]:
            if lg.lower() in title.lower():
                league = lg
                break
        # Naive team extraction: first two words after "Will"
        parts = title.replace("Will the ", "").replace("Will ", "").split()
        team_a = parts[0] if len(parts) > 0 else "Team A"
        # Look for " vs " or " beat "
        if " vs " in title:
            teams = title.split(" vs ")
            team_a = teams[0].strip().split()[-1]
            team_b = teams[1].strip().split()[0]
        elif " beat " in title.lower():
            idx = title.lower().index(" beat ")
            team_a = title[:idx].split()[-1]
            team_b = title[idx + 6:].split()[0]
        else:
            team_b = "Team B"
        return team_a, team_b, league

    def fetch_features(self, market: dict) -> tuple[np.ndarray, dict]:
        team_a, team_b, league = self._extract_teams(market)
        features, context = fetch_sports_features(team_a, team_b, league)

        # Over/under target from title
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
