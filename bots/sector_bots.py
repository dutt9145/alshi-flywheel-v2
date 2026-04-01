"""
bots/sector_bots.py  (v11 — worldwide keyword expansion)

Changes vs v10:
  WeatherBot:
    - 200+ worldwide cities added across all continents
    - Expanded weather phenomena: monsoon, cyclone, typhoon, derecho,
      polar vortex, nor'easter, dust storm, wildfire smoke, ice storm
    - Regional weather terms added
  EconomicsBot:
    - Global central banks: ECB, BOJ, BOE, RBA, PBOC, SNB, Riksbank
    - International economic terms: PMI, ISM, Eurozone, G7, G20,
      IMF, World Bank, OPEC, WTO
    - Commodity markets: oil, gold, silver, copper, wheat, corn
  CryptoBot:
    - 30+ additional coins and tokens
    - DeFi protocols: Uniswap, Aave, Compound, Curve, MakerDAO
    - Layer 2s: Arbitrum, Optimism, Polygon, Base
    - NFT platforms, crypto exchanges
  PoliticsBot:
    - International politics: UN, NATO, EU, G7, BRICS
    - World leaders and elections globally
    - Geopolitical terms: sanctions, treaty, referendum, coup
  TechBot:
    - More companies: Samsung, TSMC, Intel, AMD, Qualcomm, Arm
    - AI models: GPT, Gemini, Claude, Llama, Grok
    - Tech events: WWDC, CES, earnings seasons by company
  SportsBot:
    - International leagues: Premier League, La Liga, Serie A,
      Bundesliga, Champions League, Copa America, World Cup
    - Esports: League of Legends, Valorant, CS2
    - Combat sports: boxing, wrestling, bjj
    - Olympic sports, cricket, rugby
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
    haystack = " ".join([
        market.get("event_ticker", ""),
        market.get("ticker", ""),
        market.get("title", ""),
        market.get("subtitle", ""),
    ]).lower()
    return any(kw.lower() in haystack for kw in keywords)


def _has_sports_prefix(market: dict) -> bool:
    SPORTS_PREFIXES = (
        "kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
        "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic", "kxthail", "kxsl",
        "kxdota", "kxintlf", "kxcs2", "kxegypt", "kxvenf",
        "kxepl", "kxsoccer", "kxboxing", "kxwwe", "kxcricket",
        "kxrugby", "kxesport",
    )
    et = market.get("event_ticker", "").lower()
    tk = market.get("ticker", "").lower()
    return any(et.startswith(p) or tk.startswith(p) for p in SPORTS_PREFIXES)


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
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
        "kxbnb", "kxavax", "kxlink", "kxcoin", "kxmatic", "kxada",
        "kxdot", "kxatom", "kxnear", "kxfil", "kxapt", "kxsui",

        # ── Major coins ────────────────────────────────────────────────────
        "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "ripple",
        "dogecoin", "doge", "cardano", "ada", "polkadot", "dot",
        "avalanche", "avax", "chainlink", "link", "polygon", "matic",
        "shiba inu", "shib", "litecoin", "ltc", "bitcoin cash", "bch",
        "stellar", "xlm", "cosmos", "atom", "algorand", "algo",
        "tron", "trx", "filecoin", "fil", "near protocol", "near",
        "aptos", "apt", "sui", "sei", "injective", "inj",
        "arbitrum", "arb", "optimism", "op", "base",
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
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxpres", "kxsenate", "kxhouse", "kxgov", "kxelect",
        "kxpol", "kxcong", "kxsupct", "kxadmin", "kxintl",
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
        "nato expansion", "un resolution", "war", "conflict",
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
#  4. WEATHER BOT  (v11 — worldwide city expansion)
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherBot(BaseBot):
    """
    Temperature, precipitation, storms — worldwide city coverage.

    v11 changes:
    - 200+ worldwide cities added across North America, Europe, Asia,
      South America, Africa, Australia/Oceania
    - Extended phenomena: monsoon, cyclone, typhoon, derecho,
      polar vortex, nor'easter, dust storm, ice storm, wildfire
    - Multi-word weather phrases for unambiguous matching
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxwthr", "kxhurr", "kxsnow", "kxtmp", "kxrain",
        "kxstorm", "kxtornado", "kxblizz", "kxfrost",
        "kxcyclone", "kxtyphoon", "kxmonsoon",

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
#  5. TECH BOT  (v11 — expanded companies + AI models)
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
#  6. SPORTS BOT  (v11 — international leagues + esports + combat sports)
# ═══════════════════════════════════════════════════════════════════════════════

class SportsBot(BaseBot):
    """
    US + international sports, esports, combat sports, Olympic events.
    """

    KEYWORDS = [
        # ── Kalshi kx-prefixes ──────────────────────────────────────────────
        "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
        "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
        "kxmvecbchampionship",
        "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls", "kxufc",
        "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxtennis", "kxf1", "kxolympic", "kxepl", "kxsoccer",
        "kxboxing", "kxwwe", "kxcricket", "kxrugby", "kxesport",
        "kxthail", "kxsl", "kxdota", "kxintlf", "kxcs2",

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

        # ── Tennis ─────────────────────────────────────────────────────────
        "wimbledon", "us open tennis", "french open", "roland garros",
        "australian open", "atp", "wta", "grand slam",
        "djokovic", "alcaraz", "sinner", "nadal", "federer",
        "swiatek", "sabalenka", "gauff",

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
        "dota 2", "overwatch", "rocket league", "apex legends",
        "fortnite tournament", "esports championship", "worlds lol",
        "majors cs2", "vlr valorant",

        # ── Other sports ───────────────────────────────────────────────────
        "cricket", "test match", "odi cricket", "ipl", "ashes",
        "rugby world cup", "six nations", "rugby union",
        "olympic games", "summer olympics", "winter olympics",
        "tour de france", "cycling grand tour",

        # ── Generic sports terms (safe — not used in other sectors) ────────
        "wins by", "points scored", "spread", "mvp",
        "playoff", "championship series",
        "nba finals", "stanley cup", "nfl championship",
        "world series finals", "ncaa championship",
    ]

    LEAGUE_MAP = {
        "kxnba":  "NBA", "kxnfl":  "NFL", "kxmlb": "MLB",
        "kxnhl":  "NHL", "kxmls":  "MLS", "kxufc": "UFC",
        "kxncaa": "NCAAB", "kxcbb": "NCAAB", "kxcfb": "NCAAF",
        "kxmvesports": "NBA", "kxmvecross": "NBA",
        "kxmvecbchampionship": "NCAAB",
        "kxepl": "EPL", "kxsoccer": "MLS",
        "kxtennis": "ATP", "kxf1": "F1",
        "kxboxing": "BOXING", "kxgolf": "PGA",
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
        elif "nhl" in title_lower or " goal" in title_lower:
            league = "NHL"
        elif "ncaa" in title_lower or "college" in title_lower:
            league = "NCAAB"
        elif any(t in title_lower for t in ["premier league", "champions league", "la liga", "serie a", "bundesliga"]):
            league = "EPL"
        elif "f1" in title_lower or "formula 1" in title_lower:
            league = "F1"
        elif "tennis" in title_lower or "wimbledon" in title_lower:
            league = "ATP"

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


# ── Convenience factory ───────────────────────────────────────────────────────

def all_bots() -> list[BaseBot]:
    return [
        EconomicsBot(),
        CryptoBot(),
        PoliticsBot(),
        WeatherBot(),
        TechBot(),
        SportsBot(),
    ]