"""
shared/data_fetchers.py  (v6 — odds API TTL fix)

Changes vs v5:
  1. fetch_sports_features() odds API TTL changed from 300s to 3600s.
     At 300s (matching scan interval), the cache expired between every scan,
     consuming ~576 credits/day and burning the 2,000/month paid tier in
     under 4 days. Odds lines don't move meaningfully every 5 minutes —
     1 hour is the right refresh cadence for pre-game lines.

  2. Everything else unchanged from v5.

Tier 1 (free)   : FRED, Open-Meteo, CoinGecko, Polymarket Gamma, NewsAPI, Finnhub, ESPN
Tier 2 ($9-50)  : TheOddsAPI, OddsPapi, Polygon.io, MySportsFeeds, FMP
Tier 3 ($29-150): Glassnode, Tomorrow.io, Alpha Vantage sentiment, Coinglass, Lunarcrush
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import requests

from config.settings import (
    FRED_API_KEY, NEWSAPI_KEY, FINNHUB_API_KEY,
    ODDS_API_KEY, POLYGON_API_KEY, MYSPORTSFEEDS_KEY, FMP_API_KEY,
    GLASSNODE_API_KEY, TOMORROW_IO_KEY, ALPHA_VANTAGE_KEY,
    COINGLASS_KEY, LUNARCRUSH_KEY,
)

logger = logging.getLogger(__name__)

# ── In-memory TTL cache ───────────────────────────────────────────────────────

_cache: dict = {}

def _get(url: str, params: dict = None, headers: dict = None,
         ttl: int = 300, auth=None, retries: int = 3) -> Optional[dict]:
    key = url + str(sorted((params or {}).items()))
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < ttl):
        return entry["data"]

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {},
                             auth=auth, timeout=12)
            if r.status_code in (429, 500, 502, 503):
                wait = 2 ** attempt
                logger.warning(
                    "HTTP %s from %s — retrying in %ss (attempt %d/%d)",
                    r.status_code, url, wait, attempt + 1, retries,
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            _cache[key] = {"ts": time.time(), "data": data}
            return data
        except Exception as e:
            logger.warning("Fetch failed %s: %s", url, e)
            if attempt == retries - 1:
                return entry["data"] if entry else None

    return entry["data"] if entry else None


# ── CoinGecko rate-limit guard ────────────────────────────────────────────────

_cg_lock = threading.Lock()
_cg_last_call: float = 0.0
_CG_MIN_INTERVAL: float = 2.0

def _cg_rate_limited_get(url: str, params: dict, ttl: int) -> Optional[dict]:
    global _cg_last_call
    with _cg_lock:
        elapsed = time.time() - _cg_last_call
        if elapsed < _CG_MIN_INTERVAL:
            time.sleep(_CG_MIN_INTERVAL - elapsed)
        _cg_last_call = time.time()
    return _get(url, params, ttl=ttl)


# ── Batched CoinGecko fetch ───────────────────────────────────────────────────

_KNOWN_COINS = ["bitcoin", "ethereum", "solana", "ripple"]

def _fetch_all_cg_markets() -> dict[str, dict]:
    ids_str = ",".join(_KNOWN_COINS)
    data = _cg_rate_limited_get(
        "https://api.coingecko.com/api/v3/coins/markets",
        {"vs_currency": "usd", "ids": ids_str,
         "price_change_percentage": "24h", "sparkline": "false"},
        ttl=600,
    )
    if not data:
        return {}
    return {coin["id"]: coin for coin in data}


# =============================================================================
#  1. ECONOMICS
# =============================================================================

FRED_FALLBACKS = {
    "CPIAUCSL":        314.0,
    "PCEPI":           125.0,
    "UNRATE":          4.1,
    "FEDFUNDS":        4.33,
    "PAYEMS":          159000.0,
    "DGS10":           4.3,
    "A191RL1Q225SBEA": 2.8,
}

def fetch_economics_features() -> tuple[np.ndarray, dict]:
    context = {}

    if ALPHA_VANTAGE_KEY:
        def _av(func, interval="monthly"):
            d = _get("https://www.alphavantage.co/query", {
                "function": func, "interval": interval,
                "apikey": ALPHA_VANTAGE_KEY,
            }, ttl=3600)
            try:
                return float(list(d["data"])[0]["value"])
            except Exception:
                return 0.0

        cpi          = _av("CPI")
        unemployment = _av("UNEMPLOYMENT")
        fed_funds    = _av("FEDERAL_FUNDS_RATE", "daily")
        yield_10y    = _av("TREASURY_YIELD", "daily")
        gdp          = _av("REAL_GDP", "quarterly")
        pce          = cpi
        nfp          = _av("NONFARM_PAYROLL")

        sentiment_data = _get("https://www.alphavantage.co/query", {
            "function": "NEWS_SENTIMENT", "topics": "economy_macro,finance",
            "apikey": ALPHA_VANTAGE_KEY, "limit": "50",
        }, ttl=900)
        macro_sentiment = 0.0
        if sentiment_data:
            scores = [float(a.get("overall_sentiment_score", 0))
                      for a in sentiment_data.get("feed", [])]
            macro_sentiment = float(np.mean(scores)) if scores else 0.0
        context.update({"source": "alpha_vantage", "macro_sentiment": macro_sentiment})

    else:
        FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

        def _fred(series_id: str) -> float:
            d = _get(FRED_URL, {
                "series_id": series_id, "api_key": FRED_API_KEY,
                "file_type": "json", "sort_order": "desc", "limit": 1,
            }, ttl=3600)
            try:
                return float(d["observations"][0]["value"])
            except Exception:
                fallback = FRED_FALLBACKS.get(series_id, 0.0)
                logger.warning("FRED parse failed for %s — using fallback %.2f",
                               series_id, fallback)
                return fallback

        cpi          = _fred("CPIAUCSL")
        pce          = _fred("PCEPI")
        unemployment = _fred("UNRATE")
        fed_funds    = _fred("FEDFUNDS")
        nfp          = _fred("PAYEMS")
        yield_10y    = _fred("DGS10")
        gdp          = _fred("A191RL1Q225SBEA")
        macro_sentiment = 0.0
        context["source"] = "fred"

    futures_yield = yield_10y
    if POLYGON_API_KEY:
        ff_data = _get("https://api.polygon.io/v2/last/trade/ZQ",
                       {"apiKey": POLYGON_API_KEY}, ttl=120)
        if ff_data:
            try:
                futures_yield = float(ff_data["results"]["p"])
            except Exception:
                pass

    context.update({
        "cpi": cpi, "pce": pce, "unemployment": unemployment,
        "fed_funds": fed_funds, "nfp": nfp,
        "yield_10y": yield_10y, "futures_yield": futures_yield, "gdp": gdp,
    })

    features = np.array([
        cpi, pce, unemployment, fed_funds, nfp / 1e5,
        yield_10y, futures_yield, gdp, macro_sentiment,
    ], dtype=float)
    return features, context


# =============================================================================
#  2. CRYPTO
# =============================================================================

def fetch_crypto_features(coin_id: str = "bitcoin") -> tuple[np.ndarray, dict]:
    context = {"coin": coin_id}

    cg_all = _fetch_all_cg_markets()
    cg     = cg_all.get(coin_id, {})

    price = change_24h = rank = volume = ath_change = 0.0
    if cg:
        price      = float(cg.get("current_price", 0) or 0)
        change_24h = float(cg.get("price_change_percentage_24h", 0) or 0)
        rank       = float(cg.get("market_cap_rank", 0) or 0)
        volume     = float(cg.get("total_volume", 0) or 0)
        ath_pct    = cg.get("ath_change_percentage")
        ath_change = float(ath_pct) if ath_pct is not None else 0.0

    fg = _get("https://api.alternative.me/fng/", {}, ttl=3600)
    fear_greed = 50.0
    try:
        fear_greed = float(fg["data"][0]["value"])
    except Exception:
        pass

    if POLYGON_API_KEY:
        sym = {"bitcoin": "X:BTCUSD", "ethereum": "X:ETHUSD",
               "solana": "X:SOLUSD", "ripple": "X:XRPUSD"}.get(coin_id, "X:BTCUSD")
        poly = _get(f"https://api.polygon.io/v2/last/trade/{sym}",
                    {"apiKey": POLYGON_API_KEY}, ttl=30)
        if poly:
            try:
                price = float(poly["results"]["p"])
                context["price_source"] = "polygon_realtime"
            except Exception:
                pass

    exchange_netflow = whale_count = 0.0
    if GLASSNODE_API_KEY:
        asset = {"bitcoin": "BTC", "ethereum": "ETH"}.get(coin_id, "BTC")
        nf = _get(
            "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net",
            {"a": asset, "api_key": GLASSNODE_API_KEY, "i": "24h"}, ttl=1800)
        if nf:
            try:
                exchange_netflow = float(nf[-1]["v"])
            except Exception:
                pass
        wh = _get("https://api.glassnode.com/v1/metrics/addresses/count",
                  {"a": asset, "api_key": GLASSNODE_API_KEY, "i": "24h"}, ttl=3600)
        if wh:
            try:
                whale_count = float(wh[-1]["v"]) / 1e6
            except Exception:
                pass

    funding_rate = 0.0
    if COINGLASS_KEY:
        sym_cg = {"bitcoin": "BTC", "ethereum": "ETH"}.get(coin_id, "BTC")
        fd = _get("https://open-api.coinglass.com/public/v2/funding",
                  {"symbol": sym_cg},
                  headers={"coinglassSecret": COINGLASS_KEY}, ttl=300)
        if fd:
            try:
                funding_rate = float(fd["data"][0]["fundingRate"])
            except Exception:
                pass

    social_volume = social_sentiment = 0.0
    if LUNARCRUSH_KEY:
        sym_lc = {"bitcoin": "BTC", "ethereum": "ETH",
                  "solana": "SOL"}.get(coin_id, "BTC")
        lc = _get(f"https://lunarcrush.com/api4/public/coins/{sym_lc}/v1", {},
                  headers={"Authorization": f"Bearer {LUNARCRUSH_KEY}"}, ttl=600)
        if lc:
            try:
                d = lc["data"]
                social_volume    = float(d.get("social_volume_24h", 0)) / 1e6
                social_sentiment = float(d.get("sentiment", 50)) / 100
            except Exception:
                pass

    context.update({
        "price": price, "change_24h_pct": change_24h, "rank": rank,
        "volume_b": volume / 1e9, "ath_change_pct": ath_change,
        "fear_greed": fear_greed, "exchange_netflow": exchange_netflow,
        "whale_count_m": whale_count, "funding_rate": funding_rate,
        "social_volume_m": social_volume, "social_sentiment": social_sentiment,
    })

    features = np.array([
        price / 1e4, change_24h, rank, volume / 1e9, ath_change,
        fear_greed, exchange_netflow / 1e8, whale_count,
        funding_rate * 1000, social_volume, social_sentiment,
    ], dtype=float)
    return features, context


# =============================================================================
#  3. POLITICS
# =============================================================================

def fetch_politics_features(polymarket_slug: Optional[str] = None) -> tuple[np.ndarray, dict]:
    context = {}

    poly_prob = 0.5
    if polymarket_slug:
        pm = _get("https://gamma-api.polymarket.com/markets",
                  {"slug": polymarket_slug}, ttl=120)
        try:
            poly_prob = float(pm[0]["outcomePrices"][0])
        except Exception:
            pass

    # Metaculus disabled — requires auth as of 2026, returns 403
    metaculus_prob = 0.5

    news_sentiment = news_volume = 0.0
    if FINNHUB_API_KEY:
        fin_news = _get("https://finnhub.io/api/v1/news",
                        {"category": "politics", "token": FINNHUB_API_KEY}, ttl=1800)
        if fin_news:
            news_volume    = float(len(fin_news)) / 100.0
            news_sentiment = min(news_volume, 1.0)
    elif NEWSAPI_KEY:
        nd = _get("https://newsapi.org/v2/everything", {
            "q": "election OR congress OR policy",
            "language": "en", "apiKey": NEWSAPI_KEY, "pageSize": 1,
        }, ttl=3600)
        if nd:
            news_volume = float(nd.get("totalResults", 0)) / 1000.0

    approval_rating = 45.0
    days_to_event   = 30.0

    context.update({
        "polymarket_yes": poly_prob, "metaculus_prob": metaculus_prob,
        "news_sentiment": news_sentiment, "news_volume_k": news_volume,
        "approval_rating": approval_rating, "days_to_event": days_to_event,
    })

    features = np.array([
        poly_prob, metaculus_prob, days_to_event,
        news_sentiment, approval_rating, news_volume,
    ], dtype=float)
    return features, context


# =============================================================================
#  4. WEATHER
# =============================================================================

CITY_COORDS = {
    "new york": (40.71, -74.01), "nyc": (40.71, -74.01),
    "los angeles": (34.05, -118.24), "la": (34.05, -118.24),
    "chicago": (41.85, -87.65), "miami": (25.77, -80.19),
    "houston": (29.76, -95.37), "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33), "boston": (42.36, -71.06),
    "phoenix": (33.45, -112.07), "denver": (39.74, -104.98),
}

def fetch_weather_features(lat: float = 40.71, lon: float = -74.01,
                           city: str = "New York") -> tuple[np.ndarray, dict]:
    context = {"city": city, "lat": lat, "lon": lon}
    temp_max = precip = wind_max = spread = 0.0

    if TOMORROW_IO_KEY:
        tom = _get("https://api.tomorrow.io/v4/timelines", {
            "location": f"{lat},{lon}",
            "fields":   "temperatureMax,precipitationIntensityMax,windSpeedMax",
            "timesteps": "1d", "units": "metric", "apikey": TOMORROW_IO_KEY,
        }, ttl=1800)
        if tom:
            try:
                iv = tom["data"]["timelines"][0]["intervals"][0]["values"]
                temp_max = float(iv.get("temperatureMax", 0))
                precip   = float(iv.get("precipitationIntensityMax", 0))
                wind_max = float(iv.get("windSpeedMax", 0))
                context["source"] = "tomorrow_io"
            except Exception:
                pass

    if temp_max == 0.0:
        om = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
            "forecast_days": 7, "timezone": "UTC",
        }, ttl=1800)
        if om:
            daily    = om.get("daily", {})
            temps    = daily.get("temperature_2m_max", [0])
            precips  = daily.get("precipitation_sum", [0])
            winds    = daily.get("windspeed_10m_max", [0])
            temp_max = float(temps[0]) if temps else 0.0
            precip   = float(precips[0]) if precips else 0.0
            wind_max = float(winds[0]) if winds else 0.0
            spread   = float(np.std(temps)) if len(temps) > 1 else 0.0
            context["source"] = "open_meteo"

    hist_delta = temp_max - 15.0
    context.update({
        "temp_max_c": temp_max, "precip_mm": precip,
        "wind_max_kph": wind_max, "ensemble_spread": spread,
        "hist_delta": hist_delta,
    })
    features = np.array([temp_max, precip, wind_max, spread, hist_delta], dtype=float)
    return features, context


# =============================================================================
#  5. TECH / AI
# =============================================================================

def fetch_tech_features(company: str = "AAPL") -> tuple[np.ndarray, dict]:
    context = {"company": company}
    days_to_earnings = 45.0
    analyst_buy_pct  = 0.60
    eps_surprise     = 0.0
    insider_buy_sell = 0.0
    options_iv       = 0.0
    news_sentiment   = 0.0

    if FINNHUB_API_KEY:
        import datetime
        today = datetime.date.today()
        in_90 = today + datetime.timedelta(days=90)
        ec = _get("https://finnhub.io/api/v1/calendar/earnings", {
            "from": str(today), "to": str(in_90), "token": FINNHUB_API_KEY,
        }, ttl=3600)
        if ec:
            for ev in ec.get("earningsCalendar", []):
                if ev.get("symbol") == company:
                    try:
                        ed = datetime.date.fromisoformat(ev["date"])
                        days_to_earnings = float((ed - today).days)
                        eps_surprise     = float(ev.get("epsEstimate", 0) or 0)
                    except Exception:
                        pass
                    break
        grade = _get("https://finnhub.io/api/v1/stock/recommendation",
                     {"symbol": company, "token": FINNHUB_API_KEY}, ttl=3600)
        if grade:
            try:
                g = grade[0]
                total = g["buy"] + g["hold"] + g["sell"] + 1
                analyst_buy_pct = g["buy"] / total
            except Exception:
                pass
        news = _get("https://finnhub.io/api/v1/company-news", {
            "symbol": company, "from": str(today), "to": str(today),
            "token": FINNHUB_API_KEY,
        }, ttl=1800)
        if news:
            news_sentiment = min(len(news) / 20.0, 1.0)

    if FMP_API_KEY:
        insider = _get("https://financialmodelingprep.com/api/v4/insider-trading",
                       {"symbol": company, "page": 0, "apikey": FMP_API_KEY}, ttl=3600)
        if insider:
            buys  = sum(1 for t in insider if t.get("transactionType") == "P-Purchase")
            sells = sum(1 for t in insider if t.get("transactionType") == "S-Sale")
            insider_buy_sell = (buys - sells) / (buys + sells + 1)
        hist = _get(f"https://financialmodelingprep.com/api/v3/earnings-surprises/{company}",
                    {"apikey": FMP_API_KEY}, ttl=7200)
        if hist and len(hist) > 0:
            try:
                eps_surprise = float(hist[0].get("surprisePercentage", 0))
            except Exception:
                pass

    if POLYGON_API_KEY:
        opts = _get(f"https://api.polygon.io/v3/snapshot/options/{company}", {
            "apiKey": POLYGON_API_KEY, "limit": 10,
            "contract_type": "call", "order": "desc",
        }, ttl=300)
        if opts:
            try:
                ivs = [float(o["details"].get("implied_volatility", 0))
                       for o in opts.get("results", []) if o.get("details")]
                options_iv = float(np.mean(ivs)) if ivs else 0.0
            except Exception:
                pass

    if ALPHA_VANTAGE_KEY:
        av_news = _get("https://www.alphavantage.co/query", {
            "function": "NEWS_SENTIMENT", "tickers": company,
            "apikey": ALPHA_VANTAGE_KEY, "limit": "50",
        }, ttl=900)
        if av_news:
            scores = [float(a.get("overall_sentiment_score", 0))
                      for a in av_news.get("feed", [])]
            news_sentiment = float(np.mean(scores)) if scores else news_sentiment

    context.update({
        "days_to_earnings": days_to_earnings, "analyst_buy_pct": analyst_buy_pct,
        "eps_surprise": eps_surprise, "insider_buy_sell": insider_buy_sell,
        "options_iv": options_iv, "news_sentiment": news_sentiment,
    })
    features = np.array([
        days_to_earnings / 90, analyst_buy_pct, eps_surprise,
        insider_buy_sell, options_iv, news_sentiment,
    ], dtype=float)
    return features, context


# =============================================================================
#  6. SPORTS
# =============================================================================

def fetch_sports_features(team_a: str = "Team A", team_b: str = "Team B",
                          league: str = "NBA") -> tuple[np.ndarray, dict]:
    context = {"league": league, "team_a": team_a, "team_b": team_b}

    elo_a = elo_b = 1500.0
    home_advantage = 1.0
    vegas_spread   = 0.0
    pinnacle_prob  = 0.5
    home_win_pct   = 0.5
    away_win_pct   = 0.5

    sport_key = {
        "NBA": "basketball_nba", "NFL": "americanfootball_nfl",
        "MLB": "baseball_mlb",   "NHL": "icehockey_nhl",
    }.get(league, "basketball_nba")

    if ODDS_API_KEY:
        odds = _get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds", {
            "apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h",
            "oddsFormat": "decimal", "bookmakers": "pinnacle",
        }, ttl=3600)  # FIX v6: was 300 — burned ~576 credits/day at scan interval.
                      # Odds lines don't move meaningfully every 5 min.
                      # 3600s = ~48 calls/day = ~1,440/month, fits $30 tier.
        if odds:
            for game in odds:
                teams = [o["key"] for bm in game.get("bookmakers", [])
                         for m in bm.get("markets", []) for o in m.get("outcomes", [])]
                if any(team_a.lower() in t.lower() for t in teams):
                    for bm in game.get("bookmakers", []):
                        if bm.get("key") == "pinnacle":
                            outcomes = bm["markets"][0]["outcomes"]
                            for o in outcomes:
                                dec_odds = float(o["price"])
                                implied  = 1.0 / dec_odds
                                if team_a.lower() in o["name"].lower():
                                    pinnacle_prob = implied
                                    vegas_spread  = implied - 0.5
                    break

    if MYSPORTSFEEDS_KEY:
        lg_path = league.lower()
        msf = _get(
            f"https://api.mysportsfeeds.com/v2.1/pull/{lg_path}/current/team_stats.json",
            {}, auth=(MYSPORTSFEEDS_KEY + ":MYSPORTSFEEDS"), ttl=3600,
        )
        if msf:
            for ts in msf.get("teamStatsTotals", []):
                name   = ts.get("team", {}).get("abbreviation", "")
                wins   = float(ts.get("stats", {}).get("standings", {}).get("wins", 0))
                losses = float(ts.get("stats", {}).get("standings", {}).get("losses", 1))
                wp = wins / (wins + losses)
                if team_a.lower() in name.lower():
                    home_win_pct = wp
                    elo_a = 1200 + 600 * wp
                elif team_b.lower() in name.lower():
                    away_win_pct = wp
                    elo_b = 1200 + 600 * wp

    # ── ESPN unofficial fallback ──────────────────────────────────────────────
    if elo_a == 1500.0:
        sport_espn = {
            "NBA": "basketball/nba", "NFL": "football/nfl",
            "MLB": "baseball/mlb",   "NHL": "hockey/nhl",
            "NCAAB": "basketball/mens-college-basketball",
            "NCAAF": "football/college-football",
        }.get(league, "basketball/nba")

        espn = _get(
            f"https://site.api.espn.com/apis/site/v2/sports/{sport_espn}/scoreboard",
            {}, ttl=300,
        )
        if espn:
            for event in espn.get("events", []):
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                for c in competitors:
                    rec    = c.get("records", [{}])
                    record = rec[0].get("summary", "0-0") if rec else "0-0"

                    # Safely unpack W-L or W-L-OT (NHL has 3 parts)
                    parts = record.split("-")
                    try:
                        wins   = int(parts[0]) if len(parts) > 0 else 0
                        losses = int(parts[1]) if len(parts) > 1 else 0
                    except (ValueError, IndexError):
                        wins, losses = 0, 1

                    wp = wins / (wins + losses + 1)
                    nm = c.get("team", {}).get("displayName", "")
                    if team_a.lower() in nm.lower():
                        elo_a = 1200 + 600 * wp
                    elif team_b.lower() in nm.lower():
                        elo_b = 1200 + 600 * wp

    elo_diff = elo_a - elo_b
    context.update({
        "elo_a": elo_a, "elo_b": elo_b, "elo_diff": elo_diff,
        "home_advantage": home_advantage, "vegas_spread": vegas_spread,
        "pinnacle_prob": pinnacle_prob,
        "home_win_pct": home_win_pct, "away_win_pct": away_win_pct,
    })
    features = np.array([
        elo_a / 1500, elo_b / 1500, elo_diff / 300,
        home_advantage, vegas_spread, pinnacle_prob,
        home_win_pct, away_win_pct,
    ], dtype=float)
    return features, context