# shared/noaa_client.py
# ---------------------------------------------------------------
# NOAA Weather API client — no API key required
# ---------------------------------------------------------------

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NOAA_BASE = "https://api.weather.gov"
HEADERS = {
    "User-Agent": "KalshiFlywheelBot/1.0 (contact@yourdomain.com)",
    "Accept": "application/geo+json",
}

# ── City → (lat, lon) master lookup ─────────────────────────────
CITY_COORDS = {
    "new york":     (40.7128, -74.0060),
    "nyc":          (40.7128, -74.0060),
    "los angeles":  (34.0522, -118.2437),
    "la":           (34.0522, -118.2437),
    "chicago":      (41.8781, -87.6298),
    "houston":      (29.7604, -95.3698),
    "miami":        (25.7617, -80.1918),
    "dallas":       (32.7767, -96.7970),
    "atlanta":      (33.7490, -84.3880),
    "seattle":      (47.6062, -122.3321),
    "denver":       (39.7392, -104.9903),
    "boston":       (42.3601, -71.0589),
    "phoenix":      (33.4484, -112.0740),
    "washington":   (38.9072, -77.0369),
    "dc":           (38.9072, -77.0369),
    "philadelphia": (39.9526, -75.1652),
    "san francisco":(37.7749, -122.4194),
    "minneapolis":  (44.9778, -93.2650),
    "detroit":      (42.3314, -83.0458),
    "las vegas":    (36.1699, -115.1398),
    "new orleans":  (29.9511, -90.0715),
}


# ═══════════════════════════════════════════════════════════════
#  NOAA API CALLS
# ═══════════════════════════════════════════════════════════════

async def get_grid_meta(lat: float, lon: float) -> Optional[dict]:
    """Resolve lat/lon to NOAA grid metadata."""
    url = f"{NOAA_BASE}/points/{lat:.4f},{lon:.4f}"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            props = r.json()["properties"]
            return {
                "forecast_url":        props["forecast"],
                "forecast_hourly_url": props["forecastHourly"],
                "state":               props["relativeLocation"]["properties"]["state"],
                "city":                props["relativeLocation"]["properties"]["city"],
                "office":              props["cwa"],
                "grid_x":              props["gridX"],
                "grid_y":              props["gridY"],
            }
    except Exception as e:
        logger.warning(f"[NOAA] get_grid_meta failed for ({lat},{lon}): {e}")
        return None


async def get_forecast(forecast_url: str) -> list[dict]:
    """Fetch 7-day forecast periods."""
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
            r = await client.get(forecast_url)
            r.raise_for_status()
            return r.json()["properties"]["periods"]
    except Exception as e:
        logger.warning(f"[NOAA] get_forecast failed: {e}")
        return []


async def get_hourly_forecast(hourly_url: str) -> list[dict]:
    """Fetch hourly forecast periods."""
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
            r = await client.get(hourly_url)
            r.raise_for_status()
            return r.json()["properties"]["periods"]
    except Exception as e:
        logger.warning(f"[NOAA] get_hourly failed: {e}")
        return []


async def get_active_alerts(state: str) -> list[dict]:
    """Fetch active NWS alerts for a state (2-letter code)."""
    url = f"{NOAA_BASE}/alerts/active?area={state}"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json().get("features", [])
    except Exception as e:
        logger.warning(f"[NOAA] get_active_alerts failed for {state}: {e}")
        return []


async def fetch_weather_package(lat: float, lon: float) -> Optional[dict]:
    """
    Master call — returns forecast + hourly + alerts for a location.
    Returns None if grid metadata lookup fails.
    """
    meta = await get_grid_meta(lat, lon)
    if not meta:
        return None

    forecast, hourly, alerts = await asyncio.gather(
        get_forecast(meta["forecast_url"]),
        get_hourly_forecast(meta["forecast_hourly_url"]),
        get_active_alerts(meta["state"]),
    )

    return {
        "meta":       meta,
        "forecast":   forecast,
        "hourly":     hourly,
        "alerts":     alerts,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
#  CITY LOOKUP
# ═══════════════════════════════════════════════════════════════

def city_to_coords(city_string: str) -> Optional[tuple[float, float]]:
    """Fuzzy city name → (lat, lon). Returns None if no match."""
    normalized = city_string.lower().strip()
    for key, coords in CITY_COORDS.items():
        if key in normalized or normalized in key:
            return coords
    return None


# ═══════════════════════════════════════════════════════════════
#  MARKET TEXT PARSERS
#  Imported by sector_bots.WeatherBot._get_noaa_prior()
# ═══════════════════════════════════════════════════════════════

def parse_market_date(title: str, close_time: Optional[str] = None) -> str:
    """
    Extract target date from market title or fall back to close_time.
    Returns 'YYYY-MM-DD'.
    """
    today = datetime.now(timezone.utc)

    if re.search(r"\btoday\b", title, re.IGNORECASE):
        return today.strftime("%Y-%m-%d")
    if re.search(r"\btomorrow\b", title, re.IGNORECASE):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
    }
    m = re.search(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{1,2})",
        title, re.IGNORECASE
    )
    if m:
        month_num = months[m.group(1).lower()]
        day = int(m.group(2))
        year = today.year
        candidate = datetime(year, month_num, day, tzinfo=timezone.utc)
        if candidate < today - timedelta(days=1):
            candidate = candidate.replace(year=year + 1)
        return candidate.strftime("%Y-%m-%d")

    if close_time:
        try:
            return close_time[:10]
        except Exception:
            pass

    return (today + timedelta(days=1)).strftime("%Y-%m-%d")


def parse_temp_threshold(title: str) -> Optional[float]:
    """
    Extract temperature threshold from market title.
    e.g. 'exceed 75°F' → 75.0
    """
    m = re.search(r"(\d+\.?\d*)\s*(?:°\s*F|degrees?\s*F|°|degrees?)", title, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def classify_weather_market(title: str) -> str:
    """
    Returns one of: 'temp_high', 'temp_low', 'precipitation',
                    'severe_weather', 'snow', 'unknown'
    """
    t = title.lower()
    if any(kw in t for kw in ["snow", "snowfall", "blizzard"]):
        return "snow"
    if any(kw in t for kw in ["storm", "tornado", "hurricane", "severe", "alert"]):
        return "severe_weather"
    if any(kw in t for kw in ["rain", "precip", "shower", "wet"]):
        return "precipitation"
    if "low" in t and not any(kw in t for kw in ["temp", "°", "degree", "high"]):
        return "temp_low"
    if any(kw in t for kw in ["high", "exceed", "above", "over", "temp", "°", "degree"]):
        return "temp_high"
    return "unknown"


# ═══════════════════════════════════════════════════════════════
#  SIGNAL EXTRACTORS
# ═══════════════════════════════════════════════════════════════

def extract_temp_signals(hourly: list[dict], target_date_str: str) -> dict:
    """
    Compute high/low/mean temp for a target date from hourly forecasts.
    Returns {} if no matching periods found.
    """
    temps = []
    for period in hourly:
        start = period.get("startTime", "")
        if target_date_str in start:
            t = period.get("temperature")
            unit = period.get("temperatureUnit", "F")
            if t is not None:
                if unit == "C":
                    t = t * 9 / 5 + 32
                temps.append(float(t))

    if not temps:
        return {}

    return {
        "high_f":       max(temps),
        "low_f":        min(temps),
        "mean_f":       sum(temps) / len(temps),
        "sample_count": len(temps),
    }


def extract_precip_signals(forecast: list[dict], target_date_str: str) -> dict:
    """
    Scan forecast periods for precipitation probability on a target date.
    """
    precip_keywords = ["rain", "shower", "storm", "snow", "sleet", "drizzle", "precip"]
    best_pop = 0.0
    has_mention = False
    short = ""

    for period in forecast:
        start = period.get("startTime", "")
        if target_date_str not in start:
            continue

        pop = period.get("probabilityOfPrecipitation", {})
        if isinstance(pop, dict):
            val = pop.get("value") or 0
            best_pop = max(best_pop, float(val))

        detail = (period.get("detailedForecast") or "").lower()
        short_fc = period.get("shortForecast") or ""
        if any(kw in detail for kw in precip_keywords):
            has_mention = True
            short = short_fc

    return {
        "precip_pct":         best_pop,
        "has_precip_mention": has_mention,
        "short_forecast":     short,
    }


def extract_alert_signals(alerts: list[dict]) -> dict:
    """
    Scan active alerts for high-severity events.
    """
    severity_rank = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
    alert_types = []
    best_severity = "Unknown"

    for feature in alerts:
        props = feature.get("properties", {})
        event = props.get("event", "")
        severity = props.get("severity", "Unknown")
        if event:
            alert_types.append(event)
        if severity_rank.get(severity, 0) > severity_rank.get(best_severity, 0):
            best_severity = severity

    return {
        "has_severe_alert": best_severity in ("Extreme", "Severe"),
        "alert_types":      alert_types,
        "highest_severity": best_severity,
    }