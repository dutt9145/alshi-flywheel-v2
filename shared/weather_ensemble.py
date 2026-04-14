"""
weather_ensemble.py (v1.0)

Weather ensemble: Cross-check NOAA with Open-Meteo for calibrated forecasts.

When two independent weather models agree → high confidence
When they diverge → reduce confidence or skip

NOAA is already integrated (noaa_client.py). This module adds Open-Meteo
as a second opinion and provides ensemble logic.

Usage:
    from shared.weather_ensemble import WeatherEnsemble
    
    ensemble = WeatherEnsemble()
    
    result = ensemble.get_temperature_probability(
        city="Dallas",
        threshold_f=90,
        comparison="above",
        date="2026-04-15",
    )
    
    if result.confidence > 0.7:
        # High confidence — both models agree
        trade(result.probability)
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


@dataclass
class WeatherForecast:
    """Forecast from a single source."""
    source: str
    temperature_f: float
    temperature_c: float
    precip_probability: float  # 0-1
    precip_inches: float
    snow_inches: float
    wind_mph: float
    confidence: float  # Source's self-reported confidence
    forecast_time: Optional[datetime] = None


@dataclass
class EnsembleResult:
    """Result of ensemble forecast."""
    probability: float          # Ensemble P(event)
    confidence: float           # 0-1, based on model agreement
    noaa_prob: Optional[float]  # NOAA's probability
    openmeteo_prob: Optional[float]  # Open-Meteo's probability
    divergence: float           # |noaa - openmeteo|
    sources_agree: bool         # True if divergence < threshold
    reason: str


class WeatherEnsemble:
    """
    Ensemble weather forecasting using NOAA + Open-Meteo.
    
    Open-Meteo is free, no API key required, and uses ECMWF/GFS models.
    """
    
    # Open-Meteo API (free, no key)
    OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
    
    # City coordinates (add more as needed)
    CITY_COORDS = {
        # Texas
        "dallas": (32.7767, -96.7970),
        "houston": (29.7604, -95.3698),
        "austin": (30.2672, -97.7431),
        "san antonio": (29.4241, -98.4936),
        # Northeast
        "new york": (40.7128, -74.0060),
        "boston": (42.3601, -71.0589),
        "philadelphia": (39.9526, -75.1652),
        "washington": (38.9072, -77.0369),
        "dc": (38.9072, -77.0369),
        # Southeast
        "miami": (25.7617, -80.1918),
        "atlanta": (33.7490, -84.3880),
        "orlando": (28.5383, -81.3792),
        "tampa": (27.9506, -82.4572),
        # Midwest
        "chicago": (41.8781, -87.6298),
        "detroit": (42.3314, -83.0458),
        "minneapolis": (44.9778, -93.2650),
        "denver": (39.7392, -104.9903),
        # West
        "los angeles": (34.0522, -118.2437),
        "san francisco": (37.7749, -122.4194),
        "seattle": (47.6062, -122.3321),
        "phoenix": (33.4484, -112.0740),
        "las vegas": (36.1699, -115.1398),
        # Other
        "portland": (45.5152, -122.6784),
        "salt lake city": (40.7608, -111.8910),
        "kansas city": (39.0997, -94.5786),
        "st louis": (38.6270, -90.1994),
        "new orleans": (29.9511, -90.0715),
        "nashville": (36.1627, -86.7816),
        "charlotte": (35.2271, -80.8431),
        "raleigh": (35.7796, -78.6382),
        "indianapolis": (39.7684, -86.1581),
        "columbus": (39.9612, -82.9988),
        "cleveland": (41.4993, -81.6944),
        "pittsburgh": (40.4406, -79.9959),
        "baltimore": (39.2904, -76.6122),
        "milwaukee": (43.0389, -87.9065),
        "sacramento": (38.5816, -121.4944),
        "san diego": (32.7157, -117.1611),
        "oklahoma city": (35.4676, -97.5164),
        "memphis": (35.1495, -90.0490),
        "louisville": (38.2527, -85.7585),
        "cincinnati": (39.1031, -84.5120),
        "buffalo": (42.8864, -78.8784),
        "albany": (42.6526, -73.7562),
        "hartford": (41.7658, -72.6734),
        "providence": (41.8240, -71.4128),
    }
    
    # Divergence thresholds
    TEMP_DIVERGENCE_THRESHOLD_F = 5.0   # 5°F — models disagree
    PRECIP_DIVERGENCE_THRESHOLD = 0.20  # 20% probability difference
    
    # Confidence adjustments
    HIGH_AGREEMENT_BOOST = 1.15     # Boost when models agree within 2°F
    LOW_AGREEMENT_PENALTY = 0.70    # Penalty when models diverge >5°F
    
    def __init__(
        self,
        cache_ttl_sec: int = 600,  # Cache forecasts for 10 minutes
        enabled: bool = True,
    ):
        self.cache_ttl_sec = cache_ttl_sec
        self.enabled = enabled
        
        # Cache
        self._cache: Dict[str, Tuple[float, any]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cached(self, cache_key: str) -> Optional[any]:
        """Get cached data if not expired."""
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl_sec:
                return data
        return None
    
    def _set_cache(self, cache_key: str, data: any) -> None:
        """Cache data with timestamp."""
        self._cache[cache_key] = (time.time(), data)
    
    def _normalize_city(self, city: str) -> str:
        """Normalize city name for lookup."""
        return city.lower().strip().replace("_", " ").replace("-", " ")
    
    def _get_coords(self, city: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city."""
        normalized = self._normalize_city(city)
        return self.CITY_COORDS.get(normalized)
    
    def _c_to_f(self, c: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return c * 9/5 + 32
    
    def _f_to_c(self, f: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (f - 32) * 5/9
    
    # ── Open-Meteo API ────────────────────────────────────────────────────────
    
    def _fetch_openmeteo(
        self,
        lat: float,
        lon: float,
        date: str,  # YYYY-MM-DD
    ) -> Optional[WeatherForecast]:
        """
        Fetch forecast from Open-Meteo.
        
        Returns WeatherForecast or None on error.
        """
        cache_key = f"openmeteo_{lat}_{lon}_{date}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,snowfall_sum,wind_speed_10m_max",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "America/New_York",
                "start_date": date,
                "end_date": date,
            }
            
            resp = requests.get(self.OPENMETEO_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            daily = data.get("daily", {})
            
            if not daily.get("temperature_2m_max"):
                return None
            
            temp_max = daily["temperature_2m_max"][0]
            temp_min = daily["temperature_2m_min"][0]
            temp_avg = (temp_max + temp_min) / 2
            
            forecast = WeatherForecast(
                source="open-meteo",
                temperature_f=temp_max,  # Use high for threshold checks
                temperature_c=self._f_to_c(temp_max),
                precip_probability=(daily.get("precipitation_probability_max", [0])[0] or 0) / 100,
                precip_inches=daily.get("precipitation_sum", [0])[0] or 0,
                snow_inches=daily.get("snowfall_sum", [0])[0] or 0,
                wind_mph=daily.get("wind_speed_10m_max", [0])[0] or 0,
                confidence=0.85,  # Open-Meteo uses ECMWF which is quite good
            )
            
            self._set_cache(cache_key, forecast)
            logger.debug(
                "[WEATHER] Open-Meteo: %s %.1f°F precip=%.0f%%",
                date, forecast.temperature_f, forecast.precip_probability * 100,
            )
            return forecast
            
        except Exception as e:
            logger.warning("[WEATHER] Open-Meteo fetch failed: %s", e)
            return None
    
    # ── NOAA Integration ──────────────────────────────────────────────────────
    
    def _fetch_noaa(
        self,
        lat: float,
        lon: float,
        date: str,
    ) -> Optional[WeatherForecast]:
        """
        Fetch forecast from NOAA via weather.gov API.
        
        Note: This is a simplified version. Your existing noaa_client.py
        has more sophisticated NOAA integration — you may want to use that
        instead and just call it from here.
        """
        cache_key = f"noaa_{lat}_{lon}_{date}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            # Step 1: Get grid point
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            headers = {"User-Agent": "KalshiFlywheel/1.0"}
            
            resp = requests.get(points_url, headers=headers, timeout=10)
            resp.raise_for_status()
            points_data = resp.json()
            
            forecast_url = points_data.get("properties", {}).get("forecast")
            if not forecast_url:
                return None
            
            # Step 2: Get forecast
            self._rate_limit()
            resp = requests.get(forecast_url, headers=headers, timeout=10)
            resp.raise_for_status()
            forecast_data = resp.json()
            
            periods = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                return None
            
            # Find the period matching our date
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            
            for period in periods:
                period_start = period.get("startTime", "")
                if period_start:
                    try:
                        period_date = datetime.fromisoformat(period_start.replace("Z", "+00:00")).date()
                        if period_date == target_date and period.get("isDaytime", True):
                            temp_f = period.get("temperature", 70)
                            precip_pct = period.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
                            
                            forecast = WeatherForecast(
                                source="noaa",
                                temperature_f=temp_f,
                                temperature_c=self._f_to_c(temp_f),
                                precip_probability=precip_pct / 100,
                                precip_inches=0,  # NOAA doesn't give this in simple forecast
                                snow_inches=0,
                                wind_mph=0,
                                confidence=0.90,  # NOAA is generally reliable
                            )
                            
                            self._set_cache(cache_key, forecast)
                            logger.debug(
                                "[WEATHER] NOAA: %s %.1f°F precip=%.0f%%",
                                date, forecast.temperature_f, forecast.precip_probability * 100,
                            )
                            return forecast
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.warning("[WEATHER] NOAA fetch failed: %s", e)
            return None
    
    # ── Ensemble Logic ────────────────────────────────────────────────────────
    
    def get_temperature_probability(
        self,
        city: str,
        threshold_f: float,
        comparison: str,  # "above" or "below"
        date: str,  # YYYY-MM-DD
    ) -> EnsembleResult:
        """
        Get ensemble probability of temperature crossing a threshold.
        
        Args:
            city: City name (e.g., "Dallas", "New York")
            threshold_f: Temperature threshold in Fahrenheit
            comparison: "above" or "below"
            date: Target date (YYYY-MM-DD)
            
        Returns:
            EnsembleResult with probability and confidence
        """
        if not self.enabled:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason="Weather ensemble disabled",
            )
        
        coords = self._get_coords(city)
        if not coords:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason=f"Unknown city: {city}",
            )
        
        lat, lon = coords
        
        # Fetch from both sources
        noaa = self._fetch_noaa(lat, lon, date)
        openmeteo = self._fetch_openmeteo(lat, lon, date)
        
        if not noaa and not openmeteo:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason="No weather data available",
            )
        
        # Calculate individual probabilities
        # Using a simple model: P = sigmoid((forecast - threshold) / uncertainty)
        # For simplicity, assume ±3°F uncertainty
        
        def temp_to_prob(temp_f: float, threshold: float, above: bool) -> float:
            """Convert temperature forecast to probability of crossing threshold."""
            diff = temp_f - threshold
            if not above:
                diff = -diff
            # Sigmoid with ~3°F uncertainty
            import math
            return 1 / (1 + math.exp(-diff / 3))
        
        above = comparison.lower() == "above"
        
        noaa_prob = None
        openmeteo_prob = None
        
        if noaa:
            noaa_prob = temp_to_prob(noaa.temperature_f, threshold_f, above)
        if openmeteo:
            openmeteo_prob = temp_to_prob(openmeteo.temperature_f, threshold_f, above)
        
        # Ensemble combination
        if noaa_prob is not None and openmeteo_prob is not None:
            # Both available — weighted average
            # Weight NOAA slightly higher (0.55) due to local calibration
            ensemble_prob = 0.55 * noaa_prob + 0.45 * openmeteo_prob
            
            # Check divergence
            temp_divergence = abs(noaa.temperature_f - openmeteo.temperature_f)
            prob_divergence = abs(noaa_prob - openmeteo_prob)
            
            if temp_divergence <= 2.0:
                # Strong agreement
                confidence = min(1.0, 0.90 * self.HIGH_AGREEMENT_BOOST)
                sources_agree = True
                reason = f"Strong agreement: NOAA={noaa.temperature_f:.0f}°F Open-Meteo={openmeteo.temperature_f:.0f}°F"
            elif temp_divergence <= self.TEMP_DIVERGENCE_THRESHOLD_F:
                # Moderate agreement
                confidence = 0.80
                sources_agree = True
                reason = f"Moderate agreement: NOAA={noaa.temperature_f:.0f}°F Open-Meteo={openmeteo.temperature_f:.0f}°F"
            else:
                # Divergence
                confidence = max(0.4, 0.80 * self.LOW_AGREEMENT_PENALTY)
                sources_agree = False
                reason = f"Models diverge: NOAA={noaa.temperature_f:.0f}°F Open-Meteo={openmeteo.temperature_f:.0f}°F (Δ{temp_divergence:.0f}°F)"
            
            return EnsembleResult(
                probability=ensemble_prob,
                confidence=confidence,
                noaa_prob=noaa_prob,
                openmeteo_prob=openmeteo_prob,
                divergence=prob_divergence,
                sources_agree=sources_agree,
                reason=reason,
            )
        
        # Only one source available
        if noaa_prob is not None:
            return EnsembleResult(
                probability=noaa_prob,
                confidence=0.70,  # Single source = lower confidence
                noaa_prob=noaa_prob,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason=f"NOAA only: {noaa.temperature_f:.0f}°F",
            )
        else:
            return EnsembleResult(
                probability=openmeteo_prob,
                confidence=0.65,
                noaa_prob=None,
                openmeteo_prob=openmeteo_prob,
                divergence=0,
                sources_agree=False,
                reason=f"Open-Meteo only: {openmeteo.temperature_f:.0f}°F",
            )
    
    def get_precipitation_probability(
        self,
        city: str,
        threshold_inches: float,
        comparison: str,  # "above" or "below"
        date: str,
    ) -> EnsembleResult:
        """
        Get ensemble probability of precipitation crossing a threshold.
        """
        if not self.enabled:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason="Weather ensemble disabled",
            )
        
        coords = self._get_coords(city)
        if not coords:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason=f"Unknown city: {city}",
            )
        
        lat, lon = coords
        
        # For precipitation, we primarily use Open-Meteo since it gives amounts
        openmeteo = self._fetch_openmeteo(lat, lon, date)
        noaa = self._fetch_noaa(lat, lon, date)
        
        if not openmeteo:
            return EnsembleResult(
                probability=0.5,
                confidence=0.0,
                noaa_prob=None,
                openmeteo_prob=None,
                divergence=0,
                sources_agree=False,
                reason="No precipitation data available",
            )
        
        above = comparison.lower() == "above"
        
        # Simple threshold probability from Open-Meteo
        if above:
            # P(precip > threshold)
            if openmeteo.precip_inches > threshold_inches:
                openmeteo_prob = 0.85  # Forecast already exceeds
            else:
                # Scale by how close we are
                ratio = openmeteo.precip_inches / (threshold_inches + 0.01)
                openmeteo_prob = min(0.8, ratio * openmeteo.precip_probability)
        else:
            # P(precip < threshold)
            if openmeteo.precip_inches < threshold_inches:
                openmeteo_prob = 0.85
            else:
                ratio = threshold_inches / (openmeteo.precip_inches + 0.01)
                openmeteo_prob = min(0.8, ratio * (1 - openmeteo.precip_probability))
        
        # Cross-check with NOAA precip probability
        noaa_prob = None
        if noaa:
            # NOAA gives probability of any precip, not amounts
            if above and threshold_inches <= 0.1:
                # "Above 0.1 inches" ≈ NOAA's precip probability
                noaa_prob = noaa.precip_probability
            elif not above and threshold_inches >= 0.1:
                noaa_prob = 1 - noaa.precip_probability
        
        # Combine
        if noaa_prob is not None:
            ensemble_prob = 0.5 * openmeteo_prob + 0.5 * noaa_prob
            divergence = abs(openmeteo_prob - noaa_prob)
            sources_agree = divergence < self.PRECIP_DIVERGENCE_THRESHOLD
            confidence = 0.75 if sources_agree else 0.55
            reason = f"Ensemble: Open-Meteo={openmeteo_prob:.0%} NOAA={noaa_prob:.0%}"
        else:
            ensemble_prob = openmeteo_prob
            divergence = 0
            sources_agree = False
            confidence = 0.60
            reason = f"Open-Meteo only: {openmeteo.precip_inches:.2f}in forecast"
        
        return EnsembleResult(
            probability=ensemble_prob,
            confidence=confidence,
            noaa_prob=noaa_prob,
            openmeteo_prob=openmeteo_prob,
            divergence=divergence,
            sources_agree=sources_agree,
            reason=reason,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_ensemble: Optional[WeatherEnsemble] = None


def get_weather_ensemble() -> WeatherEnsemble:
    """Get the global weather ensemble instance."""
    global _ensemble
    if _ensemble is None:
        _ensemble = WeatherEnsemble()
    return _ensemble