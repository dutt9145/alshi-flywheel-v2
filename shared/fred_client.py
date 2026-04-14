"""
fred_client.py (v1.0)

FRED API client for economic indicators.

FRED (Federal Reserve Economic Data) provides free access to:
  - Fed Funds Rate (DFF, DFEDTARU)
  - CPI / Inflation (CPIAUCSL, CPILFESL)
  - Unemployment (UNRATE, ICSA)
  - GDP (GDP, GDPC1)
  - Treasury Yields (DGS10, DGS2)
  - And 800,000+ other series

Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    from shared.fred_client import FredClient
    
    fred = FredClient(api_key=FRED_API_KEY)
    
    # Get latest CPI
    cpi = fred.get_latest("CPIAUCSL")
    
    # Get Fed Funds target range
    fed_rate = fred.get_fed_funds_rate()
    
    # Get unemployment rate
    unemployment = fred.get_unemployment()
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
class EconomicIndicator:
    """An economic data point."""
    series_id: str
    name: str
    value: float
    date: str
    units: str
    change_pct: Optional[float] = None  # Change from previous
    yoy_change_pct: Optional[float] = None  # Year-over-year change


@dataclass
class EconomicSnapshot:
    """Snapshot of key economic indicators."""
    fed_funds_rate: Optional[float]
    fed_funds_upper: Optional[float]
    cpi_yoy: Optional[float]
    core_cpi_yoy: Optional[float]
    unemployment_rate: Optional[float]
    jobless_claims: Optional[int]
    gdp_growth: Optional[float]
    treasury_10y: Optional[float]
    treasury_2y: Optional[float]
    yield_curve_spread: Optional[float]  # 10y - 2y
    pce_yoy: Optional[float]
    timestamp: datetime


class FredClient:
    """
    Client for FRED (Federal Reserve Economic Data) API.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Key series IDs
    SERIES = {
        # Fed Funds
        "fed_funds_effective": "DFF",
        "fed_funds_upper": "DFEDTARU",
        "fed_funds_lower": "DFEDTARL",
        
        # Inflation
        "cpi_all": "CPIAUCSL",           # CPI All Urban Consumers
        "cpi_core": "CPILFESL",          # CPI Less Food & Energy
        "pce": "PCEPI",                  # PCE Price Index
        "pce_core": "PCEPILFE",          # Core PCE
        
        # Employment
        "unemployment": "UNRATE",         # Unemployment Rate
        "jobless_claims": "ICSA",         # Initial Jobless Claims
        "nonfarm_payrolls": "PAYEMS",     # Nonfarm Payrolls
        
        # GDP
        "gdp": "GDP",                     # Nominal GDP
        "gdp_real": "GDPC1",              # Real GDP
        
        # Treasury Yields
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DGS3MO",
        "treasury_30y": "DGS30",
        
        # Housing
        "housing_starts": "HOUST",
        "existing_home_sales": "EXHOSLUSM495S",
        
        # Consumer
        "consumer_sentiment": "UMCSENT",
        "retail_sales": "RSXFS",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 3600,  # Cache for 1 hour (data updates slowly)
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache_ttl_sec = cache_ttl_sec
        
        # Cache: series_id -> (timestamp, data)
        self._cache: Dict[str, Tuple[float, any]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # FRED allows ~120 requests/min
        
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — economic data unavailable")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get cached data if not expired."""
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self.cache_ttl_sec:
                return data
        return None
    
    def _set_cache(self, key: str, data: any) -> None:
        """Cache data."""
        self._cache[key] = (time.time(), data)
    
    def get_series(
        self,
        series_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Fetch a FRED series.
        
        Returns list of observations (most recent first).
        """
        if not self.api_key:
            return []
        
        cache_key = f"series_{series_id}_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            observations = data.get("observations", [])
            self._set_cache(cache_key, observations)
            
            return observations
            
        except Exception as e:
            logger.warning("[FRED] Failed to fetch %s: %s", series_id, e)
            return []
    
    def get_latest(self, series_id: str) -> Optional[EconomicIndicator]:
        """Get the latest value for a series."""
        obs = self.get_series(series_id, limit=2)
        if not obs:
            return None
        
        latest = obs[0]
        value_str = latest.get("value", ".")
        
        if value_str == ".":
            # Missing value, try next
            if len(obs) > 1:
                latest = obs[1]
                value_str = latest.get("value", ".")
        
        if value_str == ".":
            return None
        
        try:
            value = float(value_str)
        except ValueError:
            return None
        
        # Calculate change if we have previous
        change_pct = None
        if len(obs) > 1:
            prev_str = obs[1].get("value", ".")
            if prev_str != ".":
                try:
                    prev = float(prev_str)
                    if prev != 0:
                        change_pct = ((value - prev) / abs(prev)) * 100
                except ValueError:
                    pass
        
        return EconomicIndicator(
            series_id=series_id,
            name=series_id,
            value=value,
            date=latest.get("date", ""),
            units="",
            change_pct=change_pct,
        )
    
    def get_fed_funds_rate(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current Fed Funds target range.
        
        Returns (lower, upper) bounds in percent.
        """
        lower = self.get_latest(self.SERIES["fed_funds_lower"])
        upper = self.get_latest(self.SERIES["fed_funds_upper"])
        
        return (
            lower.value if lower else None,
            upper.value if upper else None,
        )
    
    def get_cpi_yoy(self) -> Optional[float]:
        """Get year-over-year CPI inflation rate."""
        obs = self.get_series(self.SERIES["cpi_all"], limit=13)
        if len(obs) < 13:
            return None
        
        try:
            current = float(obs[0]["value"])
            year_ago = float(obs[12]["value"])
            if year_ago > 0:
                return ((current - year_ago) / year_ago) * 100
        except (ValueError, KeyError):
            pass
        
        return None
    
    def get_core_cpi_yoy(self) -> Optional[float]:
        """Get year-over-year Core CPI inflation rate."""
        obs = self.get_series(self.SERIES["cpi_core"], limit=13)
        if len(obs) < 13:
            return None
        
        try:
            current = float(obs[0]["value"])
            year_ago = float(obs[12]["value"])
            if year_ago > 0:
                return ((current - year_ago) / year_ago) * 100
        except (ValueError, KeyError):
            pass
        
        return None
    
    def get_unemployment(self) -> Optional[float]:
        """Get unemployment rate."""
        indicator = self.get_latest(self.SERIES["unemployment"])
        return indicator.value if indicator else None
    
    def get_jobless_claims(self) -> Optional[int]:
        """Get initial jobless claims (weekly)."""
        indicator = self.get_latest(self.SERIES["jobless_claims"])
        return int(indicator.value * 1000) if indicator else None  # In thousands
    
    def get_treasury_yields(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get treasury yields.
        
        Returns (2y, 10y, spread).
        """
        t2y = self.get_latest(self.SERIES["treasury_2y"])
        t10y = self.get_latest(self.SERIES["treasury_10y"])
        
        y2 = t2y.value if t2y else None
        y10 = t10y.value if t10y else None
        spread = (y10 - y2) if (y10 is not None and y2 is not None) else None
        
        return (y2, y10, spread)
    
    def get_economic_snapshot(self) -> EconomicSnapshot:
        """Get a snapshot of key economic indicators."""
        fed_lower, fed_upper = self.get_fed_funds_rate()
        t2y, t10y, spread = self.get_treasury_yields()
        
        return EconomicSnapshot(
            fed_funds_rate=fed_lower,
            fed_funds_upper=fed_upper,
            cpi_yoy=self.get_cpi_yoy(),
            core_cpi_yoy=self.get_core_cpi_yoy(),
            unemployment_rate=self.get_unemployment(),
            jobless_claims=self.get_jobless_claims(),
            gdp_growth=None,  # Quarterly, complex calculation
            treasury_10y=t10y,
            treasury_2y=t2y,
            yield_curve_spread=spread,
            pce_yoy=None,  # Similar to CPI calculation
            timestamp=datetime.now(timezone.utc),
        )
    
    def predict_rate_decision(
        self,
        current_rate: float,
        target_rate: float,
        decision: str,  # "hike", "cut", "hold"
    ) -> float:
        """
        Predict probability of a Fed rate decision.
        
        Uses current data to estimate likelihood.
        
        Args:
            current_rate: Current Fed Funds upper bound
            target_rate: Rate level market is asking about
            decision: "hike", "cut", or "hold"
            
        Returns:
            Probability estimate (0-1)
        """
        snapshot = self.get_economic_snapshot()
        
        # Base probability
        prob = 0.5
        
        # CPI factor: high inflation → more likely to hike
        if snapshot.cpi_yoy is not None:
            if snapshot.cpi_yoy > 3.0:
                # Inflation above target
                if decision == "hike":
                    prob += 0.15
                elif decision == "cut":
                    prob -= 0.20
            elif snapshot.cpi_yoy < 2.0:
                # Inflation below target
                if decision == "cut":
                    prob += 0.10
                elif decision == "hike":
                    prob -= 0.15
        
        # Unemployment factor: high unemployment → more likely to cut
        if snapshot.unemployment_rate is not None:
            if snapshot.unemployment_rate > 5.0:
                if decision == "cut":
                    prob += 0.15
                elif decision == "hike":
                    prob -= 0.15
            elif snapshot.unemployment_rate < 4.0:
                if decision == "hike":
                    prob += 0.10
        
        # Yield curve factor: inverted → recession fears → likely to cut
        if snapshot.yield_curve_spread is not None:
            if snapshot.yield_curve_spread < 0:
                # Inverted yield curve
                if decision == "cut":
                    prob += 0.10
                elif decision == "hike":
                    prob -= 0.10
        
        # Distance from target
        if current_rate and target_rate:
            diff = target_rate - current_rate
            if decision == "hike" and diff > 0:
                # Market expects hike to target
                prob += min(0.15, diff * 0.05)
            elif decision == "cut" and diff < 0:
                # Market expects cut to target
                prob += min(0.15, abs(diff) * 0.05)
        
        return max(0.05, min(0.95, prob))


# ── Singleton ─────────────────────────────────────────────────────────────────

_client: Optional[FredClient] = None


def get_fred_client() -> FredClient:
    """Get the global FRED client."""
    global _client
    if _client is None:
        _client = FredClient()
    return _client