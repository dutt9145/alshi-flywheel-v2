"""
market_data_client.py (v1.0)

Market data client for stocks, VIX, and options data.

Uses Yahoo Finance (free, no API key required) for:
  - Stock prices and changes
  - VIX (volatility index)
  - Options data (put/call ratio)
  - Earnings dates

Usage:
    from shared.market_data_client import MarketDataClient
    
    mkt = MarketDataClient()
    
    # Get stock price
    price = mkt.get_stock_price("AAPL")
    
    # Get VIX
    vix = mkt.get_vix()
    
    # Get market snapshot
    snapshot = mkt.get_market_snapshot()
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """Stock price data."""
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    timestamp: datetime


@dataclass
class VIXData:
    """VIX volatility data."""
    current: float
    change: float
    change_pct: float
    high_52w: float
    low_52w: float
    level: str  # "low", "normal", "elevated", "high", "extreme"


@dataclass
class MarketSnapshot:
    """Snapshot of key market indicators."""
    spy_price: Optional[float]
    spy_change_pct: Optional[float]
    qqq_price: Optional[float]
    qqq_change_pct: Optional[float]
    vix: Optional[float]
    vix_level: str
    put_call_ratio: Optional[float]
    market_sentiment: str  # "fear", "neutral", "greed"
    timestamp: datetime


class MarketDataClient:
    """
    Client for market data using Yahoo Finance.
    
    No API key required — uses public endpoints.
    """
    
    # Yahoo Finance endpoints
    YF_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
    YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    # Key symbols
    SYMBOLS = {
        "spy": "SPY",      # S&P 500 ETF
        "qqq": "QQQ",      # Nasdaq 100 ETF
        "dia": "DIA",      # Dow Jones ETF
        "iwm": "IWM",      # Russell 2000 ETF
        "vix": "^VIX",     # VIX Volatility Index
        "vvix": "^VVIX",   # VIX of VIX
        "tlt": "TLT",      # 20+ Year Treasury ETF
        "gld": "GLD",      # Gold ETF
        "uso": "USO",      # Oil ETF
    }
    
    # VIX levels
    VIX_LEVELS = {
        "low": (0, 12),
        "normal": (12, 20),
        "elevated": (20, 25),
        "high": (25, 35),
        "extreme": (35, 100),
    }
    
    def __init__(
        self,
        cache_ttl_sec: int = 60,  # Cache for 1 minute
    ):
        self.cache_ttl_sec = cache_ttl_sec
        
        # Cache
        self._cache: Dict[str, Tuple[float, any]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.3
        
        # Session for connection reuse
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
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
    
    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "^VIX")
            
        Returns:
            StockQuote or None
        """
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            params = {
                "symbols": symbol,
                "fields": "regularMarketPrice,regularMarketChange,regularMarketChangePercent,"
                         "regularMarketVolume,marketCap,trailingPE,fiftyTwoWeekHigh,fiftyTwoWeekLow",
            }
            
            resp = self._session.get(self.YF_QUOTE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("quoteResponse", {}).get("result", [])
            if not results:
                return None
            
            q = results[0]
            
            quote = StockQuote(
                symbol=symbol,
                price=q.get("regularMarketPrice", 0),
                change=q.get("regularMarketChange", 0),
                change_pct=q.get("regularMarketChangePercent", 0),
                volume=q.get("regularMarketVolume", 0),
                market_cap=q.get("marketCap"),
                pe_ratio=q.get("trailingPE"),
                timestamp=datetime.now(timezone.utc),
            )
            
            self._set_cache(cache_key, quote)
            return quote
            
        except Exception as e:
            logger.warning("[MARKET] Quote fetch failed for %s: %s", symbol, e)
            return None
    
    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get quotes for multiple symbols."""
        # Check cache first
        result = {}
        uncached = []
        
        for sym in symbols:
            cached = self._get_cached(f"quote_{sym}")
            if cached:
                result[sym] = cached
            else:
                uncached.append(sym)
        
        if not uncached:
            return result
        
        self._rate_limit()
        
        try:
            params = {
                "symbols": ",".join(uncached),
                "fields": "regularMarketPrice,regularMarketChange,regularMarketChangePercent,"
                         "regularMarketVolume,marketCap,trailingPE",
            }
            
            resp = self._session.get(self.YF_QUOTE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            for q in data.get("quoteResponse", {}).get("result", []):
                sym = q.get("symbol", "")
                quote = StockQuote(
                    symbol=sym,
                    price=q.get("regularMarketPrice", 0),
                    change=q.get("regularMarketChange", 0),
                    change_pct=q.get("regularMarketChangePercent", 0),
                    volume=q.get("regularMarketVolume", 0),
                    market_cap=q.get("marketCap"),
                    pe_ratio=q.get("trailingPE"),
                    timestamp=datetime.now(timezone.utc),
                )
                result[sym] = quote
                self._set_cache(f"quote_{sym}", quote)
            
            return result
            
        except Exception as e:
            logger.warning("[MARKET] Batch quote fetch failed: %s", e)
            return result
    
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price."""
        quote = self.get_quote(symbol)
        return quote.price if quote else None
    
    def get_vix(self) -> Optional[VIXData]:
        """
        Get VIX volatility index data.
        
        Returns VIXData with current level and classification.
        """
        cache_key = "vix_data"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            params = {
                "symbols": "^VIX",
                "fields": "regularMarketPrice,regularMarketChange,regularMarketChangePercent,"
                         "fiftyTwoWeekHigh,fiftyTwoWeekLow",
            }
            
            resp = self._session.get(self.YF_QUOTE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("quoteResponse", {}).get("result", [])
            if not results:
                return None
            
            q = results[0]
            current = q.get("regularMarketPrice", 0)
            
            # Classify VIX level
            level = "normal"
            for level_name, (low, high) in self.VIX_LEVELS.items():
                if low <= current < high:
                    level = level_name
                    break
            
            vix_data = VIXData(
                current=current,
                change=q.get("regularMarketChange", 0),
                change_pct=q.get("regularMarketChangePercent", 0),
                high_52w=q.get("fiftyTwoWeekHigh", 0),
                low_52w=q.get("fiftyTwoWeekLow", 0),
                level=level,
            )
            
            self._set_cache(cache_key, vix_data)
            return vix_data
            
        except Exception as e:
            logger.warning("[MARKET] VIX fetch failed: %s", e)
            return None
    
    def get_market_snapshot(self) -> MarketSnapshot:
        """Get a snapshot of key market indicators."""
        # Fetch key quotes
        quotes = self.get_quotes_batch(["SPY", "QQQ", "^VIX"])
        vix_data = self.get_vix()
        
        spy = quotes.get("SPY")
        qqq = quotes.get("QQQ")
        
        # Determine market sentiment
        sentiment = "neutral"
        if vix_data:
            if vix_data.current > 25:
                sentiment = "fear"
            elif vix_data.current < 15:
                sentiment = "greed"
            
            # Adjust based on price action
            if spy and spy.change_pct < -1.5:
                sentiment = "fear"
            elif spy and spy.change_pct > 1.5:
                sentiment = "greed"
        
        return MarketSnapshot(
            spy_price=spy.price if spy else None,
            spy_change_pct=spy.change_pct if spy else None,
            qqq_price=qqq.price if qqq else None,
            qqq_change_pct=qqq.change_pct if qqq else None,
            vix=vix_data.current if vix_data else None,
            vix_level=vix_data.level if vix_data else "unknown",
            put_call_ratio=None,  # Would need options data
            market_sentiment=sentiment,
            timestamp=datetime.now(timezone.utc),
        )
    
    def predict_price_threshold(
        self,
        symbol: str,
        threshold: float,
        comparison: str,  # "above" or "below"
        timeframe: str = "day",  # "day", "week", "month"
    ) -> float:
        """
        Predict probability of price crossing a threshold.
        
        Uses current price, volatility, and momentum.
        
        Args:
            symbol: Stock symbol
            threshold: Price threshold
            comparison: "above" or "below"
            timeframe: Time window
            
        Returns:
            Probability estimate (0-1)
        """
        quote = self.get_quote(symbol)
        if not quote or quote.price <= 0:
            return 0.5
        
        current = quote.price
        
        # Calculate distance to threshold as % of price
        distance_pct = (threshold - current) / current * 100
        
        # Get VIX for volatility context
        vix_data = self.get_vix()
        implied_vol = vix_data.current if vix_data else 20
        
        # Estimate daily volatility (~VIX/16 for daily, sqrt(252) ≈ 16)
        daily_vol = implied_vol / 16
        
        # Adjust for timeframe
        if timeframe == "week":
            period_vol = daily_vol * (5 ** 0.5)  # sqrt(5 trading days)
        elif timeframe == "month":
            period_vol = daily_vol * (21 ** 0.5)  # sqrt(21 trading days)
        else:
            period_vol = daily_vol
        
        # Simple probability model using volatility
        # P(above threshold) ≈ Φ((current - threshold) / (current * vol))
        import math
        
        # Standardized distance
        z = -distance_pct / period_vol if period_vol > 0 else 0
        
        # Approximate normal CDF
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        if comparison.lower() == "above":
            prob = norm_cdf(z)
        else:
            prob = 1 - norm_cdf(z)
        
        # Adjust for momentum
        if quote.change_pct > 0 and comparison == "above":
            prob = min(0.95, prob * 1.05)
        elif quote.change_pct < 0 and comparison == "below":
            prob = min(0.95, prob * 1.05)
        
        return max(0.05, min(0.95, prob))
    
    def predict_earnings_move(
        self,
        symbol: str,
        expected_move_pct: float,
        direction: str,  # "beat", "miss", "inline"
    ) -> float:
        """
        Predict probability of earnings outcome.
        
        Very rough heuristic — real model would need whisper numbers,
        options implied move, analyst revisions, etc.
        
        Returns:
            Probability estimate (0-1)
        """
        # Base rates (historical averages)
        base_probs = {
            "beat": 0.65,    # ~65% of companies beat estimates
            "miss": 0.20,    # ~20% miss
            "inline": 0.15,  # ~15% inline
        }
        
        prob = base_probs.get(direction.lower(), 0.5)
        
        # Get market context
        vix_data = self.get_vix()
        if vix_data:
            # High VIX → more uncertainty → regress to 50%
            if vix_data.current > 25:
                prob = 0.5 + (prob - 0.5) * 0.7
        
        return prob


# ── Singleton ─────────────────────────────────────────────────────────────────

_client: Optional[MarketDataClient] = None


def get_market_data_client() -> MarketDataClient:
    """Get the global market data client."""
    global _client
    if _client is None:
        _client = MarketDataClient()
    return _client