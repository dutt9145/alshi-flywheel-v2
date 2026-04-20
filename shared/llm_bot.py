"""
shared/llm_bot.py  (v1.0 — LLM-powered qualitative market analysis)

Uses Claude API for markets where qualitative judgment matters:
  - Politics: elections, legislation, appointments, geopolitics
  - Economics: Fed decisions, inflation, employment, GDP
  - Financial Markets: earnings, IPOs, stock movements
  - Global Events: international affairs, disasters, treaties

Sports, weather, and crypto use quantitative models instead.

Features:
  - Web search integration for current events
  - Response caching to reduce API costs
  - Confidence discounting (LLMs tend to be overconfident)
  - Structured JSON output for reliable parsing
"""

import json
import logging
import os
import re
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Response cache (in-memory, resets on restart) ──────────────────────────────
_response_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _cache_key(market_title: str, yes_price: int) -> str:
    """Generate cache key from market title and price bucket."""
    # Bucket prices into 5-cent ranges to allow some price movement
    price_bucket = (yes_price // 5) * 5
    raw = f"{market_title.lower().strip()}|{price_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _get_cached(key: str) -> Optional[dict]:
    """Get cached response if still valid."""
    if key in _response_cache:
        ts, response = _response_cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            logger.debug("[LLM] Cache hit: %s", key)
            return response
        else:
            del _response_cache[key]
    return None


def _set_cached(key: str, response: dict) -> None:
    """Cache a response."""
    _response_cache[key] = (time.time(), response)
    
    # Prune old entries if cache gets too large
    if len(_response_cache) > 500:
        now = time.time()
        expired = [k for k, (ts, _) in _response_cache.items() 
                   if now - ts > CACHE_TTL_SECONDS]
        for k in expired:
            del _response_cache[k]


# ── Sector detection ───────────────────────────────────────────────────────────

_POLITICS_KEYWORDS = (
    "president", "election", "senate", "congress", "vote", "ballot",
    "governor", "mayor", "legislation", "bill", "law", "supreme court",
    "impeach", "pardon", "executive order", "cabinet", "ambassador",
    "democrat", "republican", "gop", "primary", "caucus", "poll",
    "nato", "un ", "united nations", "eu ", "european union",
    "trump", "biden", "harris", "desantis", "newsom",
    "xi jinping", "putin", "zelensky", "netanyahu", "modi",
    "tariff", "sanction", "treaty", "summit", "diplomacy",
)

_ECONOMICS_KEYWORDS = (
    "fed ", "federal reserve", "fomc", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "deflation", "stagflation",
    "gdp", "recession", "depression", "employment", "unemployment",
    "jobless", "payroll", "wage", "labor", "jobs report",
    "treasury", "yield", "bond", "debt ceiling", "deficit",
    "stimulus", "quantitative", "taper", "hawkish", "dovish",
)

_FINANCIAL_MARKETS_KEYWORDS = (
    "stock", "share", "equity", "nasdaq", "s&p", "dow jones",
    "earnings", "revenue", "profit", "eps", "guidance",
    "ipo", "merger", "acquisition", "buyout", "spinoff",
    "bankruptcy", "default", "delisting", "sec ", "filing",
    "tesla", "apple", "google", "amazon", "microsoft", "nvidia",
    "bitcoin etf", "crypto etf", "spot etf",
    "market cap", "valuation", "analyst", "upgrade", "downgrade",
)

_GLOBAL_EVENTS_KEYWORDS = (
    "war", "peace", "ceasefire", "invasion", "military",
    "attack", "strike", "bombing", "missile", "nuclear",
    "earthquake", "hurricane", "tsunami", "disaster", "emergency",
    "pandemic", "outbreak", "virus", "who ", "health emergency",
    "olympics", "world cup", "fifa", "ioc",
    "nobel", "oscar", "grammy", "emmy", "awards",
    "pope", "royal", "monarch", "coronation",
    "spacex", "nasa", "launch", "rocket", "mars", "moon landing",
    "ai ", "artificial intelligence", "chatgpt", "openai", "anthropic",
)


def _detect_sector(title: str) -> str:
    """Detect which qualitative sector a market belongs to."""
    title_lower = title.lower()
    
    # Check each sector's keywords
    politics_hits = sum(1 for kw in _POLITICS_KEYWORDS if kw in title_lower)
    economics_hits = sum(1 for kw in _ECONOMICS_KEYWORDS if kw in title_lower)
    financial_hits = sum(1 for kw in _FINANCIAL_MARKETS_KEYWORDS if kw in title_lower)
    global_hits = sum(1 for kw in _GLOBAL_EVENTS_KEYWORDS if kw in title_lower)
    
    # Return sector with most hits, default to global_events
    hits = [
        (politics_hits, "politics"),
        (economics_hits, "economics"),
        (financial_hits, "financial_markets"),
        (global_hits, "global_events"),
    ]
    hits.sort(reverse=True)
    
    if hits[0][0] > 0:
        return hits[0][1]
    return "global_events"


# ── Claude API calling ─────────────────────────────────────────────────────────

def _call_claude(
    prompt: str,
    system: str = "",
    max_tokens: int = 500,
    use_web_search: bool = True,
) -> Optional[dict]:
    """
    Call Claude API with optional web search.
    Returns parsed JSON response or None on failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("[LLM] ANTHROPIC_API_KEY not set")
        return None
    
    try:
        import httpx
    except ImportError:
        logger.error("[LLM] httpx not installed — run: pip install httpx")
        return None
    
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    
    # Build request body
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    if system:
        body["system"] = system
    
    # Add web search tool if enabled
    if use_web_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        logger.warning("[LLM] API timeout")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("[LLM] API error: %s", e.response.text[:200])
        return None
    except Exception as e:
        logger.warning("[LLM] API call failed: %s", e)
        return None
    
    # Extract text from response
    text_content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_content += block.get("text", "")
    
    if not text_content:
        logger.warning("[LLM] Empty response from API")
        return None
    
    # Parse JSON from response
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"probability"[^{}]*\}', text_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        # Try parsing the whole response as JSON
        return json.loads(text_content)
    except json.JSONDecodeError:
        logger.warning("[LLM] Failed to parse JSON from: %s", text_content[:200])
        return None


# ── Market evaluation ──────────────────────────────────────────────────────────

@dataclass
class LLMSignal:
    """Signal from LLM evaluation."""
    sector: str
    prob: float
    confidence: float
    market_prob: float
    reasoning: str
    cached: bool = False


def evaluate_market_llm(
    title: str,
    yes_price_cents: int,
    close_time: Optional[str] = None,
    ticker: str = "",
) -> Optional[LLMSignal]:
    """
    Evaluate a market using Claude with web search.
    
    Returns LLMSignal with probability estimate, or None if evaluation fails.
    """
    # Check cache first
    cache_key = _cache_key(title, yes_price_cents)
    cached = _get_cached(cache_key)
    if cached:
        return LLMSignal(
            sector=cached.get("sector", "global_events"),
            prob=cached.get("probability", 0.5),
            confidence=cached.get("confidence", 0.5) * 0.7,  # Discount
            market_prob=yes_price_cents / 100,
            reasoning=cached.get("reasoning", "cached"),
            cached=True,
        )
    
    # Detect sector
    sector = _detect_sector(title)
    
    # Build prompt
    market_prob = yes_price_cents / 100
    
    # Calculate time until close if available
    time_context = ""
    if close_time:
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_left = (close_dt - now).total_seconds() / 3600
            if hours_left > 0:
                time_context = f"This market closes in {hours_left:.1f} hours."
        except Exception:
            pass
    
    system_prompt = """You are a prediction market analyst. Your job is to estimate true probabilities for events.

CRITICAL RULES:
1. Use web search to find recent news and data
2. Be calibrated — if you say 70%, it should happen 70% of the time
3. Account for base rates and historical patterns
4. Be skeptical of your own confidence
5. Respond ONLY with valid JSON, no other text

Your response must be exactly this JSON format:
{"probability": 0.XX, "confidence": 0.XX, "reasoning": "2-3 sentence explanation"}

probability: Your estimate of P(YES) from 0.01 to 0.99
confidence: How confident you are in your estimate from 0.3 to 0.9 (be humble)
reasoning: Brief explanation citing specific evidence"""

    user_prompt = f"""Analyze this prediction market:

MARKET: {title}
CURRENT PRICE: {yes_price_cents}¢ (market implies {market_prob:.0%} probability)
{time_context}

Search for recent news about this topic, then estimate the TRUE probability this resolves YES.

Respond with JSON only: {{"probability": X.XX, "confidence": X.XX, "reasoning": "..."}}"""

    # Call Claude
    result = _call_claude(user_prompt, system=system_prompt, use_web_search=True)
    
    if not result:
        logger.debug("[LLM] No result for: %s", title[:50])
        return None
    
    # Validate and extract fields
    try:
        prob = float(result.get("probability", 0.5))
        conf = float(result.get("confidence", 0.5))
        reasoning = str(result.get("reasoning", ""))
        
        # Clamp values
        prob = max(0.01, min(0.99, prob))
        conf = max(0.3, min(0.9, conf))
        
        # Discount confidence (LLMs tend to be overconfident)
        conf *= 0.7
        
    except (TypeError, ValueError) as e:
        logger.warning("[LLM] Invalid values in response: %s", e)
        return None
    
    # Cache the result
    _set_cached(cache_key, {
        "sector": sector,
        "probability": prob,
        "confidence": conf,
        "reasoning": reasoning,
    })
    
    logger.info(
        "[LLM] %s | %s | our_p=%.2f mkt_p=%.2f conf=%.2f | %s",
        sector.upper(), title[:40], prob, market_prob, conf, reasoning[:60],
    )
    
    return LLMSignal(
        sector=sector,
        prob=prob,
        confidence=conf,
        market_prob=market_prob,
        reasoning=reasoning,
        cached=False,
    )


# ── Bot interface (compatible with sector_bots.py) ─────────────────────────────

class LLMBot:
    """
    LLM-powered bot for qualitative sectors.
    
    Handles: politics, economics, financial_markets, global_events
    
    Compatible with the existing bot interface:
      - is_relevant(market) -> bool
      - evaluate(market, news_signal) -> BotSignal or None
    """
    
    # Sectors this bot handles
    SECTORS = ("politics", "economics", "financial_markets", "global_events")
    
    # Combined keywords from all qualitative sectors
    KEYWORDS = list(_POLITICS_KEYWORDS + _ECONOMICS_KEYWORDS + 
                    _FINANCIAL_MARKETS_KEYWORDS + _GLOBAL_EVENTS_KEYWORDS)
    
    # Prefixes to SKIP (these are handled by other bots)
    SKIP_PREFIXES = (
        # Sports
        "kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
        "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
        "kxatp", "kxwta", "kxtennis", "kxpga", "kxf1", "kxolympic",
        "kxepl", "kxsoccer", "kxcricket", "kxrugby", "kxesport",
        "kxdota", "kxcs2", "kxcsgo", "kxlol", "kxvalorant",
        "kxow", "kxrl", "kxapex", "kxfort",
        # Weather
        "kxhurr", "kxtemp", "kxrain", "kxsnow", "kxweather",
        "kxlowt", "kxhight", "kxchll", "kxdens",
        # Crypto
        "kxbtc", "kxeth", "kxsol", "kxcrypto", "kxdefi",
        "kxxrp", "kxdoge", "kxbnb", "kxavax", "kxlink",
    )
    
    def __init__(self):
        self.sector_name = "llm"  # Will be overridden per-signal
        self._enabled = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if not self._enabled:
            logger.warning("[LLMBot] Disabled — ANTHROPIC_API_KEY not set")
    
    def is_relevant(self, market: dict) -> bool:
        """Check if this market should be evaluated by the LLM bot."""
        if not self._enabled:
            return False
        
        ticker = market.get("ticker", "").lower()
        event_ticker = market.get("event_ticker", "").lower()
        
        # Skip markets handled by other bots
        for prefix in self.SKIP_PREFIXES:
            if ticker.startswith(prefix) or event_ticker.startswith(prefix):
                return False
        
        # Check if title contains any of our keywords
        title = market.get("title", "").lower()
        return any(kw in title for kw in self.KEYWORDS)
    
    def evaluate(self, market: dict, news_signal=None):
        """
        Evaluate a market using Claude API.
        
        Returns a BotSignal compatible with the consensus engine.
        """
        if not self._enabled:
            return None
        
        title = market.get("title", "")
        ticker = market.get("ticker", "")
        close_time = market.get("close_time", "")
        
        # Parse price
        try:
            bid = float(market.get("yes_bid_dollars") or 0)
            ask = float(market.get("yes_ask_dollars") or 0)
            if bid > 0 and ask > 0:
                yes_price_cents = int(round((bid + ask) / 2 * 100))
            else:
                last = float(market.get("last_price_dollars") or 0)
                yes_price_cents = int(round(last * 100)) if last > 0 else 0
        except (TypeError, ValueError):
            yes_price_cents = 0
        
        if yes_price_cents <= 2 or yes_price_cents >= 98:
            return None
        
        # Call LLM
        signal = evaluate_market_llm(
            title=title,
            yes_price_cents=yes_price_cents,
            close_time=close_time,
            ticker=ticker,
        )
        
        if not signal:
            return None
        
        # v12.7: financial_markets disabled (0.55 Brier — coin-flip territory)
        if signal.sector == "financial_markets":
            logger.debug("[LLM] Skipping financial_markets: %s", ticker[:40])
            return None
        
        # Update sector_name for this signal
        self.sector_name = signal.sector
        
        # Return BotSignal compatible with consensus engine
        # Import here to avoid circular dependency
        from shared.consensus_engine import BotSignal
        
        return BotSignal(
            sector=signal.sector,
            prob=signal.prob,
            confidence=signal.confidence,
            market_prob=signal.market_prob,
            brier_score=0.20,  # Conservative estimate until calibrated
        )
    
    # ── Stub methods for compatibility with BaseBot interface ──────────────────
    
    def fetch_features(self, market: dict, skip_noaa: bool = False):
        """Stub for Bayesian update compatibility."""
        import numpy as np
        return np.array([0.5]), {}
    
    def record_outcome(self, features, resolved_yes: bool) -> None:
        """Stub for Bayesian update compatibility."""
        pass
    
    def refresh_calibration(self) -> dict:
        """Stub for calibration compatibility."""
        return {"llm_bot": "no calibration needed"}
    
    def save_model(self) -> None:
        """Stub for model saving compatibility."""
        pass


# ── Convenience function ───────────────────────────────────────────────────────

def get_llm_bot() -> LLMBot:
    """Get a singleton LLM bot instance."""
    return LLMBot()