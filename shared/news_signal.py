"""
shared/news_signal.py  (v4 — Perplexity primary signal engine)

Architecture change vs v3:
  Switched from NewsData.io keyword-count polling → Perplexity sector intelligence.

WHY:
  - NewsData.io free tier: 12-hour delayed articles → stale signal, useless for
    intraday prediction markets that price in news within minutes
  - Perplexity Sonar: real-time web search + synthesis → asks "is there breaking
    news affecting [sector] markets RIGHT NOW?" and reasons about market relevance
  - Cost: ~$0.14/day vs $199.99/month for NewsData paid tier

SIGNAL QUALITY COMPARISON:
  Old approach:  Count articles mentioning "bitcoin" → velocity ratio
                 Problem: counts press releases, opinion pieces, old news

  New approach:  Ask Perplexity "Has breaking news in last 2hr moved Bitcoin
                 prediction markets?" → returns scored signal + reasoning
                 Advantage: filters for market-relevant events, real-time,
                 understands context (Fed meeting vs random crypto article)

USAGE PATTERN:
  - One Perplexity query per sector per 30-minute interval
  - 6 sectors × 2/hr = 12 queries/hr × 24hr = ~288 queries/day
  - At $1/M tokens (sonar standard), ~500 tokens/query avg = ~$0.14/day
  - Results cached 15 minutes — rapid scan cycles hit cache, not API
  - Each sector gets a specialized prompt tuned for prediction market relevance

INTEGRATION:
  The public API is unchanged — get_velocity() and get_velocity_features()
  still return the same float values, so no changes needed in base_bot.py
  or the orchestrator. Drop-in replacement.

ENV VARS:
  PERPLEXITY_API_KEY  — your Perplexity API key (already set for Discord bot)
  NEWSAPI_KEY         — no longer used, kept for backwards compatibility
  NEWSDATA_KEY        — no longer used, kept for backwards compatibility
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_CACHE_TTL_MINUTES   = 15     # Don't re-query same sector within this window
_SECTOR_POLL_MINUTES = 30     # Full sector refresh interval
_MIN_REQUEST_GAP     = 1.0    # Seconds between Perplexity requests
_PERPLEXITY_URL      = "https://api.perplexity.ai/chat/completions"
_MODEL               = "sonar"  # Standard sonar — real-time, ~$1/M tokens
                                 # Upgrade to "sonar-pro" for deeper reasoning

# ── Sector-specific Perplexity prompts ────────────────────────────────────────

_SECTOR_PROMPTS = {
    "economics": """Search for breaking financial and economic news from the last 2 hours.
Focus on: Fed/central bank announcements, CPI/inflation data, jobs reports, GDP releases,
recession signals, major market moves (S&P, Nasdaq, Dow), commodity price spikes,
treasury yield moves, or any macro event that would affect prediction market probabilities.

Return JSON only:
{
  "velocity_score": <1-10, where 1=no news, 10=major market-moving event>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on the most market-relevant event, or 'No significant economic news'>",
  "top_event": "<specific event name or null>",
  "confidence": <0.0-1.0>
}""",

    "crypto": """Search for breaking cryptocurrency news from the last 2 hours.
Focus on: Bitcoin/Ethereum price moves >3%, exchange hacks or outages, regulatory actions
(SEC, CFTC, DOJ), ETF approvals or rejections, major protocol exploits, whale movements,
stablecoin depegs, major DeFi events, or any news that would shift crypto prediction market odds.

Return JSON only:
{
  "velocity_score": <1-10, where 1=quiet, 10=major breaking event>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on the most market-relevant crypto event, or 'Crypto markets quiet'>",
  "top_event": "<specific event name or null>",
  "confidence": <0.0-1.0>
}""",

    "politics": """Search for breaking political news from the last 2 hours.
Focus on: US political developments, election news, major legislation votes, executive orders,
Supreme Court decisions, international geopolitical events (NATO, UN, major conflicts),
government shutdown risks, major political figure announcements, or polling shifts
that would affect prediction market probabilities.

Return JSON only:
{
  "velocity_score": <1-10, where 1=routine news, 10=major breaking political event>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on most politically significant event, or 'No major political developments'>",
  "top_event": "<specific event name or null>",
  "confidence": <0.0-1.0>
}""",

    "weather": """Search for breaking severe weather news and forecasts from the last 2 hours.
Focus on: hurricane/tropical storm formations or landfalls, tornado outbreaks, major blizzard
or winter storm warnings, extreme heat events, flooding, NOAA alerts for major US cities,
or any severe weather event that would affect temperature/precipitation prediction markets.
Include specific cities affected if relevant.

Return JSON only:
{
  "velocity_score": <1-10, where 1=normal weather, 10=major severe weather event in progress>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on most significant weather event, or 'No severe weather alerts'>",
  "top_event": "<specific event or location or null>",
  "confidence": <0.0-1.0>
}""",

    "tech": """Search for breaking technology and earnings news from the last 2 hours.
Focus on: major tech earnings beats/misses (Apple, Nvidia, Microsoft, Google, Meta),
AI model releases or major announcements, semiconductor news, major product launches,
antitrust actions against tech companies, major outages (cloud providers, social platforms),
M&A announcements, or any tech event that would shift prediction market probabilities.

Return JSON only:
{
  "velocity_score": <1-10, where 1=routine tech news, 10=major market-moving event>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on most significant tech event, or 'No major tech news'>",
  "top_event": "<specific event name or null>",
  "confidence": <0.0-1.0>
}""",

    "sports": """Search for breaking sports news from the last 2 hours.
Focus on: major injury announcements for star players (NBA, NFL, MLB, NHL), game
postponements or cancellations, coaching changes, trades or signings, suspension
announcements, or any news that would significantly shift win probability or player
prop prediction markets. Include specific player names and teams.

Return JSON only:
{
  "velocity_score": <1-10, where 1=no impactful news, 10=major injury or trade announcement>,
  "is_spiking": <true if score >= 7>,
  "summary": "<one sentence on most market-relevant sports development, or 'No major sports news'>",
  "top_event": "<specific player/team/event or null>",
  "confidence": <0.0-1.0>
}""",
}

_SYSTEM_PROMPT = (
    "You are a real-time news intelligence system for a prediction market trading bot. "
    "Your job is to assess news velocity — how much breaking news is moving prediction "
    "market probabilities right now. Search the web for events from the last 2 hours only. "
    "Ignore older news. Return ONLY valid JSON matching the exact structure requested. "
    "No preamble, no explanation, no markdown."
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _db_execute(sql: str, params: tuple = (), fetch: bool = False) -> list:
    database_url = os.getenv("DATABASE_URL", "")
    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute(sql, params)
            result = []
            if fetch and cur.description:
                cols   = [d[0] for d in cur.description]
                result = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return result
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.row_factory = sqlite3.Row
            cur  = conn.execute(sql, params)
            result = [dict(r) for r in cur.fetchall()] if fetch else []
            conn.commit()
            conn.close()
            return result
    except Exception as e:
        logger.error("NewsSignal DB error: %s", e)
        return []


def init_news_tables() -> None:
    _db_execute("""
        CREATE TABLE IF NOT EXISTS news_velocity (
            id             SERIAL PRIMARY KEY,
            keyword        TEXT NOT NULL,
            article_count  INT DEFAULT 0,
            velocity_score FLOAT8 DEFAULT 0.0,
            window_start   TIMESTAMPTZ,
            window_end     TIMESTAMPTZ,
            recorded_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_news_velocity_keyword
        ON news_velocity (keyword, recorded_at DESC)
    """)


# ── Sector inference ──────────────────────────────────────────────────────────

_SECTOR_KEYWORDS = {
    "economics": ["cpi", "fed", "fomc", "gdp", "inflation", "rate", "treasury",
                  "recession", "jobs", "payroll", "copper", "gold", "oil", "sp500",
                  "dow", "nasdaq", "yield", "tariff"],
    "crypto":    ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp",
                  "defi", "nft", "blockchain", "coinbase", "binance"],
    "politics":  ["election", "president", "congress", "senate", "supreme", "vote",
                  "democrat", "republican", "ukraine", "nato", "trump", "harris"],
    "weather":   ["temperature", "hurricane", "tornado", "flood", "snow", "rain",
                  "storm", "weather", "fahrenheit", "celsius", "blizzard"],
    "tech":      ["apple", "nvidia", "microsoft", "google", "amazon", "meta",
                  "tesla", "earnings", "ai", "semiconductor", "iphone"],
    "sports":    ["nba", "nfl", "mlb", "nhl", "basketball", "football", "baseball",
                  "points", "touchdown", "player", "team", "game"],
}


def _infer_sector(market_title: str) -> str:
    title_lower = market_title.lower()
    scores = {sector: 0 for sector in _SECTOR_KEYWORDS}
    for sector, keywords in _SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                scores[sector] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "economics"


# ── Main class ────────────────────────────────────────────────────────────────

class NewsSignal:
    """
    Real-time news velocity signal using Perplexity Sonar.

    v4 key changes:
    - Perplexity sector queries instead of keyword article polling
    - Real-time web search (not 12hr delayed like NewsData.io free tier)
    - Synthesized market-relevant signal instead of raw article counts
    - ~$0.14/day at current scan rates
    - 15-minute cache per sector prevents API hammering
    - Public API unchanged — drop-in replacement for v2/v3
    """

    def __init__(
        self,
        api_key:                  Optional[str] = None,
        poll_interval_sec:        int   = 1800,  # 30 min sector refresh
        velocity_window_min:      int   = 30,
        velocity_spike_threshold: float = 2.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("PERPLEXITY_API_KEY", "")
            or os.getenv("PERPLEXITY_KEY", "")
        )
        self.poll_interval_sec        = poll_interval_sec
        self.velocity_window_min      = velocity_window_min
        self.velocity_spike_threshold = velocity_spike_threshold

        self._sector_cache: dict[str, dict] = {}
        self._lock              = threading.Lock()
        self._last_request_ts   = 0.0
        self._daily_query_count = 0
        self._daily_reset_date  = ""

        init_news_tables()

        if not self.api_key:
            logger.warning(
                "NewsSignal: PERPLEXITY_API_KEY not set — velocity signals disabled."
            )
        else:
            logger.info(
                "NewsSignal v4 ready | provider=perplexity model=%s "
                "poll_interval=%ds cache_ttl=%dmin",
                _MODEL, poll_interval_sec, _CACHE_TTL_MINUTES,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_market(self, market_title: str) -> None:
        """
        v4: sector-level tracking, no per-keyword registration needed.
        Kept for API compatibility — no-op.
        """
        pass

    def get_velocity(self, market_title: str) -> float:
        """
        Return velocity score for a market based on its sector's news signal.
        Normalizes 1-10 sector score to ratio format for feature vector compat.
        """
        if not self.api_key:
            return 0.0

        sector = _infer_sector(market_title)

        with self._lock:
            cached = self._sector_cache.get(sector)

        if cached:
            # Normalize 1-10 → ratio (1.0 = baseline, 5.0 = major spike)
            return cached.get("score", 1.0) / 2.0

        # No cache — trigger async refresh and return neutral baseline
        self._trigger_sector_refresh(sector)
        return 1.0

    def get_velocity_features(self, market_title: str) -> tuple[float, float]:
        """Return (velocity_score, is_spiking) for feature vector injection."""
        sector = _infer_sector(market_title)

        with self._lock:
            cached = self._sector_cache.get(sector)

        if cached:
            velocity   = cached.get("score", 1.0) / 2.0
            is_spiking = 1.0 if cached.get("is_spiking", False) else 0.0
            return velocity, is_spiking

        return 1.0, 0.0

    def get_sector_summary(self, sector: str) -> str:
        """
        Return the latest news summary for a sector.
        Can be used by orchestrator for enhanced logging/context.
        """
        with self._lock:
            cached = self._sector_cache.get(sector)
        return cached.get("summary", "No recent data") if cached else "No recent data"

    def start_background_poller(self) -> None:
        if not self.api_key:
            return
        t = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="news-velocity-poller",
        )
        t.start()
        logger.info(
            "NewsSignal background poller started (interval=%ds, sectors=%d)",
            self.poll_interval_sec, len(_SECTOR_PROMPTS),
        )

    # ── Background poller ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while True:
            try:
                self._refresh_all_sectors()
            except Exception as e:
                logger.error("NewsSignal poll error: %s", e)
            time.sleep(self.poll_interval_sec)

    def _refresh_all_sectors(self) -> None:
        self._reset_daily_counter_if_needed()

        logger.info(
            "NewsSignal: refreshing sectors (daily_queries=%d)",
            self._daily_query_count,
        )

        for sector in _SECTOR_PROMPTS:
            with self._lock:
                cached = self._sector_cache.get(sector)

            if cached:
                age_min = (
                    datetime.now(timezone.utc) - cached["ts"]
                ).total_seconds() / 60
                if age_min < _CACHE_TTL_MINUTES:
                    logger.debug("NewsSignal: %s cache fresh (%.1fmin)", sector, age_min)
                    continue

            self._query_sector(sector)
            time.sleep(_MIN_REQUEST_GAP)

    def _trigger_sector_refresh(self, sector: str) -> None:
        """Non-blocking single sector refresh."""
        t = threading.Thread(
            target=self._query_sector,
            args=(sector,),
            daemon=True,
            name=f"news-refresh-{sector}",
        )
        t.start()

    def _query_sector(self, sector: str) -> None:
        """Query Perplexity for news velocity in one sector."""
        if not self.api_key:
            return

        prompt = _SECTOR_PROMPTS.get(sector)
        if not prompt:
            return

        try:
            elapsed = time.time() - self._last_request_ts
            if elapsed < _MIN_REQUEST_GAP:
                time.sleep(_MIN_REQUEST_GAP - elapsed)

            raw = self._call_perplexity(prompt)
            self._last_request_ts    = time.time()
            self._daily_query_count += 1

            if not raw:
                return

            parsed = self._parse_response(raw)
            if not parsed:
                return

            score      = float(parsed.get("velocity_score", 1))
            is_spiking = bool(parsed.get("is_spiking", False))
            summary    = str(parsed.get("summary", ""))
            top_event  = parsed.get("top_event")
            confidence = float(parsed.get("confidence", 0.5))

            with self._lock:
                self._sector_cache[sector] = {
                    "score":      score,
                    "is_spiking": is_spiking,
                    "summary":    summary,
                    "top_event":  top_event,
                    "confidence": confidence,
                    "ts":         datetime.now(timezone.utc),
                }

            if is_spiking:
                logger.info(
                    "[NEWS SPIKE] sector=%s score=%.1f/10 event=%s | %s",
                    sector, score, top_event or "unknown", summary,
                )
            else:
                logger.debug(
                    "NewsSignal: %s score=%.1f/10 | %s", sector, score, summary,
                )

            self._persist_sector(sector, score)

        except Exception as e:
            logger.warning("NewsSignal query failed sector=%s: %s", sector, e)

    def _call_perplexity(self, user_prompt: str) -> Optional[str]:
        """Make a Perplexity Sonar API call."""
        import urllib.request

        payload = json.dumps({
            "model":    _MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens":       300,
            "temperature":      0.1,
            "return_citations": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            _PERPLEXITY_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
                "User-Agent":    "kalshi-flywheel/4.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"].strip()

    def _parse_response(self, raw: str) -> Optional[dict]:
        """Parse JSON response, handling markdown code fences."""
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            logger.warning("NewsSignal: could not parse response: %s", raw[:200])
            return None

    def _persist_sector(self, sector: str, score: float) -> None:
        """Persist to DB for audit trail and weekly analysis."""
        p   = _ph()
        now = datetime.now(timezone.utc)
        _db_execute(
            f"""
            INSERT INTO news_velocity
                (keyword, article_count, velocity_score,
                 window_start, window_end, recorded_at)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p})
            """,
            (
                f"sector:{sector}", 0, score,
                now - timedelta(minutes=self.velocity_window_min),
                now, now,
            ),
        )

    def _reset_daily_counter_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            if self._daily_query_count > 0:
                logger.info(
                    "NewsSignal: daily reset — %d Perplexity queries yesterday",
                    self._daily_query_count,
                )
            self._daily_query_count = 0
            self._daily_reset_date  = today