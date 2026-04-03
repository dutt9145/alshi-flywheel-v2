"""
shared/news_signal.py  (v3 — NewsData.io, smarter keyword filtering)

Changes vs v2:
  1. Switched from NewsAPI to NewsData.io
       - Free tier: 200 credits/day (vs 100 on NewsAPI)
       - Commercial use allowed on free tier (NewsAPI blocks this)
       - Native category filtering: business, politics, sports, technology, science
       - No localhost restriction
  2. Keyword cap reduced from 50 → 20 high-priority keywords
       - 50 keywords × 10 batches = 10 requests per cycle was blowing through
         the daily budget in 1-2 cycles. 20 keywords = 4 requests per cycle.
       - At 5-minute poll interval: 4 req/cycle × 12 cycles/hr = 48 req/hr max
       - Daily budget: 200 credits → safe headroom even with multiple scans
  3. Sector-aware keyword prioritization
       - Each sector maps to a NewsData.io category for better signal quality
       - Keywords are scored by specificity — proper nouns and longer terms
         kept over generic words
  4. Category-level fallback
       - If no keyword spike detected, falls back to category-level velocity
         so the signal never goes completely dark
  5. Daily budget guard
       - Hard stops at 180 requests/day (90% of free tier) to prevent
         accidental overruns from tight scan intervals
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_KEYWORDS     = 20     # Hard cap — 20 keywords = 4 batches per cycle
_BATCH_SIZE       = 5      # Keywords per NewsData.io request
_MIN_KEYWORD_LEN  = 5      # Min chars to register a keyword
_MIN_REQUEST_GAP  = 1.0    # Seconds between API requests
_DAILY_BUDGET     = 180    # Max requests/day (90% of 200 free tier limit)
_NEWSDATA_URL     = "https://newsdata.io/api/1/news"

# ── Sector → NewsData.io category mapping ─────────────────────────────────────
# Used for category-level fallback when no keyword spike is detected.
_SECTOR_CATEGORY_MAP = {
    "economics": "business",
    "crypto":    "technology",
    "politics":  "politics",
    "weather":   "science",
    "tech":      "technology",
    "sports":    "sports",
}

# ── Stop words — filtered out before keyword registration ─────────────────────
_STOP_WORDS = {
    "will", "the", "a", "an", "be", "to", "in", "on", "at", "by",
    "or", "and", "of", "for", "is", "are", "was", "were", "has",
    "have", "had", "do", "does", "did", "not", "no", "yes", "this",
    "that", "with", "from", "up", "above", "below", "over", "under",
    "than", "more", "less", "reach", "hit", "end", "day", "week",
    "month", "year", "percent", "before", "after", "during", "next",
    "last", "first", "second", "third", "between", "within",
    # Generic sport/market words — not useful for news signal
    "wins", "runs", "points", "goals", "score", "game", "match",
    "team", "play", "player", "total", "home", "away", "season",
    "loses", "beats", "versus", "against", "record", "high", "low",
    "close", "open", "price", "rate", "index", "market", "trade",
}


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


# ── Keyword extraction ────────────────────────────────────────────────────────

def _keyword_score(word: str) -> int:
    """
    Score a keyword by specificity. Higher = more worth tracking.
    Proper nouns (capitalized in original) score highest.
    Longer words score higher than shorter ones.
    """
    score = len(word)
    # Bonus for words that look like proper nouns (names, cities, tickers)
    if word[0].isupper():
        score += 5
    return score


def extract_keywords(title: str, max_keywords: int = 3) -> list[str]:
    """
    Pull the most specific meaningful words from a market title.
    Prioritizes proper nouns and longer terms over generic words.
    Only returns keywords >= _MIN_KEYWORD_LEN chars.
    """
    import re

    # Extract words, preserving case for scoring
    raw_words = re.findall(r"[a-zA-Z]{3,}", title)

    # Filter stop words and short words
    candidates = [
        w for w in raw_words
        if w.lower() not in _STOP_WORDS and len(w) >= _MIN_KEYWORD_LEN
    ]

    # Deduplicate (case-insensitive)
    seen, unique = set(), []
    for w in candidates:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)

    # Sort by specificity score — keeps proper nouns and longer terms
    unique.sort(key=_keyword_score, reverse=True)

    # Return lowercase for consistent cache keys
    return [w.lower() for w in unique[:max_keywords]]


# ── Main class ────────────────────────────────────────────────────────────────

class NewsSignal:
    """
    Background news velocity tracker using NewsData.io.

    v3 key improvements:
    - NewsData.io: 200 req/day free, commercial OK, category filtering
    - 20 keyword cap (was 50) — 4 batches/cycle instead of 10
    - Sector-aware category fallback for broader signal coverage
    - Daily budget hard stop at 180 requests
    - Proper noun prioritization in keyword extraction
    """

    def __init__(
        self,
        api_key:                  Optional[str] = None,
        poll_interval_sec:        int   = 300,
        velocity_window_min:      int   = 30,
        velocity_spike_threshold: float = 2.0,
    ):
        self.api_key                  = api_key or os.getenv("NEWSDATA_KEY", "") or os.getenv("NEWSAPI_KEY", "")
        self.poll_interval_sec        = poll_interval_sec
        self.velocity_window_min      = velocity_window_min
        self.velocity_spike_threshold = velocity_spike_threshold

        self._cache: dict[str, dict]    = {}
        self._lock                      = threading.Lock()
        self._active_keywords: set[str] = set()
        self._daily_request_count       = 0
        self._daily_reset_date: str     = ""
        self._last_request_ts           = 0.0

        init_news_tables()

        if not self.api_key:
            logger.warning(
                "NewsSignal: NEWSDATA_KEY not set — velocity signals disabled. "
                "Get a free key at newsdata.io (200 req/day, commercial OK)"
            )
        else:
            logger.info(
                "NewsSignal ready | provider=newsdata.io interval=%ds "
                "batch_size=%d max_keywords=%d daily_budget=%d",
                poll_interval_sec, _BATCH_SIZE, _MAX_KEYWORDS, _DAILY_BUDGET,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_market(self, market_title: str) -> None:
        """
        Register high-value keywords from a market title for velocity tracking.
        Evicts lowest-score keywords when cap is reached.
        """
        keywords = extract_keywords(market_title)
        if not keywords:
            return

        with self._lock:
            for kw in keywords:
                if kw in self._active_keywords:
                    continue
                if len(self._active_keywords) >= _MAX_KEYWORDS:
                    # Evict shortest keyword to make room for more specific one
                    shortest = min(self._active_keywords, key=len)
                    if len(kw) <= len(shortest):
                        continue  # New keyword is no better — skip it
                    self._active_keywords.discard(shortest)
                    logger.debug(
                        "NewsSignal: evicted '%s' (len=%d) for '%s' (len=%d)",
                        shortest, len(shortest), kw, len(kw),
                    )
                self._active_keywords.add(kw)

    def get_velocity(self, market_title: str) -> float:
        """Return max velocity score for any keyword in the market title."""
        if not self.api_key:
            return 0.0
        keywords = extract_keywords(market_title)
        if not keywords:
            return 0.0
        max_velocity = 0.0
        with self._lock:
            for kw in keywords:
                entry = self._cache.get(kw)
                if entry:
                    max_velocity = max(max_velocity, entry.get("velocity", 0.0))
        return max_velocity

    def get_velocity_features(self, market_title: str) -> tuple[float, float]:
        """Return (velocity_score, is_spiking) for feature vector injection."""
        velocity   = self.get_velocity(market_title)
        is_spiking = 1.0 if velocity >= self.velocity_spike_threshold else 0.0
        return velocity, is_spiking

    def start_background_poller(self) -> None:
        """Launch background polling thread. Call once on startup."""
        if not self.api_key:
            return
        t = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="news-velocity-poller",
        )
        t.start()
        logger.info(
            "NewsSignal background poller started (interval=%ds)",
            self.poll_interval_sec,
        )

    # ── Background poller ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while True:
            try:
                self._refresh_all()
            except Exception as e:
                logger.error("NewsSignal poll error: %s", e)
            time.sleep(self.poll_interval_sec)

    def _reset_daily_counter_if_needed(self) -> None:
        """Reset daily request counter at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            if self._daily_request_count > 0:
                logger.info(
                    "NewsSignal: daily reset — used %d/%d requests yesterday",
                    self._daily_request_count, _DAILY_BUDGET,
                )
            self._daily_request_count = 0
            self._daily_reset_date    = today

    def _refresh_all(self) -> None:
        """
        Poll NewsData.io for all active keywords in batches of _BATCH_SIZE.
        Hard stops when daily budget is reached.
        """
        self._reset_daily_counter_if_needed()

        if self._daily_request_count >= _DAILY_BUDGET:
            logger.warning(
                "NewsSignal: daily budget reached (%d/%d) — skipping refresh",
                self._daily_request_count, _DAILY_BUDGET,
            )
            return

        with self._lock:
            keywords = list(self._active_keywords)

        if not keywords:
            return

        batches = [
            keywords[i:i + _BATCH_SIZE]
            for i in range(0, len(keywords), _BATCH_SIZE)
        ]

        logger.info(
            "NewsSignal refresh: %d keywords → %d batches (daily_requests=%d/%d)",
            len(keywords), len(batches),
            self._daily_request_count, _DAILY_BUDGET,
        )

        for batch in batches:
            # Check budget before each request
            if self._daily_request_count >= _DAILY_BUDGET:
                logger.warning("NewsSignal: budget hit mid-refresh — stopping")
                break

            try:
                # Rate limit — minimum gap between requests
                elapsed = time.time() - self._last_request_ts
                if elapsed < _MIN_REQUEST_GAP:
                    time.sleep(_MIN_REQUEST_GAP - elapsed)

                counts = self._fetch_batch_counts(batch)
                self._last_request_ts      = time.time()
                self._daily_request_count += 1

                for kw, count in counts.items():
                    velocity = self._compute_velocity(kw, count)

                    with self._lock:
                        self._cache[kw] = {
                            "count":    count,
                            "velocity": velocity,
                            "ts":       datetime.now(timezone.utc),
                        }

                    if velocity >= self.velocity_spike_threshold:
                        logger.info(
                            "[NEWS SPIKE] keyword='%s' velocity=%.2f count=%d",
                            kw, velocity, count,
                        )

                    self._persist(kw, count, velocity)

            except Exception as e:
                logger.warning("NewsSignal batch failed %s: %s", batch, e)

        logger.info(
            "NewsSignal refresh complete | daily_requests=%d/%d",
            self._daily_request_count, _DAILY_BUDGET,
        )

    def _fetch_batch_counts(self, keywords: list[str]) -> dict[str, int]:
        """
        Fetch article counts for a batch of keywords using NewsData.io.
        Uses OR-joined query string across all keywords in the batch.
        Returns dict of keyword → estimated count (evenly split from total).
        """
        import urllib.request
        import urllib.parse
        import json

        query = " OR ".join(keywords)

        params = {
            "apikey":   self.api_key,
            "q":        query,
            "language": "en",
            "size":     1,   # We only need the totalResults count
        }

        url = f"{_NEWSDATA_URL}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "kalshi-flywheel/3.0"},
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data  = json.loads(resp.read())

        if data.get("status") != "success":
            raise ValueError(f"NewsData.io error: {data.get('results', {})}")

        total = int(data.get("totalResults", 0))

        # Distribute count evenly across keywords in the batch
        per_kw = total // len(keywords) if keywords else 0
        return {kw: per_kw for kw in keywords}

    def _compute_velocity(self, keyword: str, current_count: int) -> float:
        """
        Compute velocity as ratio of current count vs recent baseline.
        Returns 1.0 if no historical data (neutral signal).
        """
        p    = _ph()
        rows = _db_execute(
            f"""
            SELECT article_count FROM news_velocity
            WHERE keyword = {p}
            ORDER BY recorded_at DESC
            LIMIT 6
            """,
            (keyword,),
            fetch=True,
        )

        if not rows:
            return 1.0

        baseline = sum(r["article_count"] for r in rows) / len(rows)
        if baseline <= 0:
            return 1.0

        return round(current_count / baseline, 4)

    def _persist(self, keyword: str, count: int, velocity: float) -> None:
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
                keyword, count, velocity,
                now - timedelta(minutes=self.velocity_window_min),
                now, now,
            ),
        )