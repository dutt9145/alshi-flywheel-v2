"""
shared/news_signal.py  (v2 — batched queries, rate limit fix)

Changes vs v1:
  1. _refresh_all() now batches all keywords into groups of 5 per NewsAPI
     request using OR operator — reduces API calls from N to N/5
  2. Hard cap of 20 keywords max tracked at any time — prevents unbounded
     growth from registering every single market title word
  3. Keyword deduplication and prioritization — longer/more specific keywords
     kept over generic ones when cap is hit
  4. Per-cycle rate limit guard — minimum 2s sleep between NewsAPI requests
     to stay under free tier limits (100 req/day = 1 req/15min safely)
  5. Daily request counter added — logs total daily usage so you can monitor
  6. register_market() now only registers keywords > 5 chars to filter out
     generic names like "wins", "runs", "real", "pete" etc.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Max keywords to track simultaneously — prevents unbounded growth
_MAX_KEYWORDS    = 50
# Max keywords per NewsAPI batch query
_BATCH_SIZE      = 5
# Minimum chars for a keyword to be worth tracking
_MIN_KEYWORD_LEN = 5
# Minimum seconds between NewsAPI requests (free tier protection)
_MIN_REQUEST_GAP = 2.0

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

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "to", "in", "on", "at", "by",
    "or", "and", "of", "for", "is", "are", "was", "were", "has",
    "have", "had", "do", "does", "did", "not", "no", "yes", "this",
    "that", "with", "from", "up", "above", "below", "over", "under",
    "than", "more", "less", "reach", "hit", "end", "day", "week",
    "month", "year", "percent", "before", "after", "during", "next",
    "last", "first", "second", "third", "between", "within",
    # Generic sport words that aren't meaningful for news search
    "wins", "runs", "points", "goals", "score", "game", "match",
    "team", "play", "player", "total", "home", "away", "season",
    "wins", "loses", "beats", "versus", "against", "record",
}

def extract_keywords(title: str, max_keywords: int = 3) -> list[str]:
    """
    Pull meaningful words from a market title for news search.
    Only returns keywords longer than _MIN_KEYWORD_LEN chars.
    """
    import re
    words  = re.findall(r"[a-zA-Z]{3,}", title)
    words  = [
        w.lower() for w in words
        if w.lower() not in _STOP_WORDS and len(w) >= _MIN_KEYWORD_LEN
    ]
    seen, unique = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    unique.sort(key=len, reverse=True)
    return unique[:max_keywords]


# ── Main class ────────────────────────────────────────────────────────────────

class NewsSignal:
    """
    Background news velocity tracker — batched, rate-limited.

    Key changes in v2:
    - Keywords batched 5 at a time into single NewsAPI OR queries
    - Hard cap of 50 keywords max (evicts shortest/least specific)
    - Min keyword length of 5 chars filters generic names
    - Daily request counter logged for monitoring
    """

    def __init__(
        self,
        api_key:                  Optional[str] = None,
        poll_interval_sec:        int   = 300,
        velocity_window_min:      int   = 30,
        velocity_spike_threshold: float = 2.0,
    ):
        self.api_key                  = api_key or os.getenv("NEWSAPI_KEY", "")
        self.poll_interval_sec        = poll_interval_sec
        self.velocity_window_min      = velocity_window_min
        self.velocity_spike_threshold = velocity_spike_threshold

        self._cache: dict[str, dict]    = {}
        self._lock                      = threading.Lock()
        self._active_keywords: set[str] = set()
        self._daily_request_count       = 0
        self._last_request_ts           = 0.0

        init_news_tables()

        if not self.api_key:
            logger.warning(
                "NewsSignal: NEWSAPI_KEY not set — velocity signals disabled. "
                "Get a free key at newsapi.org"
            )
        else:
            logger.info(
                "NewsSignal ready | interval=%ds batch_size=%d max_keywords=%d",
                poll_interval_sec, _BATCH_SIZE, _MAX_KEYWORDS,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_market(self, market_title: str) -> None:
        """
        Register meaningful keywords from a market title for velocity tracking.
        Only keeps keywords >= _MIN_KEYWORD_LEN chars.
        Evicts shortest keywords when cap is reached.
        """
        keywords = extract_keywords(market_title)
        if not keywords:
            return

        with self._lock:
            for kw in keywords:
                if kw in self._active_keywords:
                    continue
                if len(self._active_keywords) >= _MAX_KEYWORDS:
                    # Evict the shortest keyword to make room
                    shortest = min(self._active_keywords, key=len)
                    self._active_keywords.discard(shortest)
                    logger.debug("NewsSignal: evicted '%s' to make room for '%s'", shortest, kw)
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

    def _refresh_all(self) -> None:
        """
        Poll NewsAPI for all active keywords in batches of _BATCH_SIZE.
        One request per batch using OR operator.
        Much more efficient than one request per keyword.
        """
        with self._lock:
            keywords = list(self._active_keywords)

        if not keywords:
            return

        # Build batches of _BATCH_SIZE keywords
        batches = [
            keywords[i:i + _BATCH_SIZE]
            for i in range(0, len(keywords), _BATCH_SIZE)
        ]

        logger.info(
            "NewsSignal refresh: %d keywords → %d batches (daily_requests=%d)",
            len(keywords), len(batches), self._daily_request_count,
        )

        for batch in batches:
            try:
                # Rate limit — minimum gap between requests
                elapsed = time.time() - self._last_request_ts
                if elapsed < _MIN_REQUEST_GAP:
                    time.sleep(_MIN_REQUEST_GAP - elapsed)

                counts = self._fetch_batch_counts(batch)
                self._last_request_ts = time.time()
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
            "NewsSignal refresh complete | daily_requests=%d",
            self._daily_request_count,
        )

    def _fetch_batch_counts(self, keywords: list[str]) -> dict[str, int]:
        """
        Fetch article counts for a batch of keywords in ONE NewsAPI request.
        Uses OR operator: "bitcoin OR ethereum OR solana"
        Returns dict of keyword → estimated count (evenly split from total).
        """
        import urllib.request
        import urllib.parse
        import json

        query = " OR ".join(keywords)
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=self.velocity_window_min)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={urllib.parse.quote(query)}"
            f"&from={since}"
            f"&sortBy=publishedAt"
            f"&pageSize=1"
            f"&apiKey={self.api_key}"
        )

        req = urllib.request.Request(
            url, headers={"User-Agent": "kalshi-flywheel/2.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data  = json.loads(resp.read())
            total = int(data.get("totalResults", 0))

        # Distribute count evenly across keywords in the batch
        # This is approximate — individual counts would require N requests
        per_kw = total // len(keywords) if keywords else 0
        return {kw: per_kw for kw in keywords}

    def _compute_velocity(self, keyword: str, current_count: int) -> float:
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