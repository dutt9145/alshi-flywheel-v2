"""
shared/news_signal.py

Real-time news velocity signal.

Tracks the *rate of new articles* on a topic — not sentiment.
When article volume on a market keyword doubles in 15 minutes,
price is about to move. Positioning before the move is the edge.

How it works:
  1. Polls NewsAPI every NEWS_POLL_INTERVAL_SEC (default 300 = 5min)
  2. For each active market title, extracts keywords and counts recent articles
  3. Computes velocity = (current_count - baseline_count) / time_window
  4. High velocity → elevated signal → bots weight their predictions higher

Usage:
    from shared.news_signal import NewsSignal
    ns = NewsSignal()
    ns.start_background_poller()  # call once on startup

    # In bot.fetch_features():
    velocity = ns.get_velocity(market_title)
    # velocity: 0.0 (no news) → 1.0+ (spiking)
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

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

# Stop words to strip from market titles before building search queries
_STOP_WORDS = {
    "will", "the", "a", "an", "be", "to", "in", "on", "at", "by",
    "or", "and", "of", "for", "is", "are", "was", "were", "has",
    "have", "had", "do", "does", "did", "not", "no", "yes", "this",
    "that", "with", "from", "up", "above", "below", "over", "under",
    "than", "more", "less", "reach", "hit", "end", "day", "week",
    "month", "year", "percent", "before", "after", "during", "next",
    "last", "first", "second", "third", "between", "within",
}

def extract_keywords(title: str, max_keywords: int = 3) -> list[str]:
    """
    Pull the most meaningful words from a market title for news search.
    Returns up to max_keywords terms, longer words preferred.
    """
    import re
    words = re.findall(r"[a-zA-Z]{3,}", title)
    words = [w.lower() for w in words if w.lower() not in _STOP_WORDS]
    # Deduplicate, sort by length descending (longer = more specific)
    seen    = set()
    unique  = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    unique.sort(key=len, reverse=True)
    return unique[:max_keywords]


# ── Main class ────────────────────────────────────────────────────────────────

class NewsSignal:
    """
    Background news velocity tracker.

    Parameters
    ----------
    api_key : str
        NewsAPI key. Falls back to NEWSAPI_KEY env var.
    poll_interval_sec : int
        How often to refresh article counts. Default 300 (5 min).
    velocity_window_min : int
        Time window for velocity calculation in minutes. Default 30.
    velocity_spike_threshold : float
        Multiplier vs baseline above which velocity is "spiking". Default 2.0.
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

        # In-memory cache: keyword → {"count": int, "velocity": float, "ts": datetime}
        self._cache: dict[str, dict] = {}
        self._lock  = threading.Lock()
        self._active_keywords: set[str] = set()

        init_news_tables()

        if not self.api_key:
            logger.warning(
                "NewsSignal: NEWSAPI_KEY not set — velocity signals disabled. "
                "Get a free key at newsapi.org"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_market(self, market_title: str) -> None:
        """
        Register a market title for velocity tracking.
        Call this when a new open market is discovered during scan.
        """
        keywords = extract_keywords(market_title)
        with self._lock:
            self._active_keywords.update(keywords)

    def get_velocity(self, market_title: str) -> float:
        """
        Return the current news velocity score for a market title.

        Score interpretation:
          0.0       → no news activity
          0.5–1.0   → moderate news (baseline)
          1.0–2.0   → elevated (worth noting)
          2.0+      → spike (price likely to move soon)

        Returns 0.0 if API key not configured or no data yet.
        """
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
        """
        Return (velocity_score, is_spiking) as a 2-tuple for injection
        into bot feature vectors.

        velocity_score : float  — raw velocity (0.0+)
        is_spiking     : float  — 1.0 if spiking, 0.0 if not (float for np.append)
        """
        velocity   = self.get_velocity(market_title)
        is_spiking = 1.0 if velocity >= self.velocity_spike_threshold else 0.0
        return velocity, is_spiking

    def start_background_poller(self) -> None:
        """
        Launch the background thread that polls NewsAPI every poll_interval_sec.
        Call once on orchestrator startup.
        """
        if not self.api_key:
            return

        t = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="news-velocity-poller",
        )
        t.start()
        logger.info("NewsSignal background poller started (interval=%ds)", self.poll_interval_sec)

    # ── Background poller ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Runs forever in a daemon thread."""
        while True:
            try:
                self._refresh_all()
            except Exception as e:
                logger.error("NewsSignal poll error: %s", e)
            time.sleep(self.poll_interval_sec)

    def _refresh_all(self) -> None:
        """Poll NewsAPI for all active keywords and update cache."""
        with self._lock:
            keywords = list(self._active_keywords)

        if not keywords:
            return

        for keyword in keywords:
            try:
                count = self._fetch_article_count(keyword)
                velocity = self._compute_velocity(keyword, count)

                with self._lock:
                    self._cache[keyword] = {
                        "count":    count,
                        "velocity": velocity,
                        "ts":       datetime.now(timezone.utc),
                    }

                if velocity >= self.velocity_spike_threshold:
                    logger.info(
                        "[NEWS SPIKE] keyword='%s' velocity=%.2f count=%d",
                        keyword, velocity, count,
                    )

                self._persist(keyword, count, velocity)

            except Exception as e:
                logger.warning("NewsSignal refresh failed for '%s': %s", keyword, e)

    def _fetch_article_count(self, keyword: str) -> int:
        """
        Query NewsAPI for articles mentioning keyword in the last velocity_window_min.
        Returns article count (integer).
        """
        import urllib.request
        import json

        since = (
            datetime.now(timezone.utc) - timedelta(minutes=self.velocity_window_min)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={urllib.parse.quote(keyword)}"
            f"&from={since}"
            f"&sortBy=publishedAt"
            f"&pageSize=1"             # we only need the totalResults count
            f"&apiKey={self.api_key}"
        )

        import urllib.parse
        req  = urllib.request.Request(url, headers={"User-Agent": "kalshi-flywheel/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data  = json.loads(resp.read())
            return int(data.get("totalResults", 0))

    def _compute_velocity(self, keyword: str, current_count: int) -> float:
        """
        Compute velocity = current_count / baseline_count.
        Baseline is the average of the last 6 recorded counts for this keyword.
        Returns 0.0 if no baseline yet.
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
            return 1.0  # neutral on first read

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
                (keyword, article_count, velocity_score, window_start, window_end, recorded_at)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p})
            """,
            (
                keyword, count, velocity,
                now - timedelta(minutes=self.velocity_window_min),
                now, now,
            ),
        )