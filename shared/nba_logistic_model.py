"""
shared/nba_logistic_model.py

NBA player prop logistic regression model with feature logging.
Mirrors shared/mlb_logistic_model.py architecture.

Phase 1 (current): Log features on every evaluation
Phase 2: Train logreg models per prop_code once 30+ resolved
Phase 3: Swap inference if logreg Brier < Poisson Brier

Feature vector (per prop):
  [0]  season_stat_pg        — season average for this stat
  [1]  rolling_stat_pg       — last 10 games average
  [2]  threshold             — prop line
  [3]  delta_season          — season_avg - threshold
  [4]  delta_rolling         — rolling_avg - threshold (0 if no rolling)
  [5]  games_played          — sample size proxy
  [6]  minutes_pg            — playing time proxy
  [7]  market_prob           — Kalshi implied probability
  [8]  poisson_prob          — our Poisson model prediction
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature indices ────────────────────────────────────────────────────────────
F_SEASON_STAT     = 0
F_ROLLING_STAT    = 1
F_THRESHOLD       = 2
F_DELTA_SEASON    = 3
F_DELTA_ROLLING   = 4
F_GAMES_PLAYED    = 5
F_MINUTES_PG      = 6
F_MARKET_PROB     = 7
F_POISSON_PROB    = 8

FEATURE_COUNT = 9


@dataclass
class NBAFeatures:
    """Feature vector for a single NBA prop evaluation."""
    prop_code:      str
    ticker:         str
    player_name:    str
    features:       np.ndarray   # shape (FEATURE_COUNT,)
    poisson_prob:   float
    market_prob:    float
    timestamp:      str


def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _db_execute(sql: str, params: tuple = (), fetch: bool = False) -> list:
    """Execute SQL against Postgres or SQLite."""
    database_url = os.getenv("DATABASE_URL", "")
    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            cur.execute(sql, params)
            result = []
            if fetch and cur.description:
                cols = [d[0] for d in cur.description]
                result = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return result
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            result = [dict(r) for r in cur.fetchall()] if fetch else []
            conn.commit()
            conn.close()
            return result
    except Exception as e:
        logger.error("nba_logistic_model DB error: %s", e)
        return []


def init_nba_features_table() -> None:
    """Create the nba_prop_features table if it doesn't exist."""
    p = _ph()
    _db_execute(f"""
        CREATE TABLE IF NOT EXISTS nba_prop_features (
            id SERIAL PRIMARY KEY,
            prop_code TEXT NOT NULL,
            ticker TEXT NOT NULL,
            player_name TEXT,
            features TEXT NOT NULL,
            poisson_prob FLOAT8,
            market_prob FLOAT8,
            outcome INT,
            logged_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_nba_features_prop_code
        ON nba_prop_features (prop_code)
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_nba_features_ticker
        ON nba_prop_features (ticker)
    """)
    logger.info("[nba_logreg] nba_prop_features table initialized")


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(
    parsed,            # NBATicker
    season_stats,      # PlayerStats
    rolling_stats,     # RollingStats or None
    poisson_prob: float,
    market_prob: float,
) -> NBAFeatures:
    """
    Build feature vector for a single NBA prop evaluation.
    """
    prop_code = parsed.prop_code
    threshold = parsed.threshold

    # Map prop_code to the relevant stat field
    stat_map = {
        "PTS": ("points_pg", "points_pg"),
        "REB": ("rebounds_pg", "rebounds_pg"),
        "AST": ("assists_pg", "assists_pg"),
        "3PT": ("three_pm_pg", "three_pm_pg"),
        "STL": ("steals_pg", "steals_pg"),
        "BLK": ("blocks_pg", "blocks_pg"),
    }

    season_field, rolling_field = stat_map.get(prop_code, ("points_pg", "points_pg"))

    season_val = getattr(season_stats, season_field, 0.0) if season_stats else 0.0
    rolling_val = getattr(rolling_stats, rolling_field, 0.0) if rolling_stats else 0.0

    games_played = season_stats.games_played if season_stats else 0
    minutes_pg = season_stats.minutes_pg if season_stats else 0.0

    features = np.array([
        season_val,                           # F_SEASON_STAT
        rolling_val,                          # F_ROLLING_STAT
        float(threshold),                     # F_THRESHOLD
        season_val - threshold,               # F_DELTA_SEASON
        rolling_val - threshold if rolling_stats else 0.0,  # F_DELTA_ROLLING
        float(games_played),                  # F_GAMES_PLAYED
        minutes_pg,                           # F_MINUTES_PG
        market_prob,                          # F_MARKET_PROB
        poisson_prob,                         # F_POISSON_PROB
    ], dtype=np.float32)

    player_name = f"{parsed.player_first}. {parsed.player_last}"

    return NBAFeatures(
        prop_code=prop_code,
        ticker=parsed.raw_ticker,
        player_name=player_name,
        features=features,
        poisson_prob=poisson_prob,
        market_prob=market_prob,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Feature logging (Phase 1) ──────────────────────────────────────────────────

def log_features(nba_features: NBAFeatures) -> None:
    """
    Log feature vector to database for future training.
    Called on every NBA prop evaluation.
    """
    p = _ph()
    try:
        features_json = json.dumps(nba_features.features.tolist())
        _db_execute(
            f"""
            INSERT INTO nba_prop_features
                (prop_code, ticker, player_name, features, poisson_prob, market_prob, logged_at)
            VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p})
            """,
            (
                nba_features.prop_code,
                nba_features.ticker,
                nba_features.player_name,
                features_json,
                nba_features.poisson_prob,
                nba_features.market_prob,
                datetime.now(timezone.utc),
            ),
        )
        logger.debug(
            "[nba_logreg] Logged %s: %s → poisson=%.3f market=%.3f",
            nba_features.prop_code, nba_features.ticker[:40],
            nba_features.poisson_prob, nba_features.market_prob,
        )
    except Exception as e:
        logger.warning("[nba_logreg] log_features failed: %s", e)


# ── Outcome recording ──────────────────────────────────────────────────────────

def record_outcome(ticker: str, outcome: int) -> None:
    """
    Record resolution outcome for a logged feature vector.
    outcome: 1 = over, 0 = under
    """
    p = _ph()
    try:
        _db_execute(
            f"""
            UPDATE nba_prop_features
            SET outcome = {p}
            WHERE ticker = {p} AND outcome IS NULL
            """,
            (outcome, ticker),
        )
        logger.debug("[nba_logreg] Recorded outcome %d for %s", outcome, ticker)
    except Exception as e:
        logger.warning("[nba_logreg] record_outcome failed: %s", e)


# ── Training data retrieval ────────────────────────────────────────────────────

def get_training_data(prop_code: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch all resolved features for a prop_code.
    Returns (X, y) where X is (n_samples, FEATURE_COUNT) and y is (n_samples,).
    """
    p = _ph()
    rows = _db_execute(
        f"""
        SELECT features, outcome
        FROM nba_prop_features
        WHERE prop_code = {p} AND outcome IS NOT NULL
        ORDER BY logged_at
        """,
        (prop_code,),
        fetch=True,
    )

    if not rows:
        return np.array([]), np.array([])

    X_list = []
    y_list = []
    for row in rows:
        try:
            features = json.loads(row["features"])
            X_list.append(features)
            y_list.append(int(row["outcome"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def get_feature_counts() -> dict[str, tuple[int, int]]:
    """
    Get counts of logged vs resolved features per prop_code.
    Returns {prop_code: (total_logged, total_resolved)}.
    """
    rows = _db_execute(
        """
        SELECT prop_code,
               COUNT(*) as total,
               COUNT(outcome) as resolved
        FROM nba_prop_features
        GROUP BY prop_code
        """,
        fetch=True,
    )
    return {
        row["prop_code"]: (int(row["total"]), int(row["resolved"]))
        for row in rows
    }


# ── Model persistence (Phase 2) ────────────────────────────────────────────────
# Placeholder for logreg model storage — will be implemented when training begins

_models: dict[str, object] = {}  # prop_code → trained LogisticRegression


def load_models() -> None:
    """Load trained models from disk/DB (Phase 2)."""
    pass  # TODO: implement when models exist


def save_model(prop_code: str, model: object) -> None:
    """Save trained model to disk/DB (Phase 2)."""
    pass  # TODO: implement when models exist


def predict_logreg(prop_code: str, features: np.ndarray) -> Optional[float]:
    """
    Run logreg inference if a model exists for this prop_code.
    Returns P(over) or None if no model available.
    """
    if prop_code not in _models:
        return None
    try:
        model = _models[prop_code]
        prob = model.predict_proba(features.reshape(1, -1))[0, 1]
        return float(prob)
    except Exception as e:
        logger.warning("[nba_logreg] predict failed for %s: %s", prop_code, e)
        return None


# ── Initialization ─────────────────────────────────────────────────────────────

# Initialize table on import
try:
    init_nba_features_table()
except Exception as e:
    logger.warning("[nba_logreg] Table init failed (will retry): %s", e)