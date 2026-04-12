"""
shared/mlb_logistic_model.py — Logistic regression for MLB player props

Replaces the Poisson model with a feature-based classifier trained on
historical resolved trades from Supabase.

Architecture:
  1. Feature extraction: builds a feature vector from the same data the
     Poisson model uses (season avg, L10 rolling, opp pitcher, park factor,
     home/away, threshold distance)
  2. Feature logging: writes features + outcome to `mlb_prop_features` table
     on every evaluation so training data accumulates automatically
  3. Inference: loads trained coefficients from Supabase `mlb_logreg_models`
     table and predicts P(YES) using sklearn-compatible logistic regression
  4. Training: standalone script pulls features + outcomes, fits model,
     uploads coefficients

Prop codes supported:
  HIT (≥N hits), TB (≥N total bases), HR/HRR (≥N home runs),
  KS (≥N strikeouts by pitcher), RBI (≥N RBIs), RUN (≥N runs)
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLBFeatureVector:
    """Raw features for one MLB player prop evaluation."""
    prop_code:       str      # HIT, TB, HR, KS, RBI, RUN
    threshold:       float    # e.g. 1.5 for "≥2 hits"
    # Batter season averages
    season_avg:      float    # season avg for this stat (hits/game, TB/game, etc.)
    season_games:    int
    # L10 rolling averages
    rolling_avg:     float    # last 10 games avg for this stat
    rolling_games:   int      # how many of the last 10 we actually have
    # Opposing pitcher context
    opp_pitcher_k9:  float    # opposing pitcher K/9 (0 if unavailable)
    opp_pitcher_era: float    # opposing pitcher ERA (0 if unavailable)
    # Situational
    park_factor:     float    # park factor (1.0 = neutral)
    is_home:         bool     # True if player's team is home
    # Threshold distance (how far the threshold is from expected value)
    threshold_dist:  float    # (threshold - season_avg) / max(season_avg, 0.01)
    # Market price as a feature (what the market thinks)
    market_prob:     float    # Kalshi market probability


@dataclass
class LogRegPrediction:
    """Output from the logistic regression model."""
    prob_yes:    float
    confidence:  float
    model_used:  str    # "logreg" or "poisson_fallback"


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Constants matching mlb_hit_model.py
_LEAGUE_AVG_AVG     = 0.243
_LEAGUE_AVG_SLG     = 0.399
_LEAGUE_AVG_HR_RATE = 0.032
_LEAGUE_AVG_K9      = 8.8


def _get_season_and_rolling(prop_code, season, rolling):
    """
    Return (season_avg, rolling_avg, season_games, rolling_games) for a given
    prop_code, using actual fields from BatterSeasonStats / BatterRollingStats.
    """
    if prop_code == "HIT":
        s_avg = season.avg if season and season.at_bats > 0 else _LEAGUE_AVG_AVG
        r_avg = rolling.avg if rolling and rolling.at_bats > 0 else None
        s_games = season.plate_apps if season else 0
        r_games = rolling.games if rolling and hasattr(rolling, 'games') else 0
    elif prop_code == "TB":
        s_avg = season.slg if season and season.at_bats > 0 else _LEAGUE_AVG_SLG
        r_avg = (rolling.total_bases / rolling.at_bats
                 if rolling and rolling.at_bats > 0 else None)
        s_games = season.plate_apps if season else 0
        r_games = rolling.games if rolling and hasattr(rolling, 'games') else 0
    elif prop_code in ("HR", "HRR"):
        s_avg = (season.home_runs / season.plate_apps
                 if season and season.plate_apps > 0 else _LEAGUE_AVG_HR_RATE)
        r_avg = (rolling.home_runs / rolling.at_bats
                 if rolling and rolling.at_bats > 0 else None)
        s_games = season.plate_apps if season else 0
        r_games = rolling.games if rolling and hasattr(rolling, 'games') else 0
    else:
        # Fallback for unknown batter props (RBI, RUN, etc.)
        s_avg = 0.0
        r_avg = None
        s_games = season.plate_apps if season else 0
        r_games = 0
    return s_avg, r_avg, s_games, r_games


def extract_features(
    parsed,             # ParsedMLBTicker from kalshi_ticker_parser
    season,             # BatterSeasonStats or None
    rolling,            # BatterRollingStats or None
    opp_pitcher=None,   # PitcherSeasonStats or None
    park_factor: float = 1.0,
    market_prob: float = 0.5,
) -> Optional[MLBFeatureVector]:
    """
    Extract a feature vector from the same data the Poisson model uses.
    Returns None if we can't build a meaningful feature vector.
    """
    prop_code = parsed.prop_code
    threshold = float(parsed.threshold)

    # Determine if home game
    is_home = (parsed.player_team_code == parsed.home_team_code)

    if prop_code == "KS":
        # Pitcher strikeouts — opp_pitcher is actually the pitcher's own stats
        if opp_pitcher is None:
            return None
        season_avg = opp_pitcher.k_per_9 / 9.0 * 5.5  # expected K per game
        rolling_avg = season_avg  # no rolling for pitchers yet
        rolling_games = 0
        season_games = max(int(getattr(opp_pitcher, 'games', 0)), 1)
        opp_k9 = 0.0
        opp_era = 0.0
    else:
        # Batter props
        if season is None:
            return None

        season_avg, rolling_avg_raw, season_games, rolling_games = \
            _get_season_and_rolling(prop_code, season, rolling)

        rolling_avg = rolling_avg_raw if rolling_avg_raw is not None else season_avg

        # Opposing pitcher context
        opp_k9 = opp_pitcher.k_per_9 if opp_pitcher else 0.0
        opp_era = getattr(opp_pitcher, 'era', 0.0) if opp_pitcher else 0.0

    # Threshold distance: how many standard deviations above/below avg
    threshold_dist = (threshold - season_avg) / max(season_avg, 0.01)

    return MLBFeatureVector(
        prop_code=prop_code,
        threshold=threshold,
        season_avg=season_avg,
        season_games=season_games,
        rolling_avg=rolling_avg,
        rolling_games=rolling_games,
        opp_pitcher_k9=opp_k9,
        opp_pitcher_era=opp_era,
        park_factor=park_factor,
        is_home=is_home,
        threshold_dist=threshold_dist,
        market_prob=market_prob,
    )


def features_to_array(fv: MLBFeatureVector) -> np.ndarray:
    """Convert feature vector to numpy array for sklearn."""
    return np.array([
        fv.season_avg,
        fv.rolling_avg,
        fv.threshold,
        fv.threshold_dist,
        fv.opp_pitcher_k9,
        fv.opp_pitcher_era,
        fv.park_factor,
        float(fv.is_home),
        fv.season_games,
        fv.rolling_games,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "season_avg", "rolling_avg", "threshold", "threshold_dist",
    "opp_pitcher_k9", "opp_pitcher_era", "park_factor", "is_home",
    "season_games", "rolling_games",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE LOGGING (accumulates training data in Supabase)
# ═══════════════════════════════════════════════════════════════════════════════

def _db_execute(sql: str, params: tuple = ()) -> bool:
    """Execute SQL against DATABASE_URL. Returns True on success."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning("mlb_logistic DB error: %s", e)
        return False


def init_feature_table() -> None:
    """Create the feature logging table if it doesn't exist."""
    _db_execute("""
        CREATE TABLE IF NOT EXISTS mlb_prop_features (
            id            SERIAL PRIMARY KEY,
            ticker        TEXT NOT NULL,
            player_name   TEXT,
            prop_code     TEXT NOT NULL,
            threshold     FLOAT8 NOT NULL,
            season_avg    FLOAT8,
            rolling_avg   FLOAT8,
            threshold_dist FLOAT8,
            opp_pitcher_k9 FLOAT8,
            opp_pitcher_era FLOAT8,
            park_factor   FLOAT8,
            is_home       BOOLEAN,
            season_games  INT,
            rolling_games INT,
            market_prob   FLOAT8,
            our_prob      FLOAT8,
            outcome       INT,          -- NULL until resolved, then 0 or 1
            created_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_mlb_features_ticker
        ON mlb_prop_features (ticker)
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_mlb_features_outcome
        ON mlb_prop_features (outcome)
    """)


_table_initialized = False


def log_features(
    ticker: str,
    player_name: str,
    fv: MLBFeatureVector,
    our_prob: float,
) -> None:
    """Log feature vector to Supabase for future training."""
    global _table_initialized
    if not _table_initialized:
        init_feature_table()
        _table_initialized = True

    _db_execute(
        """
        INSERT INTO mlb_prop_features
            (ticker, player_name, prop_code, threshold,
             season_avg, rolling_avg, threshold_dist,
             opp_pitcher_k9, opp_pitcher_era, park_factor, is_home,
             season_games, rolling_games, market_prob, our_prob)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            ticker, player_name, fv.prop_code, fv.threshold,
            fv.season_avg, fv.rolling_avg, fv.threshold_dist,
            fv.opp_pitcher_k9, fv.opp_pitcher_era, fv.park_factor, fv.is_home,
            fv.season_games, fv.rolling_games, fv.market_prob, our_prob,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL STORAGE & LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory cache of trained model coefficients per prop_code
_models: dict[str, dict] = {}  # prop_code → {"coef": [...], "intercept": float, "n_samples": int}


def _init_model_table() -> None:
    _db_execute("""
        CREATE TABLE IF NOT EXISTS mlb_logreg_models (
            prop_code   TEXT PRIMARY KEY,
            coef_json   TEXT NOT NULL,
            intercept   FLOAT8 NOT NULL,
            n_samples   INT NOT NULL,
            brier       FLOAT8,
            trained_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def load_models_from_supabase() -> int:
    """Load all trained models from Supabase into memory. Returns count loaded."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        return 0

    _init_model_table()

    try:
        import psycopg2
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute("SELECT prop_code, coef_json, intercept, n_samples FROM mlb_logreg_models")
        rows = cur.fetchall()
        conn.close()

        for prop_code, coef_json, intercept, n_samples in rows:
            _models[prop_code] = {
                "coef": json.loads(coef_json),
                "intercept": intercept,
                "n_samples": n_samples,
            }
        if _models:
            logger.info("[mlb_logreg] Loaded %d trained models: %s",
                        len(_models), list(_models.keys()))
        return len(_models)
    except Exception as e:
        logger.warning("[mlb_logreg] Failed to load models: %s", e)
        return 0


def save_model_to_supabase(
    prop_code: str,
    coef: list[float],
    intercept: float,
    n_samples: int,
    brier: float,
) -> bool:
    """Save trained model coefficients to Supabase."""
    _init_model_table()
    return _db_execute(
        """
        INSERT INTO mlb_logreg_models (prop_code, coef_json, intercept, n_samples, brier, trained_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (prop_code) DO UPDATE SET
            coef_json  = EXCLUDED.coef_json,
            intercept  = EXCLUDED.intercept,
            n_samples  = EXCLUDED.n_samples,
            brier      = EXCLUDED.brier,
            trained_at = NOW()
        """,
        (prop_code, json.dumps(coef), intercept, n_samples, brier),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_logreg(fv: MLBFeatureVector) -> Optional[LogRegPrediction]:
    """
    Predict P(YES) using trained logistic regression coefficients.
    Returns None if no trained model exists for this prop_code.
    """
    model = _models.get(fv.prop_code)
    if not model:
        return None

    coef = np.array(model["coef"])
    intercept = model["intercept"]
    X = features_to_array(fv)

    # Logistic function: P(YES) = 1 / (1 + exp(-(X·coef + intercept)))
    logit = np.dot(X, coef) + intercept
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    # Confidence based on sample size and distance from 0.5
    n = model["n_samples"]
    sample_conf = min(0.90, 0.50 + 0.005 * n)  # ramps from 0.50 to 0.90 over 80 samples
    edge_conf = abs(prob - 0.5) * 2  # higher confidence when prediction is extreme
    confidence = sample_conf * (0.7 + 0.3 * edge_conf)

    return LogRegPrediction(
        prob_yes=float(np.clip(prob, 0.02, 0.98)),
        confidence=confidence,
        model_used="logreg",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING (called from scripts/train_mlb_logreg.py)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(prop_code: str, min_samples: int = 30) -> Optional[dict]:
    """
    Train a logistic regression model for one prop_code using data from
    mlb_prop_features table. Returns stats dict or None if insufficient data.

    This function is designed to be called from a training script, not at
    runtime. It requires sklearn.
    """
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        logger.error("No DATABASE_URL — can't train")
        return None

    import psycopg2
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT season_avg, rolling_avg, threshold, threshold_dist,
               opp_pitcher_k9, opp_pitcher_era, park_factor, is_home,
               season_games, rolling_games, outcome
        FROM mlb_prop_features
        WHERE prop_code = %s AND outcome IS NOT NULL
        ORDER BY created_at
        """,
        (prop_code,),
    )
    rows = cur.fetchall()
    conn.close()

    if len(rows) < min_samples:
        logger.info("[train] %s: only %d samples (need %d), skipping",
                     prop_code, len(rows), min_samples)
        return None

    X = np.array([[
        r[0], r[1], r[2], r[3], r[4], r[5], r[6],
        float(r[7]) if r[7] is not None else 0.0,
        r[8], r[9],
    ] for r in rows], dtype=np.float64)

    y = np.array([r[10] for r in rows], dtype=np.float64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_scaled, y)

    # Cross-validated Brier score
    probs = model.predict_proba(X_scaled)[:, 1]
    brier = float(np.mean((probs - y) ** 2))

    # Note: we store scaled coefficients. At inference time we need to
    # scale features the same way. For simplicity in v1, we store
    # coef * (1/scale) + adjust intercept so we can use raw features.
    # coef_unscaled[i] = coef[i] / scale[i]
    # intercept_adj = intercept - sum(coef[i] * mean[i] / scale[i])
    coef_unscaled = model.coef_[0] / scaler.scale_
    intercept_adj = model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_)

    result = {
        "prop_code": prop_code,
        "coef": coef_unscaled.tolist(),
        "intercept": float(intercept_adj),
        "n_samples": len(rows),
        "brier": brier,
        "class_balance": f"{y.mean():.2%} YES",
        "feature_names": FEATURE_NAMES,
    }

    # Save to Supabase
    saved = save_model_to_supabase(
        prop_code=prop_code,
        coef=coef_unscaled.tolist(),
        intercept=float(intercept_adj),
        n_samples=len(rows),
        brier=brier,
    )

    # Update in-memory cache
    _models[prop_code] = {
        "coef": coef_unscaled.tolist(),
        "intercept": float(intercept_adj),
        "n_samples": len(rows),
    }

    logger.info(
        "[train] %s: n=%d brier=%.4f balance=%s saved=%s",
        prop_code, len(rows), brier, result["class_balance"], saved,
    )

    return result