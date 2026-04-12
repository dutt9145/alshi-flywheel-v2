#!/usr/bin/env python3
"""
scripts/train_nba_logreg.py

Train logistic regression models for NBA player props.
Run manually or via cron after games resolve.

Usage:
    railway run python scripts/train_nba_logreg.py

Requirements:
    - 30+ resolved props per prop_code to train
    - Features logged via nba_logistic_model.log_features()
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.nba_logistic_model import (
    get_training_data,
    get_feature_counts,
    FEATURE_COUNT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_nba_logreg")

# Minimum resolved samples to train
MIN_SAMPLES = 30

# Prop codes to train
PROP_CODES = ["PTS", "REB", "AST", "3PT", "STL", "BLK"]


def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _db_execute(sql: str, params: tuple = (), fetch: bool = False) -> list:
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
        logger.error("DB error: %s", e)
        return []


def train_model(prop_code: str) -> dict:
    """
    Train logistic regression for a single prop_code.
    Returns training stats dict.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss

    X, y = get_training_data(prop_code)

    if len(X) < MIN_SAMPLES:
        return {
            "prop_code": prop_code,
            "status": "skipped",
            "reason": f"only {len(X)} resolved (need {MIN_SAMPLES})",
        }

    # Check class balance
    pos_rate = y.mean()
    if pos_rate < 0.1 or pos_rate > 0.9:
        logger.warning(
            "[%s] Imbalanced classes: %.1f%% positive — model may be unreliable",
            prop_code, pos_rate * 100,
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )

    # Cross-validation for Brier score estimate
    cv_scores = cross_val_score(
        model, X_scaled, y,
        cv=min(5, len(X) // 10),  # 5-fold or fewer if small dataset
        scoring="neg_brier_score",
    )
    cv_brier = -cv_scores.mean()

    # Fit on full data
    model.fit(X_scaled, y)

    # In-sample Brier
    y_prob = model.predict_proba(X_scaled)[:, 1]
    train_brier = brier_score_loss(y, y_prob)

    # Compare to Poisson baseline (feature index 8 is poisson_prob)
    poisson_probs = X[:, 8]  # F_POISSON_PROB
    poisson_brier = brier_score_loss(y, poisson_probs)

    improvement = (poisson_brier - train_brier) / poisson_brier * 100

    # Save model
    model_data = {
        "prop_code": prop_code,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "train_brier": round(train_brier, 4),
        "cv_brier": round(cv_brier, 4),
        "poisson_brier": round(poisson_brier, 4),
    }

    # Persist to DB
    p = _ph()
    _db_execute(
        f"""
        INSERT INTO nba_logreg_models (prop_code, model_json, trained_at)
        VALUES ({p}, {p}, {p})
        ON CONFLICT (prop_code) DO UPDATE
        SET model_json = EXCLUDED.model_json,
            trained_at = EXCLUDED.trained_at
        """,
        (prop_code, json.dumps(model_data), datetime.now(timezone.utc)),
    )

    return {
        "prop_code": prop_code,
        "status": "trained",
        "n_samples": len(X),
        "pos_rate": round(pos_rate, 3),
        "train_brier": round(train_brier, 4),
        "cv_brier": round(cv_brier, 4),
        "poisson_brier": round(poisson_brier, 4),
        "improvement": f"{improvement:+.1f}%",
    }


def ensure_models_table():
    """Create the models table if it doesn't exist."""
    _db_execute("""
        CREATE TABLE IF NOT EXISTS nba_logreg_models (
            prop_code TEXT PRIMARY KEY,
            model_json TEXT NOT NULL,
            trained_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def main():
    logger.info("=" * 60)
    logger.info("NBA Logistic Regression Training")
    logger.info("=" * 60)

    ensure_models_table()

    # Show feature counts
    counts = get_feature_counts()
    logger.info("")
    logger.info("Feature counts by prop_code:")
    for code in PROP_CODES:
        total, resolved = counts.get(code, (0, 0))
        pct = (resolved / total * 100) if total > 0 else 0.0
        logger.info("  %s: %d logged, %d resolved (%.1f%%)", code, total, resolved, pct)

    logger.info("")

    # Train models
    results = []
    for code in PROP_CODES:
        total, resolved = counts.get(code, (0, 0))
        if resolved < MIN_SAMPLES:
            logger.info("[%s] Skipping — only %d resolved (need %d)", code, resolved, MIN_SAMPLES)
            results.append({
                "prop_code": code,
                "status": "skipped",
                "reason": f"only {resolved} resolved (need {MIN_SAMPLES})",
            })
            continue

        try:
            result = train_model(code)
            results.append(result)
            if result["status"] == "trained":
                logger.info(
                    "[%s] Trained: n=%d, train_brier=%.4f, cv_brier=%.4f, "
                    "poisson_brier=%.4f, improvement=%s",
                    code, result["n_samples"], result["train_brier"],
                    result["cv_brier"], result["poisson_brier"], result["improvement"],
                )
        except Exception as e:
            logger.error("[%s] Training failed: %s", code, e)
            results.append({
                "prop_code": code,
                "status": "error",
                "reason": str(e),
            })

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)

    trained = [r for r in results if r.get("status") == "trained"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    if not trained:
        logger.info("No models trained — need more resolved data.")
        logger.info("Features are being logged automatically. Check back after more games resolve.")
    else:
        logger.info("Trained %d models:", len(trained))
        for r in trained:
            logger.info(
                "  %s: Brier %.4f (Poisson: %.4f, %s)",
                r["prop_code"], r["cv_brier"], r["poisson_brier"], r["improvement"],
            )

    if skipped:
        logger.info("")
        logger.info("Skipped %d prop codes (insufficient data):", len(skipped))
        for r in skipped:
            logger.info("  %s: %s", r["prop_code"], r.get("reason", ""))

    if errors:
        logger.info("")
        logger.info("Errors on %d prop codes:", len(errors))
        for r in errors:
            logger.info("  %s: %s", r["prop_code"], r.get("reason", ""))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()