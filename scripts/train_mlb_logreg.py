#!/usr/bin/env python3
"""
scripts/train_mlb_logreg.py — Train logistic regression models for MLB props

Usage:
  # Set DATABASE_URL in environment, then:
  python scripts/train_mlb_logreg.py

  # Or with Railway:
  railway run python scripts/train_mlb_logreg.py

What it does:
  1. Connects to Supabase
  2. First, backfills outcomes: joins mlb_prop_features with signals table
     to fill in outcome column for resolved markets
  3. For each prop_code with enough resolved data (default 30+):
     - Pulls features + outcomes
     - Trains sklearn LogisticRegression with L2 penalty
     - Computes Brier score
     - Uploads coefficients to mlb_logreg_models table
  4. Prints summary

The trained models are automatically loaded by the orchestrator on next
restart via load_models_from_supabase().
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_mlb")


def backfill_outcomes():
    """Join mlb_prop_features with signals to fill in outcomes."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        logger.error("No DATABASE_URL set")
        sys.exit(1)

    import psycopg2
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Update mlb_prop_features.outcome from signals.outcome
    cur.execute("""
        UPDATE mlb_prop_features f
        SET outcome = s.outcome
        FROM signals s
        WHERE f.ticker = s.ticker
          AND f.outcome IS NULL
          AND s.outcome IS NOT NULL
    """)
    updated = cur.rowcount
    conn.commit()

    logger.info("Backfilled %d outcomes from signals table", updated)

    # Stats
    cur.execute("""
        SELECT prop_code,
               COUNT(*) as total,
               COUNT(outcome) as resolved,
               ROUND(AVG(CASE WHEN outcome = 1 THEN 1.0 ELSE 0.0 END)::numeric, 3) as yes_rate
        FROM mlb_prop_features
        GROUP BY prop_code
        ORDER BY total DESC
    """)
    rows = cur.fetchall()
    conn.close()

    logger.info("\n%-6s %6s %8s %8s", "PROP", "TOTAL", "RESOLVED", "YES%")
    logger.info("-" * 32)
    for prop_code, total, resolved, yes_rate in rows:
        logger.info("%-6s %6d %8d %7.1f%%", prop_code, total, resolved, (yes_rate or 0) * 100)

    return rows


def main():
    logger.info("=== MLB Logistic Regression Training ===")

    # Step 1: Backfill outcomes
    prop_stats = backfill_outcomes()

    # Step 2: Train models
    # Import here so DATABASE_URL is available
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.mlb_logistic_model import train_model

    results = []
    for prop_code, total, resolved, _ in prop_stats:
        if resolved < 30:
            logger.info("[%s] Skipping — only %d resolved (need 30)", prop_code, resolved)
            continue

        result = train_model(prop_code, min_samples=30)
        if result:
            results.append(result)

    # Step 3: Summary
    logger.info("\n=== Training Summary ===")
    if not results:
        logger.info("No models trained — need more resolved data.")
        logger.info("Features are being logged automatically. Check back after more games resolve.")
    else:
        for r in results:
            logger.info(
                "  %s: Brier=%.4f  n=%d  balance=%s",
                r["prop_code"], r["brier"], r["n_samples"], r["class_balance"],
            )
            # Print feature importance
            coef = r["coef"]
            names = r["feature_names"]
            importance = sorted(zip(names, coef), key=lambda x: abs(x[1]), reverse=True)
            for name, c in importance[:5]:
                logger.info("    %+.4f  %s", c, name)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()