"""
shared/calibration_logger.py  (v4 — Postgres interval fix + better error logging)

Fixes vs v3:
  1. Postgres INTERVAL syntax fixed — was using string interpolation
     which breaks with psycopg2; now uses parameterized query correctly
  2. log_signal and log_trade both log a confirmation line on success
     so you can verify writes are reaching Supabase in Railway logs
  3. init_db() logs which mode it's running in at INFO level (was silent)
  4. _db() context manager logs connection errors explicitly

Automatically detects which database to use:
  - DATABASE_URL set → Supabase PostgreSQL (Railway production)
  - DATABASE_URL not set → local SQLite (laptop dev mode)

Zero code changes needed when switching environments.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH      = os.getenv("DB_PATH", "flywheel.db")
USE_POSTGRES = bool(DATABASE_URL)


# ── Connection context manager ────────────────────────────────────────────────

@contextmanager
def _db():
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        try:
            conn = psycopg2.connect(DATABASE_URL)
        except Exception as e:
            logger.error("Postgres connection failed: %s", e)
            raise
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _rows_to_dicts(rows, cursor=None):
    """Normalise both sqlite3.Row and psycopg2 tuples into dicts."""
    if not rows:
        return []
    if USE_POSTGRES and cursor:
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in rows]
    return [dict(r) for r in rows]


def _ph(n: int) -> str:
    """Return n placeholders: %s for postgres, ? for sqlite."""
    p = "%s" if USE_POSTGRES else "?"
    return ", ".join([p] * n)


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Works for both SQLite and Postgres."""
    mode = "Supabase PostgreSQL" if USE_POSTGRES else f"SQLite at {DB_PATH}"
    logger.info("Initialising database — %s", mode)

    if USE_POSTGRES:
        serial  = "SERIAL PRIMARY KEY"
        int_def = "INTEGER NOT NULL DEFAULT 1"
        ref     = "REFERENCES trades(id)"
    else:
        serial  = "INTEGER PRIMARY KEY AUTOINCREMENT"
        int_def = "INTEGER NOT NULL DEFAULT 1"
        ref     = "REFERENCES trades(id)"

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS signals (
            id           {serial},
            created_at   TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            sector       TEXT NOT NULL,
            our_prob     REAL NOT NULL,
            market_prob  REAL NOT NULL,
            edge         REAL NOT NULL,
            confidence   REAL NOT NULL,
            direction    TEXT NOT NULL,
            brier_score  REAL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS trades (
            id              {serial},
            created_at      TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            contracts       INTEGER NOT NULL,
            yes_price_cents INTEGER NOT NULL,
            dollars_risked  REAL NOT NULL,
            avg_prob        REAL NOT NULL,
            avg_edge        REAL NOT NULL,
            avg_confidence  REAL NOT NULL,
            order_id        TEXT,
            demo_mode       {int_def}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS outcomes (
            id          {serial},
            logged_at   TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            resolved    TEXT NOT NULL,
            pnl_usd     REAL,
            trade_id    INTEGER {ref}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS calibration_stats (
            id             {serial},
            computed_at    TEXT NOT NULL,
            sector         TEXT NOT NULL,
            window_days    INTEGER,
            n_predictions  INTEGER,
            brier_score    REAL,
            accuracy       REAL,
            avg_edge       REAL,
            avg_confidence REAL,
            total_pnl      REAL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS arb_opportunities (
            id           {serial},
            detected_at  TEXT,
            ticker       TEXT,
            kalshi_prob  REAL,
            poly_prob    REAL,
            sharp_prob   REAL,
            spread       REAL,
            mode         TEXT,
            notes        TEXT
        )
        """,
    ]

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            for stmt in statements:
                cur.execute(stmt)
        else:
            for stmt in statements:
                conn.execute(stmt)

    logger.info("Database ready — %s", mode)


# ── Write helpers ─────────────────────────────────────────────────────────────

def log_signal(
    ticker: str,
    sector: str,
    our_prob: float,
    market_prob: float,
    edge: float,
    confidence: float,
    direction: str,
    brier_score: Optional[float] = None,
) -> None:
    """
    Write one row to the signals table.
    Called for every market that passes consensus (gates 1-3),
    regardless of whether the arb gate or Kelly sizer approves the trade.
    """
    sql = f"""
        INSERT INTO signals
        (created_at, ticker, sector, our_prob, market_prob,
         edge, confidence, direction, brier_score)
        VALUES ({_ph(9)})
    """
    vals = (
        datetime.now(timezone.utc).isoformat(),
        ticker, sector,
        our_prob, market_prob,
        edge, confidence,
        direction, brier_score,
    )
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)
    logger.info(
        "SIGNAL logged | %s | %s | prob=%.2f%% edge=%+.2f%% conf=%.0f%%",
        ticker, direction,
        our_prob * 100, edge * 100, confidence * 100,
    )


def log_trade(
    ticker: str,
    direction: str,
    contracts: int,
    yes_price_cents: int,
    dollars_risked: float,
    avg_prob: float,
    avg_edge: float,
    avg_confidence: float,
    order_id: Optional[str] = None,
    demo_mode: bool = True,
) -> int:
    """
    Write one row to the trades table. Returns the new trade ID.
    demo_mode=True means the trade was simulated (no real money moved).
    """
    vals = (
        datetime.now(timezone.utc).isoformat(),
        ticker, direction,
        contracts, yes_price_cents, dollars_risked,
        avg_prob, avg_edge, avg_confidence,
        order_id, int(demo_mode),
    )

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO trades
                (created_at, ticker, direction, contracts, yes_price_cents,
                 dollars_risked, avg_prob, avg_edge, avg_confidence,
                 order_id, demo_mode)
                VALUES ({_ph(11)}) RETURNING id
            """, vals)
            trade_id = cur.fetchone()[0]
        else:
            cur = conn.execute(f"""
                INSERT INTO trades
                (created_at, ticker, direction, contracts, yes_price_cents,
                 dollars_risked, avg_prob, avg_edge, avg_confidence,
                 order_id, demo_mode)
                VALUES ({_ph(11)})
            """, vals)
            trade_id = cur.lastrowid

    logger.info(
        "TRADE logged #%d | %s | %s %dx@%dc | $%.2f | demo=%s",
        trade_id, ticker, direction,
        contracts, yes_price_cents, dollars_risked, demo_mode,
    )
    return trade_id


def log_outcome(
    ticker: str,
    resolved: str,
    pnl_usd: Optional[float] = None,
    trade_id: Optional[int] = None,
) -> None:
    vals = (
        datetime.now(timezone.utc).isoformat(),
        ticker, resolved.upper(),
        pnl_usd, trade_id,
    )
    sql = f"""
        INSERT INTO outcomes (logged_at, ticker, resolved, pnl_usd, trade_id)
        VALUES ({_ph(5)})
    """
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)
    logger.info(
        "OUTCOME logged | %s resolved=%s P&L=%s",
        ticker, resolved,
        f"${pnl_usd:.2f}" if pnl_usd is not None else "pending",
    )


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_calibration(sector: str, window_days: int = 30) -> dict:
    """
    Compute Brier score and accuracy for a sector over the last window_days.
    Uses parameterized interval to avoid psycopg2 string interpolation issues.
    """
    if USE_POSTGRES:
        # Correct parameterized interval for psycopg2 — do NOT use f-string here
        date_sql = "s.created_at >= NOW() - INTERVAL '%s days'"
        params   = (sector, window_days)
        sql = f"""
            SELECT s.our_prob, o.resolved, s.edge, s.confidence
            FROM signals s
            LEFT JOIN outcomes o ON s.ticker = o.ticker
            WHERE s.sector = %s
              AND {date_sql}
              AND o.resolved IS NOT NULL
        """
    else:
        date_sql = "s.created_at >= datetime('now', ? || ' days')"
        params   = (sector, f"-{window_days}")
        sql = f"""
            SELECT s.our_prob, o.resolved, s.edge, s.confidence
            FROM signals s
            LEFT JOIN outcomes o ON s.ticker = o.ticker
            WHERE s.sector = ?
              AND {date_sql}
              AND o.resolved IS NOT NULL
        """

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = _rows_to_dicts(cur.fetchall(), cur)
        else:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not rows:
        logger.info("[%s] No resolved signals in last %d days", sector, window_days)
        return {"sector": sector, "n": 0, "brier": None, "accuracy": None}

    probs   = [r["our_prob"] for r in rows]
    actuals = [1 if r["resolved"] == "YES" else 0 for r in rows]
    edges   = [r["edge"] for r in rows]
    confs   = [r["confidence"] for r in rows]

    brier    = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(rows)
    accuracy = sum(
        1 for p, a in zip(probs, actuals) if (p > 0.5) == bool(a)
    ) / len(rows)

    stats = {
        "sector":          sector,
        "n":               len(rows),
        "brier":           round(brier, 4),
        "accuracy":        round(accuracy, 4),
        "avg_edge":        round(sum(edges) / len(edges), 4),
        "avg_confidence":  round(sum(confs) / len(confs), 4),
    }

    ins_vals = (
        datetime.now(timezone.utc).isoformat(),
        sector, window_days, len(rows),
        brier, accuracy,
        stats["avg_edge"], stats["avg_confidence"], 0,
    )
    ins_sql = f"""
        INSERT INTO calibration_stats
        (computed_at, sector, window_days, n_predictions,
         brier_score, accuracy, avg_edge, avg_confidence, total_pnl)
        VALUES ({_ph(9)})
    """
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(ins_sql, ins_vals)
        else:
            conn.execute(ins_sql, ins_vals)

    logger.info(
        "[%s] Calibration: Brier=%.4f Acc=%.1f%% n=%d",
        sector, brier, accuracy * 100, len(rows),
    )
    return stats