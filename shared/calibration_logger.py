"""
shared/calibration_logger.py  (v6 — log_outcome signature updated)

Changes vs v5:
  1. log_outcome() now accepts our_prob and correct parameters
     so the resolution ingestion pipeline can record what the bot
     predicted vs what actually happened — required for Brier scoring
  2. outcomes table INSERT updated to match new columns
  3. Everything else unchanged from v5
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
            id           {serial},
            logged_at    TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            resolved     TEXT NOT NULL,
            pnl_usd      REAL,
            trade_id     INTEGER {ref},
            our_prob     REAL,
            correct      BOOLEAN
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
    sql = f"""
        INSERT INTO signals
        (created_at, ticker, sector, our_prob, market_prob,
         edge, confidence, direction, brier_score)
        VALUES ({_ph(9)})
    """
    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(sector),
        float(our_prob),
        float(market_prob),
        float(edge),
        float(confidence),
        str(direction),
        float(brier_score) if brier_score is not None else None,
    )
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)
    logger.info(
        "SIGNAL logged | %s | %s | prob=%.2f%% edge=%+.2f%% conf=%.0f%%",
        ticker, direction,
        float(our_prob) * 100,
        float(edge) * 100,
        float(confidence) * 100,
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
    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(direction),
        int(contracts),
        int(yes_price_cents),
        float(dollars_risked),
        float(avg_prob),
        float(avg_edge),
        float(avg_confidence),
        str(order_id) if order_id is not None else None,
        int(demo_mode),
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
            trade_id = int(cur.fetchone()[0])
        else:
            cur = conn.execute(f"""
                INSERT INTO trades
                (created_at, ticker, direction, contracts, yes_price_cents,
                 dollars_risked, avg_prob, avg_edge, avg_confidence,
                 order_id, demo_mode)
                VALUES ({_ph(11)})
            """, vals)
            trade_id = int(cur.lastrowid)

    logger.info(
        "TRADE logged #%d | %s | %s %dx@%dc | $%.2f | demo=%s",
        trade_id, ticker, direction,
        int(contracts), int(yes_price_cents),
        float(dollars_risked), bool(demo_mode),
    )
    return trade_id


def log_outcome(
    ticker: str,
    resolved: str,
    pnl_usd: Optional[float] = None,
    trade_id: Optional[int] = None,
    our_prob: Optional[float] = None,
    correct: Optional[bool] = None,
) -> None:
    """
    Write one row to the outcomes table.

    Parameters
    ----------
    ticker    : Kalshi market ticker
    resolved  : "YES" or "NO" — what the market actually resolved to
    pnl_usd   : Realised P&L in dollars (None until fill data is available)
    trade_id  : FK to trades table (None if no trade was placed)
    our_prob  : The probability our model assigned before resolution
    correct   : True if our direction call matched the resolution
    """
    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(resolved).upper(),
        float(pnl_usd)   if pnl_usd  is not None else None,
        int(trade_id)    if trade_id is not None else None,
        float(our_prob)  if our_prob is not None else None,
        bool(correct)    if correct  is not None else None,
    )
    sql = f"""
        INSERT INTO outcomes
        (logged_at, ticker, resolved, pnl_usd, trade_id, our_prob, correct)
        VALUES ({_ph(7)})
    """
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)
    logger.info(
        "OUTCOME logged | %s resolved=%s our_prob=%s correct=%s P&L=%s",
        ticker, resolved,
        f"{float(our_prob):.3f}" if our_prob is not None else "n/a",
        correct,
        f"${float(pnl_usd):.2f}" if pnl_usd is not None else "pending",
    )


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_calibration(sector: str, window_days: int = 30) -> dict:
    if USE_POSTGRES:
        date_sql = "s.created_at >= NOW() - INTERVAL '%s days'"
        params   = (str(sector), int(window_days))
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
        params   = (str(sector), f"-{int(window_days)}")
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

    probs   = [float(r["our_prob"]) for r in rows]
    actuals = [1 if r["resolved"] == "YES" else 0 for r in rows]
    edges   = [float(r["edge"]) for r in rows]
    confs   = [float(r["confidence"]) for r in rows]

    brier    = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(rows)
    accuracy = sum(
        1 for p, a in zip(probs, actuals) if (p > 0.5) == bool(a)
    ) / len(rows)

    stats = {
        "sector":         str(sector),
        "n":              int(len(rows)),
        "brier":          round(float(brier), 4),
        "accuracy":       round(float(accuracy), 4),
        "avg_edge":       round(float(sum(edges) / len(edges)), 4),
        "avg_confidence": round(float(sum(confs) / len(confs)), 4),
    }

    ins_vals = (
        datetime.now(timezone.utc).isoformat(),
        str(sector),
        int(window_days),
        int(len(rows)),
        float(brier),
        float(accuracy),
        float(stats["avg_edge"]),
        float(stats["avg_confidence"]),
        float(0),
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
        sector, float(brier), float(accuracy) * 100, int(len(rows)),
    )
    return stats