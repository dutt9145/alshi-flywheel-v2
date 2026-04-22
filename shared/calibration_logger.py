"""
shared/calibration_logger.py  (v11 — clean slate, split signal streams)

Breaking changes from v10:
  - Removed log_signal(). Callers must use log_main_signal() for
    consensus-driven signals or log_correlation_leg() for correlation pairs.
  - Schema no longer includes a combined `signals` table. New tables are
    `main_signals` (consensus/fade/restime) and `correlation_legs` (pair arb).
  - init_db() is a no-op for the core tables now; the migration script
    (migrations/001_clean_slate.sql) owns schema creation. init_db() still
    creates the optional support tables (calibration_stats, arb_opportunities)
    since those may not be present on a fresh deploy.

Why the split:
  The old `signals` table mingled two fundamentally different kinds of rows.
  Consensus-driven signals represent a directional view where our_prob is
  independent of market_prob. Correlation legs represent one side of a pair
  arbitrage where "our_prob" is a derivation of the market price itself
  (either equal to it for the cheap leg, or 1 - other_leg_price for the NO
  side). Putting both in one table corrupted every calibration analysis and
  made it impossible to tell how the main trading path was actually
  performing. The new schema enforces separation at the DB layer via a
  CHECK constraint on main_signals.

Migration path:
  1. Halt orchestrator.
  2. Run migrations/001_clean_slate.sql in Supabase.
  3. Deploy this file and the new orchestrator together.
  4. Verify main_signals and correlation_legs accumulate rows as expected.
  5. Historical data is preserved in *_v1_archive tables for forensics.

Compatibility:
  log_trade() and log_outcome() retain their signatures. log_outcome() now
  requires `source` to be stored on the corresponding trade (which is set
  at trade creation time).
"""

import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH      = os.getenv("DB_PATH", "flywheel.db")
USE_POSTGRES = bool(DATABASE_URL)

# ── Connection pool ──────────────────────────────────────────────────────────

_pg_pool      = None
_pg_pool_lock = threading.Lock()


def _get_pool():
    global _pg_pool
    if _pg_pool is None and USE_POSTGRES:
        with _pg_pool_lock:
            if _pg_pool is None:
                try:
                    import psycopg2.pool
                    _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=5,
                        dsn=DATABASE_URL,
                    )
                    logger.info(
                        "calibration_logger: Postgres pool initialized (max=5)"
                    )
                except Exception as e:
                    logger.error(
                        "calibration_logger: pool init failed: %s", e
                    )
    return _pg_pool


@contextmanager
def _db():
    if USE_POSTGRES:
        pool = _get_pool()
        if pool is None:
            raise RuntimeError("Postgres pool unavailable")
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)
    else:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _ph(n: int) -> str:
    p = "%s" if USE_POSTGRES else "?"
    return ", ".join([p] * n)


# ── Schema init (optional support tables only) ───────────────────────────────

def init_db() -> None:
    """Create support tables if missing. Core schema is owned by migrations.

    In v11, the core tables (main_signals, correlation_legs, trades, outcomes,
    daily_pnl, model_states) are created by migrations/001_clean_slate.sql.
    This function only handles the optional support tables that may not
    have been migrated yet.
    """
    mode = "Supabase PostgreSQL" if USE_POSTGRES else f"SQLite at {DB_PATH}"
    logger.info("calibration_logger v11 — DB mode: %s", mode)

    if USE_POSTGRES:
        serial = "SERIAL PRIMARY KEY"
    else:
        serial = "INTEGER PRIMARY KEY AUTOINCREMENT"

    # Optional support tables only. Core tables come from the migration.
    optional_statements = [
        f"""
        CREATE TABLE IF NOT EXISTS calibration_stats (
            id             {serial},
            computed_at    TIMESTAMPTZ NOT NULL,
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
            detected_at  TIMESTAMPTZ,
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

    # Sanity-check that the core tables exist. If they don't, log a loud
    # warning — the orchestrator shouldn't start until the migration ran.
    sanity_check_sql = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('main_signals', 'correlation_legs', 'trades', 'outcomes')
    """

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            for stmt in optional_statements:
                cur.execute(stmt)

            cur.execute(sanity_check_sql)
            present = {row[0] for row in cur.fetchall()}
            required = {'main_signals', 'correlation_legs', 'trades', 'outcomes'}
            missing = required - present
            if missing:
                logger.error(
                    "MIGRATION NOT APPLIED: missing tables %s. "
                    "Run migrations/001_clean_slate.sql before starting.",
                    sorted(missing),
                )
                raise RuntimeError(
                    f"Required tables missing: {sorted(missing)}. "
                    "See migrations/001_clean_slate.sql"
                )
        else:
            for stmt in optional_statements:
                conn.execute(stmt)

    logger.info("calibration_logger v11 — ready")


# ── Main signal writer ───────────────────────────────────────────────────────

def log_main_signal(
    ticker:       str,
    sector:       str,
    our_prob:     float,
    market_prob:  float,
    edge:         float,
    confidence:   float,
    direction:    str,
    source:       str = "main",      # 'main', 'fade', or 'restime'
    brier_score:  Optional[float] = None,
) -> None:
    """Log a main-path signal (consensus decision).

    Raises ValueError for obviously-bad inputs. Raises on DB constraint
    violation, in particular the main_signals_not_correlation_artifact
    check which rejects our_prob == market_prob or our_prob + market_prob == 1.
    That's the DB's job — the caller is wrong if this fires.
    """
    if direction not in ("YES", "NO"):
        raise ValueError(f"direction must be YES or NO, got {direction!r}")
    if source not in ("main", "fade", "restime"):
        raise ValueError(f"invalid source {source!r}")
    if not (0.0 <= our_prob <= 1.0):
        raise ValueError(f"our_prob out of range: {our_prob}")
    if not (0.0 <= market_prob <= 1.0):
        raise ValueError(f"market_prob out of range: {market_prob}")

    # Short-circuit caller bugs: if it looks like a correlation artifact,
    # refuse rather than let the DB reject it (better error message).
    if abs(our_prob - market_prob) < 0.001:
        raise ValueError(
            f"refusing to log main signal with our_prob == market_prob "
            f"(ticker={ticker}, our_prob={our_prob:.4f}). This looks like "
            f"a correlation-engine leg — route it through log_correlation_leg()."
        )
    if abs((our_prob + market_prob) - 1.0) < 0.001:
        raise ValueError(
            f"refusing to log main signal with our_prob + market_prob == 1 "
            f"(ticker={ticker}, our_prob={our_prob:.4f}, mkt_prob={market_prob:.4f}). "
            f"This looks like a correlation NO leg — route it through log_correlation_leg()."
        )

    sql = f"""
        INSERT INTO main_signals
            (created_at, ticker, sector, source, our_prob, market_prob,
             edge, confidence, direction, brier_score)
        VALUES ({_ph(10)})
    """
    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(sector),
        str(source),
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
        "MAIN SIGNAL | %s | %s | %s | prob=%.2f%% mkt=%.2f%% edge=%+.2f%% conf=%.0f%%",
        ticker[:40], source, direction,
        float(our_prob) * 100,
        float(market_prob) * 100,
        float(edge) * 100,
        float(confidence) * 100,
    )


# ── Correlation leg writer ───────────────────────────────────────────────────

def log_correlation_leg(
    event_group:            str,
    leg_role:               str,          # 'cheap' or 'expensive'
    ticker:                 str,
    direction:              str,
    leg_price_cents:        int,
    pair_divergence_cents:  float,
    sector:                 str,
    confidence:             Optional[float] = None,
) -> int:
    """Log one leg of a correlation-engine pair. Returns row id.

    The returned id lets the caller link this leg to the resulting trade
    (via trades.id → correlation_legs.trade_id) once the order is placed.
    """
    if leg_role not in ("cheap", "expensive"):
        raise ValueError(f"leg_role must be cheap or expensive, got {leg_role!r}")
    if direction not in ("YES", "NO"):
        raise ValueError(f"direction must be YES or NO, got {direction!r}")

    sql = f"""
        INSERT INTO correlation_legs
            (created_at, event_group, leg_role, ticker, direction,
             leg_price_cents, pair_divergence_cents, sector, confidence)
        VALUES ({_ph(9)})
        RETURNING id
    """ if USE_POSTGRES else f"""
        INSERT INTO correlation_legs
            (created_at, event_group, leg_role, ticker, direction,
             leg_price_cents, pair_divergence_cents, sector, confidence)
        VALUES ({_ph(9)})
    """

    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(event_group),
        str(leg_role),
        str(ticker),
        str(direction),
        int(leg_price_cents),
        float(pair_divergence_cents),
        str(sector),
        float(confidence) if confidence is not None else None,
    )

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(sql, vals)
            row_id = int(cur.fetchone()[0])
        else:
            cur = conn.execute(sql, vals)
            row_id = int(cur.lastrowid)

    logger.info(
        "CORR LEG #%d | group=%s | %s | %s %s @ %d¢ | div=%.1f¢",
        row_id, event_group[:30], leg_role.upper(), direction,
        ticker[:30], leg_price_cents, pair_divergence_cents,
    )
    return row_id


def link_correlation_leg_to_trade(leg_id: int, trade_id: int) -> None:
    """Mark a correlation leg as executed by linking it to the trades row."""
    sql = f"UPDATE correlation_legs SET executed = TRUE, trade_id = {_ph(1)} WHERE id = {_ph(1)}"
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, (int(trade_id), int(leg_id)))
        else:
            conn.execute(sql, (int(trade_id), int(leg_id)))


# ── Trade writer ─────────────────────────────────────────────────────────────

def log_trade(
    ticker:           str,
    direction:        str,
    contracts:        int,
    yes_price_cents:  int,
    dollars_risked:   float,
    avg_prob:         float,
    avg_edge:         float,
    avg_confidence:   float,
    order_id:         Optional[str] = None,
    client_order_id:  Optional[str] = None,
    demo_mode:        bool = True,
    sector:           Optional[str] = None,
    source:           str = "main",     # NEW: 'main', 'fade', 'restime', 'correlation'
) -> Optional[int]:
    """Write a trade row. Returns trade_id on success, None if dedupe rejected.

    Handles the live-order_id uniqueness constraint gracefully. If we try
    to log a trade with an order_id that already exists (and isn't a demo
    placeholder), we treat it as a duplicate and return None without raising.
    That lets the caller decide how to handle the collision.
    """
    if direction not in ("YES", "NO"):
        raise ValueError(f"direction must be YES or NO, got {direction!r}")
    if source not in ("main", "fade", "restime", "correlation"):
        raise ValueError(f"invalid source {source!r}")
    if contracts <= 0:
        raise ValueError(f"contracts must be > 0, got {contracts}")
    if not (1 <= yes_price_cents <= 99):
        raise ValueError(f"yes_price_cents out of range: {yes_price_cents}")

    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(sector) if sector is not None else None,
        str(source),
        str(direction),
        int(contracts),
        int(yes_price_cents),
        float(dollars_risked),
        float(avg_prob),
        float(avg_edge),
        float(avg_confidence),
        str(order_id) if order_id is not None else None,
        str(client_order_id) if client_order_id is not None else None,
        int(demo_mode),
    )

    sql_pg = f"""
        INSERT INTO trades
            (created_at, ticker, sector, source, direction, contracts,
             yes_price_cents, dollars_risked, avg_prob, avg_edge,
             avg_confidence, order_id, client_order_id, demo_mode)
        VALUES ({_ph(14)})
        ON CONFLICT DO NOTHING
        RETURNING id
    """
    sql_sqlite = f"""
        INSERT INTO trades
            (created_at, ticker, sector, source, direction, contracts,
             yes_price_cents, dollars_risked, avg_prob, avg_edge,
             avg_confidence, order_id, client_order_id, demo_mode)
        VALUES ({_ph(14)})
    """

    try:
        with _db() as conn:
            if USE_POSTGRES:
                cur = conn.cursor()
                cur.execute(sql_pg, vals)
                row = cur.fetchone()
                if row is None:
                    logger.warning(
                        "TRADE INSERT DEDUPED | %s | order_id=%s already exists",
                        ticker[:40], order_id,
                    )
                    return None
                trade_id = int(row[0])
            else:
                cur = conn.execute(sql_sqlite, vals)
                trade_id = int(cur.lastrowid)
    except Exception as e:
        # On SQLite (no ON CONFLICT DO NOTHING) or unexpected PG errors,
        # surface the exception — caller should handle.
        logger.error("log_trade failed for %s: %s", ticker, e)
        raise

    logger.info(
        "TRADE #%d | %s | %s | %s | %s %dx@%d¢ | $%.2f | demo=%s",
        trade_id, ticker[:40],
        str(sector) if sector else "—",
        source, direction,
        int(contracts), int(yes_price_cents),
        float(dollars_risked), bool(demo_mode),
    )
    return trade_id


# ── Outcome writer ───────────────────────────────────────────────────────────

def log_outcome(
    ticker:    str,
    resolved:  str,
    pnl_usd:   Optional[float] = None,
    trade_id:  Optional[int]   = None,
    our_prob:  Optional[float] = None,
    correct:   Optional[bool]  = None,
    sector:    Optional[str]   = None,
) -> None:
    """Log a resolved outcome. Called by orchestrator after P&L computed."""
    resolved_upper = str(resolved).upper()
    if resolved_upper not in ("YES", "NO"):
        raise ValueError(f"resolved must be YES or NO, got {resolved!r}")

    vals = (
        datetime.now(timezone.utc).isoformat(),
        str(ticker),
        str(sector) if sector is not None else None,
        resolved_upper,
        float(pnl_usd)  if pnl_usd  is not None else None,
        int(trade_id)   if trade_id is not None else None,
        float(our_prob) if our_prob is not None else None,
        bool(correct)   if correct  is not None else None,
    )
    sql = f"""
        INSERT INTO outcomes
            (logged_at, ticker, sector, resolved, pnl_usd,
             trade_id, our_prob, correct)
        VALUES ({_ph(8)})
    """
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)

    logger.info(
        "OUTCOME | %s | sector=%s | %s | our_p=%s correct=%s P&L=%s",
        ticker[:40], sector or "—", resolved_upper,
        f"{float(our_prob):.3f}" if our_prob is not None else "n/a",
        correct,
        f"${float(pnl_usd):+.2f}" if pnl_usd is not None else "pending",
    )


# ── Model state persistence (unchanged from v10) ─────────────────────────────

def save_model_state(sector: str, blob: bytes) -> None:
    import base64
    encoded = base64.b64encode(blob).decode()

    if USE_POSTGRES:
        sql = """
            INSERT INTO model_states (sector, updated_at, model_blob)
            VALUES (%s, %s, %s)
            ON CONFLICT (sector) DO UPDATE
            SET model_blob = EXCLUDED.model_blob,
                updated_at = EXCLUDED.updated_at
        """
    else:
        sql = """
            INSERT OR REPLACE INTO model_states (sector, updated_at, model_blob)
            VALUES (?, ?, ?)
        """

    vals = (sector, datetime.now(timezone.utc).isoformat(), encoded)
    with _db() as conn:
        if USE_POSTGRES:
            conn.cursor().execute(sql, vals)
        else:
            conn.execute(sql, vals)
    logger.info("[%s] Model state saved to DB", sector)


def load_model_state(sector: str) -> Optional[bytes]:
    import base64

    sql = (
        "SELECT model_blob FROM model_states WHERE sector = %s"
        if USE_POSTGRES else
        "SELECT model_blob FROM model_states WHERE sector = ?"
    )

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(sql, (sector,))
            row = cur.fetchone()
        else:
            row = conn.execute(sql, (sector,)).fetchone()

    if not row:
        logger.info("[%s] No model state found in DB (fresh start)", sector)
        return None

    encoded = row[0] if USE_POSTGRES else row["model_blob"]
    return base64.b64decode(encoded)


# ── Calibration (operates on main_signals only) ──────────────────────────────

def compute_calibration(sector: str, window_days: int = 30) -> dict:
    """Compute Brier/accuracy from main_signals for a sector.

    v11: queries main_signals, not signals. Correlation legs are excluded
    from calibration metrics by definition — they don't represent model views.
    """
    if USE_POSTGRES:
        sql = """
            SELECT ms.our_prob, o.resolved, ms.edge, ms.confidence
            FROM main_signals ms
            LEFT JOIN outcomes o ON ms.ticker = o.ticker
            WHERE ms.sector = %s
              AND ms.created_at >= NOW() - INTERVAL '%s days'
              AND o.resolved IS NOT NULL
        """
        params = (str(sector), int(window_days))
    else:
        sql = """
            SELECT ms.our_prob, o.resolved, ms.edge, ms.confidence
            FROM main_signals ms
            LEFT JOIN outcomes o ON ms.ticker = o.ticker
            WHERE ms.sector = ?
              AND ms.created_at >= datetime('now', ? || ' days')
              AND o.resolved IS NOT NULL
        """
        params = (str(sector), f"-{int(window_days)}")

    with _db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        else:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not rows:
        logger.info(
            "[%s] No resolved main_signals in last %d days",
            sector, window_days,
        )
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