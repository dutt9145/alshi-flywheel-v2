"""
shared/circuit_breaker.py

Daily P&L tracking, circuit breaker halt logic, and real-time bankroll sync.

The circuit breaker does three things:
  1. Tracks realized P&L per calendar day in the daily_pnl table
  2. Halts all new positions if daily loss exceeds CIRCUIT_BREAKER_PCT of bankroll
  3. Provides real-time bankroll sync before every trade (not just at 2am retrain)

Usage:
    from shared.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(kalshi_client)

    # Before every trade:
    if cb.is_halted():
        return  # skip

    # After every resolved outcome:
    cb.record_pnl(pnl_usd)

    # Sync bankroll before sizing:
    bankroll = cb.sync_bankroll()
"""

import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── DB helpers (mirrors orchestrator._query_signals pattern) ──────────────────

def _db_execute(sql: str, params: tuple = (), fetch: bool = False):
    database_url = os.getenv("DATABASE_URL", "")
    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute(sql, params)
            result = None
            if fetch and cur.description:
                cols   = [d[0] for d in cur.description]
                result = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return result or []
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
        logger.error("CircuitBreaker DB error: %s", e)
        return []


def _ph() -> str:
    """Return correct SQL placeholder for the active DB."""
    return "%s" if os.getenv("DATABASE_URL") else "?"


def init_circuit_breaker_tables() -> None:
    """Create daily_pnl table if it doesn't exist. Called once on startup."""
    _db_execute("""
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date           DATE PRIMARY KEY,
            realized_pnl   FLOAT8 DEFAULT 0.0,
            trade_count    INT    DEFAULT 0,
            halted         BOOLEAN DEFAULT FALSE,
            halt_reason    TEXT
        )
    """)
    logger.info("CircuitBreaker tables ready")


# ── Main class ────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Stateful guard that tracks daily P&L and halts trading when losses
    exceed the configured threshold.

    Parameters
    ----------
    kalshi_client : KalshiClient
        Used for real-time balance sync via get_balance().
    bankroll : float
        Starting bankroll. Updated by sync_bankroll().
    daily_loss_limit_pct : float
        Fraction of bankroll that triggers a halt. Default: 0.05 (5%).
    """

    def __init__(
        self,
        kalshi_client,
        bankroll: float,
        daily_loss_limit_pct: float = 0.05,
    ):
        self.client               = kalshi_client
        self.bankroll             = bankroll
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._halted              = False
        self._halt_reason: Optional[str] = None

        init_circuit_breaker_tables()
        self._load_today_state()

    # ── State init ─────────────────────────────────────────────────────────────

    def _today(self) -> str:
        return date.today().isoformat()

    def _load_today_state(self) -> None:
        """
        On startup, restore halt state from DB in case this is a restart
        mid-day after a halt was triggered.
        """
        p   = _ph()
        rows = _db_execute(
            f"SELECT halted, halt_reason, realized_pnl FROM daily_pnl WHERE date = {p}",
            (self._today(),),
            fetch=True,
        )
        if rows:
            self._halted      = bool(rows[0]["halted"])
            self._halt_reason = rows[0].get("halt_reason")
            pnl               = rows[0].get("realized_pnl", 0.0)
            logger.info(
                "CircuitBreaker restored: date=%s pnl=$%.2f halted=%s",
                self._today(), pnl, self._halted,
            )
        else:
            # First trade of the day — seed the row
            p2 = _ph()
            _db_execute(
                f"INSERT INTO daily_pnl (date, realized_pnl, trade_count, halted) "
                f"VALUES ({p2}, 0.0, 0, FALSE) ON CONFLICT (date) DO NOTHING",
                (self._today(),),
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_halted(self) -> bool:
        """
        Returns True if trading should be paused today.
        Checks DB each call so the halt flag survives process restarts.
        """
        if self._halted:
            return True

        p    = _ph()
        rows = _db_execute(
            f"SELECT halted, halt_reason FROM daily_pnl WHERE date = {p}",
            (self._today(),),
            fetch=True,
        )
        if rows and rows[0]["halted"]:
            self._halted      = True
            self._halt_reason = rows[0].get("halt_reason")
            logger.warning(
                "CircuitBreaker: halt active — %s", self._halt_reason
            )
            return True
        return False

    def record_pnl(self, pnl_usd: float) -> None:
        """
        Add realized P&L from a settled trade.
        Triggers halt check after each update.

        Call this from _ingest_resolved_markets() for every matched trade.
        """
        if pnl_usd is None:
            return

        p = _ph()
        _db_execute(
            f"""
            INSERT INTO daily_pnl (date, realized_pnl, trade_count)
            VALUES ({p}, {p}, 1)
            ON CONFLICT (date) DO UPDATE
              SET realized_pnl = daily_pnl.realized_pnl + EXCLUDED.realized_pnl,
                  trade_count  = daily_pnl.trade_count  + 1
            """,
            (self._today(), pnl_usd),
        )
        self._check_halt_threshold()

    def _check_halt_threshold(self) -> None:
        """Read today's P&L and halt if loss limit is breached."""
        p    = _ph()
        rows = _db_execute(
            f"SELECT realized_pnl FROM daily_pnl WHERE date = {p}",
            (self._today(),),
            fetch=True,
        )
        if not rows:
            return

        daily_pnl   = float(rows[0]["realized_pnl"])
        loss_limit  = -(self.bankroll * self.daily_loss_limit_pct)

        if daily_pnl <= loss_limit:
            reason = (
                f"Daily loss ${abs(daily_pnl):.2f} exceeded "
                f"{self.daily_loss_limit_pct*100:.1f}% limit "
                f"(${abs(loss_limit):.2f})"
            )
            self._trigger_halt(reason)

    def _trigger_halt(self, reason: str) -> None:
        """Set halt flag in DB and in memory."""
        self._halted      = True
        self._halt_reason = reason
        p = _ph()
        _db_execute(
            f"UPDATE daily_pnl SET halted = TRUE, halt_reason = {p} WHERE date = {p}",
            (reason, self._today()),
        )
        logger.critical("🛑 CIRCUIT BREAKER TRIPPED: %s", reason)

    def sync_bankroll(self) -> float:
        """
        Fetch real-time balance from Kalshi and update self.bankroll.
        Returns the updated bankroll. Falls back to stored value on error.

        Call this before every trade sizing — not just at 2am retrain.
        """
        try:
            balance       = self.client.get_balance()
            self.bankroll = balance
            logger.debug("Bankroll synced: $%.2f", balance)
            return balance
        except Exception as e:
            logger.warning("Bankroll sync failed, using last known: $%.2f — %s",
                           self.bankroll, e)
            return self.bankroll

    def daily_summary(self) -> dict:
        """Return today's P&L summary for logging/dashboard."""
        p    = _ph()
        rows = _db_execute(
            f"SELECT * FROM daily_pnl WHERE date = {p}",
            (self._today(),),
            fetch=True,
        )
        return rows[0] if rows else {"date": self._today(), "realized_pnl": 0.0}

    def reset_halt(self) -> None:
        """
        Manually clear the halt flag. Use only after reviewing the loss
        cause. Does NOT reset daily P&L — that resets at midnight.
        """
        self._halted      = False
        self._halt_reason = None
        p = _ph()
        _db_execute(
            f"UPDATE daily_pnl SET halted = FALSE, halt_reason = NULL WHERE date = {p}",
            (self._today(),),
        )
        logger.warning("CircuitBreaker halt manually reset for %s", self._today())