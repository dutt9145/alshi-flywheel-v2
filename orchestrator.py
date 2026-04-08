"""
orchestrator.py  (v19.7 — provisional loss unwind fix)

Changes vs v19.6:
  1. Provisional loss double-count bug fixed.
     Previously, _ingest_resolved_markets() did:
       self._sector_daily_pnl[sec] += pnl_usd
     on top of an already-booked provisional loss. On a losing trade this
     meant: -$300 (provisional at execution) + -$300 (actual at resolution)
     = -$600 sector_pnl instead of the correct -$300.

     Fix: unwind the provisional first, then apply actual P&L:
       provisional = self._open_provisionals.pop(ticker, 0.0)
       self._sector_daily_pnl[sec] = sector_pnl - provisional + pnl_usd

     On a win:  -$300 provisional + $200 actual  → correct net -$100
     On a loss: -$300 provisional + -$300 actual → correct net -$300
                (was -$600 before this fix)

  2. _open_provisionals cleared in retrain_models() nightly reset.

  3. Carries forward all v19.6 changes: NO P&L formula fix, sports ticker
     corrections, _sector_pnl_lock, provisional loss booking.
"""

import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone

import schedule

from config.settings import (
    BANKROLL, DEMO_MODE, RETRAIN_HOUR, SCAN_INTERVAL_SEC,
    CIRCUIT_BREAKER_PCT,
    SHARP_SPREAD_THRESHOLD_PCT, SHARP_LARGE_ORDER_CONTRACTS,
    SHARP_VOLUME_SPIKE_MULTIPLIER,
    FADE_WINDOW_MINUTES, FADE_THRESHOLD_CENTS, FADE_MIN_DISAGREEMENT,
    CORR_MIN_DIVERGENCE_CENTS, CORR_MAX_GROUP_SIZE,
    RESTIME_MIN_OVERDUE_MIN, RESTIME_MIN_PROB, RESTIME_MIN_SAMPLES,
    NEWS_POLL_INTERVAL_SEC, NEWS_VELOCITY_WINDOW_MIN, NEWS_VELOCITY_SPIKE_THRESHOLD,
    NEWSAPI_KEY,
    SECTOR_MAX_DAILY_LOSS, SECTOR_MIN_RESOLVED,
    KELLY_FRACTION,
    sector_kelly_fraction, sector_loss_cap,
)
from shared.arb_layer import ArbLayer
from shared.calibration_logger import init_db, log_signal, log_trade, log_outcome
from shared.circuit_breaker import CircuitBreaker
from shared.consensus_engine import ConsensusEngine
from shared.correlation_engine import CorrelationEngine
from shared.fade_scanner import FadeScanner
from shared.kalshi_client import KalshiClient
from shared.kelly_sizer import kelly_stake, no_kelly_stake
from shared.news_signal import NewsSignal
from shared.resolution_timer import ResolutionTimer
from shared.sharp_detector import SharpDetector
from bots.sector_bots import all_bots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

# ── Duplicate trade cooldown ───────────────────────────────────────────────────
RECENT_TRADE_WINDOW_SEC = 300  # 5 minutes

# ── Ticker normalization ───────────────────────────────────────────────────────
_UUID_SUFFIX_RE = re.compile(r'-S[0-9A-Fa-f]{4,}-[0-9A-Fa-f]+$')


def _normalize_ticker(ticker: str) -> str:
    """Strip Kalshi UUID suffix from multivariate market tickers."""
    return _UUID_SUFFIX_RE.sub('', ticker)


# ── Sports prefix guard ────────────────────────────────────────────────────────
_SPORTS_PREFIXES = (
    "kxmve", "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
    "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
    "kxmvecbchampionship",
    "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
    "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
    "kxtennis", "kxf1", "kxolympic", "kxthail", "kxsl",
    "kxdota", "kxintlf", "kxcs2", "kxegypt", "kxvenf",
    "kxepl", "kxsoccer", "kxboxing", "kxwwe", "kxcricket",
    "kxrugby", "kxesport",
    "kxeuroleague", "kxcba", "kxcbagame",
)


def _has_sports_prefix(market: dict) -> bool:
    et      = market.get("event_ticker", "").lower()
    tk      = market.get("ticker", "").lower()
    tk_norm = _UUID_SUFFIX_RE.sub('', tk)
    return any(
        et.startswith(p) or tk.startswith(p) or tk_norm.startswith(p)
        for p in _SPORTS_PREFIXES
    )


# ── Structural market blocklist ────────────────────────────────────────────────
_STRUCTURAL_MARKET_BLOCKLIST = (
    "kxmvecrosscategory",
    "kxmvecrosscat",
    "kxmvecross",
    "kxmvesportsmultigameextended",
    "kxmvesportsmultigame",
    "kxmvesportsmulti",
)


def _is_blocked_structural_market(ticker: str) -> bool:
    """Return True if ticker matches an uncalibrated structural market type."""
    tk = _UUID_SUFFIX_RE.sub('', ticker.lower())
    return any(tk.startswith(p) for p in _STRUCTURAL_MARKET_BLOCKLIST)


# ── Scan page rate-limit guard ─────────────────────────────────────────────────
SCAN_PAGE_SLEEP_SEC = 0.25


# ── Sector inference from ticker ───────────────────────────────────────────────
_TICKER_SECTOR_MAP: list[tuple[tuple[str, ...], str]] = [
    (("kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
      "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
      "kxtennis", "kxf1", "kxolympic", "kxepl", "kxsoccer",
      "kxboxing", "kxwwe", "kxcricket", "kxrugby", "kxesport",
      "kxdota", "kxintlf", "kxcs2",
      "kxeuroleague", "kxcba", "kxcbagame"), "sports"),
    (("kxbtc", "kxeth", "kxsol", "kxcrypto", "kxdefi"), "crypto"),
    (("kxelect", "kxpres", "kxsen", "kxhouse", "kxgov",
      "kxpol", "kxvote", "kxapprove"), "politics"),
    (("kxhurr", "kxtemp", "kxrain", "kxsnow", "kxweather",
      "kxnoaa", "kxclimate"), "weather"),
    (("kxai", "kxtech", "kxfed", "kxcpi", "kxgdp",
      "kxjobs", "kxrate", "kxinfl"), "economics"),
]


def _infer_sector_from_ticker(ticker: str) -> str:
    tk = ticker.lower()
    for prefixes, sector in _TICKER_SECTOR_MAP:
        if any(tk.startswith(p) for p in prefixes):
            return sector
    return "economics"


# ── Connection pool ────────────────────────────────────────────────────────────
_pg_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pg_pool
    if _pg_pool is None and os.getenv("DATABASE_URL"):
        with _pool_lock:
            if _pg_pool is None:
                try:
                    from psycopg2 import pool as pg_pool
                    _pg_pool = pg_pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=8,
                        dsn=os.getenv("DATABASE_URL"),
                    )
                    logger.info("Postgres connection pool initialized (max=8)")
                except Exception as e:
                    logger.error("Connection pool init failed: %s", e)
    return _pg_pool


def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _query_signals(sql: str, params: tuple = ()) -> list:
    pool = _get_pool()
    try:
        if pool:
            conn = pool.getconn()
            try:
                cur  = conn.cursor()
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                conn.commit()
                return rows
            finally:
                pool.putconn(conn)
        else:
            import sqlite3
            db_path = os.getenv("DB_PATH", "flywheel.db")
            conn    = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows    = [dict(r) for r in conn.execute(sql, params).fetchall()]
            conn.close()
            return rows
    except Exception as e:
        logger.error("_query_signals error: %s", e)
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_yes_price_cents(market: dict) -> int:
    try:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
        if bid > 0 and ask > 0:
            return int(round((bid + ask) / 2 * 100))
    except (TypeError, ValueError):
        pass
    try:
        last = float(market.get("last_price_dollars") or 0)
        if last > 0:
            return int(round(last * 100))
    except (TypeError, ValueError):
        pass
    return 0


def _calculate_pnl(ticker: str, resolved_yes: bool) -> tuple[float | None, list[int]]:
    p = _ph()
    trade_rows = _query_signals(
        f"SELECT id, direction, contracts, yes_price_cents "
        f"FROM trades "
        f"WHERE ticker = {p} "
        f"ORDER BY created_at DESC "
        f"LIMIT 50",
        (ticker,),
    )

    if not trade_rows:
        return None, []

    total_pnl: float = 0.0
    trade_ids: list[int] = []

    for trade in trade_rows:
        raw_id    = trade.get("id")
        direction = str(trade.get("direction", "YES")).upper()
        contracts = int(trade.get("contracts", 0))
        yes_price = int(trade.get("yes_price_cents", 50))

        if contracts <= 0:
            continue

        try:
            trade_ids.append(int(raw_id))
        except (TypeError, ValueError):
            logger.warning("Skipping non-integer trade id: %r", raw_id)
            continue

        yes_price_frac = yes_price / 100.0
        no_price_frac  = 1.0 - yes_price_frac

        if direction == "YES":
            pnl = contracts * ((1.0 - yes_price_frac) if resolved_yes else (-yes_price_frac))
        else:
            # NO contract costs no_price_frac, pays $1 if NO resolves.
            # Win: +yes_price_frac per contract (the gain on a NO contract)
            # Loss: -no_price_frac per contract (what was paid for NO)
            # Bug in v19.5 and earlier used -yes_price_frac for the loss,
            # massively inflating losses on high yes_price markets.
            pnl = contracts * (no_price_frac if not resolved_yes else (-no_price_frac))

        total_pnl += pnl

    if not trade_ids:
        return None, []

    return round(total_pnl, 2), trade_ids


def _update_signal_brier(ticker: str, resolved_yes: bool) -> None:
    outcome = 1 if resolved_yes else 0
    now     = datetime.now(timezone.utc)
    p       = _ph()
    pool    = _get_pool()

    try:
        if pool:
            conn = pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    UPDATE signals
                    SET outcome     = {p},
                        outcome_at  = {p},
                        brier_score = POWER(our_prob - {p}, 2)
                    WHERE ticker    = {p}
                      AND outcome IS NULL
                    """,
                    (outcome, now, outcome, ticker),
                )
                conn.commit()
            finally:
                pool.putconn(conn)
        else:
            import sqlite3
            db_path = os.getenv("DB_PATH", "flywheel.db")
            conn    = sqlite3.connect(db_path)
            conn.execute(
                f"""
                UPDATE signals
                SET outcome     = {p},
                    outcome_at  = {p},
                    brier_score = (our_prob - {p}) * (our_prob - {p})
                WHERE ticker    = {p}
                  AND outcome IS NULL
                """,
                (outcome, now, outcome, ticker),
            )
            conn.commit()
            conn.close()
        logger.debug("Signal brier updated: %s → outcome=%d", ticker, outcome)
    except Exception as e:
        logger.warning("_update_signal_brier failed for %s: %s", ticker, e)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class FlywheelOrchestrator:

    def __init__(self):
        self.client  = KalshiClient()
        self.bots    = all_bots()
        self.engine  = ConsensusEngine()
        self.arb     = ArbLayer()

        self.circuit_breaker = CircuitBreaker(
            kalshi_client        = self.client,
            bankroll             = BANKROLL,
            daily_loss_limit_pct = CIRCUIT_BREAKER_PCT,
        )
        self.bankroll = self.circuit_breaker.sync_bankroll()

        self.sharp = SharpDetector(
            kalshi_client            = self.client,
            spread_threshold_pct     = SHARP_SPREAD_THRESHOLD_PCT,
            large_order_contracts    = SHARP_LARGE_ORDER_CONTRACTS,
            volume_spike_multiplier  = SHARP_VOLUME_SPIKE_MULTIPLIER,
        )

        self.fade_scanner = FadeScanner(
            kalshi_client          = self.client,
            sharp_detector         = self.sharp,
            fade_window_minutes    = FADE_WINDOW_MINUTES,
            fade_threshold_cents   = FADE_THRESHOLD_CENTS,
            fade_min_disagreement  = FADE_MIN_DISAGREEMENT,
        )

        self.corr_engine = CorrelationEngine(
            min_divergence_cents = CORR_MIN_DIVERGENCE_CENTS,
            max_group_size       = CORR_MAX_GROUP_SIZE,
        )

        self.res_timer = ResolutionTimer(
            min_overdue_minutes = RESTIME_MIN_OVERDUE_MIN,
            min_prob_to_trade   = RESTIME_MIN_PROB,
            min_sample_count    = RESTIME_MIN_SAMPLES,
        )

        self.news = NewsSignal(
            api_key                  = NEWSAPI_KEY,
            poll_interval_sec        = NEWS_POLL_INTERVAL_SEC,
            velocity_window_min      = NEWS_VELOCITY_WINDOW_MIN,
            velocity_spike_threshold = NEWS_VELOCITY_SPIKE_THRESHOLD,
        )

        self._exposure: dict[str, float]         = {b.sector_name: 0.0 for b in self.bots}
        self._sector_daily_pnl: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}

        # ── FIX: Lock serializing ALL _sector_daily_pnl reads/writes ──────────
        # Eliminates race between _run_ingestion_thread() writing mid-scan and
        # the cap check reading a stale pre-ingestion snapshot, which caused the
        # $149.52 gap between daily_pnl and sector_pnl in logs.
        self._sector_pnl_lock = threading.Lock()

        self._ingestion_lock                     = threading.Lock()
        self._last_bot_probs:   dict[str, float] = {}
        self._last_bot_sectors: dict[str, str]   = {}

        self._resolved_count_cache:    dict[str, int] = {}
        self._resolved_count_cache_ts: float          = 0.0

        self._scan_bankroll: float = self.bankroll
        self._scan_summary:  dict  = {}

        self._recently_traded: dict[str, float] = {}

        # ── Provisional loss tracker ───────────────────────────────────────
        # Keyed by ticker. Stores the provisional loss booked at execution
        # time so _ingest_resolved_markets() can unwind it precisely before
        # applying the actual resolved P&L. Prevents the double-count bug
        # where a losing trade would register -$300 at execution AND -$300
        # again at resolution, producing -$600 sector_pnl.
        self._open_provisionals: dict[str, float] = {}

    # ── Sector resolved count (cached) ────────────────────────────────────────

    def _get_sector_resolved_count(self, sector: str) -> int:
        now = time.monotonic()
        if now - self._resolved_count_cache_ts > 60.0:
            self._resolved_count_cache.clear()
            self._resolved_count_cache_ts = now

        if sector not in self._resolved_count_cache:
            p = _ph()
            rows = _query_signals(
                f"SELECT COUNT(*) as n FROM signals "
                f"WHERE sector = {p} AND outcome IS NOT NULL",
                (sector,),
            )
            self._resolved_count_cache[sector] = int(rows[0]["n"]) if rows else 0

        return self._resolved_count_cache[sector]

    # ── Resolution ingestion ───────────────────────────────────────────────────

    def _ingest_resolved_markets(self) -> int:
        logger.info("=== Resolution ingestion starting ===")

        try:
            min_ts = self.client.get_latest_outcome_ts()
            if min_ts:
                logger.info("Incremental ingestion — fetching markets after %s", min_ts)
            resolved_markets = self.client.get_resolved_markets(min_close_ts=min_ts)
        except Exception as e:
            logger.error("Failed to fetch resolved markets: %s", e)
            return 0

        if not resolved_markets:
            logger.info("No resolved markets returned from Kalshi.")
            return 0

        logger.info("%d settled markets returned from Kalshi", len(resolved_markets))

        p = _ph()
        if min_ts:
            already_recorded = {
                _normalize_ticker(r["ticker"])
                for r in _query_signals(
                    f"SELECT DISTINCT ticker FROM outcomes WHERE logged_at > {p}",
                    (min_ts,),
                )
            }
        else:
            already_recorded = {
                _normalize_ticker(r["ticker"])
                for r in _query_signals("SELECT DISTINCT ticker FROM outcomes")
            }

        processed_this_run: set[str] = set()
        recorded = 0

        for market in resolved_markets:
            if self.circuit_breaker.is_halted():
                logger.warning(
                    "Circuit breaker active — stopping ingestion early to prevent "
                    "further phantom P&L accumulation"
                )
                break

            ticker = _normalize_ticker(market.get("ticker", ""))
            result = market.get("result", "")

            if not ticker or result not in ("yes", "no"):
                continue
            if ticker in already_recorded or ticker in processed_this_run:
                continue

            resolved_yes = result == "yes"

            sig_rows = _query_signals(
                f"SELECT our_prob, direction FROM signals "
                f"WHERE ticker = {p} ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            )

            for bot in self.bots:
                if not bot.is_relevant(market):
                    continue
                try:
                    features, _ = bot.fetch_features(market, skip_noaa=True)
                    import numpy as np
                    features = np.append(features, [0.0, 0.0])
                    bot.record_outcome(features, resolved_yes)
                    logger.info(
                        "[%s] Bayesian update: %s → %s",
                        bot.sector_name, ticker, result.upper(),
                    )
                except Exception as e:
                    logger.warning(
                        "[%s] record_outcome failed for %s: %s",
                        bot.sector_name, ticker, e,
                    )

            if not sig_rows:
                processed_this_run.add(ticker)
                continue

            our_prob  = float(sig_rows[0]["our_prob"])
            direction = str(sig_rows[0]["direction"])

            direction_correct = (
                (direction == "YES" and resolved_yes) or
                (direction == "NO"  and not resolved_yes)
            )

            pnl_usd, trade_ids = _calculate_pnl(ticker, resolved_yes)
            primary_trade_id   = trade_ids[0] if trade_ids else None

            if pnl_usd is not None:
                logger.info(
                    "[P&L] %s resolved %s → $%+.2f (primary_trade_id=%s, all_ids=%s)",
                    ticker, result.upper(), pnl_usd, primary_trade_id, trade_ids,
                )
                self.circuit_breaker.record_pnl(pnl_usd)

                sec_rows = _query_signals(
                    f"SELECT sector FROM signals WHERE ticker = {p} LIMIT 1",
                    (ticker,),
                )
                if sec_rows:
                    sec = sec_rows[0].get("sector", "")
                    with self._sector_pnl_lock:
                        if sec in self._sector_daily_pnl:
                            # ── FIX: Unwind provisional before applying actual P&L ──
                            # v19.6 and earlier did += pnl_usd on top of the
                            # provisional, double-counting losses:
                            #   -$300 provisional + -$300 actual = -$600 (wrong)
                            # Now we pop the provisional and replace it cleanly:
                            #   -$300 provisional unwound, +actual applied = correct
                            provisional = self._open_provisionals.pop(ticker, 0.0)
                            self._sector_daily_pnl[sec] = (
                                self._sector_daily_pnl[sec] - provisional + pnl_usd
                            )
                            logger.debug(
                                "[PROVISIONAL UNWIND] %s sector=%s "
                                "provisional=$%.2f actual=$%.2f new_sector_pnl=$%.2f",
                                ticker, sec, provisional, pnl_usd,
                                self._sector_daily_pnl[sec],
                            )
            else:
                logger.debug(
                    "[P&L] %s resolved %s — no matching trade",
                    ticker, result.upper(),
                )

            processed_this_run.add(ticker)

            try:
                log_outcome(
                    ticker   = ticker,
                    resolved = result.upper(),
                    pnl_usd  = pnl_usd,
                    trade_id = primary_trade_id,
                    our_prob = our_prob,
                    correct  = direction_correct,
                )
                recorded += 1
            except Exception as e:
                logger.error(
                    "log_outcome failed for %s (trade_id=%s): %s — "
                    "outcome counted but not persisted; will retry next ingestion cycle",
                    ticker, primary_trade_id, e,
                )

            _update_signal_brier(ticker, resolved_yes)

        logger.info("=== Resolution ingestion complete — %d new outcomes ===", recorded)
        return recorded

    def _run_ingestion_thread(self) -> None:
        if not self._ingestion_lock.acquire(blocking=False):
            logger.warning("Ingestion already running — skipping this cycle")
            return

        def _run():
            try:
                self._ingest_resolved_markets()
            finally:
                self._ingestion_lock.release()

        t = threading.Thread(target=_run, daemon=True, name="resolution-ingestion")
        t.start()
        logger.info("Resolution ingestion thread started")

    # ── Per-market pipeline ────────────────────────────────────────────────────

    def _evaluate_market(self, market: dict) -> None:
        ticker = _normalize_ticker(
            market.get("ticker") or market.get("event_ticker") or "UNKNOWN"
        )
        title     = market.get("title", "")
        yes_price = _parse_yes_price_cents(market)

        if yes_price == 0:
            return
        if yes_price <= 2 or yes_price >= 98:
            return

        self.news.register_market(title)
        sharp_signal = self.sharp.analyze(market)

        signals = []
        for bot in self.bots:
            try:
                sig = bot.evaluate(market, news_signal=self.news)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.error("[%s] error on %s: %s", bot.sector_name, ticker, e)

        if not signals:
            return

        consensus = self.engine.evaluate(ticker, yes_price, signals)

        if not consensus.execute:
            logger.debug("Consensus reject %s: %s", ticker, consensus.reject_reason)
            return

        if sharp_signal.sharp_detected and not sharp_signal.aligned_with(consensus.direction):
            logger.info(
                "SHARP VETO %s | sharp_dir=%s our_dir=%s sharp_conf=%.2f",
                ticker, sharp_signal.sharp_direction, consensus.direction,
                sharp_signal.confidence,
            )
            return

        lead_sector = signals[0].sector

        self._last_bot_probs[ticker]   = consensus.avg_prob
        self._last_bot_sectors[ticker] = lead_sector

        # ── Arb check + log in single call (v19.4) ────────────────────────
        arb = self.arb.check_and_log(
            ticker        = ticker,
            kalshi_cents  = yes_price,
            our_direction = consensus.direction,
            market_title  = title,
        )

        if not arb.passes:
            logger.info(
                "ARB GATE VETO %s | dir_aligned=%s spread=%.2f%% notes=%s",
                ticker, arb.direction_aligned, arb.cross_spread * 100, arb.notes,
            )
            return

        try:
            log_signal(
                ticker      = ticker,
                sector      = lead_sector,
                our_prob    = consensus.avg_prob,
                market_prob = yes_price / 100,
                edge        = consensus.avg_edge,
                confidence  = consensus.avg_confidence,
                direction   = consensus.direction,
            )
        except Exception as e:
            logger.error("log_signal failed for %s: %s", ticker, e)

        self._execute_trade(
            ticker      = ticker,
            title       = title,
            yes_price   = yes_price,
            consensus   = consensus,
            lead_sector = lead_sector,
            arb         = arb,
        )

    def _execute_trade(
        self,
        ticker:      str,
        title:       str,
        yes_price:   int,
        consensus,
        lead_sector: str,
        arb,
        fade:        bool = False,
    ) -> None:
        # ── Duplicate trade guard ──────────────────────────────────────────
        now      = time.monotonic()
        last_ts  = self._recently_traded.get(ticker, 0.0)
        elapsed  = now - last_ts
        if elapsed < RECENT_TRADE_WINDOW_SEC:
            logger.info(
                "DUPLICATE GUARD skip %s — last trade %.0fs ago (cooldown=%ds)",
                ticker, elapsed, RECENT_TRADE_WINDOW_SEC,
            )
            return

        # ── Circuit breaker ────────────────────────────────────────────────
        if self.circuit_breaker.is_halted():
            logger.warning("CIRCUIT BREAKER HALT — skipping trade for %s", ticker)
            return

        # ── Per-sector loss cap ────────────────────────────────────────────
        cap = sector_loss_cap(lead_sector)
        if cap == 0.0:
            logger.warning(
                "SECTOR DISABLED: %s cap=0 — skipping %s", lead_sector, ticker
            )
            return

        # ── FIX: Read _sector_daily_pnl under lock to get a consistent value
        # even if ingestion thread is writing concurrently.
        with self._sector_pnl_lock:
            sector_loss_today = self._sector_daily_pnl.get(lead_sector, 0.0)

        if sector_loss_today <= -cap:
            logger.warning(
                "SECTOR LOSS CAP hit: %s daily_loss=$%.2f cap=$%.2f — skipping %s",
                lead_sector, sector_loss_today, cap, ticker,
            )
            return

        # ── Scan-cycle cached bankroll + summary ───────────────────────────
        live_bankroll   = self._scan_bankroll
        summary         = self._scan_summary
        daily_pnl       = float(summary.get("realized_pnl", 0.0))
        loss_limit      = live_bankroll * CIRCUIT_BREAKER_PCT
        drawdown_factor = (
            max(0.5, 1.0 + (daily_pnl / loss_limit) * 0.5)
            if daily_pnl < 0 and loss_limit > 0 else 1.0
        )

        # ── Exploration Kelly ──────────────────────────────────────────────
        resolved_count = self._get_sector_resolved_count(lead_sector)
        kf             = sector_kelly_fraction(lead_sector, resolved_count)
        in_exploration = resolved_count < SECTOR_MIN_RESOLVED.get(lead_sector, 30)

        if in_exploration:
            logger.info(
                "[EXPLORATION] %s resolved=%d — will scale stake to %.0f%% of Kelly",
                lead_sector, resolved_count, (kf / KELLY_FRACTION) * 100,
            )

        sector_exp = self._exposure.get(lead_sector, 0.0)

        # ── Kelly sizing ───────────────────────────────────────────────────
        if consensus.direction == "YES":
            sizing = kelly_stake(
                prob            = consensus.avg_prob,
                yes_price_cents = yes_price,
                bankroll        = self.bankroll,
                sector          = lead_sector,
                sector_exposure = sector_exp,
                live_bankroll   = live_bankroll,
                drawdown_factor = drawdown_factor,
            )
            side = "yes"
        else:
            sizing = no_kelly_stake(
                prob            = consensus.avg_prob,
                yes_price_cents = yes_price,
                bankroll        = self.bankroll,
                sector          = lead_sector,
                sector_exposure = sector_exp,
                live_bankroll   = live_bankroll,
                drawdown_factor = drawdown_factor,
            )
            side = "no"

        # ── Apply exploration scaling post-sizing ──────────────────────────
        if in_exploration and sizing["contracts"] > 0:
            exploration_scale = kf / KELLY_FRACTION
            sizing["dollars"]   = round(sizing["dollars"] * exploration_scale, 2)
            sizing["contracts"] = max(1, int(sizing["dollars"] / (yes_price / 100)))
            logger.info(
                "[EXPLORATION] %s scaled stake: $%.2f × %.2f → $%.2f (%d contracts)",
                lead_sector, sizing["dollars"] / exploration_scale,
                exploration_scale, sizing["dollars"], sizing["contracts"],
            )

        if sizing["contracts"] <= 0:
            logger.info(
                "Sizing zero for %s — rationale: %s",
                ticker, sizing.get("rationale", "unknown"),
            )
            return

        client_order_id = str(uuid.uuid4())

        if DEMO_MODE:
            logger.info(
                "DEMO TRADE %s | %s | %s %dx@%dc | $%.2f%s%s",
                ticker, title[:40], consensus.direction.upper(),
                sizing["contracts"], yes_price, sizing["dollars"],
                " [FADE]"        if fade          else "",
                " [EXPLORATION]" if in_exploration else "",
            )
            order_id = f"demo-{client_order_id}"
        else:
            try:
                order_resp = self.client.place_order(
                    ticker          = ticker,
                    side            = side,
                    count           = sizing["contracts"],
                    yes_price       = yes_price,
                    client_order_id = client_order_id,
                )
                order_id = order_resp.get("order", {}).get("order_id", client_order_id)
            except Exception as e:
                logger.error("Order failed %s: %s", ticker, e)
                return

        self._recently_traded[ticker]   = now
        self._exposure[lead_sector]     = sector_exp + sizing["dollars"]
        self.bankroll                   = live_bankroll

        # ── Provisional loss booking ───────────────────────────────────────
        # Book the full dollars_risked as a provisional loss immediately at
        # execution time. Makes the sector loss cap aware of in-flight open
        # risk, not just resolved P&L. Stored in _open_provisionals so
        # _ingest_resolved_markets() can unwind it precisely on resolution
        # rather than stacking it with the actual P&L (the v19.6 bug).
        provisional_loss = -sizing["dollars"]
        with self._sector_pnl_lock:
            self._sector_daily_pnl[lead_sector] = (
                self._sector_daily_pnl.get(lead_sector, 0.0) + provisional_loss
            )
            self._open_provisionals[ticker] = provisional_loss
        logger.debug(
            "[PROVISIONAL] %s sector=%s provisional_loss=$%.2f new_sector_pnl=$%.2f",
            ticker, lead_sector, provisional_loss,
            self._sector_daily_pnl[lead_sector],
        )

        arb_note  = f" arb_spread={arb.cross_spread:.2f}%" if arb and arb.arb_exists else ""
        fade_note = " [FADE]" if fade else ""

        trade_id = log_trade(
            ticker          = ticker,
            sector          = lead_sector,
            direction       = consensus.direction,
            contracts       = sizing["contracts"],
            yes_price_cents = yes_price,
            dollars_risked  = sizing["dollars"],
            avg_prob        = consensus.avg_prob,
            avg_edge        = consensus.avg_edge,
            avg_confidence  = consensus.avg_confidence,
            order_id        = order_id,
            demo_mode       = DEMO_MODE,
        )

        logger.info(
            "TRADE #%d | %s | %s | %s %dx@%dc | $%.2f | edge=%+.2f%% conf=%.0f%%%s%s",
            trade_id, ticker, lead_sector.upper(), consensus.direction.upper(),
            sizing["contracts"], yes_price, sizing["dollars"],
            consensus.avg_edge * 100, consensus.avg_confidence * 100,
            arb_note, fade_note,
        )

    # ── Fade execution ─────────────────────────────────────────────────────────

    def _execute_fades(self, open_markets: list[dict]) -> None:
        if not self._last_bot_probs:
            return

        fades = self.fade_scanner.scan(open_markets, self._last_bot_probs)

        for fade in fades:
            from shared.consensus_engine import ConsensusResult
            mock_consensus = ConsensusResult(
                ticker         = fade.ticker,
                signals        = [],
                execute        = True,
                direction      = fade.direction,
                avg_prob       = fade.our_prob,
                avg_edge       = abs(fade.our_prob - fade.market_prob),
                avg_confidence = fade.confidence,
                reject_reason  = "",
            )

            sector = self._last_bot_sectors.get(fade.ticker, "sports")

            self._execute_trade(
                ticker      = fade.ticker,
                title       = f"[FADE] {fade.ticker}",
                yes_price   = fade.yes_price_cents,
                consensus   = mock_consensus,
                lead_sector = sector,
                arb         = None,
                fade        = True,
            )
            self.fade_scanner.mark_executed(fade.ticker)

    # ── Correlation execution ──────────────────────────────────────────────────

    def _execute_correlations(self, open_markets: list[dict]) -> None:
        corr_signals = self.corr_engine.scan(open_markets)

        for sig in corr_signals:
            if self.circuit_breaker.is_halted():
                break

            if _is_blocked_structural_market(sig.event_group):
                logger.info(
                    "CORR BLOCKED (uncalibrated structural market): %s",
                    sig.event_group,
                )
                continue

            inferred_sector = _infer_sector_from_ticker(sig.event_group)

            logger.info(
                "[CORR TRADE] Legging into %s sector=%s (cheap=%s@%d¢, expensive=%s@%d¢) div=%.1f¢",
                sig.event_group, inferred_sector,
                sig.cheap_ticker, sig.cheap_price_cents,
                sig.expensive_ticker, sig.expensive_price_cents,
                sig.divergence_cents,
            )

            for ticker, direction, price in [
                (sig.cheap_ticker,     sig.cheap_direction,     sig.cheap_price_cents),
                (sig.expensive_ticker, sig.expensive_direction, sig.expensive_price_cents),
            ]:
                from shared.consensus_engine import ConsensusResult
                mock_consensus = ConsensusResult(
                    ticker         = ticker,
                    signals        = [],
                    execute        = True,
                    direction      = direction,
                    avg_prob       = sig.cheap_price_cents / 100 if direction == "YES" else 1 - sig.expensive_price_cents / 100,
                    avg_edge       = sig.divergence_cents / 100 / 2,
                    avg_confidence = sig.confidence,
                    reject_reason  = "",
                )

                self._execute_trade(
                    ticker      = ticker,
                    title       = f"[CORR] {sig.event_group}",
                    yes_price   = price,
                    consensus   = mock_consensus,
                    lead_sector = inferred_sector,
                    arb         = None,
                    fade        = False,
                )

    # ── Resolution timing execution ────────────────────────────────────────────

    def _execute_resolution_timing(self, open_markets: list[dict]) -> None:
        signals = self.res_timer.scan(
            open_markets,
            self._last_bot_probs,
            self._last_bot_sectors,
        )

        for sig in signals:
            if self.circuit_breaker.is_halted():
                break

            if _is_blocked_structural_market(sig.ticker):
                logger.info(
                    "RESTIME BLOCKED (uncalibrated structural market): %s",
                    sig.ticker,
                )
                continue

            from shared.consensus_engine import ConsensusResult
            mock_consensus = ConsensusResult(
                ticker         = sig.ticker,
                signals        = [],
                execute        = True,
                direction      = sig.direction,
                avg_prob       = sig.our_prob,
                avg_edge       = abs(sig.our_prob - sig.yes_price_cents / 100),
                avg_confidence = sig.confidence,
                reject_reason  = "",
            )

            self._execute_trade(
                ticker      = sig.ticker,
                title       = f"[RESTIME] {sig.ticker}",
                yes_price   = sig.yes_price_cents,
                consensus   = mock_consensus,
                lead_sector = sig.sector,
                arb         = None,
                fade        = False,
            )

    # ── Market scan ────────────────────────────────────────────────────────────

    def scan_markets(self) -> None:
        logger.info("Scanning markets...")

        self._scan_bankroll = self.circuit_breaker.sync_bankroll()
        self._scan_summary  = self.circuit_breaker.daily_summary()
        self.bankroll       = self._scan_bankroll

        try:
            markets = self.client.get_all_open_markets(
                page_sleep_sec=SCAN_PAGE_SLEEP_SEC,
            )
        except TypeError:
            try:
                markets = self.client.get_all_open_markets()
            except Exception as e:
                logger.error("Failed to fetch markets: %s", e)
                return
        except Exception as e:
            logger.error("Failed to fetch markets: %s", e)
            return

        logger.info("Fetched %d open markets", len(markets))

        for market in markets:
            self._evaluate_market(market)

        self._execute_fades(markets)
        self._execute_correlations(markets)
        self._execute_resolution_timing(markets)

        # ── Read sector P&L under lock for consistent summary log ──────────
        with self._sector_pnl_lock:
            sector_pnl_snapshot = dict(self._sector_daily_pnl)

        logger.info(
            "Scan complete | daily_pnl=$%.2f trades=%d halted=%s | sector_pnl=%s",
            self._scan_summary.get("realized_pnl", 0.0),
            self._scan_summary.get("trade_count", 0),
            self._scan_summary.get("halted", False),
            {k: f"${v:+.2f}" for k, v in sector_pnl_snapshot.items()},
        )

    # ── Nightly retrain ────────────────────────────────────────────────────────

    def retrain_models(self) -> None:
        logger.info("=== Nightly retrain ===")

        n_resolved = self._ingest_resolved_markets()
        logger.info("Ingested %d new resolved outcomes this cycle", n_resolved)

        n_patterns = self.res_timer.rebuild_patterns()
        logger.info("Rebuilt %d resolution timing patterns", n_patterns)

        for bot in self.bots:
            stats = bot.refresh_calibration()
            logger.info("[%s] calibration: %s", bot.sector_name, stats)
            bot.save_model()

        live_bankroll       = self.circuit_breaker.sync_bankroll()
        self.bankroll       = live_bankroll
        self._scan_bankroll = live_bankroll
        self._scan_summary  = {}
        logger.info("Bankroll synced: $%.2f", live_bankroll)

        self._exposure = {b.sector_name: 0.0 for b in self.bots}

        # ── Reset under lock ───────────────────────────────────────────────
        with self._sector_pnl_lock:
            self._sector_daily_pnl   = {b.sector_name: 0.0 for b in self.bots}
            self._open_provisionals  = {}

        self._last_bot_probs.clear()
        self._last_bot_sectors.clear()
        self._recently_traded.clear()

        self._resolved_count_cache.clear()
        self._resolved_count_cache_ts = 0.0

        logger.info("=== Retrain complete ===")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v19.7 | DEMO=%s | $%.2f | arb_mode=%s",
            DEMO_MODE, self.bankroll, self.arb._mode,
        )
        init_db()

        # ── NewsSignal startup — 401 on bad key is non-fatal ──────────────
        try:
            self.news.start_background_poller()
        except Exception as e:
            logger.warning(
                "NewsSignal failed to start background poller: %s — "
                "continuing without news signal (trades will not be news-gated)",
                e,
            )

        import random
        jitter = random.randint(-45, 45)
        schedule.every(SCAN_INTERVAL_SEC + jitter).seconds.do(self.scan_markets)
        schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(self.retrain_models)
        schedule.every(4).hours.do(self._run_ingestion_thread)

        self._run_ingestion_thread()

        logger.info("Waiting 30s before first scan to stagger API calls...")
        time.sleep(30)

        self.scan_markets()
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    FlywheelOrchestrator().run()