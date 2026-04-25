"""
orchestrator.py  (v20.3 — atomic correlation pair execution)

Changes vs v20.1:
  ATOMIC CORRELATION PAIRS
  Previous versions placed each correlation leg independently via
  _execute_correlations. If the second leg failed any gate (liquidity,
  cooldown, sizing, blocklist, or API rejection), the first leg was left
  as a naked directional bet — no hedge. Audit on 2026-04-24 confirmed
  this was happening 100% of the time: 40 correlation trades in history,
  0 paired, every single one a single naked leg.

  v20.3 enforces atomic placement:
    Phase 1: Pre-flight BOTH legs through every gate. If either fails,
             skip the pair entirely.
    Phase 2: Place cheap leg first. If it fails, abort cleanly.
    Phase 3: Place expensive leg. If it fails, IMMEDIATELY close the
             cheap leg at market to prevent a naked bet.
    Phase 4: Log both legs with a shared pair_uuid for joint tracking.
    Phase 5: Dedupe within a scan — one pair per event_group per scan.

  Schema change: correlation_legs.pair_uuid (TEXT, nullable) added via
  migration. Historical rows have NULL. Requires updated
  shared/calibration_logger.py v11.1 (log_correlation_leg accepts pair_uuid).

Changes vs v20 (carried forward):
  v20.1: Phase 3 expanded to cover all sports families with 24h age floor.
  v20:   Signal stream separation, DB-backed cooldown, one-outcome-per-trade,
         source tagging, gate-failure instrumentation, outcome-before-side-
         effects ordering.

Deployment order:
   1. Run migration_correlation_pairs.sql in Supabase
   2. Replace shared/calibration_logger.py (v11.1)
   3. Replace orchestrator.py (v20.3)
   4. git commit + push
   5. Railway redeploys; watch logs for [CORR-PAIR] messages
"""

import logging
import math
import os
import re
import threading
import time
import uuid
from collections import defaultdict
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
from shared.calibration_logger import (
    init_db,
    log_main_signal,
    log_correlation_leg,
    link_correlation_leg_to_trade,
    log_trade,
    log_outcome,
)
from shared.circuit_breaker import CircuitBreaker
from shared.consensus_engine import ConsensusEngine, ConsensusResult, BotSignal
from shared.correlation_engine import CorrelationEngine
from shared.correlation_tracker import CorrelationTracker
from shared.early_exit_manager import EarlyExitManager
from shared.fade_scanner import FadeScanner
from shared.kalshi_client import KalshiClient
from shared.kelly_sizer import kelly_stake, no_kelly_stake
from shared.limit_order_manager import LimitOrderManager
from shared.lineup_checker import LineupChecker
from shared.liquidity_filter import LiquidityFilter
from shared.news_signal import NewsSignal
from shared.pinnacle_reference import PinnacleReference
from shared.resolution_timer import ResolutionTimer
from shared.sharp_detector import SharpDetector
from shared.weather_ensemble import WeatherEnsemble
from bots.sector_bots import all_bots

from shared.risk_manager import RiskManager
from shared.brier_tracker import BrierTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

logging.getLogger("shared.sharp_detector").setLevel(logging.WARNING)
logging.getLogger("shared.mlb_stats_fetcher").setLevel(logging.WARNING)
logging.getLogger("shared.nba_stats_fetcher").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RECENT_TRADE_WINDOW_SEC = 1800
FINANCIAL_MARKETS_DISABLED = True
SCAN_PAGE_SLEEP_SEC = 0.25
_UUID_SUFFIX_RE = re.compile(r'-S[0-9A-Fa-f]{4,}-[0-9A-Fa-f]+$')


def _normalize_ticker(ticker: str) -> str:
    return _UUID_SUFFIX_RE.sub('', ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Sector inference
# ─────────────────────────────────────────────────────────────────────────────

_TICKER_SECTOR_MAP: list[tuple[tuple[str, ...], str]] = [
    (("kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
      "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
      "kxatp", "kxwta", "kxtennis", "kxabagame", "kxpga", "kxowga",
      "kxf1", "kxolympic", "kxepl", "kxsoccer",
      "kxboxing", "kxwwe", "kxcricket", "kxrugby", "kxesport",
      "kxdota", "kxintlf", "kxcs2", "kxcsgo",
      "kxeuroleague", "kxcba", "kxcbagame", "kxt20", "kxipl", "kxbbl",
      "kxacbgame", "kxaleaguegame", "kxaleague",
      "kxarg", "kxlali", "kxbund", "kxseri", "kxliga", "kxligu",
      "kxbras", "kxswis", "kxbelg", "kxecul", "kxpslg", "kxsaud",
      "kxjlea", "kxuclt", "kxucl", "kxfifa", "kxalea", "kxafl",
      "kxconmebol", "kxchll", "kxargpremdiv",
      "kxfiba", "kxacbg", "kxvtbg", "kxbalg", "kxbbse", "kxnpbg",
      "kxahlg", "kxshl", "kxdima", "kxufl", "kxintl",
      "kxsuperlig", "kxegypl", "kxscottishprem", "kxuel", "kxeredivisie",
      "kxmotogp", "kxallsvenskan", "kxekstraklasa", "kxurypd",
      "kxow", "kxvalorant", "kxlol", "kxleague",
      "kxrl", "kxrocketleague", "kxapex", "kxfort", "kxr6g",
      "kxitf", "kxkf", "kxnextag", "kxsurv",
      "kxdel", "kxkhl", "kxjbleague", "kxkbl", "kxkleague",
      "kxchnsl", "kxballerleague", "kxapfddh", "kxr6", "kxhnl", "kxelh",
      "kxtabletennis", "kxkbogame", "kxlnbelite",
      ), "sports"),
    (("kxbtc", "kxeth", "kxsol", "kxcrypto", "kxdefi",
      "kxxrp", "kxdoge", "kxbnb", "kxavax", "kxlink",
      "kxcoin", "kxmatic", "kxada", "kxshib",
      ), "crypto"),
    (("kxelect", "kxpres", "kxsen", "kxhouse", "kxgov",
      "kxpol", "kxvote", "kxapprove",
      "kxswalwell", "kxtrumppardons", "kxtrumpendorse",
      "kxdenmarkpm", "kxisraelpm",
      "kxtrumpact", "kxtrumptime", "kxmamdanieo",
      "kxleave", "kxhormuz", "kxca14s", "kxpressbriefing",
      ), "politics"),
    (("kxhurr", "kxtemp", "kxrain", "kxsnow", "kxweather",
      "kxnoaa", "kxclimate", "kxlowt", "kxchll", "kxdens",
      "kxhight",
      ), "weather"),
    (("kxai", "kxtech", "kxfed", "kxcpi", "kxgdp",
      "kxjobs", "kxrate", "kxinfl", "kxpayroll",
      "kxhighinfl", "kxuspspend",
      ), "economics"),
    (("kxjetfuel", "kxwti", "kxoil", "kxgold", "kxsilver", "kxnat", "kxgasoline",
      "kxhoil", "kxtrufegg",
      "kxusdjpy", "kxeurusd", "kxgbpusd", "kxusdcad", "kxusdchf", "kxaudusd",
      "kxspy", "kxqqq", "kxtsla", "kxaapl", "kxnvda", "kxamzn", "kxgoog", "kxmsft",
      "kxtruft", "kxmar", "kxhilt", "kxnatgas",
      ), "financial_markets"),
]


def _infer_sector_from_ticker(ticker: str) -> str:
    tk = ticker.lower()
    for prefixes, sector in _TICKER_SECTOR_MAP:
        if any(tk.startswith(p) for p in prefixes):
            return sector
    return "economics"


def _is_financial_markets_ticker(ticker: str) -> bool:
    return _infer_sector_from_ticker(ticker) == "financial_markets"


# ─────────────────────────────────────────────────────────────────────────────
# Structural market blocklist
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURAL_MARKET_BLOCKLIST = (
    "kxmvecrosscategory",
    "kxmvecrosscat",
    "kxmvecross",
    "kxmvesportsmultigameextended",
    "kxmvesportsmultigame",
    "kxmvesportsmulti",
    "kxcoinbase",
)


def _is_blocked_structural_market(ticker: str) -> bool:
    tk = _UUID_SUFFIX_RE.sub('', ticker.lower())
    return any(tk.startswith(p) for p in _STRUCTURAL_MARKET_BLOCKLIST)


_NBA_PROP_CODES = ("PTS", "REB", "AST", "3PT", "STL", "BLK")


def _is_nba_player_prop(ticker: str) -> bool:
    tk_upper = ticker.upper()
    return any(tk_upper.startswith(f"KXNBA{code}") for code in _NBA_PROP_CODES)


# ─────────────────────────────────────────────────────────────────────────────
# Connection pool
# ─────────────────────────────────────────────────────────────────────────────

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


def _query_db(sql: str, params: tuple = (), _retries: int = 2) -> list:
    pool = _get_pool()
    try:
        if pool:
            conn = pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                conn.commit()
                return rows
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                pool.putconn(conn)
                if _retries > 0:
                    logger.warning(
                        "_query_db error, retrying (%d left): %s",
                        _retries, e,
                    )
                    global _pg_pool
                    try:
                        _pg_pool.closeall()
                    except Exception:
                        pass
                    _pg_pool = None
                    return _query_db(sql, params, _retries=_retries - 1)
                raise
            finally:
                try:
                    pool.putconn(conn)
                except Exception:
                    pass
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            conn.close()
            return rows
    except Exception as e:
        logger.error("_query_db error: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed cooldown + signal existence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recently_traded_in_db(ticker: str, within_sec: int = RECENT_TRADE_WINDOW_SEC) -> bool:
    p = _ph()
    if os.getenv("DATABASE_URL"):
        sql = (
            f"SELECT 1 FROM trades "
            f"WHERE ticker = {p} "
            f"  AND created_at > NOW() - (INTERVAL '1 second' * {p}) "
            f"LIMIT 1"
        )
        rows = _query_db(sql, (ticker, int(within_sec)))
    else:
        sql = (
            f"SELECT 1 FROM trades "
            f"WHERE ticker = {p} "
            f"  AND created_at > datetime('now', {p}) "
            f"LIMIT 1"
        )
        rows = _query_db(sql, (ticker, f"-{int(within_sec)} seconds"))
    return len(rows) > 0


def _recent_main_signal_exists(ticker: str, source: str, within_minutes: int = 30) -> bool:
    p = _ph()
    if os.getenv("DATABASE_URL"):
        sql = (
            f"SELECT 1 FROM main_signals "
            f"WHERE ticker = {p} AND source = {p} "
            f"  AND created_at > NOW() - (INTERVAL '1 minute' * {p}) "
            f"LIMIT 1"
        )
        rows = _query_db(sql, (ticker, source, int(within_minutes)))
    else:
        sql = (
            f"SELECT 1 FROM main_signals "
            f"WHERE ticker = {p} AND source = {p} "
            f"  AND created_at > datetime('now', {p}) "
            f"LIMIT 1"
        )
        rows = _query_db(sql, (ticker, source, f"-{int(within_minutes)} minutes"))
    return len(rows) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yes_price_cents(market: dict) -> int:
    try:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
        if bid > 0 and ask > 0:
            return int(round((bid + ask) / 2 * 100))
        if ask > 0:
            return int(round(ask * 100))
        if bid > 0:
            return int(round(bid * 100))
    except (TypeError, ValueError):
        pass
    try:
        last = float(market.get("last_price_dollars") or 0)
        if last > 0:
            return int(round(last * 100))
    except (TypeError, ValueError):
        pass
    return 0


def _calculate_trade_pnl(
    direction: str,
    contracts: int,
    yes_price_cents: int,
    resolved_yes: bool,
) -> float:
    yes_price_frac = yes_price_cents / 100.0
    no_price_frac = 1.0 - yes_price_frac
    direction_upper = direction.upper()

    if direction_upper == "YES":
        per_contract = (1.0 - yes_price_frac) if resolved_yes else (-yes_price_frac)
    elif direction_upper == "NO":
        per_contract = (1.0 - no_price_frac) if not resolved_yes else (-no_price_frac)
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    return round(contracts * per_contract, 4)


def _update_main_signal_brier(ticker: str, resolved_yes: bool) -> None:
    outcome = 1 if resolved_yes else 0
    now = datetime.now(timezone.utc)
    p = _ph()
    pool = _get_pool()

    try:
        if pool:
            conn = pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    UPDATE main_signals
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
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.execute(
                f"""
                UPDATE main_signals
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
    except Exception as e:
        logger.warning("_update_main_signal_brier failed for %s: %s", ticker, e)


def _update_correlation_leg_outcome(ticker: str, resolved_yes: bool) -> None:
    outcome = 1 if resolved_yes else 0
    now = datetime.now(timezone.utc)
    p = _ph()
    pool = _get_pool()

    try:
        if pool:
            conn = pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    UPDATE correlation_legs
                    SET outcome    = {p},
                        outcome_at = {p}
                    WHERE ticker   = {p}
                      AND outcome IS NULL
                    """,
                    (outcome, now, ticker),
                )
                conn.commit()
            finally:
                pool.putconn(conn)
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.execute(
                f"""
                UPDATE correlation_legs
                SET outcome    = {p},
                    outcome_at = {p}
                WHERE ticker   = {p}
                  AND outcome IS NULL
                """,
                (outcome, now, ticker),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning("_update_correlation_leg_outcome failed for %s: %s", ticker, e)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class FlywheelOrchestrator:

    def __init__(self):
        self.client = KalshiClient()
        self.bots = all_bots()
        self.engine = ConsensusEngine()
        self.arb = ArbLayer()

        self.circuit_breaker = CircuitBreaker(
            kalshi_client=self.client,
            bankroll=BANKROLL,
            daily_loss_limit_pct=CIRCUIT_BREAKER_PCT,
        )
        _synced = self.circuit_breaker.sync_bankroll()
        self.bankroll = _synced if _synced > 0 else BANKROLL

        self.sharp = SharpDetector(
            kalshi_client=self.client,
            spread_threshold_pct=SHARP_SPREAD_THRESHOLD_PCT,
            large_order_contracts=SHARP_LARGE_ORDER_CONTRACTS,
            volume_spike_multiplier=SHARP_VOLUME_SPIKE_MULTIPLIER,
        )
        self.fade_scanner = FadeScanner(
            kalshi_client=self.client,
            sharp_detector=self.sharp,
            fade_window_minutes=FADE_WINDOW_MINUTES,
            fade_threshold_cents=FADE_THRESHOLD_CENTS,
            fade_min_disagreement=FADE_MIN_DISAGREEMENT,
        )
        self.corr_engine = CorrelationEngine(
            min_divergence_cents=CORR_MIN_DIVERGENCE_CENTS,
            max_group_size=CORR_MAX_GROUP_SIZE,
        )
        self.res_timer = ResolutionTimer(
            min_overdue_minutes=RESTIME_MIN_OVERDUE_MIN,
            min_prob_to_trade=RESTIME_MIN_PROB,
            min_sample_count=RESTIME_MIN_SAMPLES,
        )
        self.news = NewsSignal(
            api_key=NEWSAPI_KEY,
            poll_interval_sec=NEWS_POLL_INTERVAL_SEC,
            velocity_window_min=NEWS_VELOCITY_WINDOW_MIN,
            velocity_spike_threshold=NEWS_VELOCITY_SPIKE_THRESHOLD,
        )

        self._exposure: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}
        self._sector_daily_pnl: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}
        self._sector_pnl_lock = threading.Lock()
        self._ingestion_lock = threading.Lock()

        self._last_bot_probs: dict[str, float] = {}
        self._last_bot_sectors: dict[str, str] = {}

        self._resolved_count_cache: dict[str, int] = {}
        self._resolved_count_cache_ts: float = 0.0

        self._scan_bankroll: float = self.bankroll
        self._scan_summary: dict = {}
        self._open_provisionals: dict[int, float] = {}

        self._gate_stats: dict[str, int] = defaultdict(int)

        self.risk_manager = RiskManager(
            bankroll=BANKROLL,
            max_daily_drawdown_pct=0.08,
            max_player_exposure_pct=0.03,
            max_single_trade_pct=0.02,
            max_sector_exposure_pct=0.07,
            shadow_hours_required=72,
            enforce_shadow=(not DEMO_MODE and os.getenv("ENFORCE_SHADOW_MODE", "false").lower() == "true"),
        )
        if self.risk_manager.enforce_shadow:
            self.risk_manager.start_shadow()

        self.brier_tracker = BrierTracker.from_supabase(os.getenv("DATABASE_URL", ""))

        self.liquidity_filter = LiquidityFilter(enabled=True)
        self.correlation_tracker = CorrelationTracker()

        self.pinnacle = PinnacleReference(
            api_key=os.getenv("ODDS_API_KEY", ""),
            enabled=True,
        )
        self.lineup_checker = LineupChecker(enabled=True)
        self.weather_ensemble = WeatherEnsemble(enabled=True)

        self.limit_order_mgr = LimitOrderManager(
            client=self.client,
            enabled=not DEMO_MODE,
        )
        self.early_exit_mgr = EarlyExitManager(
            client=self.client,
            enabled=not DEMO_MODE,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    def _get_sector_resolved_count(self, sector: str) -> int:
        now = time.monotonic()
        if now - self._resolved_count_cache_ts > 60.0:
            self._resolved_count_cache.clear()
            self._resolved_count_cache_ts = now

        if sector not in self._resolved_count_cache:
            p = _ph()
            rows = _query_db(
                f"SELECT COUNT(*) as n FROM main_signals "
                f"WHERE sector = {p} AND outcome IS NOT NULL",
                (sector,),
            )
            self._resolved_count_cache[sector] = int(rows[0]["n"]) if rows else 0

        return self._resolved_count_cache[sector]

    def _extract_risk_entity_key(self, ticker: str, sector: str) -> str:
        parts = ticker.split("-")
        tk = ticker.upper()
        if sector == "sports" and len(parts) >= 3:
            if (tk.startswith("KXMLB") or tk.startswith("KXNBA")
                or tk.startswith("KXATP") or tk.startswith("KXWTA")
                or tk.startswith("KXTENNIS") or tk.startswith("KXITF")):
                return parts[2]
        return ticker

    def _passes_market_quality_gate(self, market: dict | None, sector: str) -> tuple[bool, str]:
        if market is None:
            return False, "missing market snapshot"

        ticker = _normalize_ticker(market.get("ticker", ""))

        if _is_blocked_structural_market(ticker):
            return False, "structural blocklist"

        if sector == "financial_markets" and FINANCIAL_MARKETS_DISABLED:
            return False, "financial_markets disabled"

        liq_result = self.liquidity_filter.check(market, sector=sector)
        if not liq_result.passes:
            return False, f"liquidity fail: {liq_result.reason}"

        sector_bot = next((b for b in self.bots if b.sector_name == sector), None)
        if sector_bot is not None and not sector_bot.is_relevant(market):
            return False, f"{sector} bot rejected market"

        return True, ""

    def _rebuild_exposure(self) -> None:
        self._exposure = {b.sector_name: 0.0 for b in self.bots}
        rows = _query_db(
            """
            SELECT t.sector, t.dollars_risked
            FROM trades t
            LEFT JOIN outcomes o ON o.trade_id = t.id
            WHERE o.trade_id IS NULL
              AND t.created_at > NOW() - INTERVAL '3 days'
            """
        )
        for row in rows:
            sector = row.get("sector")
            dollars = float(row.get("dollars_risked") or 0.0)
            if sector in self._exposure:
                self._exposure[sector] += dollars

    # ── Resolution ingestion ───────────────────────────────────────────────────

    def _process_resolved_market(
        self,
        market: dict,
        already_recorded: set[str],
        processed_this_run: set[str],
        bayesian_updated_events: set[str],
    ) -> int:
        p = _ph()

        ticker = _normalize_ticker(market.get("ticker", ""))
        result_raw = market.get("result", "") or ""
        result = result_raw.lower()

        if not ticker or result not in ("yes", "no"):
            return 0
        if ticker in already_recorded or ticker in processed_this_run:
            return 0

        resolved_yes = (result == "yes")

        event_key = _normalize_ticker(
            market.get("event_ticker", "") or ticker.rsplit("-", 1)[0]
        )
        allow_bayesian_update = event_key not in bayesian_updated_events

        if allow_bayesian_update:
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
            bayesian_updated_events.add(event_key)

        _update_main_signal_brier(ticker, resolved_yes)
        _update_correlation_leg_outcome(ticker, resolved_yes)

        main_sig_rows = _query_db(
            f"SELECT our_prob, direction, sector FROM main_signals "
            f"WHERE ticker = {p} ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        )
        if main_sig_rows:
            prop_code = None
            for code in ("HIT", "TB", "HRR", "HR", "KS"):
                if f"KXMLB{code}" in ticker.upper():
                    prop_code = code
                    break
            if prop_code:
                self.brier_tracker.record(
                    prop_code,
                    float(main_sig_rows[0]["our_prob"]),
                    1 if resolved_yes else 0,
                )

        if _is_nba_player_prop(ticker):
            try:
                from shared.nba_logistic_model import record_outcome as record_nba_outcome
                record_nba_outcome(ticker, 1 if resolved_yes else 0)
            except Exception as e:
                logger.warning("[NBA LOGREG] record_outcome failed for %s: %s", ticker, e)

        trade_rows = _query_db(
            f"""
            SELECT t.id, t.direction, t.contracts, t.yes_price_cents,
                   t.sector, t.source, t.avg_prob
            FROM trades t
            LEFT JOIN outcomes o ON o.trade_id = t.id
            WHERE t.ticker = {p}
              AND o.trade_id IS NULL
            ORDER BY t.created_at ASC
            """,
            (ticker,),
        )

        if not trade_rows:
            processed_this_run.add(ticker)
            return 0

        outcomes_written = 0

        for trade in trade_rows:
            trade_id = int(trade["id"])
            direction = str(trade.get("direction", "YES")).upper()
            contracts = int(trade.get("contracts", 0))
            yes_price = int(trade.get("yes_price_cents", 50))
            sector = trade.get("sector", "") or _infer_sector_from_ticker(ticker)
            source = trade.get("source", "main") or "main"
            trade_our_prob = float(trade.get("avg_prob") or 0.5)

            if contracts <= 0:
                continue

            try:
                pnl_usd = _calculate_trade_pnl(
                    direction=direction,
                    contracts=contracts,
                    yes_price_cents=yes_price,
                    resolved_yes=resolved_yes,
                )
            except ValueError as e:
                logger.warning("P&L calc failed for trade_id=%d: %s", trade_id, e)
                continue

            direction_correct = (
                (direction == "YES" and resolved_yes)
                or (direction == "NO" and not resolved_yes)
            )

            try:
                log_outcome(
                    ticker=ticker,
                    resolved=result.upper(),
                    pnl_usd=pnl_usd,
                    trade_id=trade_id,
                    our_prob=trade_our_prob,
                    correct=direction_correct,
                    sector=sector,
                )
            except Exception as e:
                logger.error(
                    "log_outcome failed for trade_id=%d ticker=%s: %s",
                    trade_id, ticker, e,
                )
                continue

            logger.info(
                "[P&L] trade_id=%d %s resolved %s → $%+.2f (source=%s)",
                trade_id, ticker, result.upper(), pnl_usd, source,
            )
            self.circuit_breaker.record_pnl(pnl_usd)

            with self._sector_pnl_lock:
                if sector in self._sector_daily_pnl:
                    provisional = self._open_provisionals.pop(trade_id, 0.0)
                    self._sector_daily_pnl[sector] = (
                        self._sector_daily_pnl[sector] - provisional + pnl_usd
                    )

            outcomes_written += 1

        processed_this_run.add(ticker)
        return outcomes_written

    def _ingest_resolved_markets(self) -> int:
        logger.info("=== Resolution ingestion starting ===")

        try:
            min_ts = self.client.get_latest_outcome_ts()
            if min_ts:
                logger.info("Incremental — fetching markets after %s", min_ts)
            resolved_markets = self.client.get_resolved_markets(
                min_settled_ts=min_ts,
                max_pages=10 if min_ts else 350,
            )
        except Exception as e:
            logger.error("Failed to fetch resolved markets: %s", e)
            return 0

        logger.info("%d settled markets returned from Kalshi", len(resolved_markets))

        p = _ph()
        if min_ts:
            already_recorded = {
                _normalize_ticker(r["ticker"])
                for r in _query_db(
                    f"SELECT DISTINCT ticker FROM outcomes WHERE logged_at > {p}",
                    (min_ts,),
                )
            }
        else:
            already_recorded = {
                _normalize_ticker(r["ticker"])
                for r in _query_db("SELECT DISTINCT ticker FROM outcomes")
            }

        processed_this_run: set[str] = set()
        recorded = 0
        bayesian_updated_events: set[str] = set()

        for market in resolved_markets:
            if self.circuit_breaker.is_halted():
                logger.warning("Circuit breaker active — stopping ingestion early")
                break
            recorded += self._process_resolved_market(
                market, already_recorded, processed_this_run, bayesian_updated_events
            )

        logger.info("Phase 1/2 complete — %d outcome rows written", recorded)

        logger.info("=== Phase 3: checking unresolved tickers for finalized status ===")

        unresolved_rows = _query_db(
            """
            SELECT DISTINCT t.ticker
            FROM trades t
            LEFT JOIN outcomes o ON o.trade_id = t.id
            WHERE o.trade_id IS NULL
              AND t.created_at > NOW() - INTERVAL '7 days'
              AND t.created_at < NOW() - INTERVAL '24 hours'
              AND t.sector IN ('sports', 'weather', 'crypto', 'financial_markets')
            LIMIT 1500
            """
        )

        unresolved_tickers = [
            r["ticker"] for r in unresolved_rows
            if r["ticker"] not in processed_this_run
            and r["ticker"] not in already_recorded
        ]

        if not unresolved_tickers:
            logger.info("Phase 3: no unresolved tickers to check")
        else:
            logger.info("Phase 3: checking %d unresolved tickers", len(unresolved_tickers))
            try:
                finalized_markets = self.client.get_finalized_markets(
                    tickers=unresolved_tickers,
                    rate_limit_sleep=0.15,
                )
                logger.info("Phase 3: found %d finalized", len(finalized_markets))

                phase3_recorded = 0
                for market in finalized_markets:
                    if self.circuit_breaker.is_halted():
                        break
                    phase3_recorded += self._process_resolved_market(
                        market, already_recorded, processed_this_run, bayesian_updated_events
                    )
                recorded += phase3_recorded
                logger.info("Phase 3: %d additional outcomes", phase3_recorded)
            except Exception as e:
                logger.error("Phase 3 failed: %s", e)

        if recorded > 0:
            try:
                self.brier_tracker.save_to_supabase(os.getenv("DATABASE_URL", ""))
            except Exception as e:
                logger.warning("Brier tracker save failed: %s", e)

        logger.info("=== Resolution ingestion complete — %d new outcomes ===", recorded)
        return recorded

    def _run_ingestion_thread(self) -> None:
        if not self._ingestion_lock.acquire(blocking=False):
            logger.warning("Ingestion already running — skipping")
            return

        def _run():
            try:
                self._ingest_resolved_markets()
            finally:
                self._ingestion_lock.release()

        t = threading.Thread(target=_run, daemon=True, name="resolution-ingestion")
        t.start()

    # ── Main path ──────────────────────────────────────────────────────────────

    def _evaluate_market(self, market: dict) -> None:
        self._gate_stats['markets_scanned'] += 1

        ticker = _normalize_ticker(
            market.get("ticker") or market.get("event_ticker") or "UNKNOWN"
        )
        title = market.get("title", "")
        yes_price = _parse_yes_price_cents(market)

        if yes_price == 0:
            self._gate_stats['rejected_no_price'] += 1
            return
        if yes_price <= 2 or yes_price >= 98:
            self._gate_stats['rejected_extreme_price'] += 1
            return

        if FINANCIAL_MARKETS_DISABLED and _is_financial_markets_ticker(ticker):
            self._gate_stats['rejected_fm_disabled'] += 1
            return

        liq_result = self.liquidity_filter.check(market, sector="unknown")
        if not liq_result.passes:
            self._gate_stats['rejected_liquidity'] += 1
            return

        self._gate_stats['reached_bot_evaluation'] += 1

        self.news.register_market(title)
        sharp_signal = self.sharp.analyze(market)

        signals = []
        any_bot_relevant = False
        for bot in self.bots:
            if not bot.is_relevant(market):
                continue
            any_bot_relevant = True
            try:
                sig = bot.evaluate(market, news_signal=self.news)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.error("[%s] error on %s: %s", bot.sector_name, ticker, e)

        if not any_bot_relevant:
            self._gate_stats['rejected_no_relevant_bot'] += 1
            return
        if not signals:
            self._gate_stats['rejected_bots_returned_none'] += 1
            return

        self._gate_stats['reached_consensus'] += 1

        consensus = self.engine.evaluate(ticker, yes_price, signals)

        if not consensus.execute:
            reason = consensus.reject_reason or "unknown"
            reason_lower = reason.lower()
            if "edge" in reason_lower and "floor" in reason_lower:
                self._gate_stats['rejected_consensus_edge_floor'] += 1
            elif "edge" in reason_lower and "ceiling" in reason_lower:
                self._gate_stats['rejected_consensus_edge_ceiling'] += 1
            elif "confidence" in reason_lower:
                self._gate_stats['rejected_consensus_confidence'] += 1
            elif "direction" in reason_lower and "disagree" in reason_lower:
                self._gate_stats['rejected_consensus_direction_disagree'] += 1
            elif "direction" in reason_lower and "filter" in reason_lower:
                self._gate_stats['rejected_consensus_direction_filter'] += 1
            else:
                self._gate_stats['rejected_consensus_other'] += 1
            return

        self._gate_stats['passed_consensus'] += 1

        if sharp_signal.sharp_detected and not sharp_signal.aligned_with(consensus.direction):
            self._gate_stats['rejected_sharp_veto'] += 1
            return

        lead_sector = signals[0].sector

        self._last_bot_probs[ticker] = consensus.avg_prob
        self._last_bot_sectors[ticker] = lead_sector

        arb = self.arb.check_and_log(
            ticker=ticker,
            kalshi_cents=yes_price,
            our_direction=consensus.direction,
            market_title=title,
        )
        if not arb.passes:
            self._gate_stats['rejected_arb_gate'] += 1
            return

        if not _recent_main_signal_exists(ticker, source="main", within_minutes=30):
            try:
                log_main_signal(
                    ticker=ticker,
                    sector=lead_sector,
                    our_prob=consensus.avg_prob,
                    market_prob=yes_price / 100,
                    edge=consensus.avg_edge,
                    confidence=consensus.avg_confidence,
                    direction=consensus.direction,
                    source="main",
                )
                self._gate_stats['main_signals_logged'] += 1
            except ValueError as e:
                logger.warning("log_main_signal refused %s: %s", ticker, e)
                self._gate_stats['rejected_signal_shape'] += 1
                return
            except Exception as e:
                logger.error("log_main_signal failed for %s: %s", ticker, e)

        self._execute_trade(
            ticker=ticker,
            title=title,
            yes_price=yes_price,
            consensus=consensus,
            lead_sector=lead_sector,
            arb=arb,
            source="main",
            market=market,
        )

    def _execute_trade(
        self,
        ticker: str,
        title: str,
        yes_price: int,
        consensus,
        lead_sector: str,
        arb,
        source: str = "main",
        fade: bool = False,
        market: dict = None,
    ):
        if _recently_traded_in_db(ticker, within_sec=RECENT_TRADE_WINDOW_SEC):
            self._gate_stats['rejected_cooldown'] += 1
            return None

        if FINANCIAL_MARKETS_DISABLED and lead_sector == "financial_markets":
            self._gate_stats['rejected_fm_disabled'] += 1
            return None

        if self.circuit_breaker.is_halted():
            self._gate_stats['rejected_circuit_breaker'] += 1
            return None

        if not self.risk_manager.can_trade():
            self._gate_stats['rejected_risk_halt'] += 1
            return None

        cap = sector_loss_cap(lead_sector)
        if cap == 0.0:
            self._gate_stats['rejected_sector_disabled'] += 1
            return None

        with self._sector_pnl_lock:
            sector_loss_today = self._sector_daily_pnl.get(lead_sector, 0.0)
        if sector_loss_today <= -cap:
            self._gate_stats['rejected_sector_loss_cap'] += 1
            return None

        live_bankroll = self._scan_bankroll
        daily_pnl = float(self._scan_summary.get("realized_pnl", 0.0))
        loss_limit = live_bankroll * CIRCUIT_BREAKER_PCT
        drawdown_factor = (
            max(0.5, 1.0 + (daily_pnl / loss_limit) * 0.5)
            if daily_pnl < 0 and loss_limit > 0 else 1.0
        )

        resolved_count = self._get_sector_resolved_count(lead_sector)
        kf = sector_kelly_fraction(lead_sector, resolved_count)
        in_exploration = resolved_count < SECTOR_MIN_RESOLVED.get(lead_sector, 30)

        pinnacle_adj = 1.0
        if lead_sector == "sports" and not fade:
            sharp_check = self.pinnacle.check_from_kalshi_ticker(
                ticker=ticker,
                our_prob=consensus.avg_prob,
                our_direction=consensus.direction,
            )
            if not sharp_check.passes:
                self._gate_stats['rejected_pinnacle'] += 1
                return None
            pinnacle_adj = sharp_check.confidence_adjustment

        if lead_sector == "sports" and not fade:
            lineup_check = self.lineup_checker.check_from_kalshi_ticker(ticker)
            if not lineup_check.is_active:
                self._gate_stats['rejected_lineup'] += 1
                return None
            if lineup_check.status == "questionable":
                pinnacle_adj *= 0.7

        sector_exp = self._exposure.get(lead_sector, 0.0)
        effective_drawdown = drawdown_factor * pinnacle_adj

        if consensus.direction == "YES":
            sizing = kelly_stake(
                prob=consensus.avg_prob,
                yes_price_cents=yes_price,
                bankroll=self.bankroll,
                sector=lead_sector,
                sector_exposure=sector_exp,
                live_bankroll=live_bankroll,
                drawdown_factor=effective_drawdown,
                market=market,
            )
            side = "yes"
        else:
            sizing = no_kelly_stake(
                prob=consensus.avg_prob,
                yes_price_cents=yes_price,
                bankroll=self.bankroll,
                sector=lead_sector,
                sector_exposure=sector_exp,
                live_bankroll=live_bankroll,
                drawdown_factor=effective_drawdown,
                market=market,
            )
            side = "no"

        if in_exploration and sizing["contracts"] > 0:
            exploration_scale = kf / KELLY_FRACTION
            sizing["dollars"] = round(sizing["dollars"] * exploration_scale, 2)
            sizing["contracts"] = max(1, int(sizing["dollars"] / (yes_price / 100)))

        if sizing["contracts"] <= 0:
            self._gate_stats['rejected_sizing_zero'] += 1
            return None

        player_key = self._extract_risk_entity_key(ticker, lead_sector)
        ok, reason = self.risk_manager.pre_trade_check(
            player_key, lead_sector, sizing["dollars"],
        )
        if not ok:
            self._gate_stats['rejected_risk_pretrade'] += 1
            return None

        corr_result = self.correlation_tracker.get_discount(ticker)
        if corr_result.discount < 1.0:
            sizing["dollars"] = round(sizing["dollars"] * corr_result.discount, 2)
            sizing["contracts"] = max(1, int(sizing["contracts"] * corr_result.discount))
            if sizing["contracts"] <= 0:
                self._gate_stats['rejected_correlation_discount'] += 1
                return None

        client_order_id = str(uuid.uuid4())

        if DEMO_MODE:
            order_id = f"demo-{client_order_id}"
        else:
            try:
                limit_result = self.limit_order_mgr.execute_limit_order(
                    ticker=ticker,
                    side=side,
                    contracts=sizing["contracts"],
                    max_price_cents=yes_price + 2,
                    urgency="normal",
                )
                if limit_result.contracts_filled == 0:
                    logger.warning("LIMIT FAIL %s — falling back to market", ticker[:40])
                    order_resp = self.client.place_order(
                        ticker=ticker, side=side,
                        count=sizing["contracts"],
                        yes_price=yes_price,
                        client_order_id=client_order_id,
                    )
                    order_id = order_resp.get("order", {}).get("order_id", client_order_id)
                else:
                    order_id = (limit_result.order_ids[0]
                                if limit_result.order_ids else client_order_id)
            except Exception as e:
                self._gate_stats['rejected_order_error'] += 1
                logger.error("Order failed %s: %s", ticker, e)
                return None

        self._exposure[lead_sector] = sector_exp + sizing["dollars"]
        self.bankroll = live_bankroll

        self.risk_manager.record_trade(player_key, lead_sector, sizing["dollars"])
        self.correlation_tracker.record_trade(ticker, sizing["dollars"])

        trade_id = log_trade(
            ticker=ticker,
            sector=lead_sector,
            source=source,
            direction=consensus.direction,
            contracts=sizing["contracts"],
            yes_price_cents=yes_price,
            dollars_risked=sizing["dollars"],
            avg_prob=consensus.avg_prob,
            avg_edge=consensus.avg_edge,
            avg_confidence=consensus.avg_confidence,
            order_id=order_id,
            client_order_id=client_order_id,
            demo_mode=DEMO_MODE,
        )

        if trade_id is None:
            self._gate_stats['rejected_trade_dedupe'] += 1
            return None

        self._gate_stats[f'trades_executed_{source}'] += 1

        provisional_loss = -sizing["dollars"]
        with self._sector_pnl_lock:
            self._sector_daily_pnl[lead_sector] = (
                self._sector_daily_pnl.get(lead_sector, 0.0) + provisional_loss
            )
            self._open_provisionals[trade_id] = provisional_loss

        logger.info(
            "TRADE #%d | %s | %s | %s | %s %dx@%dc | $%.2f | edge=%+.2f%% conf=%.0f%%",
            trade_id, ticker, lead_sector.upper(), source,
            consensus.direction.upper(),
            sizing["contracts"], yes_price, sizing["dollars"],
            consensus.avg_edge * 100, consensus.avg_confidence * 100,
        )

        return trade_id

    # ── Fade scanner ───────────────────────────────────────────────────────────

    def _execute_fades(self, open_markets: list[dict]) -> None:
        if not self._last_bot_probs:
            return

        market_map = {
            _normalize_ticker(m.get("ticker", "")): m
            for m in open_markets
        }

        fades = self.fade_scanner.scan(open_markets, self._last_bot_probs)

        for fade in fades:
            market = market_map.get(_normalize_ticker(fade.ticker))
            sector = self._last_bot_sectors.get(fade.ticker, "sports")

            ok, reason = self._passes_market_quality_gate(market, sector)
            if not ok:
                continue

            if abs(fade.our_prob - fade.market_prob) < 0.001:
                continue

            mock_consensus = ConsensusResult(
                ticker=fade.ticker, signals=[], execute=True,
                direction=fade.direction,
                avg_prob=fade.our_prob,
                avg_edge=abs(fade.our_prob - fade.market_prob),
                avg_confidence=fade.confidence,
                reject_reason="",
            )

            if not _recent_main_signal_exists(fade.ticker, source="fade", within_minutes=30):
                try:
                    log_main_signal(
                        ticker=fade.ticker,
                        sector=sector,
                        our_prob=fade.our_prob,
                        market_prob=fade.yes_price_cents / 100,
                        edge=mock_consensus.avg_edge,
                        confidence=mock_consensus.avg_confidence,
                        direction=mock_consensus.direction,
                        source="fade",
                    )
                except Exception as e:
                    logger.warning("log_main_signal failed for fade %s: %s", fade.ticker, e)

            self._execute_trade(
                ticker=fade.ticker,
                title=f"[FADE] {fade.ticker}",
                yes_price=fade.yes_price_cents,
                consensus=mock_consensus,
                lead_sector=sector,
                arb=None,
                source="fade",
                fade=True,
                market=market,
            )
            self.fade_scanner.mark_executed(fade.ticker)

    # ── Correlation engine (v20.3 atomic pair execution) ───────────────────────

    def _execute_correlations(self, open_markets: list[dict]) -> None:
        """Correlation engine — ATOMIC PAIR EXECUTION (v20.3).

        Previous versions placed each leg independently, which created naked
        directional bets whenever the second leg failed any gate. Audit on
        2026-04-24 showed 40/40 correlation trades in history had been single
        naked legs, not arbitrage pairs.

        This version enforces atomic placement:
          Phase 1: gate both legs; skip pair if EITHER fails
          Phase 2: place cheap leg; abort cleanly if it fails
          Phase 3: place expensive leg; close cheap at market if it fails
          Phase 4: link both trades to correlation_legs with shared pair_uuid
        """
        market_map = {
            _normalize_ticker(m.get("ticker", "")): m
            for m in open_markets
        }

        corr_signals = self.corr_engine.scan(open_markets)

        # Dedupe within scan: only fire one pair per event_group per scan.
        seen_groups: set[str] = set()

        for sig in corr_signals:
            if self.circuit_breaker.is_halted():
                logger.warning("[CORR] circuit breaker halted — stopping")
                break

            if sig.event_group in seen_groups:
                logger.debug(
                    "[CORR] %s — skipping, already fired pair this scan",
                    sig.event_group,
                )
                continue

            pair_ok, reject_reason = self._preflight_correlation_pair(sig, market_map)
            if not pair_ok:
                logger.debug(
                    "[CORR] %s pair REJECTED (%s): cheap=%s expensive=%s",
                    sig.event_group, reject_reason,
                    sig.cheap_ticker, sig.expensive_ticker,
                )
                continue

            try:
                self._execute_correlation_pair(sig, market_map)
                seen_groups.add(sig.event_group)
            except Exception as e:
                logger.error(
                    "[CORR] unexpected error executing pair %s: %s",
                    sig.event_group, e,
                )

    def _preflight_correlation_pair(self, sig, market_map: dict) -> tuple[bool, str]:
        """Check that BOTH legs pass every gate before placing either."""
        inferred_sector = _infer_sector_from_ticker(sig.event_group)

        allowed_sectors = ("sports", "weather", "crypto")
        if not FINANCIAL_MARKETS_DISABLED:
            allowed_sectors = allowed_sectors + ("financial_markets",)
        if inferred_sector not in allowed_sectors:
            return False, f"sector {inferred_sector} not allowed"

        if _is_blocked_structural_market(sig.event_group):
            return False, "event_group in structural blocklist"

        cheap_market = market_map.get(_normalize_ticker(sig.cheap_ticker))
        expensive_market = market_map.get(_normalize_ticker(sig.expensive_ticker))

        if cheap_market is None:
            return False, "cheap leg market not in open_markets"
        if expensive_market is None:
            return False, "expensive leg market not in open_markets"

        if _is_blocked_structural_market(sig.cheap_ticker):
            return False, "cheap ticker in structural blocklist"
        if _is_blocked_structural_market(sig.expensive_ticker):
            return False, "expensive ticker in structural blocklist"

        cheap_ok, cheap_reason = self._passes_market_quality_gate(
            cheap_market, inferred_sector,
        )
        if not cheap_ok:
            return False, f"cheap leg quality gate: {cheap_reason}"

        expensive_ok, expensive_reason = self._passes_market_quality_gate(
            expensive_market, inferred_sector,
        )
        if not expensive_ok:
            return False, f"expensive leg quality gate: {expensive_reason}"

        if _recently_traded_in_db(sig.cheap_ticker, within_sec=RECENT_TRADE_WINDOW_SEC):
            return False, "cheap ticker in cooldown"
        if _recently_traded_in_db(sig.expensive_ticker, within_sec=RECENT_TRADE_WINDOW_SEC):
            return False, "expensive ticker in cooldown"

        if self.circuit_breaker.is_halted():
            return False, "circuit breaker halted"
        if not self.risk_manager.can_trade():
            return False, "risk manager halted"

        cap = sector_loss_cap(inferred_sector)
        if cap == 0.0:
            return False, "sector disabled (zero loss cap)"

        with self._sector_pnl_lock:
            sector_loss_today = self._sector_daily_pnl.get(inferred_sector, 0.0)
        if sector_loss_today <= -cap:
            return False, f"sector at daily loss cap ({sector_loss_today:.2f})"

        return True, ""

    def _execute_correlation_pair(self, sig, market_map: dict) -> None:
        """Execute a correlation pair atomically.

        If cheap leg fills but expensive leg fails, IMMEDIATELY close the
        cheap leg at market to avoid holding a naked directional bet.
        """
        pair_uuid = str(uuid.uuid4())

        inferred_sector = _infer_sector_from_ticker(sig.event_group)
        cheap_market = market_map.get(_normalize_ticker(sig.cheap_ticker))
        expensive_market = market_map.get(_normalize_ticker(sig.expensive_ticker))

        cheap_consensus = ConsensusResult(
            ticker=sig.cheap_ticker, signals=[], execute=True,
            direction=sig.cheap_direction,
            avg_prob=sig.cheap_price_cents / 100.0,
            avg_edge=sig.divergence_cents / 100.0 / 2.0,
            avg_confidence=sig.confidence,
            reject_reason="",
        )

        expensive_consensus = ConsensusResult(
            ticker=sig.expensive_ticker, signals=[], execute=True,
            direction=sig.expensive_direction,
            avg_prob=1.0 - (sig.expensive_price_cents / 100.0),
            avg_edge=sig.divergence_cents / 100.0 / 2.0,
            avg_confidence=sig.confidence,
            reject_reason="",
        )

        logger.info(
            "[CORR-PAIR] %s START | sector=%s | pair_uuid=%s",
            sig.event_group, inferred_sector, pair_uuid,
        )
        logger.info(
            "[CORR-PAIR]   cheap:     %s YES @%d cents",
            sig.cheap_ticker, sig.cheap_price_cents,
        )
        logger.info(
            "[CORR-PAIR]   expensive: %s NO  @%d cents (buy-NO price=%d cents)",
            sig.expensive_ticker, sig.expensive_price_cents,
            100 - sig.expensive_price_cents,
        )

        cheap_leg_id = None
        expensive_leg_id = None
        try:
            cheap_leg_id = log_correlation_leg(
                event_group=sig.event_group,
                leg_role="cheap",
                ticker=sig.cheap_ticker,
                direction=sig.cheap_direction,
                leg_price_cents=sig.cheap_price_cents,
                pair_divergence_cents=sig.divergence_cents,
                sector=inferred_sector,
                confidence=sig.confidence,
                pair_uuid=pair_uuid,
            )
            expensive_leg_id = log_correlation_leg(
                event_group=sig.event_group,
                leg_role="expensive",
                ticker=sig.expensive_ticker,
                direction=sig.expensive_direction,
                leg_price_cents=sig.expensive_price_cents,
                pair_divergence_cents=sig.divergence_cents,
                sector=inferred_sector,
                confidence=sig.confidence,
                pair_uuid=pair_uuid,
            )
        except Exception as e:
            logger.error("[CORR-PAIR] %s leg logging failed: %s", sig.event_group, e)
            return

        cheap_trade_id = self._execute_trade(
            ticker=sig.cheap_ticker,
            title=f"[CORR] {sig.event_group} cheap",
            yes_price=sig.cheap_price_cents,
            consensus=cheap_consensus,
            lead_sector=inferred_sector,
            arb=None,
            source="correlation",
            fade=False,
            market=cheap_market,
        )

        if cheap_trade_id is None:
            logger.info(
                "[CORR-PAIR] %s ABORTED at cheap leg — no trades placed (clean)",
                sig.event_group,
            )
            self._gate_stats['correlation_pair_aborted_cheap'] += 1
            return

        try:
            if cheap_leg_id is not None:
                link_correlation_leg_to_trade(cheap_leg_id, cheap_trade_id)
        except Exception as e:
            logger.warning(
                "[CORR-PAIR] cheap leg link failed leg=%s trade=%d: %s",
                cheap_leg_id, cheap_trade_id, e,
            )

        expensive_trade_id = self._execute_trade(
            ticker=sig.expensive_ticker,
            title=f"[CORR] {sig.event_group} expensive",
            yes_price=sig.expensive_price_cents,
            consensus=expensive_consensus,
            lead_sector=inferred_sector,
            arb=None,
            source="correlation",
            fade=False,
            market=expensive_market,
        )

        if expensive_trade_id is None:
            logger.error(
                "[CORR-PAIR] %s BROKEN — expensive leg rejected. Rolling back cheap leg.",
                sig.event_group,
            )
            self._rollback_cheap_leg(
                pair_uuid=pair_uuid,
                cheap_trade_id=cheap_trade_id,
                cheap_ticker=sig.cheap_ticker,
                cheap_market=cheap_market,
                event_group=sig.event_group,
            )
            self._gate_stats['correlation_pair_rollback_expensive'] += 1
            return

        try:
            if expensive_leg_id is not None:
                link_correlation_leg_to_trade(expensive_leg_id, expensive_trade_id)
        except Exception as e:
            logger.warning(
                "[CORR-PAIR] expensive leg link failed leg=%s trade=%d: %s",
                expensive_leg_id, expensive_trade_id, e,
            )

        logger.info(
            "[CORR-PAIR] %s SUCCESS | cheap_trade=%d expensive_trade=%d | pair_uuid=%s",
            sig.event_group, cheap_trade_id, expensive_trade_id, pair_uuid,
        )
        self._gate_stats['correlation_pairs_placed'] += 1

    def _rollback_cheap_leg(
        self,
        pair_uuid: str,
        cheap_trade_id: int,
        cheap_ticker: str,
        cheap_market: dict,
        event_group: str,
    ) -> None:
        """Close a cheap leg at market when the expensive leg rejected.

        In DEMO_MODE, logs only (no real API call). In live mode, sells
        the contracts back at the current best bid, eating up to ~5 cents of
        spread to ensure the position doesn't stay naked.
        """
        logger.warning(
            "[CORR-ROLLBACK] pair=%s | closing cheap leg trade_id=%d ticker=%s",
            pair_uuid, cheap_trade_id, cheap_ticker,
        )

        try:
            p = _ph()
            pool = _get_pool()
            if pool:
                conn = pool.getconn()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        f"""
                        UPDATE correlation_legs
                        SET notes = COALESCE(notes, '') || ' ROLLED_BACK_AT ' || {p}
                        WHERE pair_uuid = {p}
                        """,
                        (datetime.now(timezone.utc).isoformat(), pair_uuid),
                    )
                    conn.commit()
                finally:
                    pool.putconn(conn)
        except Exception as e:
            logger.debug("[CORR-ROLLBACK] could not update notes: %s", e)

        if DEMO_MODE:
            logger.info(
                "[CORR-ROLLBACK] DEMO mode — no real close needed for trade_id=%d",
                cheap_trade_id,
            )
            return

        p = _ph()
        try:
            trade_rows = _query_db(
                f"SELECT direction, contracts, yes_price_cents FROM trades WHERE id = {p}",
                (cheap_trade_id,),
            )
            if not trade_rows:
                logger.error(
                    "[CORR-ROLLBACK] cannot find trade_id=%d!", cheap_trade_id,
                )
                return
            trade = trade_rows[0]
            direction = str(trade["direction"]).upper()
            contracts = int(trade["contracts"])
        except Exception as e:
            logger.error(
                "[CORR-ROLLBACK] failed to look up trade %d: %s",
                cheap_trade_id, e,
            )
            return

        close_side = direction.lower()

        try:
            bid = float(cheap_market.get("yes_bid_dollars") or 0) * 100
            ask = float(cheap_market.get("yes_ask_dollars") or 0) * 100
            if direction == "YES":
                close_price = max(1, int(round(bid - 5))) if bid > 0 else 1
            else:
                no_bid = 100 - ask
                close_price = max(1, int(round(no_bid - 5))) if ask > 0 else 1
        except (TypeError, ValueError):
            close_price = 1

        try:
            close_client_order_id = f"rollback-{uuid.uuid4()}"
            order_resp = self.client.place_order(
                ticker=cheap_ticker,
                side=close_side,
                count=contracts,
                yes_price=close_price,
                action="sell",
                client_order_id=close_client_order_id,
            )
            close_order_id = order_resp.get("order", {}).get("order_id", close_client_order_id)
            logger.warning(
                "[CORR-ROLLBACK] CLOSED cheap leg trade_id=%d | ticker=%s | close_order=%s | price=%d cents",
                cheap_trade_id, cheap_ticker, close_order_id, close_price,
            )
        except Exception as e:
            logger.error(
                "[CORR-ROLLBACK] FAILED to close cheap leg trade_id=%d ticker=%s: %s",
                cheap_trade_id, cheap_ticker, e,
            )
            logger.error(
                "[CORR-ROLLBACK] MANUAL INTERVENTION REQUIRED: close %s manually on Kalshi",
                cheap_ticker,
            )

    # ── Resolution timing ──────────────────────────────────────────────────────

    def _execute_resolution_timing(self, open_markets: list[dict]) -> None:
        market_map = {
            _normalize_ticker(m.get("ticker", "")): m
            for m in open_markets
        }

        signals = self.res_timer.scan(
            open_markets, self._last_bot_probs, self._last_bot_sectors,
        )

        for sig in signals:
            if self.circuit_breaker.is_halted():
                break

            if _is_blocked_structural_market(sig.ticker):
                continue

            market = market_map.get(_normalize_ticker(sig.ticker))
            ok, reason = self._passes_market_quality_gate(market, sig.sector)
            if not ok:
                continue

            mock_consensus = ConsensusResult(
                ticker=sig.ticker, signals=[], execute=True,
                direction=sig.direction,
                avg_prob=sig.our_prob,
                avg_edge=abs(sig.our_prob - sig.yes_price_cents / 100),
                avg_confidence=sig.confidence,
                reject_reason="",
            )

            if abs(sig.our_prob - sig.yes_price_cents / 100) < 0.001:
                continue
            if abs((sig.our_prob + sig.yes_price_cents / 100) - 1.0) < 0.001:
                continue

            if not _recent_main_signal_exists(sig.ticker, source="restime", within_minutes=30):
                try:
                    log_main_signal(
                        ticker=sig.ticker,
                        sector=sig.sector,
                        our_prob=sig.our_prob,
                        market_prob=sig.yes_price_cents / 100,
                        edge=mock_consensus.avg_edge,
                        confidence=mock_consensus.avg_confidence,
                        direction=mock_consensus.direction,
                        source="restime",
                    )
                except Exception as e:
                    logger.warning("log_main_signal failed for restime %s: %s", sig.ticker, e)

            self._execute_trade(
                ticker=sig.ticker,
                title=f"[RESTIME] {sig.ticker}",
                yes_price=sig.yes_price_cents,
                consensus=mock_consensus,
                lead_sector=sig.sector,
                arb=None,
                source="restime",
                fade=False,
                market=market,
            )

    # ── Scan loop ──────────────────────────────────────────────────────────────

    def scan_markets(self) -> None:
        logger.info("=== SCAN START ===")

        self._gate_stats = defaultdict(int)

        _synced = self.circuit_breaker.sync_bankroll()
        self._scan_bankroll = _synced if _synced > 0 else BANKROLL
        self._scan_summary = self.circuit_breaker.daily_summary()
        self.bankroll = self._scan_bankroll

        self._rebuild_exposure()

        self.risk_manager.update_daily_pnl(
            self._scan_summary.get("realized_pnl", 0.0)
        )
        self.correlation_tracker.reset()

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

        self._log_gate_stats()

        logger.info("=== SCAN END ===")

    def _log_gate_stats(self) -> None:
        stats = dict(self._gate_stats)
        total = stats.get('markets_scanned', 0)

        if total == 0:
            logger.info("[GATE STATS] No markets scanned")
            return

        ordering = [
            'markets_scanned',
            'rejected_no_price',
            'rejected_extreme_price',
            'rejected_fm_disabled',
            'rejected_liquidity',
            'reached_bot_evaluation',
            'rejected_no_relevant_bot',
            'rejected_bots_returned_none',
            'reached_consensus',
            'rejected_consensus_edge_floor',
            'rejected_consensus_edge_ceiling',
            'rejected_consensus_confidence',
            'rejected_consensus_direction_disagree',
            'rejected_consensus_direction_filter',
            'rejected_consensus_other',
            'passed_consensus',
            'rejected_sharp_veto',
            'rejected_arb_gate',
            'rejected_signal_shape',
            'main_signals_logged',
            'rejected_cooldown',
            'rejected_circuit_breaker',
            'rejected_risk_halt',
            'rejected_sector_disabled',
            'rejected_sector_loss_cap',
            'rejected_pinnacle',
            'rejected_lineup',
            'rejected_sizing_zero',
            'rejected_risk_pretrade',
            'rejected_correlation_discount',
            'rejected_order_error',
            'rejected_trade_dedupe',
            'trades_executed_main',
            'trades_executed_fade',
            'trades_executed_restime',
            'trades_executed_correlation',
            'correlation_pair_aborted_cheap',
            'correlation_pair_rollback_expensive',
            'correlation_pairs_placed',
        ]

        logger.info("[GATE STATS] ===== main-path pipeline =====")
        for key in ordering:
            if key in stats and stats[key] > 0:
                pct = 100.0 * stats[key] / total if total > 0 else 0
                logger.info("[GATE STATS]   %-42s %6d  (%.1f%%)", key, stats[key], pct)

        unexpected = set(stats.keys()) - set(ordering)
        for key in sorted(unexpected):
            logger.info("[GATE STATS]   %-42s %6d  (unexpected)", key, stats[key])

    # ── Nightly retrain ────────────────────────────────────────────────────────

    def retrain_models(self) -> None:
        logger.info("=== Nightly retrain ===")

        n_resolved = self._ingest_resolved_markets()
        logger.info("Ingested %d new outcomes", n_resolved)

        n_patterns = self.res_timer.rebuild_patterns()
        logger.info("Rebuilt %d resolution timing patterns", n_patterns)

        for bot in self.bots:
            try:
                stats = bot.refresh_calibration()
                logger.info("[%s] calibration: %s", bot.sector_name, stats)
                bot.save_model()
            except Exception as e:
                logger.warning("[%s] retrain failed: %s", bot.sector_name, e)

        _synced = self.circuit_breaker.sync_bankroll()
        live_bankroll = _synced if _synced > 0 else BANKROLL
        self.bankroll = live_bankroll
        self._scan_bankroll = live_bankroll
        self._scan_summary = {}

        self._exposure = {b.sector_name: 0.0 for b in self.bots}
        with self._sector_pnl_lock:
            self._sector_daily_pnl = {b.sector_name: 0.0 for b in self.bots}
            self._open_provisionals = {}

        self._last_bot_probs.clear()
        self._last_bot_sectors.clear()
        self._resolved_count_cache.clear()
        self._resolved_count_cache_ts = 0.0

        self.risk_manager = RiskManager(
            bankroll=live_bankroll,
            max_daily_drawdown_pct=0.08,
            max_player_exposure_pct=0.03,
            max_single_trade_pct=0.02,
            max_sector_exposure_pct=0.07,
            shadow_hours_required=72,
            enforce_shadow=(not DEMO_MODE and os.getenv("ENFORCE_SHADOW_MODE", "false").lower() == "true"),
        )
        if self.risk_manager.enforce_shadow:
            self.risk_manager.start_shadow()

        self.correlation_tracker.reset()

        for stat in self.brier_tracker.all_stats():
            if stat:
                logger.info(
                    "[BRIER] %s: %.4f (n=%d trend=%s x%.1f)",
                    stat["prop_code"], stat["brier"] or 0,
                    stat["count"], stat["trend"], stat["multiplier"],
                )

        logger.info("=== Retrain complete ===")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v20.3 | DEMO=%s | bankroll=$%.2f | arb_mode=%s | FM_DISABLED=%s",
            DEMO_MODE, self.bankroll, self.arb._mode, FINANCIAL_MARKETS_DISABLED,
        )
        init_db()

        try:
            self.news.start_background_poller()
        except Exception as e:
            logger.warning("NewsSignal poller failed to start: %s", e)

        import random
        jitter = random.randint(-45, 45)
        schedule.every(SCAN_INTERVAL_SEC + jitter).seconds.do(self.scan_markets)
        schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(self.retrain_models)
        schedule.every(2).hours.do(self._run_ingestion_thread)

        self._run_ingestion_thread()

        logger.info("Waiting 30s before first scan...")
        time.sleep(30)

        self.scan_markets()
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    FlywheelOrchestrator().run()