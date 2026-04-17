"""
orchestrator.py  (v19.27 — Signal logging for all trade paths)

Changes vs v19.26:
  1. _execute_fades: Now calls log_signal() before _execute_trade().
  2. _execute_correlations: Now calls log_signal() for each leg before _execute_trade().
  3. _execute_resolution_timing: Now calls log_signal() before _execute_trade().
  
  BUG FIX: Resolution ingestion matches against the `signals` table, but fade/
  correlation/restime trades bypassed signal logging — they called _execute_trade()
  directly. This caused 174 trades with 0 resolved outcomes because there were no
  signals to match against. Now all trade paths log signals properly.

Changes vs v19.25:
  1. Limit order execution: Instead of market orders, place limits at mid
     and reprice if not filled. Saves 2-4¢ per trade on average.
  2. Early exit manager: Monitor open positions and exit when edge evaporates,
     stop loss triggers, or take profit threshold hit. Frees capital for
     better opportunities.

Changes vs v19.24:
  1. Lineup checker: Before MLB/NBA player prop trades, verify the player is
     actually in the starting lineup (MLB) or not injured/out (NBA). Skip if
     player is scratched, on IL, or listed as OUT/DOUBTFUL.
  2. Weather ensemble: Cross-check NOAA with Open-Meteo. When models agree
     (within 2°F) → boost confidence. When they diverge (>5°F) → reduce
     confidence. Integrated into weather sector trades.

Changes vs v19.23:
  1. Pinnacle sharp line check: Before executing sports trades, compare our
     probability to Pinnacle's line via Odds API. Skip if divergence > 10%,
     reduce confidence if > 5%, boost if aligned within 3%.
  2. Integrated into _execute_trade for MLB/NBA player props.

Changes vs v19.22:
  1. Liquidity filter: Skip thin/illiquid markets (spread > 8¢ or volume < 100).
     Markets failing liquidity check are logged and skipped before bot evaluation.
  2. Time-decay Kelly: Scale down position size as expiry approaches.
     - 24+ hours out: 100% Kelly
     - 12 hours out: ~75%
     - 6 hours out: ~55%
     - 1 hour out: 25%
     - <15 min out: 0% (no bet)
  3. Enhanced correlation tracker: Detects same-game/same-player props and
     discounts Kelly by 1/sqrt(1 + prior_trades). Replaces the simple
     _scan_player_trades dict with full ticker parsing and group tracking.

Changes vs v19.21:
  1. _evaluate_market: Replaced in-memory dedupe guard with database check.
     In-memory dict (_last_bot_probs) reset on every container restart,
     causing 10+ duplicate signals per ticker. Now checks signals table
     directly before inserting.
  2. _signal_exists_in_db: New helper function for dedupe check.

Changes vs v19.20:
  1. _execute_correlations: Added sports-only sector gate. Politics/economics
     ladder markets (KXTRUMPACT thresholds, KXUSPSPEND thresholds, KXHORMUZNORM
     dates) have legitimate price spreads between rungs — not mispricings.
     Correlation arb only makes sense for sports spread markets.

Changes vs v19.19:
  1. _TICKER_SECTOR_MAP: Added missing politics prefixes that were falling
     back to economics: kxtrumpact, kxtrumptime, kxmamdanieo, kxleave,
     kxhormuz, kxca14s, kxpressbriefing.
  2. _SPORTS_PREFIXES: Added random leagues that were leaking to economics
     via correlation engine: kxchnsl, kxballerleague, kxapfddh, kxr6,
     kxhnl, kxelh.
  3. _TICKER_SECTOR_MAP sports tuple: Synced with _SPORTS_PREFIXES additions.

Changes vs v19.18:
  1. _ingest_resolved_markets: Added NBA player prop outcome recording.
     Calls record_nba_outcome() for KXNBAPTS/REB/AST/3PT/STL/BLK tickers
     so resolved NBA props feed the logreg training pipeline (mirrors MLB).

Changes vs v19.17:
  1. _evaluate_market: Added `if not bot.is_relevant(market): continue` gate
     before bot.evaluate(). Fixes news-spike path in BaseBot.evaluate()
     bypassing sector classification — weather bot was evaluating politics,
     economics, entertainment markets at 0.920 conf via news velocity signal.
  2. _SPORTS_PREFIXES: synced with sector_bots.py v11.8 (~40 missing prefixes
     added). Fixes misclassified sector P&L on resolution ingestion.
  3. _TICKER_SECTOR_MAP: synced with all sector prefixes for correct sector
     inference during resolution ingestion.

Changes vs v19.16:
  1. RiskManager: drawdown kill switch (-20%), per-player cap (5%),
     per-trade cap (3%), per-sector cap (30%). Checked before every trade.
  2. Same-game prop correlation: tracks trades per player+game within a scan,
     downscales Kelly by 1/sqrt(n) for correlated bets.
  3. BrierTracker: records per-prop outcomes on resolution, persists to Supabase.
  4. All v19.16 changes carried forward intact.
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
from shared.correlation_tracker import CorrelationTracker  # v19.23: Enhanced correlation
from shared.early_exit_manager import EarlyExitManager  # v19.26: Exit when edge evaporates
from shared.fade_scanner import FadeScanner
from shared.kalshi_client import KalshiClient
from shared.kelly_sizer import kelly_stake, no_kelly_stake  # v3: time-decay built-in
from shared.limit_order_manager import LimitOrderManager  # v19.26: Smart limit orders
from shared.lineup_checker import LineupChecker  # v19.25: Verify player in lineup
from shared.liquidity_filter import LiquidityFilter  # v19.23: Liquidity filter
from shared.news_signal import NewsSignal
from shared.pinnacle_reference import PinnacleReference  # v19.24: Sharp line comparison
from shared.resolution_timer import ResolutionTimer
from shared.sharp_detector import SharpDetector
from shared.weather_ensemble import WeatherEnsemble  # v19.25: NOAA + Open-Meteo
from bots.sector_bots import all_bots

# v19.17: Going-live safety + tracking
import math
from shared.risk_manager import RiskManager
from shared.brier_tracker import BrierTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

# v19.18d: Reduce log volume to stay under Railway's 500 logs/sec limit.
# sharp_detector emits an INFO [SHARP] line for every market with a spread,
# generating 200+ lines/scan. MLB stats fetcher logs every opposing pitcher
# lookup. Both are noise at INFO — promote to WARNING so only actionable
# SHARP VETO lines (logged in orchestrator at INFO) survive.
logging.getLogger("shared.sharp_detector").setLevel(logging.WARNING)
logging.getLogger("shared.mlb_stats_fetcher").setLevel(logging.WARNING)
logging.getLogger("shared.nba_stats_fetcher").setLevel(logging.WARNING)

# ── Duplicate trade cooldown ───────────────────────────────────────────────────
RECENT_TRADE_WINDOW_SEC = 300  # 5 minutes

# ── Ticker normalization ───────────────────────────────────────────────────────
_UUID_SUFFIX_RE = re.compile(r'-S[0-9A-Fa-f]{4,}-[0-9A-Fa-f]+$')


def _normalize_ticker(ticker: str) -> str:
    """Strip Kalshi UUID suffix from multivariate market tickers."""
    return _UUID_SUFFIX_RE.sub('', ticker)


# ── Sports prefix guard (v19.20: added random leagues) ─────────────────────────
_SPORTS_PREFIXES = (
    # MVE family
    "kxmve", "kxmvecross", "kxmvecrosscategory", "kxmvecrosscat",
    "kxmvesports", "kxmvesportsmulti", "kxmvesportsmultigame",
    "kxmvecbchampionship",
    # US major sports
    "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
    "kxufc", "kxncaa", "kxcbb", "kxcfb", "kxnascar", "kxgolf",
    # Tennis
    "kxatp", "kxwta", "kxtennis", "kxabagame",
    # Golf
    "kxpga", "kxowga",
    # International soccer
    "kxepl", "kxsoccer",
    "kxarg", "kxlali", "kxbund", "kxseri", "kxliga", "kxligu",
    "kxbras", "kxswis", "kxbelg", "kxecul", "kxpslg", "kxsaud",
    "kxjlea", "kxuclt", "kxucl", "kxfifa", "kxalea", "kxafl",
    "kxconmebol", "kxchll", "kxargpremdiv",
    "kxsuperlig", "kxegypl", "kxscottishprem",
    "kxuel", "kxeredivisie", "kxallsvenskan",
    # Cricket
    "kxcba", "kxcricket", "kxt20", "kxipl", "kxbbl",
    # International basketball
    "kxfiba", "kxacbg", "kxvtbg", "kxbalg", "kxbbse", "kxnpbg",
    "kxeuroleague",
    # Hockey
    "kxahlg", "kxshl",
    # Track & field
    "kxdima",
    # Combat sports
    "kxboxing", "kxwwe",
    # Racing
    "kxf1", "kxmotogp",
    # Olympic / other
    "kxolympic", "kxrugby", "kxthail", "kxsl", "kxufl",
    "kxintl", "kxkf", "kxnextag",
    # Esports
    "kxow", "kxvalorant", "kxlol", "kxleague",
    "kxrl", "kxrocketleague",
    "kxcsgo", "kxcs2", "kxdota", "kxintlf",
    "kxapex", "kxfort", "kxhalo", "kxsc2",
    "kxesport", "kxegypt", "kxvenf", "kxr6g",
    # Entertainment / unmodelable (block from non-sports)
    "kxsurv",
    # Legacy aliases
    "kxcbagame", "kxacbgame", "kxaleaguegame", "kxaleague",
    # ITF tennis
    "kxitf",
    # Ekstraklasa / Uruguayan
    "kxekstraklasa", "kxurypd",
    # v11.8d: new leagues 2026-04-12
    "kxdel", "kxkhl", "kxjbleague", "kxkbl", "kxkleague",
    # Entertainment (block from sports path, route to GlobalEventsBot)
    "kxartiststream",
    # v19.20: Random leagues that were leaking to economics via correlation
    "kxchnsl",        # Chinese Super League
    "kxballerleague", # Baller League
    "kxapfddh",       # Unknown league
    "kxr6",           # Rainbow Six esports (KXR6MAP, KXR6GAME)
    "kxhnl",          # HNL league
    "kxelh",          # ELH league
    "kxtabletennis",  # Table tennis
    "kxkbogame",      # Korean baseball
    "kxlnbelite",     # French basketball
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


# ── Sector inference from ticker (v19.20: added politics + random leagues) ────
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
      # v19.20: Random leagues that were falling back to economics
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
      # v19.20: Additional politics prefixes (were falling back to economics)
      "kxtrumpact",      # Trump executive actions
      "kxtrumptime",     # Trump timing markets
      "kxmamdanieo",     # Mamdani EO
      "kxleave",         # KXLEAVECHERFILUS, KXLEAVEGONZALES, KXLEAVEMILLS
      "kxhormuz",        # Strait of Hormuz geopolitics (KXHORMUZNORM)
      "kxca14s",         # CA-14 special election (KXCA14SWINNER)
      "kxpressbriefing", # Press briefing counts
      ), "politics"),
    (("kxhurr", "kxtemp", "kxrain", "kxsnow", "kxweather",
      "kxnoaa", "kxclimate", "kxlowt", "kxchll", "kxdens",
      "kxhight",  # v19.22: Was missing — KXHIGHTDAL, KXHIGHTDC, etc.
      ), "weather"),
    (("kxai", "kxtech", "kxfed", "kxcpi", "kxgdp",
      "kxjobs", "kxrate", "kxinfl", "kxwti", "kxpayroll",
      "kxhighinfl", "kxuspspend",
      ), "economics"),
]


def _infer_sector_from_ticker(ticker: str) -> str:
    tk = ticker.lower()
    for prefixes, sector in _TICKER_SECTOR_MAP:
        if any(tk.startswith(p) for p in prefixes):
            return sector
    return "economics"


# ── v19.22: Database-side signal dedupe ────────────────────────────────────────
def _signal_exists_in_db(ticker: str) -> bool:
    """Check if a signal already exists for this ticker.
    
    v19.22: Database-side dedupe guard. Survives container restarts unlike
    the in-memory _last_bot_probs dict which reset on every Railway deploy,
    causing 10+ duplicate signals per ticker and poisoning the Bayesian model.
    """
    p = _ph()
    rows = _query_signals(
        f"SELECT 1 FROM signals WHERE ticker = {p} LIMIT 1",
        (ticker,),
    )
    return len(rows) > 0


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


def _query_signals(sql: str, params: tuple = (), _retries: int = 2) -> list:
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
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                pool.putconn(conn)
                if _retries > 0:
                    logger.warning(
                        "_query_signals SSL/connection error, retrying (%d left): %s",
                        _retries, e,
                    )
                    global _pg_pool
                    try:
                        _pg_pool.closeall()
                    except Exception:
                        pass
                    _pg_pool = None
                    return _query_signals(sql, params, _retries=_retries - 1)
                raise
            finally:
                try:
                    pool.putconn(conn)
                except Exception:
                    pass
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


# ── v19.19: NBA prop codes for outcome recording ───────────────────────────────
_NBA_PROP_CODES = ("PTS", "REB", "AST", "3PT", "STL", "BLK")


def _is_nba_player_prop(ticker: str) -> bool:
    """Return True if ticker is an NBA player prop market."""
    tk_upper = ticker.upper()
    return any(tk_upper.startswith(f"KXNBA{code}") for code in _NBA_PROP_CODES)


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
        _synced = self.circuit_breaker.sync_bankroll()
        self.bankroll = _synced if _synced > 0 else BANKROLL

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

        self._sector_pnl_lock    = threading.Lock()
        self._ingestion_lock     = threading.Lock()
        self._last_bot_probs:    dict[str, float] = {}
        self._last_bot_sectors:  dict[str, str]   = {}

        self._resolved_count_cache:    dict[str, int] = {}
        self._resolved_count_cache_ts: float          = 0.0

        self._scan_bankroll: float = self.bankroll
        self._scan_summary:  dict  = {}

        self._recently_traded: dict[str, float] = {}

        self._open_provisionals: dict[str, float] = {}

        # v19.17: Risk manager + Brier tracker + correlation tracking
        self.risk_manager = RiskManager(bankroll=BANKROLL)
        self.brier_tracker = BrierTracker.from_supabase(os.getenv("DATABASE_URL", ""))
        self._scan_player_trades: dict[str, int] = {}  # Legacy, kept for backward compat

        # v19.23: Liquidity filter + enhanced correlation tracker
        self.liquidity_filter = LiquidityFilter(enabled=True)
        self.correlation_tracker = CorrelationTracker()
        
        # v19.24: Pinnacle sharp line reference
        self.pinnacle = PinnacleReference(
            api_key=os.getenv("ODDS_API_KEY", ""),
            enabled=True,
        )
        
        # v19.25: Lineup checker + weather ensemble
        self.lineup_checker = LineupChecker(enabled=True)
        self.weather_ensemble = WeatherEnsemble(enabled=True)
        
        # v19.26: Limit order manager + early exit manager
        self.limit_order_mgr = LimitOrderManager(
            client=self.client,
            enabled=not DEMO_MODE,  # Only use in live mode
        )
        self.early_exit_mgr = EarlyExitManager(
            client=self.client,
            enabled=not DEMO_MODE,  # Only use in live mode
        )

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
            resolved_markets = self.client.get_resolved_markets(
                min_settled_ts = min_ts,
                max_pages    = 10 if min_ts else 350,
            )
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

        bayesian_updated_events: set[str] = set()

        for market in resolved_markets:
            if self.circuit_breaker.is_halted():
                logger.warning(
                    "Circuit breaker active — stopping ingestion early to prevent "
                    "further phantom P&L accumulation"
                )
                break

            ticker = _normalize_ticker(market.get("ticker", ""))
            result = market.get("result", "")

            result = result.lower() if result else ""
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

            event_key = _normalize_ticker(
                market.get("event_ticker", "") or ticker.rsplit("-", 1)[0]
            )
            allow_bayesian_update = event_key not in bayesian_updated_events

            for bot in self.bots:
                if not bot.is_relevant(market):
                    continue
                if not allow_bayesian_update:
                    logger.debug(
                        "[%s] Skipping duplicate Bayesian update for event %s (ticker=%s)",
                        bot.sector_name, event_key, ticker,
                    )
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

            if allow_bayesian_update:
                bayesian_updated_events.add(event_key)

            _update_signal_brier(ticker, resolved_yes)

            # v19.17: Brier tracker per-prop recording (MLB)
            if sig_rows:
                _prop_code = None
                for _code in ("HIT", "TB", "HRR", "HR", "KS"):
                    if f"KXMLB{_code}" in ticker.upper():
                        _prop_code = _code
                        break
                if _prop_code:
                    self.brier_tracker.record(
                        _prop_code, float(sig_rows[0]["our_prob"]),
                        1 if resolved_yes else 0,
                    )

            # v19.19: NBA player prop outcome recording for logreg training
            if _is_nba_player_prop(ticker):
                try:
                    from shared.nba_logistic_model import record_outcome as record_nba_outcome
                    record_nba_outcome(ticker, 1 if resolved_yes else 0)
                    logger.debug("[NBA LOGREG] Recorded outcome: %s → %s", ticker, result.upper())
                except Exception as e:
                    logger.warning("[NBA LOGREG] record_outcome failed for %s: %s", ticker, e)

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

            if primary_trade_id is None:
                logger.debug(
                    "[OUTCOME SKIP] %s — signal exists but no trade, skipping outcome log",
                    ticker,
                )
                processed_this_run.add(ticker)
                continue

            sec_rows = _query_signals(
                f"SELECT sector FROM signals WHERE ticker = {p} LIMIT 1",
                (ticker,),
            )
            sec = sec_rows[0].get("sector", "") if sec_rows else ""
            if not sec:
                sec = _infer_sector_from_ticker(ticker)

            if pnl_usd is not None:
                logger.info(
                    "[P&L] %s resolved %s → $%+.2f (primary_trade_id=%s, all_ids=%s)",
                    ticker, result.upper(), pnl_usd, primary_trade_id, trade_ids,
                )
                self.circuit_breaker.record_pnl(pnl_usd)

                with self._sector_pnl_lock:
                    if sec in self._sector_daily_pnl:
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
                    sector   = sec,
                )
                recorded += 1
            except Exception as e:
                logger.error(
                    "log_outcome failed for %s (trade_id=%s): %s — "
                    "outcome counted but not persisted; will retry next ingestion cycle",
                    ticker, primary_trade_id, e,
                )

        # v19.17: Save Brier tracker state
        if recorded > 0:
            try:
                self.brier_tracker.save_to_supabase(os.getenv("DATABASE_URL", ""))
            except Exception as e:
                logger.warning("Brier tracker save failed: %s", e)

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

        if not hasattr(self, '_debug_logged'):
            logger.info("DEBUG market keys: %s", list(market.keys()))
            logger.info("DEBUG price fields: bid=%s ask=%s last=%s yes_ask=%s",
                market.get("yes_bid"), market.get("yes_ask"),
                market.get("last_price"), market.get("yes_ask_price"))
            logger.info("DEBUG yes_bid_dollars=%s yes_ask_dollars=%s last_price_dollars=%s",
                market.get("yes_bid_dollars"), market.get("yes_ask_dollars"),
                market.get("last_price_dollars"))
            self._debug_logged = True

        if yes_price == 0:
            return
        if yes_price <= 2 or yes_price >= 98:
            return

        # v19.23: Liquidity filter — skip thin/illiquid markets
        # Check before any bot evaluation to save compute
        liq_result = self.liquidity_filter.check(market, sector="unknown")
        if not liq_result.passes:
            logger.debug(
                "LIQUIDITY SKIP %s: %s (spread=%.1f¢ vol=%s)",
                ticker[:40], liq_result.reason,
                liq_result.spread_cents or 0, liq_result.volume_24h,
            )
            return

        self.news.register_market(title)
        sharp_signal = self.sharp.analyze(market)

        # v19.18: Gate bot.evaluate() behind bot.is_relevant()
        # Fixes: news-spike path in BaseBot.evaluate() was bypassing sector
        # classification, causing weather bot to evaluate politics/economics
        # markets at 0.920 conf via news velocity signal.
        signals = []
        for bot in self.bots:
            if not bot.is_relevant(market):
                continue
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

        # v19.18: Track if this is a new ticker we haven't seen before
        _already_seen = ticker in self._last_bot_probs

        self._last_bot_probs[ticker]   = consensus.avg_prob
        self._last_bot_sectors[ticker] = lead_sector

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

        # v19.22: Database-side dedupe — survives container restarts.
        # The in-memory _last_bot_probs dict was resetting on every Railway
        # deploy, causing 10+ duplicate signals per ticker and poisoning
        # the Bayesian model (especially weather).
        if not _already_seen and not _signal_exists_in_db(ticker):
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
            market      = market,  # v19.23: For time-decay Kelly
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
        market:      dict = None,  # v19.23: For time-decay Kelly
    ) -> None:
        now      = time.monotonic()
        last_ts  = self._recently_traded.get(ticker, 0.0)
        elapsed  = now - last_ts
        if elapsed < RECENT_TRADE_WINDOW_SEC:
            logger.info(
                "DUPLICATE GUARD skip %s — last trade %.0fs ago (cooldown=%ds)",
                ticker, elapsed, RECENT_TRADE_WINDOW_SEC,
            )
            return

        if self.circuit_breaker.is_halted():
            logger.warning("CIRCUIT BREAKER HALT — skipping trade for %s", ticker)
            return

        # v19.17: Risk manager master gate
        if not self.risk_manager.can_trade():
            logger.warning("RISK HALT: %s — skipping %s", self.risk_manager.halt_reason, ticker)
            return

        cap = sector_loss_cap(lead_sector)
        if cap == 0.0:
            logger.warning(
                "SECTOR DISABLED: %s cap=0 — skipping %s", lead_sector, ticker
            )
            return

        with self._sector_pnl_lock:
            sector_loss_today = self._sector_daily_pnl.get(lead_sector, 0.0)

        if sector_loss_today <= -cap:
            logger.warning(
                "SECTOR LOSS CAP hit: %s daily_loss=$%.2f cap=$%.2f — skipping %s",
                lead_sector, sector_loss_today, cap, ticker,
            )
            return

        live_bankroll   = self._scan_bankroll
        summary         = self._scan_summary
        daily_pnl       = float(summary.get("realized_pnl", 0.0))
        loss_limit      = live_bankroll * CIRCUIT_BREAKER_PCT
        drawdown_factor = (
            max(0.5, 1.0 + (daily_pnl / loss_limit) * 0.5)
            if daily_pnl < 0 and loss_limit > 0 else 1.0
        )

        resolved_count = self._get_sector_resolved_count(lead_sector)
        kf             = sector_kelly_fraction(lead_sector, resolved_count)
        in_exploration = resolved_count < SECTOR_MIN_RESOLVED.get(lead_sector, 30)

        if in_exploration:
            logger.info(
                "[EXPLORATION] %s resolved=%d — will scale stake to %.0f%% of Kelly",
                lead_sector, resolved_count, (kf / KELLY_FRACTION) * 100,
            )

        # v19.24: Pinnacle sharp line check (sports only)
        pinnacle_adj = 1.0
        if lead_sector == "sports" and not fade:
            sharp_check = self.pinnacle.check_from_kalshi_ticker(
                ticker=ticker,
                our_prob=consensus.avg_prob,
                our_direction=consensus.direction,
            )
            
            if not sharp_check.passes:
                logger.info(
                    "PINNACLE VETO %s: %s (our=%.1f%% sharp=%.1f%% div=%+.1f%%)",
                    ticker[:40], sharp_check.reason,
                    consensus.avg_prob * 100,
                    (sharp_check.sharp_prob or 0) * 100,
                    (sharp_check.divergence or 0) * 100,
                )
                return
            
            pinnacle_adj = sharp_check.confidence_adjustment
            if pinnacle_adj != 1.0:
                logger.info(
                    "[PINNACLE] %s: %s → conf adj %.2f",
                    ticker[:35], sharp_check.reason, pinnacle_adj,
                )

        # v19.25: Lineup checker (sports player props only)
        if lead_sector == "sports" and not fade:
            lineup_check = self.lineup_checker.check_from_kalshi_ticker(ticker)
            
            if not lineup_check.is_active:
                logger.info(
                    "LINEUP VETO %s: %s (status=%s)",
                    ticker[:40], lineup_check.reason, lineup_check.status,
                )
                return
            
            if lineup_check.status == "questionable":
                # Player is questionable — reduce confidence
                pinnacle_adj *= 0.7
                logger.info(
                    "[LINEUP] %s: %s → additional conf penalty",
                    ticker[:35], lineup_check.reason,
                )

        sector_exp = self._exposure.get(lead_sector, 0.0)

        # v19.24: Apply Pinnacle confidence adjustment to drawdown factor
        effective_drawdown = drawdown_factor * pinnacle_adj

        # v19.23: Pass market dict for time-decay Kelly
        if consensus.direction == "YES":
            sizing = kelly_stake(
                prob            = consensus.avg_prob,
                yes_price_cents = yes_price,
                bankroll        = self.bankroll,
                sector          = lead_sector,
                sector_exposure = sector_exp,
                live_bankroll   = live_bankroll,
                drawdown_factor = effective_drawdown,  # v19.24: includes Pinnacle adj
                market          = market,  # v19.23: time-decay
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
                drawdown_factor = effective_drawdown,  # v19.24: includes Pinnacle adj
                market          = market,  # v19.23: time-decay
            )
            side = "no"
        
        # v19.23: Log time-decay if applied
        td_factor = sizing.get("time_decay", 1.0)
        if td_factor < 1.0:
            logger.info(
                "[TIME DECAY] %s: %.0f%% Kelly (expiry approaching)",
                ticker[:40], td_factor * 100,
            )

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

        # v19.17: Risk manager pre-trade checks
        player_key = ticker.split("-")[2] if len(ticker.split("-")) >= 3 and ticker.upper().startswith("KXMLB") else ticker
        ok, reason = self.risk_manager.pre_trade_check(player_key, lead_sector, sizing["dollars"])
        if not ok:
            logger.info("RISK BLOCKED %s: %s", ticker, reason)
            return

        # v19.23: Enhanced correlation tracking (replaces v19.17 simple dict)
        # Detects same-game, same-player props and discounts Kelly accordingly
        corr_result = self.correlation_tracker.get_discount(ticker)
        if corr_result.discount < 1.0:
            old_dollars = sizing["dollars"]
            sizing["dollars"] = round(sizing["dollars"] * corr_result.discount, 2)
            sizing["contracts"] = max(1, int(sizing["contracts"] * corr_result.discount))
            logger.info(
                "[CORRELATION] %s: %s → $%.2f × %.2f = $%.2f",
                ticker[:35], corr_result.rationale,
                old_dollars, corr_result.discount, sizing["dollars"],
            )
            if sizing["contracts"] <= 0:
                logger.info("CORRELATION SKIP %s: discount too steep", ticker[:40])
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
            # v19.26: Use limit order manager for better fills
            try:
                limit_result = self.limit_order_mgr.execute_limit_order(
                    ticker=ticker,
                    side=side,
                    contracts=sizing["contracts"],
                    max_price_cents=yes_price + 2,  # Allow 2¢ slippage
                    urgency="normal",
                )
                
                if limit_result.contracts_filled == 0:
                    logger.warning(
                        "LIMIT ORDER FAILED %s: %s — falling back to market",
                        ticker[:40], limit_result.reason,
                    )
                    # Fall back to direct market order
                    order_resp = self.client.place_order(
                        ticker          = ticker,
                        side            = side,
                        count           = sizing["contracts"],
                        yes_price       = yes_price,
                        client_order_id = client_order_id,
                    )
                    order_id = order_resp.get("order", {}).get("order_id", client_order_id)
                else:
                    order_id = limit_result.order_ids[0] if limit_result.order_ids else client_order_id
                    if limit_result.spread_saved > 0:
                        logger.info(
                            "[LIMIT] %s filled %d/%d @ %d¢ — saved %d¢",
                            ticker[:30], limit_result.contracts_filled,
                            sizing["contracts"], limit_result.fill_price,
                            limit_result.spread_saved,
                        )
                        
            except Exception as e:
                logger.error("Order failed %s: %s", ticker, e)
                return

        self._recently_traded[ticker]   = now
        self._exposure[lead_sector]     = sector_exp + sizing["dollars"]
        self.bankroll                   = live_bankroll

        # v19.23: Record in risk manager + enhanced correlation tracker
        self.risk_manager.record_trade(player_key, lead_sector, sizing["dollars"])
        self.correlation_tracker.record_trade(ticker, sizing["dollars"])

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

            # v19.27: Log signal for fade trade so resolution can match
            if not _signal_exists_in_db(fade.ticker):
                try:
                    log_signal(
                        ticker      = fade.ticker,
                        sector      = sector,
                        our_prob    = mock_consensus.avg_prob,
                        market_prob = fade.yes_price_cents / 100,
                        edge        = mock_consensus.avg_edge,
                        confidence  = mock_consensus.avg_confidence,
                        direction   = mock_consensus.direction,
                    )
                except Exception as e:
                    logger.warning("log_signal failed for fade trade %s: %s", fade.ticker, e)

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

            # v19.21: Only trade sports correlations — politics/economics ladders
            # (KXTRUMPACT thresholds, KXUSPSPEND levels, KXHORMUZNORM dates) have
            # legitimate price spreads between rungs, not mispricings.
            inferred_sector = _infer_sector_from_ticker(sig.event_group)
            if inferred_sector != "sports":
                logger.debug(
                    "CORR SKIP (non-sports sector=%s): %s",
                    inferred_sector, sig.event_group,
                )
                continue

            if _is_blocked_structural_market(sig.event_group):
                logger.info(
                    "CORR BLOCKED (uncalibrated structural market): %s",
                    sig.event_group,
                )
                continue

            # v19.18d: CONMEBOL duplicate fix — track event_group, not just
            # individual tickers. The correlation engine produces multiple
            # divergent pairs per event_group per scan. Without this guard,
            # CONMEBOL groups generated 26 duplicate trades because each pair
            # had a unique ticker that passed the per-ticker duplicate guard.
            corr_key = f"__corr__{sig.event_group}"
            now = time.monotonic()
            last_corr = self._recently_traded.get(corr_key, 0.0)
            if (now - last_corr) < RECENT_TRADE_WINDOW_SEC:
                logger.debug(
                    "CORR DUPLICATE GUARD skip %s — already traded this group",
                    sig.event_group,
                )
                continue
            self._recently_traded[corr_key] = now

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

                # v19.27: Log signal for correlation trade so resolution can match
                if not _signal_exists_in_db(ticker):
                    try:
                        log_signal(
                            ticker      = ticker,
                            sector      = inferred_sector,
                            our_prob    = mock_consensus.avg_prob,
                            market_prob = price / 100,
                            edge        = mock_consensus.avg_edge,
                            confidence  = mock_consensus.avg_confidence,
                            direction   = mock_consensus.direction,
                        )
                    except Exception as e:
                        logger.warning("log_signal failed for corr trade %s: %s", ticker, e)

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

            # v19.27: Log signal for resolution timing trade so resolution can match
            if not _signal_exists_in_db(sig.ticker):
                try:
                    log_signal(
                        ticker      = sig.ticker,
                        sector      = sig.sector,
                        our_prob    = mock_consensus.avg_prob,
                        market_prob = sig.yes_price_cents / 100,
                        edge        = mock_consensus.avg_edge,
                        confidence  = mock_consensus.avg_confidence,
                        direction   = mock_consensus.direction,
                    )
                except Exception as e:
                    logger.warning("log_signal failed for restime trade %s: %s", sig.ticker, e)

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

        _synced = self.circuit_breaker.sync_bankroll()
        self._scan_bankroll = _synced if _synced > 0 else BANKROLL
        self._scan_summary  = self.circuit_breaker.daily_summary()
        self.bankroll       = self._scan_bankroll

        # ── FIX v19.16: Reset exposure each scan ──────────────────────────
        # Previous bug: exposure accumulated across scans and never decreased
        # when trades resolved. After 5 trades ($150 cap / $30 per trade),
        # every subsequent trade sized to zero for the rest of the day.
        # In demo mode there's no real capital at risk. In live mode this
        # should be replaced with real position querying from Kalshi API.
        self._exposure = {b.sector_name: 0.0 for b in self.bots}

        # v19.17: Update risk manager + reset correlation tracking
        self.risk_manager.update_daily_pnl(
            self._scan_summary.get("realized_pnl", 0.0)
        )
        self._scan_player_trades = {}  # Legacy
        
        # v19.23: Reset enhanced correlation tracker
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

        # ── DIAGNOSTIC: price parsing audit ──
        nonzero = 0
        for m in markets[:100]:
            if _parse_yes_price_cents(m) > 0:
                nonzero += 1
        if markets:
            sample = markets[0]
            logger.info(
                "PRICE DIAG: %d/100 markets have nonzero price | "
                "response_price_units=%s | tick_size=%s | "
                "sample yes_bid=%r yes_ask=%r last=%r",
                nonzero,
                sample.get("response_price_units"),
                sample.get("tick_size"),
                sample.get("yes_bid_dollars"),
                sample.get("yes_ask_dollars"),
                sample.get("last_price_dollars"),
            )

       # ── DIAGNOSTIC: bot signal audit ──
        _diag_priced = 0
        _diag_relevant = 0
        for market in markets:
            yp = _parse_yes_price_cents(market)
            if yp > 2 and yp < 98:
                _diag_priced += 1
                for bot in self.bots:
                    if bot.is_relevant(market):
                        _diag_relevant += 1
                        break
            self._evaluate_market(market)
        logger.info(
            "SIGNAL DIAG: total=%d priced=%d relevant=%d",
            len(markets), _diag_priced, _diag_relevant,
        )

        # ── DIAGNOSTIC 2: sample tickers from priced markets ──
        _mve_count = 0
        _extreme_single = 0
        _single_game = []
        for m in markets:
            tk = m.get("ticker", "").lower()
            yp = _parse_yes_price_cents(m)
            if tk.startswith("kxmve"):
                _mve_count += 1
            else:
                if yp > 2 and yp < 98 and len(_single_game) < 5:
                    _single_game.append(f"{m.get('ticker','')[:45]}@{yp}c")
                elif yp <= 2 or yp >= 98:
                    _extreme_single += 1
        logger.info(
            "MARKET DIAG: total=%d mve=%d extreme_single=%d tradeable_single=%s",
            len(markets), _mve_count, _extreme_single, _single_game,
        )

        # ── v19.16: Secondary engines wrapped in try/except ───────────────
        try:
            self._execute_fades(markets)
        except Exception as e:
            logger.error("Fade scanner crashed (non-fatal): %s", e)

        try:
            self._execute_correlations(markets)
        except Exception as e:
            logger.error("Correlation engine crashed (non-fatal): %s", e)

        try:
            self._execute_resolution_timing(markets)
        except Exception as e:
            logger.error("Resolution timing crashed (non-fatal): %s", e)

        # v19.26: Early exit check — close positions when edge evaporates
        try:
            exits = self.early_exit_mgr.check_and_exit()
            if exits:
                for exit in exits:
                    if exit.success:
                        logger.info(
                            "EARLY EXIT %s: %s | entry=%d¢ exit=%d¢ | P&L=$%.2f",
                            exit.reason.value.upper(), exit.ticker[:35],
                            exit.entry_price, exit.exit_price, exit.pnl,
                        )
        except Exception as e:
            logger.error("Early exit check crashed (non-fatal): %s", e)

        with self._sector_pnl_lock:
            sector_pnl_snapshot = dict(self._sector_daily_pnl)

        # v19.17: Log risk manager status
        risk_status = self.risk_manager.status()
        logger.info(
            "Scan complete | daily_pnl=$%.2f trades=%d halted=%s | "
            "risk: %d player exposures, %s | sector_pnl=%s",
            self._scan_summary.get("realized_pnl", 0.0),
            self._scan_summary.get("trade_count", 0),
            self._scan_summary.get("halted", False),
            len(risk_status["player_exposures"]),
            "HALTED" if risk_status["halted"] else "OK",
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

        _synced = self.circuit_breaker.sync_bankroll()
        live_bankroll       = _synced if _synced > 0 else BANKROLL
        self.bankroll       = live_bankroll
        self._scan_bankroll = live_bankroll
        self._scan_summary  = {}
        logger.info("Bankroll synced: $%.2f", live_bankroll)

        self._exposure = {b.sector_name: 0.0 for b in self.bots}

        with self._sector_pnl_lock:
            self._sector_daily_pnl   = {b.sector_name: 0.0 for b in self.bots}
            self._open_provisionals  = {}

        self._last_bot_probs.clear()
        self._last_bot_sectors.clear()
        self._recently_traded.clear()

        self._resolved_count_cache.clear()
        self._resolved_count_cache_ts = 0.0

        # v19.17: Reset risk manager + log Brier stats
        # v19.23: Also reset correlation tracker
        self.risk_manager = RiskManager(bankroll=live_bankroll)
        self.correlation_tracker.reset()
        for stat in self.brier_tracker.all_stats():
            if stat:
                logger.info(
                    "[BRIER] %s: %.4f (%d samples, trend=%s, ×%.1f)",
                    stat["prop_code"], stat["brier"] or 0,
                    stat["count"], stat["trend"], stat["multiplier"],
                )

        logger.info("=== Retrain complete ===")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v19.27 | DEMO=%s | $%.2f | arb_mode=%s",
            DEMO_MODE, self.bankroll, self.arb._mode,
        )
        init_db()

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