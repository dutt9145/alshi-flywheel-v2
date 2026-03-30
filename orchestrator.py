"""
orchestrator.py  (v12 — full feature suite)

Changes vs v11:
  1. CircuitBreaker integrated — daily P&L tracking, halt flag, real-time
     bankroll sync before every trade
  2. SharpDetector integrated — order book analysis before every evaluate;
     trades blocked if sharp money opposes our direction
  3. FadeScanner integrated — runs after every market scan cycle to catch
     closing-time crowding opportunities; executes fades as trades
  4. NewsSignal integrated — background poller started on init; velocity
     features injected into every bot.evaluate() call
  5. CorrelationEngine integrated — runs after every scan to detect
     divergent market pairs and leg into both sides
  6. ResolutionTimer integrated — rebuilds patterns during nightly retrain;
     scans for overdue markets during each cycle
  7. Ingestion thread lock added — prevents concurrent resolution threads
  8. Postgres placeholder consistent — all SQL uses os.getenv("DATABASE_URL")
     check, not self.client.base heuristic
  9. Connection pooling — psycopg2 ThreadedConnectionPool replaces
     per-call psycopg2.connect()
 10. Bayesian models now learn from ALL relevant resolved markets, not
     just ones we signaled (sig_rows guard removed from bot update loop)
 11. log_outcome() only called when we have an actual signal (sig_rows)
     to prevent phantom correct/incorrect metrics
 12. _update_signal_brier() per-row calculation — DB computes brier
     per-row using stored our_prob, not single last-signal value
 13. Nightly retrain rebuilds resolution timing patterns
"""

import logging
import os
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

# ── Connection pool ────────────────────────────────────────────────────────────
_pg_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Lazy-init a psycopg2 ThreadedConnectionPool (Postgres only)."""
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
    """
    Execute a SQL query and return rows as list of dicts.
    Uses connection pool for Postgres, direct connection for SQLite.
    Placeholder selection is consistent: DATABASE_URL env var, not client heuristic.
    """
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


def _calculate_pnl(ticker: str, resolved_yes: bool) -> tuple[float | None, str | None]:
    """
    Aggregate P&L across ALL trades for this ticker (not just the last one).
    Returns (total_pnl_usd, comma-joined trade ids).
    """
    p = _ph()
    trade_rows = _query_signals(
        f"SELECT id, direction, contracts, yes_price_cents FROM trades WHERE ticker = {p}",
        (ticker,),
    )

    if not trade_rows:
        return None, None

    total_pnl  = 0.0
    trade_ids  = []

    for trade in trade_rows:
        trade_id   = str(trade.get("id", ""))
        direction  = str(trade.get("direction", "YES")).upper()
        contracts  = int(trade.get("contracts", 0))
        yes_price  = int(trade.get("yes_price_cents", 50))

        if contracts <= 0:
            continue

        yes_price_frac = yes_price / 100.0
        no_price_frac  = 1.0 - yes_price_frac

        if direction == "YES":
            pnl = contracts * ((1.0 - yes_price_frac) if resolved_yes else (-yes_price_frac))
        else:
            pnl = contracts * (no_price_frac if not resolved_yes else (-yes_price_frac))

        total_pnl += pnl
        trade_ids.append(trade_id)

    if not trade_ids:
        return None, None

    return round(total_pnl, 2), ",".join(trade_ids)


def _update_signal_brier(ticker: str, resolved_yes: bool) -> None:
    """
    Write outcome + brier_score back to ALL signal rows for this ticker.
    brier_score is computed per-row using each row's stored our_prob,
    not a single last-signal value. This is the correct calculation.
    """
    outcome = 1 if resolved_yes else 0
    now     = datetime.now(timezone.utc)
    p       = _ph()

    pool = _get_pool()
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

        # ── Circuit breaker (capital protection) ───────────────────────────
        self.circuit_breaker = CircuitBreaker(
            kalshi_client        = self.client,
            bankroll             = BANKROLL,
            daily_loss_limit_pct = CIRCUIT_BREAKER_PCT,
        )
        self.bankroll = self.circuit_breaker.sync_bankroll()

        # ── Sharp detector ────────────────────────────────────────────────
        self.sharp = SharpDetector(
            kalshi_client            = self.client,
            spread_threshold_pct     = SHARP_SPREAD_THRESHOLD_PCT,
            large_order_contracts    = SHARP_LARGE_ORDER_CONTRACTS,
            volume_spike_multiplier  = SHARP_VOLUME_SPIKE_MULTIPLIER,
        )

        # ── Fade scanner ──────────────────────────────────────────────────
        self.fade_scanner = FadeScanner(
            kalshi_client          = self.client,
            sharp_detector         = self.sharp,
            fade_window_minutes    = FADE_WINDOW_MINUTES,
            fade_threshold_cents   = FADE_THRESHOLD_CENTS,
            fade_min_disagreement  = FADE_MIN_DISAGREEMENT,
        )

        # ── Correlation engine ────────────────────────────────────────────
        self.corr_engine = CorrelationEngine(
            min_divergence_cents = CORR_MIN_DIVERGENCE_CENTS,
            max_group_size       = CORR_MAX_GROUP_SIZE,
        )

        # ── Resolution timer ──────────────────────────────────────────────
        self.res_timer = ResolutionTimer(
            min_overdue_minutes = RESTIME_MIN_OVERDUE_MIN,
            min_prob_to_trade   = RESTIME_MIN_PROB,
            min_sample_count    = RESTIME_MIN_SAMPLES,
        )

        # ── News signal ───────────────────────────────────────────────────
        self.news = NewsSignal(
            api_key                  = NEWSAPI_KEY,
            poll_interval_sec        = NEWS_POLL_INTERVAL_SEC,
            velocity_window_min      = NEWS_VELOCITY_WINDOW_MIN,
            velocity_spike_threshold = NEWS_VELOCITY_SPIKE_THRESHOLD,
        )

        # ── Exposure tracking ─────────────────────────────────────────────
        self._exposure: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}

        # ── Ingestion thread lock (prevents concurrent ingestion runs) ────
        self._ingestion_lock = threading.Lock()

        # ── Per-scan state (bot_probs, bot_sectors for cross-system use) ──
        self._last_bot_probs:   dict[str, float] = {}
        self._last_bot_sectors: dict[str, str]   = {}

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

        already_recorded = {
            r["ticker"]
            for r in _query_signals("SELECT DISTINCT ticker FROM outcomes")
        }

        recorded = 0
        p        = _ph()

        for market in resolved_markets:
            ticker = market.get("ticker", "")
            result = market.get("result", "")

            if not ticker or result not in ("yes", "no"):
                continue
            if ticker in already_recorded:
                continue

            resolved_yes = result == "yes"

            # Only use actual signal data — no phantom 0.5 defaults
            sig_rows = _query_signals(
                f"SELECT our_prob, direction FROM signals "
                f"WHERE ticker = {p} ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            )

            # Feed resolved outcome into ALL relevant bot models
            # (not gated on sig_rows — we want to learn from every resolved market)
            for bot in self.bots:
                if not bot.is_relevant(market):
                    continue
                try:
                    features, _ = bot.fetch_features(market, skip_noaa=True)
                    # Pad news features since we're in historical ingestion
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

            # Only record outcome + P&L + Brier when we had an actual signal
            if not sig_rows:
                continue

            our_prob  = float(sig_rows[0]["our_prob"])
            direction = str(sig_rows[0]["direction"])

            direction_correct = (
                (direction == "YES" and resolved_yes) or
                (direction == "NO"  and not resolved_yes)
            )

            pnl_usd, trade_id = _calculate_pnl(ticker, resolved_yes)

            if pnl_usd is not None:
                logger.info(
                    "[P&L] %s resolved %s → $%+.2f (trade_id=%s)",
                    ticker, result.upper(), pnl_usd, trade_id,
                )
                # Record resolved P&L in circuit breaker
                self.circuit_breaker.record_pnl(pnl_usd)
            else:
                logger.debug(
                    "[P&L] %s resolved %s — no matching trade",
                    ticker, result.upper(),
                )

            try:
                log_outcome(
                    ticker   = ticker,
                    resolved = result.upper(),
                    pnl_usd  = pnl_usd,
                    trade_id = trade_id,
                    our_prob = our_prob,
                    correct  = direction_correct,
                )
                recorded += 1
            except Exception as e:
                logger.error("log_outcome failed for %s: %s", ticker, e)

            # Per-row Brier writeback (uses stored our_prob in DB, not variable)
            _update_signal_brier(ticker, resolved_yes)

        logger.info("=== Resolution ingestion complete — %d new outcomes ===", recorded)
        return recorded

    def _run_ingestion_thread(self) -> None:
        """Launch _ingest_resolved_markets() in a background daemon thread with lock."""
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
        ticker    = market.get("ticker", "UNKNOWN")
        title     = market.get("title", "")
        yes_price = _parse_yes_price_cents(market)

        if yes_price == 0:
            return
        if yes_price <= 2 or yes_price >= 98:
            return

        # Register market with news signal tracker
        self.news.register_market(title)

        # Sharp money check — analyze order book before evaluating
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

        # Sharp money veto — don't trade against sharp money
        if sharp_signal.sharp_detected and not sharp_signal.aligned_with(consensus.direction):
            logger.info(
                "SHARP VETO %s | sharp_dir=%s our_dir=%s sharp_conf=%.2f",
                ticker, sharp_signal.sharp_direction, consensus.direction,
                sharp_signal.confidence,
            )
            return

        lead_sector = signals[0].sector

        # Store bot prob for fade scanner and resolution timer
        self._last_bot_probs[ticker]   = consensus.avg_prob
        self._last_bot_sectors[ticker] = lead_sector

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

        arb = self.arb.check(
            ticker        = ticker,
            kalshi_cents  = yes_price,
            our_direction = consensus.direction,
            market_title  = title,
        )
        self.arb.log_arb_opportunity(arb)

        if not arb.passes:
            logger.info(
                "ARB GATE VETO %s | dir_aligned=%s spread=%.2f%% notes=%s",
                ticker, arb.direction_aligned, arb.cross_spread * 100, arb.notes,
            )
            return

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
        """
        Common trade execution path used by both regular signals and fades.
        Handles circuit breaker check, bankroll sync, Kelly sizing, and order.
        """
        # Circuit breaker check before every trade
        if self.circuit_breaker.is_halted():
            logger.warning("CIRCUIT BREAKER HALT — skipping trade for %s", ticker)
            return

        # Real-time bankroll sync before sizing
        live_bankroll = self.circuit_breaker.sync_bankroll()

        # Drawdown factor — reduces Kelly when we've been losing
        summary      = self.circuit_breaker.daily_summary()
        daily_pnl    = float(summary.get("realized_pnl", 0.0))
        loss_limit   = live_bankroll * CIRCUIT_BREAKER_PCT
        # Scale from 1.0 (no loss) down to 0.5 (at 50% of loss limit)
        drawdown_factor = max(0.5, 1.0 + (daily_pnl / loss_limit) * 0.5) if daily_pnl < 0 and loss_limit > 0 else 1.0

        sector_exp = self._exposure.get(lead_sector, 0.0)

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

        if sizing["contracts"] <= 0:
            logger.info("Sizing zero for %s — skip", ticker)
            return

        client_order_id = str(uuid.uuid4())

        if DEMO_MODE:
            logger.info(
                "DEMO TRADE %s | %s | %s %dx@%dc | $%.2f%s",
                ticker, title[:40], consensus.direction.upper(),
                sizing["contracts"], yes_price, sizing["dollars"],
                " [FADE]" if fade else "",
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

        self._exposure[lead_sector] = sector_exp + sizing["dollars"]
        self.bankroll = live_bankroll  # keep in sync

        arb_note = f" arb_spread={arb.cross_spread:.2f}%" if arb and arb.arb_exists else ""
        fade_note = " [FADE]" if fade else ""

        trade_id = log_trade(
            ticker          = ticker,
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
            "TRADE #%d | %s | %s %dx@%dc | $%.2f | edge=%+.2f%% conf=%.0f%%%s%s",
            trade_id, ticker, consensus.direction.upper(),
            sizing["contracts"], yes_price, sizing["dollars"],
            consensus.avg_edge * 100, consensus.avg_confidence * 100,
            arb_note, fade_note,
        )

    # ── Fade execution ─────────────────────────────────────────────────────────

    def _execute_fades(self, open_markets: list[dict]) -> None:
        """
        Run fade scanner and execute qualifying fade trades.
        Called after every scan_markets() cycle.
        """
        if not self._last_bot_probs:
            return

        fades = self.fade_scanner.scan(open_markets, self._last_bot_probs)

        for fade in fades:
            # Build a mock consensus from the fade signal
            from shared.consensus_engine import ConsensusResult
            mock_consensus = ConsensusResult(
                execute         = True,
                direction       = fade.direction,
                avg_prob        = fade.our_prob,
                avg_edge        = abs(fade.our_prob - fade.market_prob),
                avg_confidence  = fade.confidence,
                reject_reason   = "",
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
        """
        Run correlation engine and leg into divergent pairs.
        Called after every scan_markets() cycle.
        """
        corr_signals = self.corr_engine.scan(open_markets)

        for sig in corr_signals:
            if self.circuit_breaker.is_halted():
                break

            logger.info(
                "[CORR TRADE] Legging into %s (cheap=%s@%d¢, expensive=%s@%d¢) div=%.1f¢",
                sig.event_group,
                sig.cheap_ticker, sig.cheap_price_cents,
                sig.expensive_ticker, sig.expensive_price_cents,
                sig.divergence_cents,
            )

            # Leg 1: buy the cheap side
            # Leg 2: buy the expensive side in the opposite direction
            # We execute as two separate orders with reduced sizing (half Kelly each)
            for ticker, direction, price in [
                (sig.cheap_ticker,      sig.cheap_direction,      sig.cheap_price_cents),
                (sig.expensive_ticker,  sig.expensive_direction,  sig.expensive_price_cents),
            ]:
                from shared.consensus_engine import ConsensusResult
                mock_consensus = ConsensusResult(
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
                    lead_sector = "economics",   # correlation trades are sector-agnostic
                    arb         = None,
                    fade        = False,
                )

    # ── Resolution timing execution ────────────────────────────────────────────

    def _execute_resolution_timing(self, open_markets: list[dict]) -> None:
        """
        Scan for markets that should have resolved by now and enter positions.
        Called after every scan_markets() cycle.
        """
        signals = self.res_timer.scan(
            open_markets,
            self._last_bot_probs,
            self._last_bot_sectors,
        )

        for sig in signals:
            if self.circuit_breaker.is_halted():
                break

            from shared.consensus_engine import ConsensusResult
            mock_consensus = ConsensusResult(
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

        try:
            markets = self.client.get_all_open_markets()
        except Exception as e:
            logger.error("Failed to fetch markets: %s", e)
            return

        logger.info("Fetched %d open markets", len(markets))

        # Core evaluation loop
        for market in markets:
            self._evaluate_market(market)

        # Post-scan strategies (run after all bot probs are populated)
        self._execute_fades(markets)
        self._execute_correlations(markets)
        self._execute_resolution_timing(markets)

        summary = self.circuit_breaker.daily_summary()
        logger.info(
            "Scan complete | daily_pnl=$%.2f trades=%d halted=%s",
            summary.get("realized_pnl", 0.0),
            summary.get("trade_count", 0),
            summary.get("halted", False),
        )

    # ── Nightly retrain ────────────────────────────────────────────────────────

    def retrain_models(self) -> None:
        logger.info("=== Nightly retrain ===")

        n_resolved = self._ingest_resolved_markets()
        logger.info("Ingested %d new resolved outcomes this cycle", n_resolved)

        # Rebuild resolution timing patterns from accumulated outcomes
        n_patterns = self.res_timer.rebuild_patterns()
        logger.info("Rebuilt %d resolution timing patterns", n_patterns)

        for bot in self.bots:
            stats = bot.refresh_calibration()
            logger.info("[%s] calibration: %s", bot.sector_name, stats)
            bot.save_model()

        # Sync bankroll via circuit breaker
        live_bankroll = self.circuit_breaker.sync_bankroll()
        self.bankroll = live_bankroll
        logger.info("Bankroll synced: $%.2f", live_bankroll)

        # Reset sector exposure tracking
        self._exposure = {b.sector_name: 0.0 for b in self.bots}

        # Reset bot prob cache
        self._last_bot_probs.clear()
        self._last_bot_sectors.clear()

        logger.info("=== Retrain complete ===")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v12 | DEMO=%s | $%.2f | arb_mode=%s",
            DEMO_MODE, self.bankroll, self.arb._mode,
        )
        init_db()

        # Start news velocity background poller
        self.news.start_background_poller()

        # Market scanning — every SCAN_INTERVAL_SEC
        schedule.every(SCAN_INTERVAL_SEC).seconds.do(self.scan_markets)

        # Nightly retrain — at RETRAIN_HOUR (2am)
        schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(self.retrain_models)

        # Resolution watchdog — every 4 hours in background thread
        schedule.every(4).hours.do(self._run_ingestion_thread)

        # Startup: fire ingestion immediately to catch overnight resolutions
        self._run_ingestion_thread()

        self.scan_markets()
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    FlywheelOrchestrator().run()