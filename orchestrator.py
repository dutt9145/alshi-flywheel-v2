"""
orchestrator.py  (v7 — 4-hour resolution reconciliation watchdog)

Changes vs v6:
  1. Added schedule.every(4).hours.do(self._ingest_resolved_markets)
     — Resolution ingestion now runs every 4 hours automatically
     — If the 2am retrain misses (Railway restart, redeploy timing, etc.)
       the next 4-hour cycle catches and ingests any missed outcomes
     — Fully automatic, no manual intervention or notification needed
     — Tables will populate silently in the background
  2. Everything else unchanged from v6
"""

import logging
import time
import uuid
from datetime import datetime, timezone

import schedule

from config.settings import (
    BANKROLL, DEMO_MODE, RETRAIN_HOUR, SCAN_INTERVAL_SEC,
)
from shared.arb_layer import ArbLayer
from shared.calibration_logger import init_db, log_signal, log_trade, log_outcome, compute_calibration
from shared.consensus_engine import ConsensusEngine
from shared.kalshi_client import KalshiClient
from shared.kelly_sizer import kelly_stake, no_kelly_stake
from bots.sector_bots import all_bots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


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


def _query_signals(sql: str, params: tuple = ()) -> list:
    """
    Lightweight inline query helper for resolution ingestion.
    Mirrors dashboard.py's query() but lives here to avoid circular imports.
    """
    import os
    database_url = os.getenv("DATABASE_URL", "")
    db_path      = os.getenv("DB_PATH", "flywheel.db")

    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
            return rows
        else:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            conn.close()
            return rows
    except Exception as e:
        logger.error("_query_signals error: %s", e)
        return []


class FlywheelOrchestrator:

    def __init__(self):
        self.client   = KalshiClient()
        self.bots     = all_bots()
        self.engine   = ConsensusEngine()
        self.arb      = ArbLayer()
        self.bankroll = BANKROLL
        self._exposure: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}

    # ── Resolution ingestion ───────────────────────────────────────────────────

    def _ingest_resolved_markets(self) -> int:
        """
        Fetch all settled markets from Kalshi, match against our signal log,
        update Bayesian models, and write outcomes to DB.

        Runs:
          - Every 4 hours (watchdog — catches anything the 2am job missed)
          - As part of nightly retrain at 2am

        Returns count of new outcomes recorded this cycle.
        """
        logger.info("=== Resolution ingestion starting ===")

        try:
            resolved_markets = self.client.get_resolved_markets()
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

        our_tickers = {
            r["ticker"]
            for r in _query_signals("SELECT DISTINCT ticker FROM signals")
        }

        recorded = 0

        for market in resolved_markets:
            ticker = market.get("ticker", "")
            result = market.get("result", "")

            if not ticker or result not in ("yes", "no"):
                continue

            if ticker in already_recorded:
                continue

            if ticker not in our_tickers:
                continue

            resolved_yes = result == "yes"

            for bot in self.bots:
                if not bot.is_relevant(market):
                    continue
                try:
                    features, _ = bot.fetch_features(market)
                    bot.record_outcome(features, resolved_yes)
                    logger.info(
                        "[%s] Bayesian update: %s → %s",
                        bot.sector_name, ticker, result.upper()
                    )
                except Exception as e:
                    logger.warning(
                        "[%s] record_outcome failed for %s: %s",
                        bot.sector_name, ticker, e
                    )

            sig_rows = _query_signals(
                "SELECT our_prob, direction FROM signals "
                "WHERE ticker = %s ORDER BY created_at DESC LIMIT 1"
                if self.client.base.startswith("https") else
                "SELECT our_prob, direction FROM signals "
                "WHERE ticker = ? ORDER BY created_at DESC LIMIT 1",
                (ticker,)
            )

            our_prob  = float(sig_rows[0]["our_prob"])  if sig_rows else 0.5
            direction = str(sig_rows[0]["direction"])   if sig_rows else "YES"

            direction_correct = (
                (direction == "YES" and resolved_yes) or
                (direction == "NO"  and not resolved_yes)
            )

            try:
                log_outcome(
                    ticker       = ticker,
                    resolved     = result.upper(),
                    pnl_usd      = None,
                    trade_id     = None,
                    our_prob     = our_prob,
                    correct      = direction_correct,
                )
                recorded += 1
            except Exception as e:
                logger.error("log_outcome failed for %s: %s", ticker, e)

        logger.info("=== Resolution ingestion complete — %d new outcomes ===", recorded)
        return recorded

    # ── Per-market pipeline ────────────────────────────────────────────────────

    def _evaluate_market(self, market: dict) -> None:
        ticker    = market.get("ticker", "UNKNOWN")
        title     = market.get("title", "")
        yes_price = _parse_yes_price_cents(market)

        if yes_price == 0:
            return

        if yes_price <= 2 or yes_price >= 98:
            return

        # ── Collect bot signals ────────────────────────────────────────────────
        signals = []
        for bot in self.bots:
            try:
                sig = bot.evaluate(market)
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

        lead_sector = signals[0].sector
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

        if arb.sharp_line_prob is not None:
            logger.info(
                "SHARP CONFIRM %s | kalshi=%.2f%% sharp=%.2f%% spread=%+.2f%%",
                ticker,
                yes_price,
                arb.sharp_line_prob * 100,
                yes_price - arb.sharp_line_prob * 100,
            )

        sector_exp = self._exposure.get(lead_sector, 0.0)

        if consensus.direction == "YES":
            sizing = kelly_stake(
                prob            = consensus.avg_prob,
                yes_price_cents = yes_price,
                bankroll        = self.bankroll,
                sector          = lead_sector,
                sector_exposure = sector_exp,
            )
            side = "yes"
        else:
            sizing = no_kelly_stake(
                prob            = consensus.avg_prob,
                yes_price_cents = yes_price,
                bankroll        = self.bankroll,
                sector          = lead_sector,
                sector_exposure = sector_exp,
            )
            side = "no"

        if sizing["contracts"] <= 0:
            logger.info("Sizing zero for %s — skip", ticker)
            return

        client_order_id = str(uuid.uuid4())
        try:
            order_resp = self.client.place_order(
                ticker          = ticker,
                side            = side,
                count           = sizing["contracts"],
                yes_price       = yes_price,
                client_order_id = client_order_id,
            )
        except Exception as e:
            logger.error("Order failed %s: %s", ticker, e)
            return

        order_id = order_resp.get("order", {}).get("order_id", client_order_id)
        self._exposure[lead_sector] = sector_exp + sizing["dollars"]

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

        arb_note = f" arb_spread={arb.cross_spread:.2f}%" if arb.arb_exists else ""
        logger.info(
            "TRADE #%d | %s | %s %dx@%dc | $%.2f | edge=%+.2f%% conf=%.0f%%%s",
            trade_id, ticker, consensus.direction.upper(),
            sizing["contracts"], yes_price, sizing["dollars"],
            consensus.avg_edge * 100, consensus.avg_confidence * 100, arb_note,
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
        for market in markets:
            self._evaluate_market(market)
        logger.info("Scan complete")

    # ── Nightly retrain ────────────────────────────────────────────────────────

    def retrain_models(self) -> None:
        logger.info("=== Nightly retrain ===")

        n_resolved = self._ingest_resolved_markets()
        logger.info("Ingested %d new resolved outcomes this cycle", n_resolved)

        for bot in self.bots:
            stats = bot.refresh_calibration()
            logger.info("[%s] calibration: %s", bot.sector_name, stats)
            bot.save_model()

        try:
            self.bankroll = self.client.get_balance()
            logger.info("Bankroll synced: $%.2f", self.bankroll)
        except Exception as e:
            logger.warning("Bankroll sync failed: %s", e)

        self._exposure = {b.sector_name: 0.0 for b in self.bots}

        logger.info("=== Retrain complete ===")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v7 | DEMO=%s | $%.2f | arb_mode=%s",
            DEMO_MODE, self.bankroll, self.arb._mode,
        )
        init_db()

        # Market scanning — runs every SCAN_INTERVAL_SEC seconds
        schedule.every(SCAN_INTERVAL_SEC).seconds.do(self.scan_markets)

        # Nightly retrain — runs at 2am, includes resolution ingestion
        schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(self.retrain_models)

        # Watchdog — resolution ingestion every 4 hours
        # Automatically catches any outcomes missed by the 2am retrain
        # (e.g. redeploy timing, Railway restart, Kalshi API blip)
        schedule.every(4).hours.do(self._ingest_resolved_markets)

        self.scan_markets()
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    FlywheelOrchestrator().run()