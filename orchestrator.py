"""
orchestrator.py  (v3 — signals table now populated)

Fixes vs v2:
  1. log_signal() is now called for EVERY market that passes consensus (gates 1-3)
     — this is what populates the signals table in Supabase
     — previously log_signal was never called, so signals table stayed at 0 rows
  2. log_signal() is also called when the arb gate vetoes a trade
     — you can now see everything the bot considered, not just what it traded
  3. Fixed 3x format string bug: %.2%% → %.2f%% (was causing silent log crashes)
  4. log_signal imported alongside log_trade

Every SCAN_INTERVAL_SEC:
  1. Pull all open Kalshi markets
  2. Each sector bot evaluates relevance + generates BotSignal
  3. ConsensusEngine: direction + edge + confidence gates (gates 1-3)
  4. log_signal() ← WRITTEN HERE regardless of gate 4 outcome
  5. ArbLayer: cross-venue check vs Polymarket / OddsPapi (gate 4)
  6. Kelly size + place order + log_trade()

Nightly at RETRAIN_HOUR:
  7. Feed resolved contracts back into Bayesian models
  8. Recompute calibration — Brier score updates bot weights
  9. Save models to disk
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
from shared.calibration_logger import init_db, log_signal, log_trade, compute_calibration
from shared.consensus_engine import ConsensusEngine
from shared.kalshi_client import KalshiClient
from shared.kelly_sizer import kelly_stake, no_kelly_stake
from bots.sector_bots import all_bots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


class FlywheelOrchestrator:

    def __init__(self):
        self.client   = KalshiClient()
        self.bots     = all_bots()
        self.engine   = ConsensusEngine()
        self.arb      = ArbLayer()
        self.bankroll = BANKROLL
        self._exposure: dict[str, float] = {b.sector_name: 0.0 for b in self.bots}

    # ── Per-market pipeline ────────────────────────────────────────────────────

    def _evaluate_market(self, market: dict) -> None:
        ticker    = market.get("ticker", "UNKNOWN")
        yes_bid   = market.get("yes_bid", 0)
        yes_ask   = market.get("yes_ask", 100)
        yes_price = int((yes_bid + yes_ask) / 2)
        title     = market.get("title", "")

        if yes_price <= 1 or yes_price >= 99:
            return

        # ── Gates 1-3: Bot consensus ───────────────────────────────────────────
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

        # ── Write signal to Supabase regardless of gate 4 outcome ─────────────
        # This is what populates the signals table. We log every market that
        # passes consensus so you can review what the bot saw, not just traded.
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

        if not consensus.execute:
            logger.debug("Consensus PASS %s: %s", ticker, consensus.reject_reason)
            return

        # ── Gate 4: Cross-venue arb check ─────────────────────────────────────
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

        # Log arb enhancement to consensus avg_prob if we have sharp data
        if arb.sharp_line_prob is not None:
            logger.info(
                "SHARP CONFIRM %s | kalshi=%.2f%% sharp=%.2f%% spread=%+.2f%%",
                ticker,
                yes_price / 100 * 100,
                arb.sharp_line_prob * 100,
                (yes_price / 100 - arb.sharp_line_prob) * 100,
            )

        # ── Kelly sizing ───────────────────────────────────────────────────────
        sector_exp = self._exposure.get(lead_sector, 0.0)

        if consensus.direction == "YES":
            sizing = kelly_stake(
                prob             = consensus.avg_prob,
                yes_price_cents  = yes_price,
                bankroll         = self.bankroll,
                sector           = lead_sector,
                sector_exposure  = sector_exp,
            )
            side = "yes"
        else:
            sizing = no_kelly_stake(
                prob             = consensus.avg_prob,
                yes_price_cents  = yes_price,
                bankroll         = self.bankroll,
                sector           = lead_sector,
                sector_exposure  = sector_exp,
            )
            side = "no"

        if sizing["contracts"] <= 0:
            logger.info("Sizing zero for %s — skip", ticker)
            return

        # ── Execute ────────────────────────────────────────────────────────────
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
            trade_id,
            ticker,
            consensus.direction.upper(),
            sizing["contracts"],
            yes_price,
            sizing["dollars"],
            consensus.avg_edge * 100,
            consensus.avg_confidence * 100,
            arb_note,
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

        logger.info("SAMPLE TITLES: %s", [m.get("title","") for m in markets[:10]])
        
        for market in markets:
            self._evaluate_market(market)
        logger.info("Scan complete")

    # ── Nightly retrain ────────────────────────────────────────────────────────

    def retrain_models(self) -> None:
        logger.info("=== Nightly retrain ===")
        for bot in self.bots:
            stats = bot.refresh_calibration()
            logger.info("[%s] %s", bot.sector_name, stats)
            bot.save_model()
        try:
            self.bankroll = self.client.get_balance()
            logger.info("Bankroll: $%.2f", self.bankroll)
        except Exception as e:
            logger.warning("Bankroll sync failed: %s", e)
        self._exposure = {b.sector_name: 0.0 for b in self.bots}
        logger.info("=== Retrain complete ===")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Kalshi Flywheel v3 | DEMO=%s | $%.2f | arb_mode=%s",
            DEMO_MODE, self.bankroll, self.arb._mode,
        )
        init_db()
        schedule.every(SCAN_INTERVAL_SEC).seconds.do(self.scan_markets)
        schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(self.retrain_models)
        self.scan_markets()
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    FlywheelOrchestrator().run()