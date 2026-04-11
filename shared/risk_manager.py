"""
shared/risk_manager.py

Going-live safety layer. Three guards:

  1. Drawdown kill switch — halt ALL trading if daily P&L breaches -20% bankroll
  2. Per-player exposure cap — no single player gets more than 5% of bankroll
  3. Shadow mode gate — require 48hr clean shadow run before going live

The orchestrator calls risk_manager.can_trade() before every trade execution.
If it returns False, the trade is logged but NOT executed.

Usage in orchestrator.py:
    from shared.risk_manager import RiskManager

    risk = RiskManager(bankroll=1000.0)

    # At start of each scan cycle:
    risk.update_daily_pnl(get_todays_pnl())

    # Before each trade:
    if not risk.can_trade():
        logger.warning("HALTED: %s", risk.halt_reason)
        continue

    ok, reason = risk.check_player_exposure(player_name, dollars_to_risk)
    if not ok:
        logger.info("SKIPPED: %s", reason)
        continue

    # After trade executes:
    risk.record_trade(player_name, dollars_risked)
"""

import logging
import os
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks for going live safely."""

    def __init__(
        self,
        bankroll:               float = 1000.0,
        max_daily_drawdown_pct: float = 0.20,   # halt at -20% bankroll
        max_player_exposure_pct: float = 0.05,   # 5% of bankroll per player
        max_single_trade_pct:   float = 0.03,    # 3% of bankroll per trade
        max_sector_exposure_pct: float = 0.30,   # 30% of bankroll per sector
        shadow_hours_required:  int   = 48,      # 48hr shadow before live
    ):
        self.bankroll = bankroll
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_player_exposure_pct = max_player_exposure_pct
        self.max_single_trade_pct = max_single_trade_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.shadow_hours_required = shadow_hours_required

        # Daily state (reset each day)
        self._today = date.today()
        self._daily_pnl = 0.0
        self._player_exposure: dict[str, float] = defaultdict(float)
        self._sector_exposure: dict[str, float] = defaultdict(float)
        self._trade_count = 0

        # Halt state
        self._halted = False
        self._halt_reason = ""

        # Shadow mode tracking
        self._shadow_start: Optional[datetime] = None

    # ── Daily reset ──────────────────────────────────────────────────────

    def _maybe_reset_day(self):
        """Auto-reset counters at midnight."""
        today = date.today()
        if today != self._today:
            logger.info(
                "[risk] Day rolled %s → %s. Reset: %d trades, P&L $%.2f, %d players tracked",
                self._today, today, self._trade_count, self._daily_pnl,
                len(self._player_exposure),
            )
            self._today = today
            self._daily_pnl = 0.0
            self._player_exposure.clear()
            self._sector_exposure.clear()
            self._trade_count = 0
            self._halted = False
            self._halt_reason = ""

    # ── Core checks ──────────────────────────────────────────────────────

    def update_daily_pnl(self, pnl: float):
        """Update today's P&L from resolved outcomes. Call each scan cycle."""
        self._maybe_reset_day()
        self._daily_pnl = pnl

    def can_trade(self) -> bool:
        """Master gate — check if trading is allowed right now."""
        self._maybe_reset_day()

        # Check drawdown
        threshold = -self.bankroll * self.max_daily_drawdown_pct
        if self._daily_pnl <= threshold:
            self._halted = True
            self._halt_reason = (
                f"DRAWDOWN HALT: daily P&L ${self._daily_pnl:.2f} "
                f"breached limit ${threshold:.2f} "
                f"(-{self.max_daily_drawdown_pct*100:.0f}% of ${self.bankroll:.0f} bankroll)"
            )
            return False

        if self._halted:
            return False

        return True

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def check_player_exposure(
        self, player_name: str, new_dollars: float,
    ) -> tuple[bool, str]:
        """Check if adding this trade would exceed per-player cap."""
        self._maybe_reset_day()
        current = self._player_exposure[player_name]
        cap = self.bankroll * self.max_player_exposure_pct

        if current + new_dollars > cap:
            return False, (
                f"Player exposure cap: {player_name} at "
                f"${current:.2f}+${new_dollars:.2f} = ${current+new_dollars:.2f} "
                f"> cap ${cap:.2f} ({self.max_player_exposure_pct*100:.0f}%)"
            )
        return True, ""

    def check_sector_exposure(
        self, sector: str, new_dollars: float,
    ) -> tuple[bool, str]:
        """Check if adding this trade would exceed per-sector cap."""
        self._maybe_reset_day()
        current = self._sector_exposure[sector]
        cap = self.bankroll * self.max_sector_exposure_pct

        if current + new_dollars > cap:
            return False, (
                f"Sector exposure cap: {sector} at "
                f"${current:.2f}+${new_dollars:.2f} > cap ${cap:.2f}"
            )
        return True, ""

    def check_single_trade(self, dollars: float) -> tuple[bool, str]:
        """Check if a single trade exceeds the per-trade cap."""
        cap = self.bankroll * self.max_single_trade_pct
        if dollars > cap:
            return False, (
                f"Single trade cap: ${dollars:.2f} > ${cap:.2f} "
                f"({self.max_single_trade_pct*100:.0f}%)"
            )
        return True, ""

    def pre_trade_check(
        self, player_name: str, sector: str, dollars: float,
    ) -> tuple[bool, str]:
        """Run ALL pre-trade checks. Returns (ok, reason)."""
        if not self.can_trade():
            return False, self._halt_reason

        ok, reason = self.check_single_trade(dollars)
        if not ok:
            return False, reason

        ok, reason = self.check_player_exposure(player_name, dollars)
        if not ok:
            return False, reason

        ok, reason = self.check_sector_exposure(sector, dollars)
        if not ok:
            return False, reason

        return True, ""

    # ── Recording ────────────────────────────────────────────────────────

    def record_trade(self, player_name: str, sector: str, dollars: float):
        """Record a trade for exposure tracking."""
        self._player_exposure[player_name] += dollars
        self._sector_exposure[sector] += dollars
        self._trade_count += 1
        logger.debug(
            "[risk] Recorded: %s (%s) $%.2f | player_total=$%.2f sector_total=$%.2f",
            player_name, sector, dollars,
            self._player_exposure[player_name],
            self._sector_exposure[sector],
        )

    # ── Shadow mode ──────────────────────────────────────────────────────

    def start_shadow(self):
        """Begin shadow mode tracking."""
        self._shadow_start = datetime.utcnow()
        logger.info("[risk] Shadow mode started at %s", self._shadow_start)

    def shadow_complete(self) -> bool:
        """Check if shadow period has elapsed."""
        if self._shadow_start is None:
            return False
        elapsed = (datetime.utcnow() - self._shadow_start).total_seconds() / 3600
        return elapsed >= self.shadow_hours_required

    def shadow_hours_elapsed(self) -> float:
        """How many hours of shadow mode have passed."""
        if self._shadow_start is None:
            return 0.0
        return (datetime.utcnow() - self._shadow_start).total_seconds() / 3600

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Current risk state for logging/dashboard."""
        self._maybe_reset_day()
        return {
            "halted":           self._halted,
            "halt_reason":      self._halt_reason,
            "daily_pnl":        self._daily_pnl,
            "drawdown_limit":   -self.bankroll * self.max_daily_drawdown_pct,
            "trade_count":      self._trade_count,
            "player_exposures": dict(self._player_exposure),
            "sector_exposures": dict(self._sector_exposure),
            "shadow_hours":     self.shadow_hours_elapsed(),
            "shadow_complete":  self.shadow_complete(),
        }