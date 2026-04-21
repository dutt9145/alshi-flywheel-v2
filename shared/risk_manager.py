import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks for going live safely."""

    def __init__(
        self,
        bankroll: float = 1000.0,
        max_daily_drawdown_pct: float = 0.20,
        max_player_exposure_pct: float = 0.05,
        max_single_trade_pct: float = 0.03,
        max_sector_exposure_pct: float = 0.30,
        shadow_hours_required: int = 48,
        enforce_shadow: bool = False,
    ):
        self.bankroll = bankroll
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_player_exposure_pct = max_player_exposure_pct
        self.max_single_trade_pct = max_single_trade_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.shadow_hours_required = shadow_hours_required
        self.enforce_shadow = enforce_shadow

        # Daily state
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

    def _maybe_reset_day(self) -> None:
        today = date.today()
        if today != self._today:
            logger.info(
                "[risk] Day rolled %s → %s. Reset: %d trades, P&L $%.2f, %d players tracked",
                self._today,
                today,
                self._trade_count,
                self._daily_pnl,
                len(self._player_exposure),
            )
            self._today = today
            self._daily_pnl = 0.0
            self._player_exposure.clear()
            self._sector_exposure.clear()
            self._trade_count = 0
            self._halted = False
            self._halt_reason = ""

    def update_daily_pnl(self, pnl: float) -> None:
        self._maybe_reset_day()
        self._daily_pnl = pnl

    def start_shadow(self) -> None:
        if self._shadow_start is None:
            self._shadow_start = datetime.utcnow()
            logger.info("[risk] Shadow mode started at %s", self._shadow_start)

    def shadow_complete(self) -> bool:
        if self._shadow_start is None:
            return False
        elapsed = (datetime.utcnow() - self._shadow_start).total_seconds() / 3600
        return elapsed >= self.shadow_hours_required

    def shadow_hours_elapsed(self) -> float:
        if self._shadow_start is None:
            return 0.0
        return (datetime.utcnow() - self._shadow_start).total_seconds() / 3600

    def can_trade(self) -> bool:
        """Master gate — check if trading is allowed right now."""
        self._maybe_reset_day()

        if self.enforce_shadow:
            if self._shadow_start is None:
                self._halt_reason = "SHADOW MODE NOT STARTED"
                return False
            if not self.shadow_complete():
                self._halt_reason = (
                    f"SHADOW MODE: {self.shadow_hours_elapsed():.1f}/"
                    f"{self.shadow_hours_required}h complete"
                )
                return False

        threshold = -self.bankroll * self.max_daily_drawdown_pct
        if self._daily_pnl <= threshold:
            self._halted = True
            self._halt_reason = (
                f"DRAWDOWN HALT: daily P&L ${self._daily_pnl:.2f} "
                f"breached limit ${threshold:.2f} "
                f"(-{self.max_daily_drawdown_pct * 100:.0f}% of ${self.bankroll:.0f} bankroll)"
            )
            return False

        if self._halted:
            return False

        return True

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def check_player_exposure(self, player_name: str, new_dollars: float) -> tuple[bool, str]:
        self._maybe_reset_day()
        current = self._player_exposure[player_name]
        cap = self.bankroll * self.max_player_exposure_pct

        if current + new_dollars > cap:
            return False, (
                f"Player exposure cap: {player_name} at "
                f"${current:.2f}+${new_dollars:.2f} = ${current + new_dollars:.2f} "
                f"> cap ${cap:.2f} ({self.max_player_exposure_pct * 100:.0f}%)"
            )
        return True, ""

    def check_sector_exposure(self, sector: str, new_dollars: float) -> tuple[bool, str]:
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
        cap = self.bankroll * self.max_single_trade_pct
        if dollars > cap:
            return False, (
                f"Single trade cap: ${dollars:.2f} > ${cap:.2f} "
                f"({self.max_single_trade_pct * 100:.0f}%)"
            )
        return True, ""

    def pre_trade_check(self, player_name: str, sector: str, dollars: float) -> tuple[bool, str]:
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

    def record_trade(self, player_name: str, sector: str, dollars: float) -> None:
        self._player_exposure[player_name] += dollars
        self._sector_exposure[sector] += dollars
        self._trade_count += 1
        logger.debug(
            "[risk] Recorded: %s (%s) $%.2f | player_total=$%.2f sector_total=$%.2f",
            player_name,
            sector,
            dollars,
            self._player_exposure[player_name],
            self._sector_exposure[sector],
        )

    def status(self) -> dict:
        self._maybe_reset_day()
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self._daily_pnl,
            "drawdown_limit": -self.bankroll * self.max_daily_drawdown_pct,
            "trade_count": self._trade_count,
            "player_exposures": dict(self._player_exposure),
            "sector_exposures": dict(self._sector_exposure),
            "shadow_hours": self.shadow_hours_elapsed(),
            "shadow_complete": self.shadow_complete(),
            "enforce_shadow": self.enforce_shadow,
        }