"""
early_exit_manager.py (v1.0)

Monitor open positions and exit early when edge evaporates.

Why exit early?
  1. Edge evaporated: Market moved to our price → no more edge, free up capital
  2. Cut losses: Market moved against us significantly → limit downside
  3. Lock profits: Market moved in our favor → take the win

Strategy:
  - Track all open positions (from trades table)
  - On each scan, check current market prices
  - If exit conditions met, close the position
  - Log the early exit and realized P&L

Exit Conditions:
  1. EDGE_GONE: Market moved to within 3% of our entry → edge evaporated
  2. STOP_LOSS: Market moved against us >15% → cut losses
  3. TAKE_PROFIT: Market moved in our favor >20% → lock profit
  4. TIME_DECAY: <2 hours to expiry and position underwater → exit

Usage:
    from shared.early_exit_manager import EarlyExitManager
    
    manager = EarlyExitManager(client=kalshi_client)
    
    # Run on each scan
    exits = manager.check_and_exit()
    for exit in exits:
        logger.info("EARLY EXIT: %s — %s (P&L: $%.2f)", 
            exit.ticker, exit.reason, exit.pnl)
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    EDGE_GONE = "edge_gone"        # Market moved to our price
    STOP_LOSS = "stop_loss"        # Market moved against us
    TAKE_PROFIT = "take_profit"    # Market moved in our favor
    TIME_DECAY = "time_decay"      # Near expiry, position underwater
    MANUAL = "manual"              # Manual exit request


@dataclass
class Position:
    """An open position."""
    ticker: str
    side: str  # "yes" or "no"
    contracts: int
    entry_price: int  # cents
    entry_time: datetime
    sector: str
    trade_id: int


@dataclass
class ExitResult:
    """Result of an early exit."""
    ticker: str
    reason: ExitReason
    exit_price: int
    entry_price: int
    contracts: int
    pnl: float  # Realized P&L in dollars
    held_duration_sec: float
    success: bool
    order_id: Optional[str] = None


class EarlyExitManager:
    """
    Monitor positions and execute early exits when conditions are met.
    """
    
    # Exit thresholds (configurable)
    EDGE_GONE_THRESHOLD = 0.03      # 3% — market at our price, edge gone
    STOP_LOSS_THRESHOLD = 0.15      # 15% — cut losses
    TAKE_PROFIT_THRESHOLD = 0.20    # 20% — lock in profit
    TIME_DECAY_HOURS = 2.0          # Exit if <2 hours and underwater
    MIN_HOLD_SECONDS = 300          # Don't exit within 5 min of entry
    
    def __init__(
        self,
        client,  # KalshiClient
        enabled: bool = True,
        db_url: Optional[str] = None,
    ):
        self.client = client
        self.enabled = enabled
        self.db_url = db_url or os.getenv("DATABASE_URL", "")
        
        # Cache positions
        self._positions: Dict[str, Position] = {}
        self._last_position_load: float = 0
        self._position_cache_ttl: float = 60  # Reload every 60 seconds
        
        # Statistics
        self._total_exits: int = 0
        self._total_pnl: float = 0.0
        self._exits_by_reason: Dict[ExitReason, int] = {r: 0 for r in ExitReason}
    
    def _get_db_connection(self):
        """Get database connection."""
        if not self.db_url:
            return None
        
        try:
            import psycopg2
            return psycopg2.connect(self.db_url)
        except Exception as e:
            logger.warning("[EXIT] DB connection failed: %s", e)
            return None
    
    def _load_open_positions(self) -> Dict[str, Position]:
        """
        Load open positions from trades table.
        
        "Open" = traded but not yet resolved
        """
        now = time.time()
        if now - self._last_position_load < self._position_cache_ttl:
            return self._positions
        
        conn = self._get_db_connection()
        if not conn:
            return self._positions
        
        try:
            cur = conn.cursor()
            
            # Get trades that haven't been resolved yet
            cur.execute("""
                SELECT 
                    t.id,
                    t.ticker,
                    t.direction,
                    t.contracts,
                    t.yes_price_cents,
                    t.created_at,
                    t.sector
                FROM trades t
                LEFT JOIN outcomes o ON t.ticker = o.ticker
                WHERE o.ticker IS NULL
                  AND t.created_at > NOW() - INTERVAL '7 days'
                ORDER BY t.created_at DESC
            """)
            
            rows = cur.fetchall()
            
            positions = {}
            for row in rows:
                trade_id, ticker, direction, contracts, price, created_at, sector = row
                
                # Skip if already exited (check for closing trade)
                # For now, assume one position per ticker
                if ticker in positions:
                    continue
                
                positions[ticker] = Position(
                    ticker=ticker,
                    side=direction.lower(),
                    contracts=contracts,
                    entry_price=price,
                    entry_time=created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc),
                    sector=sector or "unknown",
                    trade_id=trade_id,
                )
            
            self._positions = positions
            self._last_position_load = now
            
            logger.debug("[EXIT] Loaded %d open positions", len(positions))
            return positions
            
        except Exception as e:
            logger.warning("[EXIT] Load positions failed: %s", e)
            return self._positions
        finally:
            conn.close()
    
    def _get_current_price(self, ticker: str) -> Optional[int]:
        """Get current mid price for a ticker."""
        try:
            market = self.client.get_market(ticker)
            if not market:
                return None
            
            bid = market.get("yes_bid") or market.get("yes_bid_dollars", 0)
            ask = market.get("yes_ask") or market.get("yes_ask_dollars", 0)
            
            if isinstance(bid, float) and bid < 1:
                bid = int(bid * 100)
            if isinstance(ask, float) and ask < 1:
                ask = int(ask * 100)
            
            if bid and ask:
                return (int(bid) + int(ask)) // 2
            
            # Fallback to last price
            last = market.get("last_price") or market.get("last_price_dollars", 0)
            if isinstance(last, float) and last < 1:
                last = int(last * 100)
            
            return int(last) if last else None
            
        except Exception as e:
            logger.debug("[EXIT] Get price failed for %s: %s", ticker, e)
            return None
    
    def _get_expiry_time(self, ticker: str) -> Optional[datetime]:
        """Get expiry time for a market."""
        try:
            market = self.client.get_market(ticker)
            if not market:
                return None
            
            for field in ["close_time", "expiration_time", "end_time"]:
                raw = market.get(field)
                if raw:
                    if isinstance(raw, datetime):
                        return raw
                    if isinstance(raw, str):
                        try:
                            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        except:
                            pass
            return None
        except:
            return None
    
    def _should_exit(self, position: Position, current_price: int) -> Optional[ExitReason]:
        """
        Determine if position should be exited.
        
        Returns ExitReason if should exit, None otherwise.
        """
        entry = position.entry_price
        
        # Check minimum hold time
        now = datetime.now(timezone.utc)
        if position.entry_time.tzinfo is None:
            position.entry_time = position.entry_time.replace(tzinfo=timezone.utc)
        
        held_seconds = (now - position.entry_time).total_seconds()
        if held_seconds < self.MIN_HOLD_SECONDS:
            return None
        
        # Calculate price movement
        if position.side == "yes":
            # We bought YES at entry, current YES price matters
            price_change_pct = (current_price - entry) / entry if entry > 0 else 0
        else:
            # We bought NO, so we care about NO price = 100 - YES
            entry_no = 100 - entry
            current_no = 100 - current_price
            price_change_pct = (current_no - entry_no) / entry_no if entry_no > 0 else 0
        
        # Check exit conditions
        
        # 1. Edge gone — market at our entry price
        if abs(price_change_pct) < self.EDGE_GONE_THRESHOLD:
            # Market hasn't moved much, but if we're slightly profitable, exit
            # Actually, edge_gone means market IS at our price, so we had edge and now don't
            # This is more about: market moved TO our price FROM being different
            # For simplicity: if current ≈ entry and we've held >10 min, consider edge gone
            if held_seconds > 600:  # 10 minutes
                return ExitReason.EDGE_GONE
        
        # 2. Stop loss — market moved against us
        if price_change_pct < -self.STOP_LOSS_THRESHOLD:
            return ExitReason.STOP_LOSS
        
        # 3. Take profit — market moved in our favor
        if price_change_pct > self.TAKE_PROFIT_THRESHOLD:
            return ExitReason.TAKE_PROFIT
        
        # 4. Time decay — near expiry and underwater
        expiry = self._get_expiry_time(position.ticker)
        if expiry:
            hours_to_expiry = (expiry - now).total_seconds() / 3600
            if hours_to_expiry < self.TIME_DECAY_HOURS and price_change_pct < 0:
                return ExitReason.TIME_DECAY
        
        return None
    
    def _execute_exit(self, position: Position, current_price: int, reason: ExitReason) -> ExitResult:
        """
        Execute an early exit by closing the position.
        """
        try:
            # To close a YES position, sell YES (or buy NO)
            # To close a NO position, sell NO (or buy YES)
            if position.side == "yes":
                close_side = "no"  # Sell our YES by buying NO... actually on Kalshi
                # Selling YES = placing a NO order? Let me think...
                # Actually to close a YES position you just sell YES
                close_side = "yes"
                close_action = "sell"
            else:
                close_side = "no"
                close_action = "sell"
            
            # Place closing order at current price (market-ish)
            import uuid
            client_order_id = str(uuid.uuid4())
            
            # For closing, we use the opposite action
            # If we bought YES, we sell YES
            # Kalshi API: to sell, we place a sell order
            response = self.client.place_order(
                ticker=position.ticker,
                side=close_side,
                count=position.contracts,
                yes_price=current_price,
                client_order_id=client_order_id,
                action="sell" if position.side == "yes" else "sell",
            )
            
            order_id = response.get("order", {}).get("order_id", client_order_id)
            
            # Calculate P&L
            if position.side == "yes":
                # Bought at entry, selling at current
                pnl = (current_price - position.entry_price) / 100 * position.contracts
            else:
                # Bought NO at (100 - entry), selling at (100 - current)
                entry_no = 100 - position.entry_price
                current_no = 100 - current_price
                pnl = (current_no - entry_no) / 100 * position.contracts
            
            held_duration = (datetime.now(timezone.utc) - position.entry_time).total_seconds()
            
            logger.info(
                "[EARLY EXIT] %s %s | %s | entry=%d¢ exit=%d¢ | P&L=$%.2f | held=%.0fs",
                reason.value.upper(), position.side.upper(), position.ticker[:35],
                position.entry_price, current_price, pnl, held_duration,
            )
            
            # Update stats
            self._total_exits += 1
            self._total_pnl += pnl
            self._exits_by_reason[reason] += 1
            
            # Remove from positions cache
            if position.ticker in self._positions:
                del self._positions[position.ticker]
            
            return ExitResult(
                ticker=position.ticker,
                reason=reason,
                exit_price=current_price,
                entry_price=position.entry_price,
                contracts=position.contracts,
                pnl=pnl,
                held_duration_sec=held_duration,
                success=True,
                order_id=order_id,
            )
            
        except Exception as e:
            logger.error("[EXIT] Failed to close %s: %s", position.ticker, e)
            
            return ExitResult(
                ticker=position.ticker,
                reason=reason,
                exit_price=current_price,
                entry_price=position.entry_price,
                contracts=position.contracts,
                pnl=0,
                held_duration_sec=0,
                success=False,
            )
    
    def check_and_exit(self) -> List[ExitResult]:
        """
        Check all open positions and exit if conditions met.
        
        Call this on each market scan.
        
        Returns list of exits executed.
        """
        if not self.enabled:
            return []
        
        positions = self._load_open_positions()
        if not positions:
            return []
        
        exits = []
        
        for ticker, position in list(positions.items()):
            current_price = self._get_current_price(ticker)
            if current_price is None:
                continue
            
            reason = self._should_exit(position, current_price)
            if reason:
                result = self._execute_exit(position, current_price, reason)
                exits.append(result)
        
        return exits
    
    def get_stats(self) -> Dict:
        """Get exit statistics."""
        return {
            "total_exits": self._total_exits,
            "total_pnl": round(self._total_pnl, 2),
            "by_reason": {r.value: c for r, c in self._exits_by_reason.items()},
            "open_positions": len(self._positions),
        }
    
    def get_open_positions(self) -> List[Position]:
        """Get list of current open positions."""
        self._load_open_positions()
        return list(self._positions.values())


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: Optional[EarlyExitManager] = None


def get_early_exit_manager(client=None) -> EarlyExitManager:
    """Get the global early exit manager instance."""
    global _manager
    if _manager is None:
        if client is None:
            raise ValueError("Must provide client on first call")
        _manager = EarlyExitManager(client=client)
    return _manager