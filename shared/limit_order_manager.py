"""
limit_order_manager.py (v1.0)

Smart limit order execution to stop paying the spread.

Instead of market orders (hit the ask), place limit orders at mid
and let them fill. Saves 2-4¢ per trade.

Strategy:
  1. Calculate mid price from bid/ask
  2. Place limit order at mid (or mid + 1¢ for urgency)
  3. Monitor for fill
  4. If not filled in N seconds, re-price closer to market
  5. After max attempts, either fill at market or abandon

Usage:
    from shared.limit_order_manager import LimitOrderManager
    
    manager = LimitOrderManager(client=kalshi_client)
    
    result = manager.execute_limit_order(
        ticker="KXMLBPTS-...",
        side="yes",
        contracts=10,
        max_price_cents=52,  # Won't pay more than 52¢
        urgency="normal",    # "low", "normal", "high"
    )
    
    if result.filled:
        logger.info("Filled %d @ %d¢ (saved %d¢)", 
            result.contracts_filled, result.fill_price, result.spread_saved)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Callable
import uuid

logger = logging.getLogger(__name__)


class OrderUrgency(Enum):
    LOW = "low"        # Wait longer for fill, more aggressive on price
    NORMAL = "normal"  # Balanced
    HIGH = "high"      # Fill quickly, less price improvement


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class LimitOrderResult:
    """Result of limit order execution."""
    status: OrderStatus
    ticker: str
    side: str
    contracts_requested: int
    contracts_filled: int
    fill_price: Optional[int]  # Average fill price in cents
    spread_saved: int  # Cents saved vs market order
    attempts: int
    total_time_sec: float
    order_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class OpenOrder:
    """Tracks an open limit order."""
    order_id: str
    ticker: str
    side: str
    contracts: int
    limit_price: int
    placed_at: float
    attempt: int
    max_price: int


class LimitOrderManager:
    """
    Manage limit order execution for better fills.
    
    Instead of paying the ask (market order), we:
    1. Place limit at mid
    2. Wait for fill
    3. Re-price if needed
    4. Track savings
    """
    
    # Timing configuration by urgency
    URGENCY_CONFIG = {
        OrderUrgency.LOW: {
            "initial_offset": 0,      # Start at mid
            "reprice_interval_sec": 10,
            "max_attempts": 5,
            "final_market_order": False,
        },
        OrderUrgency.NORMAL: {
            "initial_offset": 1,      # Start at mid + 1¢
            "reprice_interval_sec": 5,
            "max_attempts": 3,
            "final_market_order": True,
        },
        OrderUrgency.HIGH: {
            "initial_offset": 2,      # Start at mid + 2¢
            "reprice_interval_sec": 3,
            "max_attempts": 2,
            "final_market_order": True,
        },
    }
    
    def __init__(
        self,
        client,  # KalshiClient
        enabled: bool = True,
        poll_interval_sec: float = 0.5,
    ):
        self.client = client
        self.enabled = enabled
        self.poll_interval_sec = poll_interval_sec
        
        # Track open orders
        self._open_orders: Dict[str, OpenOrder] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self._total_spread_saved: int = 0
        self._total_orders: int = 0
        self._total_fills: int = 0
    
    def _get_market_prices(self, ticker: str) -> Optional[Dict[str, int]]:
        """
        Get current bid/ask/mid for a ticker.
        
        Returns dict with keys: bid, ask, mid, spread
        """
        try:
            market = self.client.get_market(ticker)
            if not market:
                return None
            
            # Extract prices (handle both cents and dollars formats)
            bid = market.get("yes_bid") or market.get("yes_bid_dollars", 0)
            ask = market.get("yes_ask") or market.get("yes_ask_dollars", 0)
            
            # Convert to cents if in dollars
            if isinstance(bid, float) and bid < 1:
                bid = int(bid * 100)
            if isinstance(ask, float) and ask < 1:
                ask = int(ask * 100)
            
            bid = int(bid) if bid else 0
            ask = int(ask) if ask else 0
            
            if bid <= 0 or ask <= 0:
                return None
            
            mid = (bid + ask) // 2
            spread = ask - bid
            
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
            }
            
        except Exception as e:
            logger.warning("[LIMIT] Failed to get prices for %s: %s", ticker, e)
            return None
    
    def _place_limit_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        limit_price: int,
    ) -> Optional[str]:
        """
        Place a limit order and return order_id.
        """
        try:
            client_order_id = str(uuid.uuid4())
            
            response = self.client.place_order(
                ticker=ticker,
                side=side,
                count=contracts,
                yes_price=limit_price,
                client_order_id=client_order_id,
                order_type="limit",  # Explicitly limit
            )
            
            order_id = response.get("order", {}).get("order_id", client_order_id)
            logger.debug(
                "[LIMIT] Placed %s %s %dx @ %d¢ → order_id=%s",
                side.upper(), ticker[:30], contracts, limit_price, order_id[:8],
            )
            return order_id
            
        except Exception as e:
            logger.warning("[LIMIT] Place order failed: %s", e)
            return None
    
    def _check_order_status(self, order_id: str) -> Dict:
        """
        Check status of an order.
        
        Returns dict with: status, filled_contracts, remaining_contracts
        """
        try:
            order = self.client.get_order(order_id)
            if not order:
                return {"status": "unknown", "filled": 0, "remaining": 0}
            
            status = order.get("status", "unknown").lower()
            filled = order.get("filled_count", 0) or order.get("filled_contracts", 0)
            remaining = order.get("remaining_count", 0) or order.get("remaining_contracts", 0)
            
            return {
                "status": status,
                "filled": int(filled),
                "remaining": int(remaining),
            }
            
        except Exception as e:
            logger.warning("[LIMIT] Check order failed: %s", e)
            return {"status": "error", "filled": 0, "remaining": 0}
    
    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.client.cancel_order(order_id)
            logger.debug("[LIMIT] Cancelled order %s", order_id[:8])
            return True
        except Exception as e:
            logger.warning("[LIMIT] Cancel failed for %s: %s", order_id[:8], e)
            return False
    
    def execute_limit_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        max_price_cents: int,
        urgency: str = "normal",
    ) -> LimitOrderResult:
        """
        Execute a limit order with smart repricing.
        
        Args:
            ticker: Market ticker
            side: "yes" or "no"
            contracts: Number of contracts
            max_price_cents: Maximum price willing to pay
            urgency: "low", "normal", "high"
            
        Returns:
            LimitOrderResult with fill details
        """
        if not self.enabled:
            # Fall back to market order behavior
            return LimitOrderResult(
                status=OrderStatus.PENDING,
                ticker=ticker,
                side=side,
                contracts_requested=contracts,
                contracts_filled=0,
                fill_price=None,
                spread_saved=0,
                attempts=0,
                total_time_sec=0,
                reason="Limit orders disabled — use market order",
            )
        
        start_time = time.time()
        
        try:
            urgency_enum = OrderUrgency(urgency.lower())
        except ValueError:
            urgency_enum = OrderUrgency.NORMAL
        
        config = self.URGENCY_CONFIG[urgency_enum]
        
        # Get current market prices
        prices = self._get_market_prices(ticker)
        if not prices:
            return LimitOrderResult(
                status=OrderStatus.FAILED,
                ticker=ticker,
                side=side,
                contracts_requested=contracts,
                contracts_filled=0,
                fill_price=None,
                spread_saved=0,
                attempts=0,
                total_time_sec=time.time() - start_time,
                reason="Could not get market prices",
            )
        
        market_price = prices["ask"] if side == "yes" else (100 - prices["bid"])
        initial_spread = prices["spread"]
        
        # Calculate initial limit price
        if side == "yes":
            limit_price = min(prices["mid"] + config["initial_offset"], max_price_cents)
        else:
            # For NO, we're selling YES / buying NO
            limit_price = max(prices["mid"] - config["initial_offset"], 100 - max_price_cents)
        
        order_ids = []
        total_filled = 0
        attempts = 0
        last_fill_price = None
        
        remaining = contracts
        
        while remaining > 0 and attempts < config["max_attempts"]:
            attempts += 1
            
            # Place limit order
            order_id = self._place_limit_order(ticker, side, remaining, limit_price)
            if not order_id:
                break
            
            order_ids.append(order_id)
            
            # Wait for fill
            wait_start = time.time()
            while time.time() - wait_start < config["reprice_interval_sec"]:
                status = self._check_order_status(order_id)
                
                if status["status"] in ("filled", "closed"):
                    total_filled += status["filled"]
                    remaining -= status["filled"]
                    last_fill_price = limit_price
                    logger.info(
                        "[LIMIT] Filled %d/%d @ %d¢ (attempt %d)",
                        status["filled"], contracts, limit_price, attempts,
                    )
                    break
                elif status["status"] == "cancelled":
                    break
                elif status["filled"] > 0:
                    # Partial fill
                    total_filled += status["filled"]
                    remaining = status["remaining"]
                    last_fill_price = limit_price
                
                time.sleep(self.poll_interval_sec)
            
            # If not fully filled, cancel and reprice
            if remaining > 0:
                self._cancel_order(order_id)
                
                # Get fresh prices
                prices = self._get_market_prices(ticker)
                if prices:
                    # Move price closer to market
                    step = (attempts + 1)  # Increasingly aggressive
                    if side == "yes":
                        limit_price = min(prices["mid"] + step, max_price_cents)
                    else:
                        limit_price = max(prices["mid"] - step, 100 - max_price_cents)
        
        # Final market order if configured and still have remaining
        if remaining > 0 and config["final_market_order"]:
            logger.info(
                "[LIMIT] Falling back to market order for remaining %d contracts",
                remaining,
            )
            # Place at max price (effectively market)
            order_id = self._place_limit_order(ticker, side, remaining, max_price_cents)
            if order_id:
                order_ids.append(order_id)
                time.sleep(1)  # Brief wait for fill
                status = self._check_order_status(order_id)
                if status["filled"] > 0:
                    total_filled += status["filled"]
                    remaining -= status["filled"]
                    last_fill_price = max_price_cents
        
        # Calculate results
        elapsed = time.time() - start_time
        
        if total_filled > 0:
            # Calculate spread saved
            if last_fill_price:
                spread_saved = (market_price - last_fill_price) * total_filled
            else:
                spread_saved = 0
            
            self._total_spread_saved += max(0, spread_saved)
            self._total_fills += 1
            
            if total_filled >= contracts:
                status = OrderStatus.FILLED
            else:
                status = OrderStatus.PARTIAL
        else:
            spread_saved = 0
            status = OrderStatus.FAILED if remaining == contracts else OrderStatus.PARTIAL
        
        self._total_orders += 1
        
        return LimitOrderResult(
            status=status,
            ticker=ticker,
            side=side,
            contracts_requested=contracts,
            contracts_filled=total_filled,
            fill_price=last_fill_price,
            spread_saved=max(0, spread_saved),
            attempts=attempts,
            total_time_sec=elapsed,
            order_ids=order_ids,
            reason=f"Filled {total_filled}/{contracts} in {attempts} attempts",
        )
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            "total_orders": self._total_orders,
            "total_fills": self._total_fills,
            "fill_rate": self._total_fills / max(1, self._total_orders),
            "total_spread_saved_cents": self._total_spread_saved,
            "avg_spread_saved": self._total_spread_saved / max(1, self._total_fills),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: Optional[LimitOrderManager] = None


def get_limit_order_manager(client=None) -> LimitOrderManager:
    """Get the global limit order manager instance."""
    global _manager
    if _manager is None:
        if client is None:
            raise ValueError("Must provide client on first call")
        _manager = LimitOrderManager(client=client)
    return _manager