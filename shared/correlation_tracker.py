"""
correlation_tracker.py (v1.0)

Enhanced correlation-aware sizing for prediction markets.

Problem: If you bet on 5 props in the same MLB game, those aren't 5 independent
bets. Game-level factors (weather, umpire, pitcher) affect all props. Treating
them as independent overstates your edge and leads to overexposure.

Solution: Track bets by correlated groups (game, event, player) and discount
Kelly by 1/sqrt(n) for each additional correlated bet.

Correlation Groups:
  1. Same Game: MLB/NBA/NFL props in same game (highest correlation)
  2. Same Event: Props in same event series (medium correlation)  
  3. Same Player: Multiple props on same player (high correlation)
  4. Same Team: Multiple bets on same team across games (low correlation)

Usage:
    from shared.correlation_tracker import CorrelationTracker
    
    tracker = CorrelationTracker()
    
    # Before sizing
    discount = tracker.get_discount("KXMLBPTS-26APR141905LAANYY-NYYAJUDGE99-2")
    adjusted_stake = raw_stake * discount
    
    # After trade
    tracker.record_trade("KXMLBPTS-26APR141905LAANYY-NYYAJUDGE99-2", 25.0)
    
    # Reset at start of each scan
    tracker.reset()
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class CorrelationGroup:
    """Tracks a group of correlated positions."""
    group_key: str
    group_type: str  # "game", "player", "team", "event"
    tickers: List[str] = field(default_factory=list)
    total_exposure: float = 0.0
    trade_count: int = 0


@dataclass
class CorrelationResult:
    """Result of correlation check."""
    discount: float               # Multiplier (0-1) to apply to Kelly
    groups: List[str]             # Group keys this ticker belongs to
    prior_trades_in_groups: int   # Total prior trades across groups
    rationale: str                # Explanation


class CorrelationTracker:
    """
    Track correlated positions and compute sizing discounts.
    
    Discount formula: 1 / sqrt(1 + prior_trades_in_same_group)
    
    With multiple groups, we use the most restrictive (lowest) discount.
    """
    
    # Correlation strength by group type (higher = more correlated)
    GROUP_WEIGHTS = {
        "game": 1.0,      # Same game — very correlated
        "player": 0.9,    # Same player — highly correlated
        "event": 0.7,     # Same event series — moderately correlated
        "team": 0.3,      # Same team diff games — weakly correlated
    }
    
    # Minimum discount (floor to prevent positions going to zero)
    MIN_DISCOUNT = 0.15
    
    # Ticker parsing patterns
    PATTERNS = {
        # MLB: KXMLBPTS-26APR141905LAANYY-NYYAJUDGE99-2
        "mlb_player_prop": re.compile(
            r'^(KXMLB[A-Z]+)-(\d{2}[A-Z]{3}\d{6}[A-Z]{6})-([A-Z]{3})([A-Z]+\d+)-(\d+)$',
            re.IGNORECASE
        ),
        # MLB game-level: KXMLBF5-26APR141905LAANYY-NYY
        "mlb_game": re.compile(
            r'^(KXMLB[A-Z0-9]+)-(\d{2}[A-Z]{3}\d{6}[A-Z]{6})-([A-Z]+)$',
            re.IGNORECASE
        ),
        # NBA: KXNBAPTS-26APR15GSWLAC-GSWKPORZINGIS7-2
        "nba_player_prop": re.compile(
            r'^(KXNBA[A-Z]+)-(\d{2}[A-Z]{3}\d{2}[A-Z]{6})-([A-Z]{3})([A-Z]+\d+)-(\d+)$',
            re.IGNORECASE
        ),
        # NBA game: KXNBAGAME-26APR18MINDEN-MIN
        "nba_game": re.compile(
            r'^(KXNBA[A-Z0-9]+)-(\d{2}[A-Z]{3}\d{2}[A-Z]{6})-([A-Z]+)$',
            re.IGNORECASE
        ),
        # Generic sports: KXATPCHALLENGERMATCH-26APR14BERLAJ-BER
        "sports_generic": re.compile(
            r'^(KX[A-Z0-9]+)-(\d{2}[A-Z]{3}\d{2,6}[A-Z0-9]+)-([A-Z0-9]+)$',
            re.IGNORECASE
        ),
    }
    
    def __init__(self):
        # group_key -> CorrelationGroup
        self._groups: Dict[str, CorrelationGroup] = {}
        # ticker -> set of group_keys
        self._ticker_groups: Dict[str, Set[str]] = defaultdict(set)
        # Exposure tracking
        self._exposures: Dict[str, float] = defaultdict(float)
    
    def reset(self) -> None:
        """Reset tracker at start of each scan."""
        self._groups.clear()
        self._ticker_groups.clear()
        self._exposures.clear()
        logger.debug("[CORR TRACKER] Reset")
    
    def _parse_ticker(self, ticker: str) -> Dict[str, str]:
        """
        Extract components from a ticker.
        
        Returns dict with keys: market_type, game_id, team, player, line
        """
        result = {
            "market_type": "",
            "game_id": "",
            "team": "",
            "player": "",
            "line": "",
            "raw": ticker,
        }
        
        ticker_upper = ticker.upper()
        
        # Try MLB player prop
        m = self.PATTERNS["mlb_player_prop"].match(ticker_upper)
        if m:
            result["market_type"] = m.group(1)  # KXMLBPTS, KXMLBHR, etc.
            result["game_id"] = m.group(2)       # 26APR141905LAANYY
            result["team"] = m.group(3)          # NYY
            result["player"] = m.group(4)        # AJUDGE99
            result["line"] = m.group(5)          # 2
            return result
        
        # Try NBA player prop
        m = self.PATTERNS["nba_player_prop"].match(ticker_upper)
        if m:
            result["market_type"] = m.group(1)
            result["game_id"] = m.group(2)
            result["team"] = m.group(3)
            result["player"] = m.group(4)
            result["line"] = m.group(5)
            return result
        
        # Try MLB game-level
        m = self.PATTERNS["mlb_game"].match(ticker_upper)
        if m:
            result["market_type"] = m.group(1)
            result["game_id"] = m.group(2)
            result["team"] = m.group(3)
            return result
        
        # Try NBA game-level
        m = self.PATTERNS["nba_game"].match(ticker_upper)
        if m:
            result["market_type"] = m.group(1)
            result["game_id"] = m.group(2)
            result["team"] = m.group(3)
            return result
        
        # Try generic sports
        m = self.PATTERNS["sports_generic"].match(ticker_upper)
        if m:
            result["market_type"] = m.group(1)
            result["game_id"] = m.group(2)
            result["team"] = m.group(3)
            return result
        
        return result
    
    def _get_group_keys(self, ticker: str) -> List[Tuple[str, str]]:
        """
        Get all correlation group keys for a ticker.
        
        Returns list of (group_key, group_type) tuples.
        """
        parsed = self._parse_ticker(ticker)
        groups = []
        
        # Game-level correlation (most important)
        if parsed["game_id"]:
            game_key = f"game:{parsed['game_id']}"
            groups.append((game_key, "game"))
        
        # Player-level correlation
        if parsed["player"]:
            player_key = f"player:{parsed['team']}{parsed['player']}"
            groups.append((player_key, "player"))
        
        # Team-level correlation (within same game)
        if parsed["team"] and parsed["game_id"]:
            team_game_key = f"team:{parsed['team']}@{parsed['game_id']}"
            groups.append((team_game_key, "team"))
        
        # Event-level (for tournament/series)
        # Extract event from market_type if present
        if parsed["market_type"] and "SERIES" in parsed["market_type"]:
            event_key = f"event:{parsed['market_type']}"
            groups.append((event_key, "event"))
        
        return groups
    
    def get_discount(self, ticker: str) -> CorrelationResult:
        """
        Calculate correlation discount for a new trade.
        
        Should be called BEFORE recording the trade.
        """
        group_keys = self._get_group_keys(ticker)
        
        if not group_keys:
            return CorrelationResult(
                discount=1.0,
                groups=[],
                prior_trades_in_groups=0,
                rationale="No correlation groups detected",
            )
        
        # Find most restrictive group (highest prior trade count × weight)
        max_weighted_count = 0
        total_prior = 0
        group_names = []
        
        for group_key, group_type in group_keys:
            group_names.append(group_key)
            
            if group_key in self._groups:
                group = self._groups[group_key]
                prior = group.trade_count
                total_prior += prior
                
                weight = self.GROUP_WEIGHTS.get(group_type, 0.5)
                weighted = prior * weight
                
                if weighted > max_weighted_count:
                    max_weighted_count = weighted
        
        if max_weighted_count == 0:
            return CorrelationResult(
                discount=1.0,
                groups=group_names,
                prior_trades_in_groups=0,
                rationale="First trade in correlation groups",
            )
        
        # Discount = 1 / sqrt(1 + weighted_count)
        discount = 1.0 / math.sqrt(1 + max_weighted_count)
        discount = max(discount, self.MIN_DISCOUNT)
        
        return CorrelationResult(
            discount=round(discount, 3),
            groups=group_names,
            prior_trades_in_groups=total_prior,
            rationale=f"{total_prior} prior trades in groups → {discount:.1%} of Kelly",
        )
    
    def record_trade(self, ticker: str, dollars: float) -> None:
        """
        Record a trade for correlation tracking.
        
        Should be called AFTER the trade is executed.
        """
        group_keys = self._get_group_keys(ticker)
        
        for group_key, group_type in group_keys:
            if group_key not in self._groups:
                self._groups[group_key] = CorrelationGroup(
                    group_key=group_key,
                    group_type=group_type,
                )
            
            group = self._groups[group_key]
            group.tickers.append(ticker)
            group.total_exposure += dollars
            group.trade_count += 1
            
            self._ticker_groups[ticker].add(group_key)
        
        self._exposures[ticker] = dollars
        
        logger.debug(
            "[CORR TRACKER] Recorded %s ($%.2f) in groups: %s",
            ticker[:40], dollars, [g[0] for g in group_keys],
        )
    
    def get_group_exposure(self, group_key: str) -> float:
        """Get total exposure in a correlation group."""
        if group_key in self._groups:
            return self._groups[group_key].total_exposure
        return 0.0
    
    def get_ticker_groups(self, ticker: str) -> Set[str]:
        """Get all groups a ticker belongs to."""
        return self._ticker_groups.get(ticker, set())
    
    def summary(self) -> Dict[str, any]:
        """Get summary of current correlation state."""
        return {
            "total_groups": len(self._groups),
            "total_tickers": len(self._exposures),
            "groups_by_type": {
                gtype: sum(1 for g in self._groups.values() if g.group_type == gtype)
                for gtype in ["game", "player", "team", "event"]
            },
            "top_groups": sorted(
                [(k, v.trade_count, v.total_exposure) for k, v in self._groups.items()],
                key=lambda x: -x[1]
            )[:5],
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracker: Optional[CorrelationTracker] = None


def get_correlation_tracker() -> CorrelationTracker:
    """Get the global correlation tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = CorrelationTracker()
    return _tracker


def get_correlation_discount(ticker: str) -> float:
    """Convenience function to get discount for a ticker."""
    return get_correlation_tracker().get_discount(ticker).discount


def record_correlated_trade(ticker: str, dollars: float) -> None:
    """Convenience function to record a trade."""
    get_correlation_tracker().record_trade(ticker, dollars)