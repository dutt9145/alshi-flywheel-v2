"""
pinnacle_reference.py (v1.0)

Sharp line comparison using Pinnacle odds via The Odds API.

Pinnacle is the sharpest mainstream book — their lines reflect the true
market probability better than any model we can build. Use them as a
sanity check:

  - If Pinnacle agrees (within 5%): Confirmation — proceed
  - If Pinnacle disagrees (>5%): Either skip or fade our signal
  - If Pinnacle strongly disagrees (>10%): Hard skip

Usage:
    from shared.pinnacle_reference import PinnacleReference
    
    pinnacle = PinnacleReference(api_key=ODDS_API_KEY)
    
    # For MLB player props
    check = pinnacle.check_mlb_player_prop(
        player_name="Aaron Judge",
        prop_type="batter_hits",
        line=1.5,
        our_prob=0.65,  # Our P(over)
    )
    
    if not check.passes:
        logger.info("PINNACLE VETO: %s", check.reason)
        return
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


@dataclass
class SharpCheckResult:
    """Result of sharp line comparison."""
    passes: bool
    reason: str
    our_prob: float
    sharp_prob: Optional[float]
    divergence: Optional[float]  # our_prob - sharp_prob
    confidence_adjustment: float  # Multiplier: 1.0 = no change, 0.8 = reduce, 1.1 = boost
    sharp_direction: Optional[str]  # "OVER" or "UNDER" — what sharp money favors
    

class PinnacleReference:
    """
    Compare our probabilities to Pinnacle sharp lines.
    
    Uses The Odds API to fetch Pinnacle odds for MLB/NBA player props
    and game lines.
    """
    
    # Odds API endpoints
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for Odds API
    SPORT_KEYS = {
        "mlb": "baseball_mlb",
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "nhl": "icehockey_nhl",
    }
    
    # Player prop market keys (Odds API format)
    PROP_MARKETS = {
        # MLB
        "batter_hits": "batter_hits",
        "batter_home_runs": "batter_home_runs",
        "batter_rbis": "batter_rbis",
        "batter_runs": "batter_runs_scored",
        "batter_total_bases": "batter_total_bases",
        "batter_walks": "batter_walks",
        "batter_strikeouts": "batter_strikeouts",
        "pitcher_strikeouts": "pitcher_strikeouts",
        "pitcher_outs": "pitcher_outs",
        # NBA
        "player_points": "player_points",
        "player_rebounds": "player_rebounds",
        "player_assists": "player_assists",
        "player_threes": "player_threes",
        "player_steals": "player_steals",
        "player_blocks": "player_blocks",
        "player_pra": "player_points_rebounds_assists",
    }
    
    # Kalshi prop code to Odds API market mapping
    KALSHI_TO_ODDS_API = {
        "PTS": "player_points",
        "REB": "player_rebounds",
        "AST": "player_assists",
        "3PT": "player_threes",
        "STL": "player_steals",
        "BLK": "player_blocks",
        "HR": "batter_home_runs",
        "H": "batter_hits",
        "RBI": "batter_rbis",
        "R": "batter_runs_scored",
        "TB": "batter_total_bases",
        "K": "pitcher_strikeouts",  # Context-dependent
        "SO": "batter_strikeouts",
    }
    
    # Thresholds
    SOFT_DIVERGENCE_THRESHOLD = 0.05   # 5% — reduce confidence
    HARD_DIVERGENCE_THRESHOLD = 0.10   # 10% — skip trade
    ALIGNMENT_BOOST_THRESHOLD = 0.03   # 3% — boost confidence if aligned
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 300,  # Cache odds for 5 minutes
        enabled: bool = True,
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.cache_ttl_sec = cache_ttl_sec
        self.enabled = enabled
        
        # Cache: (sport, event_id, market) -> (timestamp, odds_data)
        self._cache: Dict[str, Tuple[float, dict]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # 1 second between requests
        
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set — Pinnacle reference disabled")
            self.enabled = False
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _american_to_prob(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Examples:
            -150 → 0.60 (60% implied)
            +150 → 0.40 (40% implied)
        """
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Get cached data if not expired."""
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl_sec:
                return data
        return None
    
    def _set_cache(self, cache_key: str, data: dict) -> None:
        """Cache data with timestamp."""
        self._cache[cache_key] = (time.time(), data)
    
    def _fetch_player_props(
        self,
        sport: str,
        market: str,
    ) -> List[dict]:
        """
        Fetch player prop odds from Odds API.
        
        Returns list of events with player prop odds.
        """
        sport_key = self.SPORT_KEYS.get(sport.lower())
        if not sport_key:
            logger.warning("Unknown sport: %s", sport)
            return []
        
        cache_key = f"{sport_key}:{market}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/sports/{sport_key}/events"
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": market,
                "bookmakers": "pinnacle",
                "oddsFormat": "american",
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            self._set_cache(cache_key, data)
            
            logger.debug(
                "[PINNACLE] Fetched %d events for %s/%s",
                len(data), sport_key, market,
            )
            return data
            
        except requests.RequestException as e:
            logger.warning("[PINNACLE] API error: %s", e)
            return []
        except Exception as e:
            logger.error("[PINNACLE] Unexpected error: %s", e)
            return []
    
    def _find_player_line(
        self,
        events: List[dict],
        player_name: str,
        line: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Find Pinnacle's odds for a specific player prop line.
        
        Returns (over_prob, under_prob) if found, None otherwise.
        """
        player_lower = player_name.lower().strip()
        
        for event in events:
            bookmakers = event.get("bookmakers", [])
            
            for book in bookmakers:
                if book.get("key") != "pinnacle":
                    continue
                
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        outcome_player = outcome.get("description", "").lower().strip()
                        
                        # Fuzzy match player name
                        if player_lower not in outcome_player and outcome_player not in player_lower:
                            # Try last name only
                            player_last = player_lower.split()[-1] if " " in player_lower else player_lower
                            if player_last not in outcome_player:
                                continue
                        
                        outcome_line = outcome.get("point")
                        if outcome_line is None:
                            continue
                        
                        # Check if line matches (within 0.5)
                        if abs(float(outcome_line) - line) > 0.5:
                            continue
                        
                        outcome_name = outcome.get("name", "").upper()
                        price = outcome.get("price", 0)
                        
                        if not price:
                            continue
                        
                        prob = self._american_to_prob(price)
                        
                        if outcome_name == "OVER":
                            return (prob, 1.0 - prob)
                        elif outcome_name == "UNDER":
                            return (1.0 - prob, prob)
        
        return None
    
    def check_player_prop(
        self,
        sport: str,
        player_name: str,
        prop_type: str,
        line: float,
        our_prob: float,
        our_direction: str = "OVER",
    ) -> SharpCheckResult:
        """
        Check our player prop probability against Pinnacle's line.
        
        Args:
            sport: "mlb", "nba", "nfl", "nhl"
            player_name: Player's full name
            prop_type: Prop type (e.g., "batter_hits", "player_points")
            line: The line (e.g., 1.5, 24.5)
            our_prob: Our probability of OVER
            our_direction: "OVER" or "UNDER" — which side we're betting
            
        Returns:
            SharpCheckResult with pass/fail and diagnostics
        """
        if not self.enabled:
            return SharpCheckResult(
                passes=True,
                reason="Pinnacle reference disabled",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Map prop type to Odds API market
        market = self.PROP_MARKETS.get(prop_type) or self.KALSHI_TO_ODDS_API.get(prop_type.upper())
        if not market:
            return SharpCheckResult(
                passes=True,
                reason=f"Unknown prop type: {prop_type}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Fetch Pinnacle odds
        events = self._fetch_player_props(sport, market)
        if not events:
            logger.debug(
                "[PINNACLE] %s: No Odds API data for %s/%s",
                player_name, sport, market,
            )
            return SharpCheckResult(
                passes=True,
                reason="No Pinnacle data available",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Find the specific player/line
        line_probs = self._find_player_line(events, player_name, line)
        if not line_probs:
            logger.debug(
                "[PINNACLE] %s @ %.1f: not found in %d events",
                player_name, line, len(events),
            )
            return SharpCheckResult(
                passes=True,
                reason=f"Player/line not found: {player_name} {line}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        over_prob, under_prob = line_probs
        
        # Determine sharp probability for our direction
        if our_direction.upper() == "OVER":
            sharp_prob = over_prob
        else:
            sharp_prob = under_prob
            our_prob = 1.0 - our_prob  # Flip to compare apples to apples
        
        # Calculate divergence (positive = we're higher than sharp)
        divergence = our_prob - sharp_prob
        
        # Determine sharp's lean
        sharp_direction = "OVER" if over_prob > 0.5 else "UNDER"
        
        # Decision logic
        abs_div = abs(divergence)
        
        if abs_div > self.HARD_DIVERGENCE_THRESHOLD:
            # Hard skip — Pinnacle strongly disagrees
            return SharpCheckResult(
                passes=False,
                reason=f"Sharp divergence too high: {divergence:+.1%} (threshold: {self.HARD_DIVERGENCE_THRESHOLD:.0%})",
                our_prob=our_prob,
                sharp_prob=sharp_prob,
                divergence=divergence,
                confidence_adjustment=0.0,
                sharp_direction=sharp_direction,
            )
        
        if abs_div > self.SOFT_DIVERGENCE_THRESHOLD:
            # Soft warning — reduce confidence
            adj = 0.7  # 30% confidence reduction
            return SharpCheckResult(
                passes=True,
                reason=f"Sharp divergence warning: {divergence:+.1%} — reducing confidence",
                our_prob=our_prob,
                sharp_prob=sharp_prob,
                divergence=divergence,
                confidence_adjustment=adj,
                sharp_direction=sharp_direction,
            )
        
        if abs_div < self.ALIGNMENT_BOOST_THRESHOLD:
            # Strong alignment — boost confidence
            adj = 1.1  # 10% confidence boost
            return SharpCheckResult(
                passes=True,
                reason=f"Sharp alignment confirmed: {divergence:+.1%}",
                our_prob=our_prob,
                sharp_prob=sharp_prob,
                divergence=divergence,
                confidence_adjustment=adj,
                sharp_direction=sharp_direction,
            )
        
        # Moderate alignment — no adjustment
        return SharpCheckResult(
            passes=True,
            reason=f"Sharp check OK: {divergence:+.1%}",
            our_prob=our_prob,
            sharp_prob=sharp_prob,
            divergence=divergence,
            confidence_adjustment=1.0,
            sharp_direction=sharp_direction,
        )
    
    def check_mlb_player_prop(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        our_prob: float,
        our_direction: str = "OVER",
    ) -> SharpCheckResult:
        """Convenience method for MLB player props."""
        return self.check_player_prop(
            sport="mlb",
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
        )
    
    def check_nba_player_prop(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        our_prob: float,
        our_direction: str = "OVER",
    ) -> SharpCheckResult:
        """Convenience method for NBA player props."""
        return self.check_player_prop(
            sport="nba",
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
        )
    
    def check_from_kalshi_ticker(
        self,
        ticker: str,
        our_prob: float,
        our_direction: str,
    ) -> SharpCheckResult:
        """
        Parse a Kalshi ticker and check against Pinnacle.
        
        Ticker format examples:
            KXMLBPTS-26APR141905LAANYY-NYYAJUDGE99-2
            KXNBAPTS-26APR15GSWLAC-GSWSCURRY30-25
        """
        if not self.enabled:
            return SharpCheckResult(
                passes=True,
                reason="Pinnacle reference disabled",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        ticker_upper = ticker.upper()
        
        # Detect sport and prop type from ticker prefix
        sport = None
        prop_code = None
        
        if ticker_upper.startswith("KXMLB"):
            sport = "mlb"
            # v19.27: Expanded prop code recognition
            # Note: TOTAL, F5TOTAL are game markets, not player props — skip Pinnacle
            if "TOTAL" in ticker_upper or "F5" in ticker_upper:
                # Game totals — not a player prop, skip Pinnacle
                return SharpCheckResult(
                    passes=True,
                    reason="Game total market (not player prop)",
                    our_prob=our_prob,
                    sharp_prob=None,
                    divergence=None,
                    confidence_adjustment=1.0,
                    sharp_direction=None,
                )
            # Extract prop code: KXMLBPTS -> PTS, KXMLBHR -> HR
            # v19.27: Added KS (strikeouts), HIT (hits), HRR (home runs)
            for code in ["PTS", "HRR", "HR", "HIT", "H", "RBI", "R", "TB", "KS", "K", "SO", "BB"]:
                if f"KXMLB{code}" in ticker_upper:
                    # Map variant codes to standard
                    if code == "HRR":
                        prop_code = "HR"
                    elif code == "HIT":
                        prop_code = "H"
                    elif code == "KS":
                        prop_code = "K"  # Strikeouts
                    else:
                        prop_code = code
                    break
        elif ticker_upper.startswith("KXNBA"):
            sport = "nba"
            for code in ["PTS", "REB", "AST", "3PT", "STL", "BLK"]:
                if f"KXNBA{code}" in ticker_upper:
                    prop_code = code
                    break
        
        if not sport or not prop_code:
            # v19.27: Log when we can't parse the ticker format
            logger.debug(
                "[PINNACLE] Can't parse ticker: %s (sport=%s, prop_code=%s)",
                ticker[:35], sport, prop_code,
            )
            return SharpCheckResult(
                passes=True,
                reason=f"Cannot parse ticker: {ticker[:30]}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Parse player name and line from ticker
        # Format: KXMLBPTS-GAMEINFO-TEAMPLAYER-LINE
        parts = ticker.split("-")
        if len(parts) < 4:
            return SharpCheckResult(
                passes=True,
                reason=f"Cannot parse ticker parts: {ticker[:30]}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        try:
            # Player part: NYYAJUDGE99 -> extract AJUDGE
            player_part = parts[2]
            # Remove team prefix (3 chars) and jersey number suffix
            team = player_part[:3]
            player_raw = player_part[3:]
            # Remove trailing numbers (jersey)
            player_name = "".join(c for c in player_raw if not c.isdigit())
            
            # Line is the last part
            line = float(parts[3])
            
        except (IndexError, ValueError) as e:
            return SharpCheckResult(
                passes=True,
                reason=f"Cannot parse player/line: {e}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Map Kalshi prop code to Odds API market
        prop_type = self.KALSHI_TO_ODDS_API.get(prop_code)
        if not prop_type:
            return SharpCheckResult(
                passes=True,
                reason=f"Unknown prop code: {prop_code}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # v19.27: Log successful ticker parse for debugging
        logger.debug(
            "[PINNACLE] Parsed %s: sport=%s player=%s prop=%s line=%.1f",
            ticker[:30], sport, player_name, prop_type, line,
        )
        
        return self.check_player_prop(
            sport=sport,
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_reference: Optional[PinnacleReference] = None


def get_pinnacle_reference() -> PinnacleReference:
    """Get the global Pinnacle reference instance."""
    global _reference
    if _reference is None:
        _reference = PinnacleReference()
    return _reference


def check_sharp_line(ticker: str, our_prob: float, our_direction: str) -> SharpCheckResult:
    """Convenience function to check a ticker against Pinnacle."""
    return get_pinnacle_reference().check_from_kalshi_ticker(ticker, our_prob, our_direction)