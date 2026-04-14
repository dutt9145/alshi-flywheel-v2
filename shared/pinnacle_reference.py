"""
pinnacle_reference.py (v1.11 — Fixed Team Extraction)

Changes vs v1.10:
  1. FIXED: Team extraction now handles 2-letter codes (TB, SF, KC, AZ, etc.)
     - Was grabbing "0TB" instead of "TB" from "26APR141940TBCWS"
     - Now tries all valid splits (2+2, 2+3, 3+2, 3+3) and validates both codes

Changes vs v1.9:
  1. MLB/NBA ONLY: Explicitly skips all other sports (NFL, NHL, etc.)
     to save API calls on sports we don't trade props on.

Changes vs v1.8:
  1. SMART EVENT MATCHING: Extract game teams from Kalshi ticker
     and query ONLY the matching event (1 API call instead of 30)
  2. Added team code mappings for MLB and NBA
  3. Added _extract_game_teams() to parse ticker game info
  4. Added _find_event_id() to find matching event by teams
  5. Modified _fetch_player_props() to accept optional game_teams filter

API call reduction: ~97% fewer calls (1 vs 30)
Free tier (500 req/mo) should now be sufficient.

Sharp line comparison using Pinnacle odds via The Odds API.
"""

import logging
import os
import re
import time
import unicodedata
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
    

def _normalize_to_kalshi_format(full_name: str) -> str:
    """
    Convert Odds API full name to Kalshi ticker format.
    
    Kalshi format: First initial + remaining name parts joined (no spaces), uppercase
    
    Examples:
        "Yordan Alvarez"      → "YALVAREZ"
        "Ronald Acuña Jr."    → "RACUNA"
        "Elly De La Cruz"     → "EDELACRUZ"
        "Stephen Curry"       → "SCURRY"
        "José Ramírez"        → "JRAMIREZ"
        "Shohei Ohtani"       → "SOHTANI"
        "Paolo Banchero"      → "PBANCHERO"
        "Matt Olson"          → "MOLSON"
    """
    if not full_name:
        return ""
    
    # Strip accents (ñ → n, é → e, í → i, etc.)
    name = unicodedata.normalize('NFD', full_name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    
    # Remove common suffixes
    for suffix in [' Jr.', ' Jr', ' Sr.', ' Sr', ' III', ' II', ' IV', ' V', '.']:
        name = name.replace(suffix, '')
    
    # Remove any non-alphabetic characters except spaces
    name = re.sub(r"[^a-zA-Z\s]", "", name)
    
    # Split into parts and filter empty
    parts = [p.strip() for p in name.split() if p.strip()]
    
    if not parts:
        return ""
    
    if len(parts) == 1:
        return parts[0].upper()
    
    # First initial + remaining parts joined (no spaces)
    first_initial = parts[0][0].upper()
    rest = ''.join(parts[1:]).upper()
    
    return first_initial + rest


def _normalize_kalshi_player(kalshi_name: str) -> str:
    """
    Normalize Kalshi player format for comparison.
    
    Already in format like "YALVAREZ", just uppercase and strip.
    """
    return kalshi_name.upper().strip()


# ── Team Code Mappings ────────────────────────────────────────────────────────

# Kalshi 3-letter codes → Odds API team name substring for matching
MLB_TEAM_CODES = {
    "ARI": ["Arizona Diamondbacks", "Diamondbacks"],
    "ATL": ["Atlanta Braves", "Braves"],
    "BAL": ["Baltimore Orioles", "Orioles"],
    "BOS": ["Boston Red Sox", "Red Sox"],
    "CHC": ["Chicago Cubs", "Cubs"],
    "CWS": ["Chicago White Sox", "White Sox"],
    "CHW": ["Chicago White Sox", "White Sox"],  # Alternate code
    "CIN": ["Cincinnati Reds", "Reds"],
    "CLE": ["Cleveland Guardians", "Guardians"],
    "COL": ["Colorado Rockies", "Rockies"],
    "DET": ["Detroit Tigers", "Tigers"],
    "HOU": ["Houston Astros", "Astros"],
    "KC": ["Kansas City Royals", "Royals"],
    "KCR": ["Kansas City Royals", "Royals"],  # Alternate
    "LAA": ["Los Angeles Angels", "LA Angels", "Angels"],
    "LAD": ["Los Angeles Dodgers", "LA Dodgers", "Dodgers"],
    "MIA": ["Miami Marlins", "Marlins"],
    "MIL": ["Milwaukee Brewers", "Brewers"],
    "MIN": ["Minnesota Twins", "Twins"],
    "NYM": ["New York Mets", "NY Mets", "Mets"],
    "NYY": ["New York Yankees", "NY Yankees", "Yankees"],
    "OAK": ["Oakland Athletics", "Athletics", "A's"],
    "PHI": ["Philadelphia Phillies", "Phillies"],
    "PIT": ["Pittsburgh Pirates", "Pirates"],
    "SD": ["San Diego Padres", "Padres"],
    "SDP": ["San Diego Padres", "Padres"],  # Alternate
    "SF": ["San Francisco Giants", "Giants"],
    "SFG": ["San Francisco Giants", "Giants"],  # Alternate
    "SEA": ["Seattle Mariners", "Mariners"],
    "STL": ["St. Louis Cardinals", "Cardinals"],
    "TB": ["Tampa Bay Rays", "Rays"],
    "TBR": ["Tampa Bay Rays", "Rays"],  # Alternate
    "TEX": ["Texas Rangers", "Rangers"],
    "TOR": ["Toronto Blue Jays", "Blue Jays"],
    "WSH": ["Washington Nationals", "Nationals"],
    "WAS": ["Washington Nationals", "Nationals"],  # Alternate
}

NBA_TEAM_CODES = {
    "ATL": ["Atlanta Hawks", "Hawks"],
    "BOS": ["Boston Celtics", "Celtics"],
    "BKN": ["Brooklyn Nets", "Nets"],
    "BRK": ["Brooklyn Nets", "Nets"],  # Alternate
    "CHA": ["Charlotte Hornets", "Hornets"],
    "CHI": ["Chicago Bulls", "Bulls"],
    "CLE": ["Cleveland Cavaliers", "Cavaliers", "Cavs"],
    "DAL": ["Dallas Mavericks", "Mavericks", "Mavs"],
    "DEN": ["Denver Nuggets", "Nuggets"],
    "DET": ["Detroit Pistons", "Pistons"],
    "GSW": ["Golden State Warriors", "Warriors"],
    "GS": ["Golden State Warriors", "Warriors"],  # Alternate
    "HOU": ["Houston Rockets", "Rockets"],
    "IND": ["Indiana Pacers", "Pacers"],
    "LAC": ["Los Angeles Clippers", "LA Clippers", "Clippers"],
    "LAL": ["Los Angeles Lakers", "LA Lakers", "Lakers"],
    "MEM": ["Memphis Grizzlies", "Grizzlies"],
    "MIA": ["Miami Heat", "Heat"],
    "MIL": ["Milwaukee Bucks", "Bucks"],
    "MIN": ["Minnesota Timberwolves", "Timberwolves", "Wolves"],
    "NOP": ["New Orleans Pelicans", "Pelicans"],
    "NO": ["New Orleans Pelicans", "Pelicans"],  # Alternate
    "NYK": ["New York Knicks", "NY Knicks", "Knicks"],
    "OKC": ["Oklahoma City Thunder", "Thunder"],
    "ORL": ["Orlando Magic", "Magic"],
    "PHI": ["Philadelphia 76ers", "76ers", "Sixers"],
    "PHX": ["Phoenix Suns", "Suns"],
    "POR": ["Portland Trail Blazers", "Trail Blazers", "Blazers"],
    "SAC": ["Sacramento Kings", "Kings"],
    "SAS": ["San Antonio Spurs", "Spurs"],
    "SA": ["San Antonio Spurs", "Spurs"],  # Alternate
    "TOR": ["Toronto Raptors", "Raptors"],
    "UTA": ["Utah Jazz", "Jazz"],
    "WAS": ["Washington Wizards", "Wizards"],
}


class PinnacleReference:
    """
    Compare our probabilities to Pinnacle sharp lines.
    
    Uses The Odds API to fetch Pinnacle odds for MLB/NBA player props
    and game lines.
    
    v1.9: Smart event matching — extracts game teams from Kalshi ticker
    and queries only the matching event. 97% API call reduction.
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
        
        # v1.2: Player name cache - maps normalized Kalshi name to Odds API description
        # This avoids repeated normalization and helps with logging
        self._player_name_cache: Dict[str, str] = {}
        
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
    
    def _extract_game_teams(
        self,
        ticker: str,
        sport: str,
    ) -> Optional[Tuple[str, str]]:
        """
        v1.9: Extract away and home team codes from Kalshi ticker.
        v1.10: Fixed to handle 2-letter codes (TB, SF, KC, AZ, etc.)
        
        Ticker format: KXMLBKS-26APR141940TBCWS-CWSNSCHULTZ75-8
          - parts[1] = game info = "26APR141940TBCWS"
          - Teams are at the END, but can be 2+2, 2+3, 3+2, or 3+3 chars
        
        Returns (away_team_code, home_team_code) or None if can't parse.
        """
        parts = ticker.upper().split("-")
        if len(parts) < 2:
            return None
        
        game_info = parts[1]
        
        # Teams are at the end of game_info, after the datetime
        # Datetime is typically 11-12 chars: "26APR141940" (date + time)
        # Try extracting teams from the end by testing known team codes
        
        team_map = MLB_TEAM_CODES if sport.lower() == "mlb" else NBA_TEAM_CODES
        
        # Try different team string lengths (4 to 6 chars from the end)
        # Possible combos: 2+2=4, 2+3=5, 3+2=5, 3+3=6
        for teams_len in [6, 5, 4]:
            if len(game_info) < teams_len:
                continue
            
            teams_str = game_info[-teams_len:]
            
            # Try all possible split points
            for split_pos in range(2, min(4, teams_len - 1)):
                away_code = teams_str[:split_pos]
                home_code = teams_str[split_pos:]
                
                # Check if BOTH codes are valid
                if away_code in team_map and home_code in team_map:
                    logger.info(
                        "[PINNACLE] v1.10 Extracted teams: %s @ %s from ticker",
                        away_code, home_code,
                    )
                    return (away_code, home_code)
        
        # Fallback: couldn't match teams
        logger.info(
            "[PINNACLE] v1.10 Could not extract teams from: %s",
            game_info[-8:] if len(game_info) >= 8 else game_info,
        )
        return None
    
    def _find_event_id(
        self,
        events: List[dict],
        game_teams: Tuple[str, str],
        sport: str,
    ) -> Optional[str]:
        """
        v1.9: Find the event ID matching the game teams.
        
        Matches Kalshi team codes to Odds API team names.
        
        Returns event_id or None if no match.
        """
        away_code, home_code = game_teams
        team_map = MLB_TEAM_CODES if sport.lower() == "mlb" else NBA_TEAM_CODES
        
        # Get possible team name variants
        away_names = team_map.get(away_code, [away_code])
        home_names = team_map.get(home_code, [home_code])
        
        for event in events:
            event_home = event.get("home_team", "")
            event_away = event.get("away_team", "")
            
            # Check if any variant matches
            away_match = any(
                name.lower() in event_away.lower() or event_away.lower() in name.lower()
                for name in away_names
            )
            home_match = any(
                name.lower() in event_home.lower() or event_home.lower() in name.lower()
                for name in home_names
            )
            
            if away_match and home_match:
                event_id = event.get("id")
                logger.info(
                    "[PINNACLE] v1.9 SMART MATCH: %s @ %s → event %s (%s vs %s)",
                    away_code, home_code, event_id[:12] if event_id else "?",
                    event_away, event_home,
                )
                return event_id
        
        logger.info(
            "[PINNACLE] v1.9 No event match for %s @ %s in %d events",
            away_code, home_code, len(events),
        )
        return None
    
    def _fetch_player_props(
        self,
        sport: str,
        market: str,
        game_teams: Optional[Tuple[str, str]] = None,
    ) -> List[dict]:
        """
        Fetch player prop odds from Odds API.
        
        v1.9: SMART EVENT MATCHING
          - If game_teams provided, query ONLY the matching event (1 API call)
          - Falls back to limited batch if no match found
        
        v1.5: Player props require a two-step approach:
          1. Get event IDs from /sports/{sport}/events
          2. Query /sports/{sport}/events/{event_id}/odds for each event
        
        Returns list of events with player prop odds.
        """
        sport_key = self.SPORT_KEYS.get(sport.lower())
        if not sport_key:
            logger.warning("Unknown sport: %s", sport)
            return []
        
        # v1.9: Include game_teams in cache key for targeted lookups
        if game_teams:
            cache_key = f"{sport_key}:{market}:{game_teams[0]}@{game_teams[1]}"
        else:
            cache_key = f"{sport_key}:{market}"
        
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            # Step 1: Get list of events for this sport
            events_cache_key = f"{sport_key}:events"
            events_list = self._get_cached(events_cache_key)
            
            if not events_list:
                events_url = f"{self.BASE_URL}/sports/{sport_key}/events"
                events_resp = requests.get(
                    events_url,
                    params={"apiKey": self.api_key},
                    timeout=10,
                )
                events_resp.raise_for_status()
                events_list = events_resp.json()
                self._set_cache(events_cache_key, events_list)
                logger.info("[PINNACLE] Fetched %d events for %s", len(events_list), sport_key)
            
            if not events_list:
                return []
            
            # v1.9: SMART EVENT MATCHING
            events_to_query = []
            
            if game_teams:
                # Find the specific event for this game
                target_event_id = self._find_event_id(events_list, game_teams, sport)
                if target_event_id:
                    # Query ONLY this event (1 API call instead of 30!)
                    events_to_query = [{"id": target_event_id}]
                    logger.info(
                        "[PINNACLE] v1.9 SMART MODE: querying 1 event (vs 30 batch)",
                    )
                else:
                    # Fallback: couldn't match teams, query limited batch
                    events_to_query = events_list[:10]
                    logger.info(
                        "[PINNACLE] v1.9 FALLBACK: no team match, querying %d events",
                        len(events_to_query),
                    )
            else:
                # No game_teams provided, use old batch mode
                events_to_query = events_list[:30]
            
            # Step 2: Query odds for selected events
            all_odds = []
            events_queried = 0
            
            for event in events_to_query:
                event_id = event.get("id")
                if not event_id:
                    continue
                
                self._rate_limit()
                events_queried += 1
                
                odds_url = f"{self.BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
                params = {
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": market,
                    "bookmakers": "pinnacle",
                    "oddsFormat": "american",
                }
                
                try:
                    odds_resp = requests.get(odds_url, params=params, timeout=10)
                    if odds_resp.status_code == 200:
                        odds_data = odds_resp.json()
                        # The response is a single event with bookmakers
                        if odds_data and odds_data.get("bookmakers"):
                            all_odds.append(odds_data)
                except requests.RequestException:
                    # Individual event failed, continue to next
                    continue
            
            if all_odds:
                logger.info(
                    "[PINNACLE] Fetched odds from %d/%d events for %s/%s",
                    len(all_odds), events_queried, sport_key, market,
                )
            else:
                logger.info(
                    "[PINNACLE] No odds found in %d events for %s/%s",
                    events_queried, sport_key, market,
                )
            
            self._set_cache(cache_key, all_odds)
            return all_odds
            
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
        
        v1.2: Uses normalized name comparison:
          1. Normalize Kalshi player name (e.g., "YALVAREZ")
          2. Normalize each Odds API player name to same format
          3. Compare normalized forms
          4. Fall back to fuzzy matching if needed
        
        v1.3: Added debug logging to show Pinnacle names being compared
        
        Returns (over_prob, under_prob) if found, None otherwise.
        """
        # Normalize the Kalshi player name
        kalshi_normalized = _normalize_kalshi_player(player_name)
        
        # v1.3: Track all Pinnacle names for debug logging
        pinnacle_names_seen = []
        
        for event in events:
            bookmakers = event.get("bookmakers", [])
            
            for book in bookmakers:
                if book.get("key") != "pinnacle":
                    continue
                
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        outcome_player = outcome.get("description", "").strip()
                        
                        if not outcome_player:
                            continue
                        
                        # v1.3: Track names for debug logging (first 10 unique)
                        if len(pinnacle_names_seen) < 10 and outcome_player not in pinnacle_names_seen:
                            pinnacle_names_seen.append(outcome_player)
                        
                        # v1.2: Normalize Odds API name to Kalshi format
                        odds_api_normalized = _normalize_to_kalshi_format(outcome_player)
                        
                        # Primary: Exact match on normalized names
                        if kalshi_normalized != odds_api_normalized:
                            # Secondary: Check if Kalshi name is contained (for partial matches)
                            # e.g., "CURRY" in "SCURRY" for "S. Curry" abbreviations
                            if len(kalshi_normalized) > 3:
                                # Try matching last name portion
                                kalshi_lastname = kalshi_normalized[1:]  # Remove first initial
                                odds_lastname = odds_api_normalized[1:] if odds_api_normalized else ""
                                if kalshi_lastname != odds_lastname:
                                    continue
                            else:
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
                        
                        # v1.2: Cache successful match and log it
                        if kalshi_normalized not in self._player_name_cache:
                            self._player_name_cache[kalshi_normalized] = outcome_player
                            logger.info(
                                "[PINNACLE] MATCH: %s → '%s' (normalized: %s)",
                                kalshi_normalized, outcome_player, odds_api_normalized,
                            )
                        
                        if outcome_name == "OVER":
                            return (prob, 1.0 - prob)
                        elif outcome_name == "UNDER":
                            return (1.0 - prob, prob)
        
        # v1.3: Log debug info when player not found
        if pinnacle_names_seen:
            logger.info(
                "[PINNACLE] DEBUG %s: Not found. Sample Pinnacle names: %s",
                kalshi_normalized, pinnacle_names_seen[:5],
            )
        else:
            logger.info(
                "[PINNACLE] DEBUG %s: No player outcomes found in %d events",
                kalshi_normalized, len(events),
            )
        
        return None
    
    # v1.10: Only check Pinnacle for these sports (saves API calls)
    SUPPORTED_SPORTS = {"mlb", "nba"}
    
    def check_player_prop(
        self,
        sport: str,
        player_name: str,
        prop_type: str,
        line: float,
        our_prob: float,
        our_direction: str = "OVER",
        game_teams: Optional[Tuple[str, str]] = None,
    ) -> SharpCheckResult:
        """
        Check a player prop against Pinnacle sharp line.
        
        v1.9: Accepts optional game_teams for smart event matching.
        v1.10: MLB and NBA only — skips other sports to save API calls.
        
        Args:
            sport: "mlb" or "nba"
            player_name: Kalshi format player name (e.g., "YALVAREZ")
            prop_type: Odds API market key (e.g., "pitcher_strikeouts")
            line: The line value (e.g., 7.5)
            our_prob: Our probability for the OVER
            our_direction: "OVER" or "UNDER"
            game_teams: Optional (away_code, home_code) for smart matching
        
        Returns:
            SharpCheckResult with pass/fail and adjustments
        """
        # v1.10: MLB/NBA only — skip other sports
        if sport.lower() not in self.SUPPORTED_SPORTS:
            return SharpCheckResult(
                passes=True,
                reason=f"Pinnacle skipped for {sport} (MLB/NBA only)",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
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
        
        # Map to Odds API market
        market = self.PROP_MARKETS.get(prop_type)
        if not market:
            return SharpCheckResult(
                passes=True,
                reason=f"No Odds API mapping for {prop_type}",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # v1.9: Pass game_teams to fetch (smart matching)
        events = self._fetch_player_props(sport, market, game_teams=game_teams)
        
        if not events:
            return SharpCheckResult(
                passes=True,
                reason="No Pinnacle odds available",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        # Find player's line
        odds = self._find_player_line(events, player_name, line)
        
        if not odds:
            return SharpCheckResult(
                passes=True,
                reason=f"Player {player_name} not found in Pinnacle odds",
                our_prob=our_prob,
                sharp_prob=None,
                divergence=None,
                confidence_adjustment=1.0,
                sharp_direction=None,
            )
        
        over_prob, under_prob = odds
        
        # Compare our probability to Pinnacle's
        if our_direction == "OVER":
            sharp_prob = over_prob
        else:
            sharp_prob = under_prob
        
        divergence = our_prob - sharp_prob
        abs_div = abs(divergence)
        
        # Determine sharp direction (what sharp money favors)
        sharp_direction = "OVER" if over_prob > 0.5 else "UNDER"
        
        # Log the comparison
        logger.info(
            "[PINNACLE] %s @ %.1f: sharp=%.1f%% ours=%.1f%% div=%+.1f%%",
            player_name, line, sharp_prob * 100, our_prob * 100, divergence * 100,
        )
        
        if abs_div > self.HARD_DIVERGENCE_THRESHOLD:
            # Hard veto — skip this trade
            return SharpCheckResult(
                passes=False,
                reason=f"Sharp VETO: {divergence:+.1%} divergence",
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
        game_teams: Optional[Tuple[str, str]] = None,
    ) -> SharpCheckResult:
        """Convenience method for MLB player props."""
        return self.check_player_prop(
            sport="mlb",
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
            game_teams=game_teams,
        )
    
    def check_nba_player_prop(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        our_prob: float,
        our_direction: str = "OVER",
        game_teams: Optional[Tuple[str, str]] = None,
    ) -> SharpCheckResult:
        """Convenience method for NBA player props."""
        return self.check_player_prop(
            sport="nba",
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
            game_teams=game_teams,
        )
    
    def check_from_kalshi_ticker(
        self,
        ticker: str,
        our_prob: float,
        our_direction: str,
    ) -> SharpCheckResult:
        """
        Parse a Kalshi ticker and check against Pinnacle.
        
        v1.9: Extracts game teams for smart event matching.
        
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
            # v1.1: Log at INFO when we can't parse the ticker format
            logger.info(
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
        
        # v1.9: Extract game teams for smart matching
        game_teams = self._extract_game_teams(ticker, sport)
        
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
        
        # v1.9: Log with game teams
        logger.info(
            "[PINNACLE] Checking %s: sport=%s player=%s prop=%s line=%.1f teams=%s",
            ticker[:30], sport, player_name, prop_type, line,
            f"{game_teams[0]}@{game_teams[1]}" if game_teams else "unknown",
        )
        
        return self.check_player_prop(
            sport=sport,
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            our_prob=our_prob,
            our_direction=our_direction,
            game_teams=game_teams,
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