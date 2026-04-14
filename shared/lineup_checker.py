"""
lineup_checker.py (v1.0)

Verify players are actually in the lineup before betting their props.

Nothing worse than betting Judge O2.5 hits and he's a late scratch.
This module checks MLB/NBA lineups before trade execution.

Sources:
  - MLB: MLB Stats API (free, official)
  - NBA: NBA API / ESPN (free)

Usage:
    from shared.lineup_checker import LineupChecker
    
    checker = LineupChecker()
    
    result = checker.check_player_active(
        sport="mlb",
        player_name="Aaron Judge",
        team="NYY",
    )
    
    if not result.is_active:
        logger.info("LINEUP SKIP: %s", result.reason)
        return
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


@dataclass
class LineupCheckResult:
    """Result of lineup check."""
    is_active: bool
    reason: str
    player_found: bool
    lineup_available: bool
    batting_order: Optional[int] = None  # MLB: 1-9, None if not starting
    status: Optional[str] = None  # "starting", "bench", "injured", "unknown"
    game_time: Optional[datetime] = None


class LineupChecker:
    """
    Check if a player is in today's lineup before betting their props.
    
    MLB: Uses MLB Stats API to get confirmed lineups
    NBA: Uses NBA API / balldontlie for injury reports
    """
    
    # MLB Stats API (free, no key needed)
    MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
    
    # NBA injury/player data
    NBA_BASE_URL = "https://www.balldontlie.io/api/v1"
    
    # Team code mappings
    MLB_TEAM_CODES = {
        "NYY": 147, "BOS": 111, "TBR": 139, "TOR": 141, "BAL": 110,
        "CLE": 114, "DET": 116, "KCR": 118, "MIN": 142, "CHW": 145,
        "HOU": 117, "LAA": 108, "OAK": 133, "SEA": 136, "TEX": 140,
        "NYM": 121, "ATL": 144, "MIA": 146, "PHI": 143, "WSN": 120,
        "CHC": 112, "MIL": 158, "STL": 138, "CIN": 113, "PIT": 134,
        "LAD": 119, "SDP": 135, "SFG": 137, "ARI": 109, "COL": 115,
    }
    
    NBA_TEAM_CODES = {
        "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5,
        "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10,
        "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
        "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
        "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
        "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30,
    }
    
    def __init__(
        self,
        cache_ttl_sec: int = 180,  # Cache lineups for 3 minutes
        enabled: bool = True,
    ):
        self.cache_ttl_sec = cache_ttl_sec
        self.enabled = enabled
        
        # Cache: cache_key -> (timestamp, data)
        self._cache: Dict[str, Tuple[float, any]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cached(self, cache_key: str) -> Optional[any]:
        """Get cached data if not expired."""
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl_sec:
                return data
        return None
    
    def _set_cache(self, cache_key: str, data: any) -> None:
        """Cache data with timestamp."""
        self._cache[cache_key] = (time.time(), data)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        # Remove periods, extra spaces, convert to lowercase
        name = name.lower().strip()
        name = re.sub(r'\.', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two player names match (fuzzy)."""
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        
        # Exact match
        if n1 == n2:
            return True
        
        # Last name match
        last1 = n1.split()[-1] if ' ' in n1 else n1
        last2 = n2.split()[-1] if ' ' in n2 else n2
        if last1 == last2:
            # Check first initial if available
            if ' ' in n1 and ' ' in n2:
                first1 = n1.split()[0][0]
                first2 = n2.split()[0][0]
                if first1 == first2:
                    return True
            else:
                return True
        
        # One contains the other
        if n1 in n2 or n2 in n1:
            return True
        
        return False
    
    # ── MLB Lineup Check ──────────────────────────────────────────────────────
    
    def _get_mlb_todays_games(self) -> List[dict]:
        """Get today's MLB games."""
        cache_key = "mlb_games_today"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = f"{self.MLB_BASE_URL}/schedule"
            params = {
                "sportId": 1,
                "date": today,
                "hydrate": "lineups,probablePitcher",
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            games = []
            for date_entry in data.get("dates", []):
                games.extend(date_entry.get("games", []))
            
            self._set_cache(cache_key, games)
            logger.debug("[LINEUP] Fetched %d MLB games for today", len(games))
            return games
            
        except Exception as e:
            logger.warning("[LINEUP] MLB schedule fetch failed: %s", e)
            return []
    
    def _get_mlb_game_lineup(self, game_pk: int) -> Optional[dict]:
        """Get lineup for a specific MLB game."""
        cache_key = f"mlb_lineup_{game_pk}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            url = f"{self.MLB_BASE_URL}/game/{game_pk}/boxscore"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            self._set_cache(cache_key, data)
            return data
            
        except Exception as e:
            logger.warning("[LINEUP] MLB boxscore fetch failed for game %d: %s", game_pk, e)
            return None
    
    def check_mlb_player(
        self,
        player_name: str,
        team: str,
    ) -> LineupCheckResult:
        """
        Check if an MLB player is in today's starting lineup.
        
        Args:
            player_name: Player's full name (e.g., "Aaron Judge")
            team: Team code (e.g., "NYY")
            
        Returns:
            LineupCheckResult with active status and batting order
        """
        if not self.enabled:
            return LineupCheckResult(
                is_active=True,
                reason="Lineup checker disabled",
                player_found=False,
                lineup_available=False,
            )
        
        team_upper = team.upper()
        team_id = self.MLB_TEAM_CODES.get(team_upper)
        
        if not team_id:
            return LineupCheckResult(
                is_active=True,
                reason=f"Unknown team code: {team}",
                player_found=False,
                lineup_available=False,
            )
        
        # Get today's games
        games = self._get_mlb_todays_games()
        if not games:
            return LineupCheckResult(
                is_active=True,
                reason="No MLB games found for today",
                player_found=False,
                lineup_available=False,
            )
        
        # Find game for this team
        target_game = None
        for game in games:
            away_id = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
            home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
            if team_id in (away_id, home_id):
                target_game = game
                break
        
        if not target_game:
            return LineupCheckResult(
                is_active=True,
                reason=f"No game found for {team} today",
                player_found=False,
                lineup_available=False,
            )
        
        game_pk = target_game.get("gamePk")
        game_time_str = target_game.get("gameDate")
        game_time = None
        if game_time_str:
            try:
                game_time = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
            except:
                pass
        
        # Check if lineup is posted (usually ~2-3 hours before game)
        # First check the schedule hydrate for lineups
        is_home = target_game.get("teams", {}).get("home", {}).get("team", {}).get("id") == team_id
        team_key = "home" if is_home else "away"
        
        lineup_data = target_game.get("lineups", {})
        team_lineup = lineup_data.get(f"{team_key}Players", [])
        
        if not team_lineup:
            # Try fetching boxscore for more detailed lineup
            boxscore = self._get_mlb_game_lineup(game_pk)
            if boxscore:
                team_data = boxscore.get("teams", {}).get(team_key, {})
                batting_order = team_data.get("battingOrder", [])
                players = team_data.get("players", {})
                
                for i, player_id in enumerate(batting_order):
                    player_key = f"ID{player_id}"
                    player_info = players.get(player_key, {})
                    full_name = player_info.get("person", {}).get("fullName", "")
                    
                    if self._names_match(player_name, full_name):
                        return LineupCheckResult(
                            is_active=True,
                            reason=f"In lineup, batting #{i+1}",
                            player_found=True,
                            lineup_available=True,
                            batting_order=i + 1,
                            status="starting",
                            game_time=game_time,
                        )
                
                # Player not in batting order — check if on roster
                for player_key, player_info in players.items():
                    full_name = player_info.get("person", {}).get("fullName", "")
                    if self._names_match(player_name, full_name):
                        status = player_info.get("status", {}).get("code", "")
                        if status in ("IL", "IL10", "IL60"):
                            return LineupCheckResult(
                                is_active=False,
                                reason=f"Player on injured list: {status}",
                                player_found=True,
                                lineup_available=True,
                                status="injured",
                                game_time=game_time,
                            )
                        return LineupCheckResult(
                            is_active=False,
                            reason="Not in starting lineup (bench)",
                            player_found=True,
                            lineup_available=True,
                            status="bench",
                            game_time=game_time,
                        )
                
                # Player not found at all
                return LineupCheckResult(
                    is_active=False,
                    reason=f"Player not found in game roster: {player_name}",
                    player_found=False,
                    lineup_available=True,
                    game_time=game_time,
                )
        
        # Lineup not available yet — game probably too far out
        return LineupCheckResult(
            is_active=True,
            reason="Lineup not yet posted (game >3hrs out)",
            player_found=False,
            lineup_available=False,
            game_time=game_time,
        )
    
    # ── NBA Lineup Check ──────────────────────────────────────────────────────
    
    def _get_nba_injuries(self) -> Dict[str, str]:
        """
        Get current NBA injury report.
        
        Returns dict of player_name_lower -> injury_status
        """
        cache_key = "nba_injuries"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        self._rate_limit()
        
        # Use ESPN's public injury data
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            injuries = {}
            for team in data.get("items", []):
                for injury in team.get("injuries", []):
                    athlete = injury.get("athlete", {})
                    name = athlete.get("displayName", "").lower()
                    status = injury.get("status", "")
                    if name:
                        injuries[name] = status
            
            self._set_cache(cache_key, injuries)
            logger.debug("[LINEUP] Fetched %d NBA injuries", len(injuries))
            return injuries
            
        except Exception as e:
            logger.warning("[LINEUP] NBA injury fetch failed: %s", e)
            return {}
    
    def check_nba_player(
        self,
        player_name: str,
        team: str,
    ) -> LineupCheckResult:
        """
        Check if an NBA player is active (not injured/out).
        
        Args:
            player_name: Player's full name (e.g., "Stephen Curry")
            team: Team code (e.g., "GSW")
            
        Returns:
            LineupCheckResult with active status
        """
        if not self.enabled:
            return LineupCheckResult(
                is_active=True,
                reason="Lineup checker disabled",
                player_found=False,
                lineup_available=False,
            )
        
        # Check injury report
        injuries = self._get_nba_injuries()
        
        player_lower = self._normalize_name(player_name)
        
        for injured_name, status in injuries.items():
            if self._names_match(player_name, injured_name):
                status_lower = status.lower()
                if "out" in status_lower:
                    return LineupCheckResult(
                        is_active=False,
                        reason=f"Player OUT: {status}",
                        player_found=True,
                        lineup_available=True,
                        status="injured",
                    )
                elif "doubtful" in status_lower:
                    return LineupCheckResult(
                        is_active=False,
                        reason=f"Player DOUBTFUL: {status}",
                        player_found=True,
                        lineup_available=True,
                        status="doubtful",
                    )
                elif "questionable" in status_lower:
                    # Questionable is risky but not a hard skip
                    return LineupCheckResult(
                        is_active=True,
                        reason=f"Player QUESTIONABLE: {status} (proceed with caution)",
                        player_found=True,
                        lineup_available=True,
                        status="questionable",
                    )
                elif "probable" in status_lower:
                    return LineupCheckResult(
                        is_active=True,
                        reason=f"Player PROBABLE: {status}",
                        player_found=True,
                        lineup_available=True,
                        status="probable",
                    )
        
        # Not on injury report — assume active
        return LineupCheckResult(
            is_active=True,
            reason="Not on injury report",
            player_found=False,
            lineup_available=True,
            status="active",
        )
    
    # ── Generic Check ─────────────────────────────────────────────────────────
    
    def check_player_active(
        self,
        sport: str,
        player_name: str,
        team: str,
    ) -> LineupCheckResult:
        """
        Check if a player is active for their sport.
        
        Args:
            sport: "mlb" or "nba"
            player_name: Player's full name
            team: Team code
            
        Returns:
            LineupCheckResult
        """
        sport_lower = sport.lower()
        
        if sport_lower == "mlb":
            return self.check_mlb_player(player_name, team)
        elif sport_lower == "nba":
            return self.check_nba_player(player_name, team)
        else:
            return LineupCheckResult(
                is_active=True,
                reason=f"Unknown sport: {sport}",
                player_found=False,
                lineup_available=False,
            )
    
    def check_from_kalshi_ticker(self, ticker: str) -> LineupCheckResult:
        """
        Parse a Kalshi ticker and check player lineup status.
        
        Ticker format examples:
            KXMLBPTS-26APR141905LAANYY-NYYAJUDGE99-2
            KXNBAPTS-26APR15GSWLAC-GSWSCURRY30-25
        """
        if not self.enabled:
            return LineupCheckResult(
                is_active=True,
                reason="Lineup checker disabled",
                player_found=False,
                lineup_available=False,
            )
        
        ticker_upper = ticker.upper()
        
        # Detect sport
        if ticker_upper.startswith("KXMLB"):
            sport = "mlb"
        elif ticker_upper.startswith("KXNBA"):
            sport = "nba"
        else:
            return LineupCheckResult(
                is_active=True,
                reason=f"Not a player prop ticker: {ticker[:20]}",
                player_found=False,
                lineup_available=False,
            )
        
        # Parse ticker: KXMLBPTS-GAMEINFO-TEAMPLAYER-LINE
        parts = ticker.split("-")
        if len(parts) < 4:
            return LineupCheckResult(
                is_active=True,
                reason=f"Cannot parse ticker: {ticker[:30]}",
                player_found=False,
                lineup_available=False,
            )
        
        try:
            # Player part: NYYAJUDGE99 -> team=NYY, player=AJUDGE
            player_part = parts[2]
            team = player_part[:3]
            player_raw = player_part[3:]
            # Remove trailing numbers (jersey)
            player_name = "".join(c for c in player_raw if not c.isdigit())
            
            # Try to reconstruct a reasonable name
            # AJUDGE -> A. Judge, SCURRY -> S. Curry
            if len(player_name) > 1:
                first_initial = player_name[0]
                last_name = player_name[1:].capitalize()
                player_name = f"{first_initial}. {last_name}"
            
        except Exception as e:
            return LineupCheckResult(
                is_active=True,
                reason=f"Cannot parse player: {e}",
                player_found=False,
                lineup_available=False,
            )
        
        return self.check_player_active(sport, player_name, team)


# ── Singleton ─────────────────────────────────────────────────────────────────

_checker: Optional[LineupChecker] = None


def get_lineup_checker() -> LineupChecker:
    """Get the global lineup checker instance."""
    global _checker
    if _checker is None:
        _checker = LineupChecker()
    return _checker


def check_player_lineup(ticker: str) -> LineupCheckResult:
    """Convenience function to check lineup from ticker."""
    return get_lineup_checker().check_from_kalshi_ticker(ticker)