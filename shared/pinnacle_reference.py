"""
pinnacle_reference.py (v1.13 — name-matching bug fixes)

Changes vs v1.12:
  THREE BUG FIXES discovered during 2026-04-25 production audit:

  1. SUFFIX-STRIP BUG (HIGHEST IMPACT)
     The suffix-stripping loop in _normalize_to_kalshi_format used
     str.replace() with non-anchored substrings like " V", " II", etc.
     This caused FALSE-POSITIVE replacements anywhere in the name:
       "Mark Vientos"       → "Markientos" → "MARKIENTOS" ❌ (should be MVIENTOS)
       "Justin Verlander"   → "Justinerlander" → "JUSTINERLANDER" ❌
       "David Iverson"      → "David erson" → "DERSON" ❌

     Fix: anchor suffix removal to end-of-string with a regex. Now only
     true Roman-numeral / Jr/Sr suffixes at the END are stripped.
     "Mark Vientos" now correctly normalizes to "MVIENTOS".

  2. MISSING ATHLETICS TEAM CODE
     The Athletics rebrand (Oakland → Sacramento → Las Vegas, 2025+)
     introduced new codes "ATH" and "SAC" used by Kalshi tickers.
     The team_codes dict only had "OAK". Result: every Athletics game
     failed _extract_game_teams, which cascaded into smart-mode failure
     and player-prop failure for that game.

     Fix: added "ATH" and "SAC" entries pointing to the same Athletics
     metadata. Also added a few other commonly-missing aliases.

  3. WRONG ERROR LOG TRUNCATION
     The "Could not extract teams from: %s" log was showing only the
     LAST 8 chars of game_info, which obscured the actual input. Logs
     showed "05ATHTEX" when the real input was "26APR242005ATHTEX".

     Fix: log the full game_info string so the regression is visible.

Changes vs v1.11:
  v1.12: get_game_total() method for dynamic AB estimation.

Changes vs v1.10:
  v1.11: Team extraction handles 2-letter codes (TB, SF, KC, AZ, etc.)

Changes vs v1.9:
  v1.10: MLB/NBA only — explicitly skips other sports.

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
    divergence: Optional[float]
    confidence_adjustment: float
    sharp_direction: Optional[str]


# v1.13 BUGFIX: anchor suffix-stripping to end-of-string only.
# The old approach used str.replace() which matched anywhere in the name,
# breaking players like "Mark Vientos" (' V' substring stripped → "Markientos").
# Order matters: longer suffixes first, so " Jr." is tried before " Jr".
_SUFFIX_PATTERN = re.compile(
    r"\s+(?:Jr\.?|Sr\.?|III|II|IV|V)\s*$",
    flags=re.IGNORECASE,
)


def _normalize_to_kalshi_format(full_name: str) -> str:
    """Convert a Pinnacle-style full name to Kalshi's compact format.

    Examples:
      "Zach Gelof"    → "ZGELOF"
      "Mark Vientos"  → "MVIENTOS"  (FIXED in v1.13; was "MARKIENTOS")
      "J.P. Crawford" → "JCRAWFORD"
      "Bobby Witt Jr."→ "BWITT"
    """
    if not full_name:
        return ""
    # Strip diacritics (José → Jose)
    name = unicodedata.normalize('NFD', full_name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    # v1.13 FIX: strip Jr/Sr/Roman-numeral suffixes ONLY at end of string.
    name = _SUFFIX_PATTERN.sub('', name)
    # Strip remaining punctuation (e.g., periods in "J.P.")
    name = re.sub(r"[^a-zA-Z\s]", "", name)
    parts = [p.strip() for p in name.split() if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].upper()
    first_initial = parts[0][0].upper()
    rest = ''.join(parts[1:]).upper()
    return first_initial + rest


def _normalize_kalshi_player(kalshi_name: str) -> str:
    return kalshi_name.upper().strip()


# ── Team Code Mappings ────────────────────────────────────────────────────────

# v1.13 FIX: Added ATH and SAC for Athletics relocation (2025+ era).
# The Athletics moved from Oakland to Sacramento for 2025-2026 (transitioning
# to Las Vegas), and Kalshi now uses ATH (and sometimes SAC) instead of OAK.
# Added other common alias variants for robustness.
MLB_TEAM_CODES = {
    "AZ":  ["Arizona Diamondbacks", "Diamondbacks"],
    "ARI": ["Arizona Diamondbacks", "Diamondbacks"],   # alt code
    "ATL": ["Atlanta Braves", "Braves"],
    "BAL": ["Baltimore Orioles", "Orioles"],
    "BOS": ["Boston Red Sox", "Red Sox"],
    "CHC": ["Chicago Cubs", "Cubs"],
    "CWS": ["Chicago White Sox", "White Sox"],
    "CHW": ["Chicago White Sox", "White Sox"],
    "CIN": ["Cincinnati Reds", "Reds"],
    "CLE": ["Cleveland Guardians", "Guardians"],
    "COL": ["Colorado Rockies", "Rockies"],
    "DET": ["Detroit Tigers", "Tigers"],
    "HOU": ["Houston Astros", "Astros"],
    "KC":  ["Kansas City Royals", "Royals"],
    "KCR": ["Kansas City Royals", "Royals"],
    "LAA": ["Los Angeles Angels", "LA Angels", "Angels"],
    "LAD": ["Los Angeles Dodgers", "LA Dodgers", "Dodgers"],
    "MIA": ["Miami Marlins", "Marlins"],
    "MIL": ["Milwaukee Brewers", "Brewers"],
    "MIN": ["Minnesota Twins", "Twins"],
    "NYM": ["New York Mets", "NY Mets", "Mets"],
    "NYY": ["New York Yankees", "NY Yankees", "Yankees"],
    "OAK": ["Oakland Athletics", "Athletics", "A's"],
    "ATH": ["Athletics", "Sacramento Athletics", "Oakland Athletics", "A's"],  # v1.13
    "SAC": ["Sacramento Athletics", "Athletics", "A's"],                       # v1.13
    "PHI": ["Philadelphia Phillies", "Phillies"],
    "PIT": ["Pittsburgh Pirates", "Pirates"],
    "SD":  ["San Diego Padres", "Padres"],
    "SDP": ["San Diego Padres", "Padres"],
    "SF":  ["San Francisco Giants", "Giants"],
    "SFG": ["San Francisco Giants", "Giants"],
    "SEA": ["Seattle Mariners", "Mariners"],
    "STL": ["St. Louis Cardinals", "Cardinals"],
    "TB":  ["Tampa Bay Rays", "Rays"],
    "TBR": ["Tampa Bay Rays", "Rays"],
    "TEX": ["Texas Rangers", "Rangers"],
    "TOR": ["Toronto Blue Jays", "Blue Jays"],
    "WSH": ["Washington Nationals", "Nationals"],
    "WAS": ["Washington Nationals", "Nationals"],
}

NBA_TEAM_CODES = {
    "ATL": ["Atlanta Hawks", "Hawks"],
    "BOS": ["Boston Celtics", "Celtics"],
    "BKN": ["Brooklyn Nets", "Nets"],
    "BRK": ["Brooklyn Nets", "Nets"],
    "CHA": ["Charlotte Hornets", "Hornets"],
    "CHO": ["Charlotte Hornets", "Hornets"],   # alt code
    "CHI": ["Chicago Bulls", "Bulls"],
    "CLE": ["Cleveland Cavaliers", "Cavaliers", "Cavs"],
    "DAL": ["Dallas Mavericks", "Mavericks", "Mavs"],
    "DEN": ["Denver Nuggets", "Nuggets"],
    "DET": ["Detroit Pistons", "Pistons"],
    "GSW": ["Golden State Warriors", "Warriors"],
    "GS":  ["Golden State Warriors", "Warriors"],
    "HOU": ["Houston Rockets", "Rockets"],
    "IND": ["Indiana Pacers", "Pacers"],
    "LAC": ["Los Angeles Clippers", "LA Clippers", "Clippers"],
    "LAL": ["Los Angeles Lakers", "LA Lakers", "Lakers"],
    "MEM": ["Memphis Grizzlies", "Grizzlies"],
    "MIA": ["Miami Heat", "Heat"],
    "MIL": ["Milwaukee Bucks", "Bucks"],
    "MIN": ["Minnesota Timberwolves", "Timberwolves", "Wolves"],
    "NOP": ["New Orleans Pelicans", "Pelicans"],
    "NO":  ["New Orleans Pelicans", "Pelicans"],
    "NYK": ["New York Knicks", "NY Knicks", "Knicks"],
    "NY":  ["New York Knicks", "NY Knicks", "Knicks"],   # alt code
    "OKC": ["Oklahoma City Thunder", "Thunder"],
    "ORL": ["Orlando Magic", "Magic"],
    "PHI": ["Philadelphia 76ers", "76ers", "Sixers"],
    "PHX": ["Phoenix Suns", "Suns"],
    "PHO": ["Phoenix Suns", "Suns"],   # alt code
    "POR": ["Portland Trail Blazers", "Trail Blazers", "Blazers"],
    "SAC": ["Sacramento Kings", "Kings"],
    "SAS": ["San Antonio Spurs", "Spurs"],
    "SA":  ["San Antonio Spurs", "Spurs"],
    "TOR": ["Toronto Raptors", "Raptors"],
    "UTA": ["Utah Jazz", "Jazz"],
    "UTAH": ["Utah Jazz", "Jazz"],   # alt code
    "WAS": ["Washington Wizards", "Wizards"],
    "WSH": ["Washington Wizards", "Wizards"],
}


class PinnacleReference:
    """
    Compare our probabilities to Pinnacle sharp lines.

    Uses The Odds API to fetch Pinnacle odds for MLB/NBA player props
    and game lines.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    SPORT_KEYS = {
        "mlb": "baseball_mlb",
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "nhl": "icehockey_nhl",
    }

    PROP_MARKETS = {
        "batter_hits": "batter_hits",
        "batter_home_runs": "batter_home_runs",
        "batter_rbis": "batter_rbis",
        "batter_runs": "batter_runs_scored",
        "batter_total_bases": "batter_total_bases",
        "batter_walks": "batter_walks",
        "batter_strikeouts": "batter_strikeouts",
        "pitcher_strikeouts": "pitcher_strikeouts",
        "pitcher_outs": "pitcher_outs",
        "player_points": "player_points",
        "player_rebounds": "player_rebounds",
        "player_assists": "player_assists",
        "player_threes": "player_threes",
        "player_steals": "player_steals",
        "player_blocks": "player_blocks",
        "player_pra": "player_points_rebounds_assists",
    }

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
        "K": "pitcher_strikeouts",
        "SO": "batter_strikeouts",
    }

    SOFT_DIVERGENCE_THRESHOLD = 0.05
    HARD_DIVERGENCE_THRESHOLD = 0.10
    ALIGNMENT_BOOST_THRESHOLD = 0.03

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 300,
        enabled: bool = True,
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.cache_ttl_sec = cache_ttl_sec
        self.enabled = enabled
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._player_name_cache: Dict[str, str] = {}
        self._last_request_time = 0.0
        self._min_request_interval = 1.0

        if not self.api_key:
            logger.warning("ODDS_API_KEY not set — Pinnacle reference disabled")
            self.enabled = False

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _american_to_prob(self, american_odds: int) -> float:
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    def _get_cached(self, cache_key: str) -> Optional[dict]:
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl_sec:
                return data
        return None

    def _set_cache(self, cache_key: str, data: dict) -> None:
        self._cache[cache_key] = (time.time(), data)

    def _extract_game_teams(
        self,
        ticker: str,
        sport: str,
    ) -> Optional[Tuple[str, str]]:
        parts = ticker.upper().split("-")
        if len(parts) < 2:
            return None

        game_info = parts[1]
        team_map = MLB_TEAM_CODES if sport.lower() == "mlb" else NBA_TEAM_CODES

        for teams_len in [6, 5, 4]:
            if len(game_info) < teams_len:
                continue

            teams_str = game_info[-teams_len:]

            for split_pos in range(2, min(4, teams_len - 1)):
                away_code = teams_str[:split_pos]
                home_code = teams_str[split_pos:]

                if away_code in team_map and home_code in team_map:
                    logger.info(
                        "[PINNACLE] Extracted teams: %s @ %s from ticker",
                        away_code, home_code,
                    )
                    return (away_code, home_code)

        # v1.13 FIX: Log the FULL game_info string, not the trimmed last 8 chars.
        # Old log showed "05ATHTEX" obscuring the real input "26APR242005ATHTEX".
        logger.info(
            "[PINNACLE] Could not extract teams from game_info=%r (ticker=%s)",
            game_info, ticker[:50],
        )
        return None

    def _find_event_id(
        self,
        events: List[dict],
        game_teams: Tuple[str, str],
        sport: str,
    ) -> Optional[str]:
        away_code, home_code = game_teams
        team_map = MLB_TEAM_CODES if sport.lower() == "mlb" else NBA_TEAM_CODES

        away_names = team_map.get(away_code, [away_code])
        home_names = team_map.get(home_code, [home_code])

        for event in events:
            event_home = event.get("home_team", "")
            event_away = event.get("away_team", "")

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
                    "[PINNACLE] SMART MATCH: %s @ %s → event %s (%s vs %s)",
                    away_code, home_code, event_id[:12] if event_id else "?",
                    event_away, event_home,
                )
                return event_id

        logger.info(
            "[PINNACLE] No event match for %s @ %s in %d events",
            away_code, home_code, len(events),
        )
        return None

    def _fetch_player_props(
        self,
        sport: str,
        market: str,
        game_teams: Optional[Tuple[str, str]] = None,
    ) -> List[dict]:
        sport_key = self.SPORT_KEYS.get(sport.lower())
        if not sport_key:
            logger.warning("Unknown sport: %s", sport)
            return []

        if game_teams:
            cache_key = f"{sport_key}:{market}:{game_teams[0]}@{game_teams[1]}"
        else:
            cache_key = f"{sport_key}:{market}"

        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()

        try:
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

            events_to_query = []

            if game_teams:
                target_event_id = self._find_event_id(events_list, game_teams, sport)
                if target_event_id:
                    events_to_query = [{"id": target_event_id}]
                    logger.info("[PINNACLE] SMART MODE: querying 1 event (vs 30 batch)")
                else:
                    events_to_query = events_list[:10]
                    logger.info("[PINNACLE] FALLBACK: no team match, querying %d events", len(events_to_query))
            else:
                events_to_query = events_list[:30]

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
                        if odds_data and odds_data.get("bookmakers"):
                            all_odds.append(odds_data)
                except requests.RequestException:
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
        kalshi_normalized = _normalize_kalshi_player(player_name)
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

                        if len(pinnacle_names_seen) < 10 and outcome_player not in pinnacle_names_seen:
                            pinnacle_names_seen.append(outcome_player)

                        odds_api_normalized = _normalize_to_kalshi_format(outcome_player)

                        if kalshi_normalized != odds_api_normalized:
                            if len(kalshi_normalized) > 3:
                                kalshi_lastname = kalshi_normalized[1:]
                                odds_lastname = odds_api_normalized[1:] if odds_api_normalized else ""
                                if kalshi_lastname != odds_lastname:
                                    continue
                            else:
                                continue

                        outcome_line = outcome.get("point")
                        if outcome_line is None:
                            continue

                        if abs(float(outcome_line) - line) > 0.5:
                            continue

                        outcome_name = outcome.get("name", "").upper()
                        price = outcome.get("price", 0)

                        if not price:
                            continue

                        prob = self._american_to_prob(price)

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

        if our_direction == "OVER":
            sharp_prob = over_prob
        else:
            sharp_prob = under_prob

        divergence = our_prob - sharp_prob
        abs_div = abs(divergence)

        sharp_direction = "OVER" if over_prob > 0.5 else "UNDER"

        logger.info(
            "[PINNACLE] %s @ %.1f: sharp=%.1f%% ours=%.1f%% div=%+.1f%%",
            player_name, line, sharp_prob * 100, our_prob * 100, divergence * 100,
        )

        if abs_div > self.HARD_DIVERGENCE_THRESHOLD:
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
            adj = 0.7
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
            adj = 1.1
            return SharpCheckResult(
                passes=True,
                reason=f"Sharp alignment confirmed: {divergence:+.1%}",
                our_prob=our_prob,
                sharp_prob=sharp_prob,
                divergence=divergence,
                confidence_adjustment=adj,
                sharp_direction=sharp_direction,
            )

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

        sport = None
        prop_code = None

        if ticker_upper.startswith("KXMLB"):
            sport = "mlb"
            if "TOTAL" in ticker_upper or "F5" in ticker_upper:
                return SharpCheckResult(
                    passes=True,
                    reason="Game total market (not player prop)",
                    our_prob=our_prob,
                    sharp_prob=None,
                    divergence=None,
                    confidence_adjustment=1.0,
                    sharp_direction=None,
                )
            for code in ["PTS", "HRR", "HR", "HIT", "H", "RBI", "R", "TB", "KS", "K", "SO", "BB"]:
                if f"KXMLB{code}" in ticker_upper:
                    if code == "HRR":
                        prop_code = "HR"
                    elif code == "HIT":
                        prop_code = "H"
                    elif code == "KS":
                        prop_code = "K"
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

        game_teams = self._extract_game_teams(ticker, sport)

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
            player_part = parts[2]
            team = player_part[:3]
            player_raw = player_part[3:]
            player_name = "".join(c for c in player_raw if not c.isdigit())
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

    # ── v1.12: Game Total Fetching ─────────────────────────────────────────────

    def get_game_total(
        self,
        ticker: str,
        sport: str = "mlb",
    ) -> Optional[float]:
        """
        v1.12: Fetch Vegas O/U (game total) for the game in a Kalshi ticker.

        Uses the same smart event matching as player prop checks.
        Returns the total line (e.g., 8.5) or None if unavailable.

        Cost: Usually 0 extra API calls because the events list is already
        cached from the player prop Pinnacle check that runs in _execute_trade.
        Only costs 1 call for the totals odds if event is found.
        """
        if not self.enabled:
            return None

        game_teams = self._extract_game_teams(ticker, sport)
        if not game_teams:
            return None

        cache_key = f"game_total:{sport}:{game_teams[0]}@{game_teams[1]}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached.get("total")

        sport_key = self.SPORT_KEYS.get(sport.lower())
        if not sport_key:
            return None

        try:
            events_cache_key = f"{sport_key}:events"
            events_list = self._get_cached(events_cache_key)

            if not events_list:
                self._rate_limit()
                events_url = f"{self.BASE_URL}/sports/{sport_key}/events"
                resp = requests.get(
                    events_url,
                    params={"apiKey": self.api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                events_list = resp.json()
                self._set_cache(events_cache_key, events_list)

            if not events_list:
                return None

            event_id = self._find_event_id(events_list, game_teams, sport)
            if not event_id:
                return None

            self._rate_limit()
            odds_url = f"{self.BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "totals",
                "bookmakers": "pinnacle",
                "oddsFormat": "american",
            }

            resp = requests.get(odds_url, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()

            for book in data.get("bookmakers", []):
                if book.get("key") != "pinnacle":
                    continue
                for market in book.get("markets", []):
                    if market.get("key") != "totals":
                        continue
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        if point is not None:
                            total = float(point)
                            self._set_cache(cache_key, {"total": total})
                            logger.info(
                                "[PINNACLE] Game total: %s @ %s → O/U %.1f",
                                game_teams[0], game_teams[1], total,
                            )
                            return total

            return None

        except Exception as e:
            logger.debug("[PINNACLE] Game total fetch failed: %s", e)
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_reference: Optional[PinnacleReference] = None


def get_pinnacle_reference() -> PinnacleReference:
    global _reference
    if _reference is None:
        _reference = PinnacleReference()
    return _reference


def check_sharp_line(ticker: str, our_prob: float, our_direction: str) -> SharpCheckResult:
    return get_pinnacle_reference().check_from_kalshi_ticker(ticker, our_prob, our_direction)