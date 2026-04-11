"""
shared/nba_stats_fetcher.py  (v4 — ESPN roster endpoint)

Uses ESPN's team roster endpoint which returns full player data
without needing $ref dereferencing. Much more reliable than search.

Endpoints:
  - Roster: site.api.espn.com/.../teams/{espn_team_id}/roster
  - Stats:  site.api.espn.com/.../athletes/{id}/statistics
  - Log:    site.api.espn.com/.../athletes/{id}/gamelog
"""

import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_ESPN_V3   = "https://site.api.espn.com/apis/common/v3/sports/basketball/nba"

# ── ESPN team ID mapping (Kalshi team code → ESPN numeric ID) ──────────────────
_KALSHI_TO_ESPN_TEAM_ID = {
    "ATL": 1,  "BOS": 2,  "BKN": 17, "CHA": 30,
    "CHI": 4,  "CLE": 5,  "DAL": 6,  "DEN": 7,
    "DET": 8,  "GSW": 9,  "HOU": 10, "IND": 11,
    "LAC": 12, "LAL": 13, "MEM": 29, "MIA": 14,
    "MIL": 15, "MIN": 16, "NOP": 3,  "NYK": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21,
    "POR": 22, "SAC": 23, "SAS": 24, "TOR": 28,
    "UTA": 26, "WAS": 27,
}

# ── Cache ──────────────────────────────────────────────────────────────────────
_roster_cache: dict[int, list] = {}       # espn_team_id → list of athlete dicts
_player_cache: dict[str, "NBAPlayer"] = {}  # cache_key → player
_stats_cache: dict[int, "PlayerStats"] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600


@dataclass
class NBAPlayer:
    player_id:  int
    full_name:  str
    team_id:    int
    jersey:     str


@dataclass
class PlayerStats:
    player_id:      int
    games_played:   int
    minutes_pg:     float
    points_pg:      float
    rebounds_pg:    float
    assists_pg:     float
    steals_pg:      float
    blocks_pg:      float
    three_pm_pg:    float
    three_pa_pg:    float
    three_pct:      float
    fga_pg:         float
    fta_pg:         float
    turnovers_pg:   float
    usage_rate:     float


@dataclass
class RollingStats:
    n_games:        int
    points_pg:      float
    rebounds_pg:    float
    assists_pg:     float
    three_pm_pg:    float
    steals_pg:      float
    blocks_pg:      float
    minutes_pg:     float


def _espn_get(url: str, timeout: int = 15) -> Optional[dict]:
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.info("[nba_stats] ESPN GET failed: %s → %s", url[:100], e)
        return None


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def _check_cache():
    global _cache_ts
    now = time.monotonic()
    if now - _cache_ts > _CACHE_TTL:
        _roster_cache.clear()
        _player_cache.clear()
        _stats_cache.clear()
        _cache_ts = now


# ── Roster fetch via ESPN team roster ──────────────────────────────────────────

def _fetch_espn_roster(kalshi_team_code: str) -> list[dict]:
    """Fetch full roster from ESPN's team roster endpoint."""
    _check_cache()

    espn_id = _KALSHI_TO_ESPN_TEAM_ID.get(kalshi_team_code)
    if not espn_id:
        logger.info("[nba_stats] no ESPN team ID for %s", kalshi_team_code)
        return []

    if espn_id in _roster_cache:
        return _roster_cache[espn_id]

    url = f"{_ESPN_BASE}/teams/{espn_id}/roster"
    data = _espn_get(url)
    if not data:
        return []

    athletes = []

    # Log top-level keys for debugging
    logger.info("[nba_stats] ESPN roster keys for %s: %s", kalshi_team_code, list(data.keys())[:10])

    # Format A: grouped by position → {"athletes": [{"position":"Guard","items":[...]}]}
    # Format B: flat athlete list → {"athletes": [{"id":"123","displayName":"X"},...]}
    # Format C: nested under team → {"team": {"athletes": [...]}}

    raw_athletes = data.get("athletes", [])
    if not raw_athletes:
        # Try nested under "team"
        raw_athletes = data.get("team", {}).get("athletes", [])

    for entry in raw_athletes:
        if "items" in entry:
            # Format A: position-grouped
            for athlete in entry["items"]:
                athletes.append(athlete)
        elif "id" in entry and ("displayName" in entry or "fullName" in entry):
            # Format B: flat athlete dict
            athletes.append(entry)
        elif "athlete" in entry:
            # Format C: wrapped in {"athlete": {...}}
            athletes.append(entry["athlete"])

    logger.info("[nba_stats] ESPN roster for %s: %d players", kalshi_team_code, len(athletes))

    if athletes:
        # Log first player for format verification
        first = athletes[0]
        logger.info("[nba_stats] sample player: %s #%s (id=%s) keys=%s",
                     first.get("displayName", first.get("fullName", "?")),
                     first.get("jersey", "?"),
                     first.get("id", "?"),
                     list(first.keys())[:8])

    _roster_cache[espn_id] = athletes
    return athletes


# ── Player lookup ──────────────────────────────────────────────────────────────

def lookup_player(
    team_id: int,       # NBA Stats API team_id (ignored — we use team code from parser)
    first_initial: str,
    last_name: str,
    jersey: str,
    team_code: str = "",  # Kalshi team code like "HOU", "MIN"
) -> Optional[NBAPlayer]:
    """Find player on ESPN roster by name + jersey."""
    _check_cache()

    cache_key = f"{team_code or team_id}-{first_initial}-{last_name}-{jersey}"
    if cache_key in _player_cache:
        return _player_cache[cache_key]

    # If we have a team code, use ESPN roster
    # If not, fall back to search
    if team_code:
        roster = _fetch_espn_roster(team_code)
    else:
        roster = []

    if not roster:
        # Fallback: ESPN athlete search
        return _search_player_espn(first_initial, last_name, jersey, team_id, cache_key)

    target_first = first_initial.upper()
    target_last = _normalize(last_name)
    target_jersey = str(jersey).lstrip("0") or "0"

    best = None
    best_score = -1

    for athlete in roster:
        espn_name = athlete.get("displayName", "") or athlete.get("fullName", "")
        espn_id = athlete.get("id")
        espn_jersey = str(athlete.get("jersey", "")).lstrip("0") or "0"

        if not espn_name or not espn_id:
            continue

        parts = espn_name.strip().split()
        if len(parts) < 2:
            continue

        p_first = _normalize(parts[0])
        p_last = _normalize(" ".join(parts[1:]))

        if not p_first.startswith(target_first):
            continue

        last_exact = (p_last == target_last)
        last_subseq = _is_subseq(target_last, p_last) if not last_exact else False
        jersey_match = (espn_jersey == target_jersey)

        if not last_exact and not last_subseq:
            continue

        score = 0
        if last_exact: score += 10
        if jersey_match: score += 5
        if last_subseq and not last_exact: score += 3

        if score > best_score:
            best_score = score
            best = NBAPlayer(
                player_id=int(espn_id),
                full_name=espn_name,
                team_id=team_id,
                jersey=espn_jersey,
            )

    if best:
        _player_cache[cache_key] = best
        logger.info("[nba_stats] found: %s (ESPN ID: %d)", best.full_name, best.player_id)
    else:
        logger.info("[nba_stats] NOT FOUND: %s. %s #%s on %s (%d roster entries checked)",
                     first_initial, last_name, jersey, team_code, len(roster))

    return best


def _search_player_espn(first_initial, last_name, jersey, team_id, cache_key) -> Optional[NBAPlayer]:
    """Fallback: search ESPN for player by name."""
    search_name = last_name.capitalize()
    url = f"{_ESPN_BASE}/athletes?search={search_name}&limit=25"
    data = _espn_get(url)
    if not data:
        return None

    target_first = first_initial.upper()
    target_last = _normalize(last_name)
    target_jersey = str(jersey).lstrip("0") or "0"

    items = data.get("items", [])

    best = None
    best_score = -1

    for athlete in items:
        # If $ref link, fetch it
        if "$ref" in athlete and "displayName" not in athlete:
            athlete = _espn_get(athlete["$ref"]) or {}

        espn_name = athlete.get("displayName", "")
        espn_id = athlete.get("id")
        espn_jersey = str(athlete.get("jersey", "")).lstrip("0") or "0"

        if not espn_name or not espn_id:
            continue

        parts = espn_name.strip().split()
        if len(parts) < 2:
            continue

        p_first = _normalize(parts[0])
        p_last = _normalize(" ".join(parts[1:]))

        if not p_first.startswith(target_first):
            continue
        if p_last != target_last and not _is_subseq(target_last, p_last):
            continue

        score = 0
        if p_last == target_last: score += 10
        if espn_jersey == target_jersey: score += 5

        if score > best_score:
            best_score = score
            best = NBAPlayer(
                player_id=int(espn_id), full_name=espn_name,
                team_id=team_id, jersey=espn_jersey,
            )

    if best:
        _player_cache[cache_key] = best

    return best


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


# ── Season stats ───────────────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    _check_cache()
    if player_id in _stats_cache:
        return _stats_cache[player_id]

    # Try the per-athlete statistics endpoint
    url = f"{_ESPN_BASE}/athletes/{player_id}/statistics"
    data = _espn_get(url)

    if not data:
        # Fallback: try the v2 athlete summary which includes stats
        url2 = f"{_ESPN_BASE}/athletes/{player_id}"
        data2 = _espn_get(url2)
        if data2:
            return _parse_athlete_summary_stats(player_id, data2)
        return None

    return _parse_statistics_response(player_id, data)


def _parse_statistics_response(player_id: int, data: dict) -> Optional[PlayerStats]:
    """Parse ESPN /athletes/{id}/statistics response."""
    try:
        # Navigate to the stats — ESPN nests differently sometimes
        stat_map = {}

        # Try: statistics.splits.categories[].stats[]
        splits = data.get("statistics", data.get("stats", {}))
        if isinstance(splits, list):
            for split in splits:
                _extract_stats_from_categories(split.get("categories", []), stat_map)
        elif isinstance(splits, dict):
            cats = splits.get("splits", {}).get("categories", splits.get("categories", []))
            _extract_stats_from_categories(cats, stat_map)

        if not stat_map:
            logger.info("[nba_stats] ESPN stats empty for player %d", player_id)
            return None

        gp = int(stat_map.get("gamesPlayed", stat_map.get("GP", 0)))
        if gp == 0:
            return None

        stats = PlayerStats(
            player_id=player_id,
            games_played=gp,
            minutes_pg=stat_map.get("avgMinutes", stat_map.get("minutes", 0)) / max(gp, 1) if "minutes" in stat_map and "avgMinutes" not in stat_map else stat_map.get("avgMinutes", 0),
            points_pg=stat_map.get("avgPoints", stat_map.get("points", 0) / max(gp, 1)),
            rebounds_pg=stat_map.get("avgRebounds", stat_map.get("rebounds", 0) / max(gp, 1)),
            assists_pg=stat_map.get("avgAssists", stat_map.get("assists", 0) / max(gp, 1)),
            steals_pg=stat_map.get("avgSteals", stat_map.get("steals", 0) / max(gp, 1)),
            blocks_pg=stat_map.get("avgBlocks", stat_map.get("blocks", 0) / max(gp, 1)),
            three_pm_pg=stat_map.get("avgThreePointFieldGoalsMade", stat_map.get("threePointFieldGoalsMade", 0) / max(gp, 1)),
            three_pa_pg=stat_map.get("avgThreePointFieldGoalsAttempted", stat_map.get("threePointFieldGoalsAttempted", 0) / max(gp, 1)),
            three_pct=stat_map.get("threePointFieldGoalPct", 0),
            fga_pg=stat_map.get("avgFieldGoalsAttempted", stat_map.get("fieldGoalsAttempted", 0) / max(gp, 1)),
            fta_pg=stat_map.get("avgFreeThrowsAttempted", stat_map.get("freeThrowsAttempted", 0) / max(gp, 1)),
            turnovers_pg=stat_map.get("avgTurnovers", stat_map.get("turnovers", 0) / max(gp, 1)),
            usage_rate=0.0,
        )
        logger.info("[nba_stats] stats for %d: %dG %.1fppg %.1frpg %.1fapg",
                     player_id, gp, stats.points_pg, stats.rebounds_pg, stats.assists_pg)
        _stats_cache[player_id] = stats
        return stats

    except Exception as e:
        logger.info("[nba_stats] ESPN stats parse error for %d: %s", player_id, e)
        return None


def _parse_athlete_summary_stats(player_id: int, data: dict) -> Optional[PlayerStats]:
    """Parse stats from ESPN /athletes/{id} summary endpoint."""
    try:
        # The summary embeds stats in "statistics" or "stats"
        stats_section = data.get("statistics", data.get("stats", []))
        stat_map = {}

        if isinstance(stats_section, list):
            for section in stats_section:
                _extract_stats_from_categories(section.get("categories", []), stat_map)
                # Also try direct "stats" array
                for s in section.get("stats", []):
                    if isinstance(s, dict):
                        stat_map[s.get("name", "")] = float(s.get("value", 0))

        if not stat_map:
            return None

        return _parse_statistics_response(player_id, {"statistics": {"categories": [{"stats": [{"name": k, "value": v} for k, v in stat_map.items()]}]}})
    except Exception as e:
        logger.info("[nba_stats] ESPN summary parse error for %d: %s", player_id, e)
        return None


def _extract_stats_from_categories(categories: list, stat_map: dict):
    """Extract stat name→value from ESPN categories structure."""
    for cat in categories:
        for stat in cat.get("stats", []):
            name = stat.get("name", stat.get("abbreviation", ""))
            value = stat.get("value", stat.get("displayValue", 0))
            if name and value is not None:
                try:
                    stat_map[name] = float(value)
                except (ValueError, TypeError):
                    pass


# ── Rolling stats ──────────────────────────────────────────────────────────────

def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch rolling averages — returns None if game log unavailable.
    
    The model handles None rolling gracefully (uses 100% season weight).
    """
    url = f"{_ESPN_BASE}/athletes/{player_id}/gamelog"
    data = _espn_get(url)
    if not data:
        return None

    try:
        # ESPN gamelog: seasonTypes[].categories[].events[]
        # Each event has a stats array matching the category headers
        season_types = data.get("seasonTypes", [])
        
        all_games = []
        labels = []
        
        for st in season_types:
            st_name = st.get("displayName", "").lower()
            if "regular" not in st_name:
                continue
                
            categories = st.get("categories", [])
            events_section = st.get("events", {})
            
            # Get stat labels from first category
            if categories and not labels:
                for cat in categories:
                    for stat_entry in cat.get("stats", []):
                        labels.append(stat_entry.get("name", stat_entry.get("abbreviation", "")))
            
            # Get event stats
            if isinstance(events_section, list):
                all_games = events_section[:n_games]

        if not all_games:
            return None
            
        # Parse game stats using labels
        pts, reb, ast, tpm, stl, blk, mins = [], [], [], [], [], [], []
        
        for game in all_games[:n_games]:
            game_stats = game.get("stats", [])
            if isinstance(game_stats, list) and labels:
                stat_dict = {}
                for i, val in enumerate(game_stats):
                    if i < len(labels):
                        try:
                            stat_dict[labels[i]] = float(val)
                        except (ValueError, TypeError):
                            pass
                
                pts.append(stat_dict.get("points", stat_dict.get("PTS", 0)))
                reb.append(stat_dict.get("rebounds", stat_dict.get("REB", 0)))
                ast.append(stat_dict.get("assists", stat_dict.get("AST", 0)))
                tpm.append(stat_dict.get("threePointFieldGoalsMade", stat_dict.get("3PM", 0)))
                stl.append(stat_dict.get("steals", stat_dict.get("STL", 0)))
                blk.append(stat_dict.get("blocks", stat_dict.get("BLK", 0)))
                mins.append(stat_dict.get("minutes", stat_dict.get("MIN", 0)))

        n = len(pts)
        if n == 0:
            return None

        return RollingStats(
            n_games=n,
            points_pg=sum(pts) / n,
            rebounds_pg=sum(reb) / n,
            assists_pg=sum(ast) / n,
            three_pm_pg=sum(tpm) / n,
            steals_pg=sum(stl) / n,
            blocks_pg=sum(blk) / n,
            minutes_pg=sum(mins) / n,
        )

    except Exception as e:
        logger.info("[nba_stats] ESPN gamelog parse error for %d: %s", player_id, e)
        return None