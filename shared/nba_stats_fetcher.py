"""
shared/nba_stats_fetcher.py  (v3 — ESPN API)

stats.nba.com blocks cloud IPs. ESPN's public API works everywhere,
no auth needed, no rate limiting issues.

Endpoints used:
  - Player search: site.api.espn.com/.../athletes?search=LASTNAME
  - Player stats:  site.api.espn.com/.../athletes/{id}/statistics
  - Player log:    site.api.espn.com/.../athletes/{id}/gamelog
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

_ESPN_BASE = "https://site.api.espn.com/apis/common/v3/sports/basketball/nba"

# ── Cache ──────────────────────────────────────────────────────────────────────
_player_cache: dict[str, "NBAPlayer"] = {}   # "TEAM-LAST-JERSEY" → player
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


# ── ESPN team abbreviation mapping ─────────────────────────────────────────────
# Maps Kalshi/NBA team codes → ESPN team abbreviations (mostly the same)
_KALSHI_TO_ESPN_TEAM = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "CHA": "CHA",
    "CHI": "CHI", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GSW": "GS",  "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NOP": "NO",  "NYK": "NY",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHX": "PHX",
    "POR": "POR", "SAC": "SAC", "SAS": "SA",  "TOR": "TOR",
    "UTA": "UTAH", "WAS": "WSH",
}


def _espn_get(url: str, timeout: int = 15) -> Optional[dict]:
    """Make a request to the ESPN API."""
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as e:
        logger.warning("[nba_stats] ESPN request failed: %s → %s", url[:80], e)
        return None
    except Exception as e:
        logger.warning("[nba_stats] ESPN unexpected error: %s", e)
        return None


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def _check_cache():
    global _cache_ts
    now = time.monotonic()
    if now - _cache_ts > _CACHE_TTL:
        _player_cache.clear()
        _stats_cache.clear()
        _cache_ts = now


# ── Player lookup via ESPN search ──────────────────────────────────────────────

def lookup_player(
    team_id: int,       # NBA team_id (used for fallback only)
    first_initial: str,
    last_name: str,
    jersey: str,
) -> Optional[NBAPlayer]:
    """
    Find an NBA player via ESPN's athlete search API.

    ESPN search is by last name → filter by first initial + jersey.
    """
    _check_cache()

    cache_key = f"{first_initial}-{last_name}-{jersey}"
    if cache_key in _player_cache:
        return _player_cache[cache_key]

    # Search ESPN for the player by last name
    search_name = last_name.capitalize()
    url = f"{_ESPN_BASE}/athletes?search={search_name}&limit=50"

    data = _espn_get(url)
    if not data:
        return None

    athletes = data.get("items", [])
    if not athletes:
        # ESPN sometimes wraps differently
        athletes = data.get("athletes", [])

    target_first = first_initial.upper()
    target_jersey = str(jersey).lstrip("0") or "0"
    target_last = _normalize(last_name)

    best = None
    best_score = -1

    for athlete in athletes:
        # ESPN can return full athlete objects or $ref links
        # If it's a $ref, we need to fetch it
        if "$ref" in athlete and "displayName" not in athlete:
            athlete = _espn_get(athlete["$ref"]) or {}

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

        first_match = p_first.startswith(target_first)
        last_match = (p_last == target_last)
        jersey_match = (espn_jersey == target_jersey)

        if not first_match:
            continue
        if not last_match:
            # Try subsequence for names like ANTETOKOUNMPO
            if not _is_subseq(target_last, p_last):
                continue

        score = 0
        if last_match: score += 10
        if jersey_match: score += 5
        if first_match: score += 2

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
        logger.debug("[nba_stats] ESPN found: %s (ID: %d)", best.full_name, best.player_id)

    return best


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


# ── Season stats via ESPN ──────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    """Fetch season averages from ESPN."""
    _check_cache()
    if player_id in _stats_cache:
        return _stats_cache[player_id]

    url = f"{_ESPN_BASE}/athletes/{player_id}/statistics"
    data = _espn_get(url)
    if not data:
        return None

    try:
        # ESPN statistics response structure
        # Look for the regular season split
        splits = data.get("statistics", {})

        # Sometimes it's nested under splits.categories
        categories = None
        if isinstance(splits, dict):
            categories = splits.get("splits", {}).get("categories", [])
            if not categories:
                # Try alternate structure
                categories = splits.get("categories", [])
        elif isinstance(splits, list):
            # Direct list of stat categories
            for s in splits:
                if "categories" in s:
                    categories = s["categories"]
                    break

        if not categories:
            logger.debug("[nba_stats] ESPN no categories for player %d", player_id)
            return None

        # Build stat map from categories
        stat_map = {}
        for cat in categories:
            cat_name = cat.get("name", "")
            for stat_entry in cat.get("stats", []):
                stat_name = stat_entry.get("name", "")
                stat_value = stat_entry.get("value", 0)
                stat_map[stat_name] = float(stat_value) if stat_value else 0.0

        gp = int(stat_map.get("gamesPlayed", stat_map.get("GP", 0)))
        if gp == 0:
            return None

        stats = PlayerStats(
            player_id=player_id,
            games_played=gp,
            minutes_pg=stat_map.get("avgMinutes", stat_map.get("MIN", 0)),
            points_pg=stat_map.get("avgPoints", stat_map.get("PTS", 0)),
            rebounds_pg=stat_map.get("avgRebounds", stat_map.get("REB", 0)),
            assists_pg=stat_map.get("avgAssists", stat_map.get("AST", 0)),
            steals_pg=stat_map.get("avgSteals", stat_map.get("STL", 0)),
            blocks_pg=stat_map.get("avgBlocks", stat_map.get("BLK", 0)),
            three_pm_pg=stat_map.get("threePointFieldGoalsMade", stat_map.get("avgThreePointFieldGoalsMade", 0)) / max(gp, 1) if "threePointFieldGoalsMade" in stat_map else stat_map.get("avgThreePointFieldGoalsMade", 0),
            three_pa_pg=stat_map.get("threePointFieldGoalsAttempted", 0) / max(gp, 1) if "threePointFieldGoalsAttempted" in stat_map else 0,
            three_pct=stat_map.get("threePointFieldGoalPct", stat_map.get("3P%", 0)),
            fga_pg=stat_map.get("avgFieldGoalsAttempted", stat_map.get("FGA", 0)),
            fta_pg=stat_map.get("avgFreeThrowsAttempted", stat_map.get("FTA", 0)),
            turnovers_pg=stat_map.get("avgTurnovers", stat_map.get("TO", 0)),
            usage_rate=0.0,
        )

        _stats_cache[player_id] = stats
        return stats

    except Exception as e:
        logger.warning("[nba_stats] ESPN stats parse error for %d: %s", player_id, e)
        return None


# ── Rolling stats via ESPN game log ────────────────────────────────────────────

def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch rolling averages from ESPN game log."""
    url = f"{_ESPN_BASE}/athletes/{player_id}/gamelog"
    data = _espn_get(url)
    if not data:
        return None

    try:
        # ESPN gamelog structure varies; look for events/stats
        events = []

        # Try "categories" → "events" structure
        categories = data.get("categories", [])
        if categories:
            # Each category has events with stats
            events_data = data.get("events", {})
            if isinstance(events_data, list):
                events = events_data[:n_games]
            elif isinstance(events_data, dict):
                events = list(events_data.values())[:n_games]

        # Try alternate "seasonTypes" → "categories" structure
        if not events:
            season_types = data.get("seasonTypes", [])
            for st in season_types:
                if st.get("displayName", "").lower() in ("regular season", "regular"):
                    categories = st.get("categories", [])
                    events = st.get("events", [])
                    break

        # If we can't parse the game log, just return None
        # The model will use season-only stats (70% weight becomes 100%)
        if not events:
            logger.debug("[nba_stats] ESPN game log empty for player %d — using season only", player_id)
            return None

        # Parse events into stats
        pts_list, reb_list, ast_list = [], [], []
        tpm_list, stl_list, blk_list, min_list = [], [], [], []

        for event in events[:n_games]:
            stats = event.get("stats", [])
            if isinstance(stats, dict):
                pts_list.append(float(stats.get("PTS", stats.get("points", 0))))
                reb_list.append(float(stats.get("REB", stats.get("rebounds", 0))))
                ast_list.append(float(stats.get("AST", stats.get("assists", 0))))
                tpm_list.append(float(stats.get("FG3M", stats.get("threePointFieldGoalsMade", 0))))
                stl_list.append(float(stats.get("STL", stats.get("steals", 0))))
                blk_list.append(float(stats.get("BLK", stats.get("blocks", 0))))
                min_list.append(float(stats.get("MIN", stats.get("minutes", 0))))

        n = len(pts_list)
        if n == 0:
            return None

        return RollingStats(
            n_games=n,
            points_pg=sum(pts_list) / n,
            rebounds_pg=sum(reb_list) / n,
            assists_pg=sum(ast_list) / n,
            three_pm_pg=sum(tpm_list) / n,
            steals_pg=sum(stl_list) / n,
            blocks_pg=sum(blk_list) / n,
            minutes_pg=sum(min_list) / n,
        )

    except Exception as e:
        logger.warning("[nba_stats] ESPN game log parse error for %d: %s", player_id, e)
        return None