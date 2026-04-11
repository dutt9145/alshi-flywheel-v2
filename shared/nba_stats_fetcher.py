"""
shared/nba_stats_fetcher.py

Fetches NBA player stats from the official NBA Stats API.
Mirrors shared/mlb_stats_fetcher.py architecture.

Data source: stats.nba.com (free, no auth, requires specific headers)

Usage:
    from shared.nba_stats_fetcher import lookup_player, fetch_player_stats, fetch_player_rolling
    player = lookup_player("BOS", "J", "TATUM", "0")
    stats = fetch_player_stats(player.player_id)
    rolling = fetch_player_rolling(player.player_id, n_games=10)
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# ── NBA Stats API headers (required to avoid 403) ─────────────────────────────
_NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

_NBA_BASE = "https://stats.nba.com/stats"

# ── Cache for roster/player lookups ────────────────────────────────────────────
_roster_cache: dict[int, list] = {}  # team_id → list of player dicts
_player_stats_cache: dict[int, "PlayerStats"] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600  # 1 hour


def _nba_get(endpoint: str, params: dict, timeout: int = 10) -> Optional[dict]:
    """Make a request to the NBA Stats API."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_NBA_BASE}/{endpoint}?{query}"

    req = Request(url)
    for k, v in _NBA_HEADERS.items():
        req.add_header(k, v)

    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        logger.warning("[nba_stats] HTTP %d for %s", e.code, endpoint)
        return None
    except (URLError, TimeoutError) as e:
        logger.warning("[nba_stats] request failed for %s: %s", endpoint, e)
        return None
    except Exception as e:
        logger.warning("[nba_stats] unexpected error for %s: %s", endpoint, e)
        return None


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class NBAPlayer:
    player_id:  int
    full_name:  str
    team_id:    int
    jersey:     str


@dataclass
class PlayerStats:
    """Season averages for an NBA player."""
    player_id:      int
    games_played:   int
    minutes_pg:     float
    points_pg:      float
    rebounds_pg:    float
    assists_pg:     float
    steals_pg:      float
    blocks_pg:      float
    three_pm_pg:    float   # 3-pointers made per game
    three_pa_pg:    float   # 3-point attempts per game
    three_pct:      float   # 3-point percentage
    fga_pg:         float   # field goal attempts per game
    fta_pg:         float   # free throw attempts per game
    turnovers_pg:   float
    usage_rate:     float   # usage percentage if available


@dataclass
class RollingStats:
    """Rolling averages over last N games."""
    n_games:        int
    points_pg:      float
    rebounds_pg:    float
    assists_pg:     float
    three_pm_pg:    float
    steals_pg:      float
    blocks_pg:      float
    minutes_pg:     float


# ── Player lookup ──────────────────────────────────────────────────────────────

def _get_current_season() -> str:
    """Return current NBA season string, e.g. '2025-26'."""
    from datetime import datetime
    now = datetime.now()
    # NBA season starts in October
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[2:]}"


def _fetch_roster(team_id: int) -> list[dict]:
    """Fetch current roster for a team."""
    global _cache_ts

    now = time.monotonic()
    if now - _cache_ts > _CACHE_TTL:
        _roster_cache.clear()
        _player_stats_cache.clear()
        _cache_ts = now

    if team_id in _roster_cache:
        return _roster_cache[team_id]

    season = _get_current_season()
    data = _nba_get("commonteamroster", {
        "TeamID": team_id,
        "Season": season,
    })

    if not data:
        return []

    try:
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        roster = [dict(zip(headers, row)) for row in rows]
        _roster_cache[team_id] = roster
        return roster
    except (KeyError, IndexError) as e:
        logger.warning("[nba_stats] roster parse error for team %d: %s", team_id, e)
        return []


def _normalize_name(name: str) -> str:
    """Strip diacritics and normalize for comparison."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def lookup_player(
    team_id: int,
    first_initial: str,
    last_name: str,
    jersey: str,
) -> Optional[NBAPlayer]:
    """
    Find a player on a team by first initial, last name, and jersey number.

    Uses fuzzy matching similar to MLB fetcher:
    1. Exact last name + first initial match
    2. Subsequence match for diacritic-stripped names
    3. Jersey number tiebreaker
    """
    roster = _fetch_roster(team_id)
    if not roster:
        return None

    target_last = last_name.upper()
    target_first = first_initial.upper()
    target_jersey = str(jersey).lstrip("0") or "0"

    candidates = []
    for player in roster:
        # Player names: "PLAYER" field has full name, or use "PLAYER_NAME"
        full_name = player.get("PLAYER", "")
        p_num = str(player.get("NUM", "")).strip().lstrip("0") or "0"
        p_id = player.get("PLAYER_ID") or player.get("TeamID")

        if not full_name or not p_id:
            continue

        name_parts = full_name.strip().split()
        if len(name_parts) < 2:
            continue

        p_first = _normalize_name(name_parts[0])
        p_last = _normalize_name(" ".join(name_parts[1:]))

        # Score matching
        first_match = p_first.startswith(target_first)
        last_exact = (p_last == target_last)
        last_subseq = _is_subsequence(target_last, p_last)
        jersey_match = (p_num == target_jersey)

        if first_match and (last_exact or last_subseq):
            score = 0
            if last_exact:
                score += 10
            if jersey_match:
                score += 5
            if last_subseq and not last_exact:
                score += 3

            candidates.append((score, NBAPlayer(
                player_id=int(p_id),
                full_name=full_name,
                team_id=team_id,
                jersey=p_num,
            )))

    if not candidates:
        # Fallback: jersey number only
        for player in roster:
            p_num = str(player.get("NUM", "")).strip().lstrip("0") or "0"
            if p_num == target_jersey:
                full_name = player.get("PLAYER", "Unknown")
                p_id = player.get("PLAYER_ID")
                if p_id:
                    return NBAPlayer(
                        player_id=int(p_id),
                        full_name=full_name,
                        team_id=team_id,
                        jersey=p_num,
                    )
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _is_subsequence(short: str, long: str) -> bool:
    """Check if short is a subsequence of long."""
    it = iter(long)
    return all(c in it for c in short)


# ── Player stats fetching ──────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    """Fetch season averages for a player."""
    if player_id in _player_stats_cache:
        return _player_stats_cache[player_id]

    season = _get_current_season()
    data = _nba_get("playerdashboardbygeneralsplits", {
        "PlayerID": player_id,
        "Season": season,
        "MeasureType": "Base",
        "PerMode": "PerGame",
        "SeasonType": "Regular Season",
    })

    if not data:
        return None

    try:
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        if not rows:
            return None
        overall = dict(zip(headers, rows[0]))

        gp = int(overall.get("GP", 0))
        if gp == 0:
            return None

        stats = PlayerStats(
            player_id=player_id,
            games_played=gp,
            minutes_pg=float(overall.get("MIN", 0)),
            points_pg=float(overall.get("PTS", 0)),
            rebounds_pg=float(overall.get("REB", 0)),
            assists_pg=float(overall.get("AST", 0)),
            steals_pg=float(overall.get("STL", 0)),
            blocks_pg=float(overall.get("BLK", 0)),
            three_pm_pg=float(overall.get("FG3M", 0)),
            three_pa_pg=float(overall.get("FG3A", 0)),
            three_pct=float(overall.get("FG3_PCT", 0)),
            fga_pg=float(overall.get("FGA", 0)),
            fta_pg=float(overall.get("FTA", 0)),
            turnovers_pg=float(overall.get("TOV", 0)),
            usage_rate=0.0,  # would need advanced stats endpoint
        )

        _player_stats_cache[player_id] = stats
        return stats

    except (KeyError, IndexError, TypeError) as e:
        logger.warning("[nba_stats] stats parse error for player %d: %s", player_id, e)
        return None


def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch rolling averages from the player's game log."""
    season = _get_current_season()
    data = _nba_get("playergamelog", {
        "PlayerID": player_id,
        "Season": season,
        "SeasonType": "Regular Season",
    })

    if not data:
        return None

    try:
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        if not rows:
            return None

        # Take last n_games (rows are most recent first)
        recent = [dict(zip(headers, row)) for row in rows[:n_games]]
        n = len(recent)

        if n == 0:
            return None

        return RollingStats(
            n_games=n,
            points_pg=sum(float(g.get("PTS", 0)) for g in recent) / n,
            rebounds_pg=sum(float(g.get("REB", 0)) for g in recent) / n,
            assists_pg=sum(float(g.get("AST", 0)) for g in recent) / n,
            three_pm_pg=sum(float(g.get("FG3M", 0)) for g in recent) / n,
            steals_pg=sum(float(g.get("STL", 0)) for g in recent) / n,
            blocks_pg=sum(float(g.get("BLK", 0)) for g in recent) / n,
            minutes_pg=sum(float(str(g.get("MIN", "0")).replace(":", ".")) for g in recent) / n,
        )

    except (KeyError, IndexError, TypeError) as e:
        logger.warning("[nba_stats] game log parse error for player %d: %s", player_id, e)
        return None


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("NBA stats fetcher — self-test")
    print("=" * 78)

    from shared.nba_ticker_parser import NBA_TEAMS

    test_players = [
        ("BOS", "J", "TATUM", "0"),
        ("LAL", "L", "JAMES", "23"),
        ("MIL", "G", "ANTETOKOUNMPO", "34"),
        ("DEN", "N", "JOKIC", "15"),
        ("DAL", "L", "DONCIC", "77"),
    ]

    for team_code, first, last, jersey in test_players:
        team_info = NBA_TEAMS.get(team_code)
        if not team_info:
            print(f"✗ Unknown team: {team_code}")
            continue

        team_id = team_info[0]
        player = lookup_player(team_id, first, last, jersey)

        if player:
            print(f"✓ {first}. {last} #{jersey} ({team_code}) → {player.full_name} (ID: {player.player_id})")

            stats = fetch_player_stats(player.player_id)
            if stats:
                print(f"  Season: {stats.games_played}G | "
                      f"{stats.points_pg:.1f}ppg {stats.rebounds_pg:.1f}rpg "
                      f"{stats.assists_pg:.1f}apg {stats.three_pm_pg:.1f} 3pm/g")
            else:
                print(f"  ✗ Could not fetch season stats")

            rolling = fetch_player_rolling(player.player_id, n_games=5)
            if rolling:
                print(f"  L{rolling.n_games}: {rolling.points_pg:.1f}ppg "
                      f"{rolling.rebounds_pg:.1f}rpg {rolling.assists_pg:.1f}apg")
            else:
                print(f"  ✗ Could not fetch rolling stats")
        else:
            print(f"✗ Player not found: {first}. {last} #{jersey} ({team_code})")

        time.sleep(0.5)  # rate limit

    print(f"\n{'=' * 78}")
    print("Done.")
    print(f"{'=' * 78}")