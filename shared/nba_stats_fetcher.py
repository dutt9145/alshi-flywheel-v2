"""
shared/nba_stats_fetcher.py  (v7 — ESPN primary, BDL fallback)

v7 changes
----------
ESPN public API (no auth required) is now the PRIMARY source for:
  - Player lookup  → bulk team roster fetch (30 calls caches ALL players)
  - Season stats   → per-player statistics endpoint
  - Rolling stats  → per-player gamelog endpoint

BallDontLie remains as FALLBACK when ESPN fails.

This eliminates the hard dependency on a BDL API key that was causing
100% of NBA player prop lookups to fail (401 Unauthorized).

ESPN endpoints (no auth):
  Roster:  site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster
  Stats:   site.api.espn.com/apis/site/v2/sports/basketball/nba/players/{id}/statistics
  Gamelog: site.api.espn.com/apis/site/v2/sports/basketball/nba/players/{id}/gamelog

BDL endpoints (auth required):
  Players:         api.balldontlie.io/v1/players?search=LASTNAME
  Season averages: api.balldontlie.io/v1/season_averages
  Game log:        api.balldontlie.io/v1/stats
"""

import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_BDL_BASE = "https://api.balldontlie.io/v1"
_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

if _API_KEY:
    logger.info("[nba_stats] BDL API key loaded (%d chars) — available as fallback", len(_API_KEY))
else:
    logger.info("[nba_stats] No BDL API key — ESPN-only mode (this is fine)")


# ── ESPN team IDs ──────────────────────────────────────────────────────────────
_ESPN_TEAM_IDS: dict[str, int] = {
    "ATL": 1,  "BOS": 2,  "BKN": 17, "CHA": 30, "CHI": 4,
    "CLE": 5,  "DAL": 6,  "DEN": 7,  "DET": 8,  "GSW": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
    "MIA": 14, "MIL": 15, "MIN": 16, "NOP": 3,  "NYK": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
    "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
}


# ── Data classes (unchanged interface from v6) ────────────────────────────────

@dataclass
class NBAPlayer:
    player_id:  int
    full_name:  str
    team_id:    int
    jersey:     str
    source:     str = "espn"    # "espn" or "bdl" — tells stats which API to hit


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


# ── Caches ─────────────────────────────────────────────────────────────────────
_player_cache:   dict[str, NBAPlayer] = {}   # cache_key → NBAPlayer
_roster_loaded:  set[str] = set()            # team codes whose rosters are cached
_roster_ts:      float = 0.0
_ROSTER_TTL      = 21600                     # 6 hours

_stats_cache:    dict[int, PlayerStats] = {}
_rolling_cache:  dict[int, RollingStats] = {}
_stats_ts:       float = 0.0
_STATS_TTL       = 3600                      # 1 hour


def _check_roster_cache():
    global _roster_ts
    now = time.monotonic()
    if now - _roster_ts > _ROSTER_TTL:
        _player_cache.clear()
        _roster_loaded.clear()
        _roster_ts = now


def _check_stats_cache():
    global _stats_ts
    now = time.monotonic()
    if now - _stats_ts > _STATS_TTL:
        _stats_cache.clear()
        _rolling_cache.clear()
        _stats_ts = now


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _espn_get(url: str, timeout: int = 15) -> Optional[dict]:
    """Fetch from ESPN public API (no auth required)."""
    req = Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "KalshiFlywheel/1.0")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        logger.warning("[nba_stats] ESPN %d: %s", e.code, url[:100])
        return None
    except Exception as e:
        logger.warning("[nba_stats] ESPN request failed: %s → %s", url[:80], e)
        return None


def _bdl_get(endpoint: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """Authenticated request to BallDontLie (fallback)."""
    if not _API_KEY:
        return None
    query = ""
    if params:
        parts = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    parts.append(f"{k}[]={item}")
            else:
                parts.append(f"{k}={v}")
        query = "&".join(parts)
    url = f"{_BDL_BASE}/{endpoint}"
    if query:
        url += f"?{query}"
    req = Request(url)
    req.add_header("Authorization", _API_KEY)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "KalshiFlywheel/1.0")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        logger.info("[nba_stats] BDL %d: %s", e.code, endpoint)
        return None
    except Exception as e:
        logger.info("[nba_stats] BDL failed: %s → %s", endpoint, e)
        return None


def _normalize(name: str) -> str:
    """Strip accents/diacritics and uppercase."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def _get_current_season() -> int:
    from datetime import datetime
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def _parse_minutes(min_str) -> float:
    """Parse '32:15' or '32.25' or '32' → float minutes."""
    s = str(min_str or "0")
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  ESPN ROSTER LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _espn_load_team_roster(team_code: str) -> list[NBAPlayer]:
    """Fetch one team's full roster from ESPN (no auth)."""
    espn_id = _ESPN_TEAM_IDS.get(team_code)
    if not espn_id:
        return []

    url = f"{_ESPN_BASE}/teams/{espn_id}/roster"
    data = _espn_get(url)
    if not data:
        return []

    players: list[NBAPlayer] = []
    athletes = data.get("athletes", [])

    for entry in athletes:
        # ESPN groups players by position category:
        #   {"position": "Guards", "items": [{player}, ...]}
        # OR returns a flat list of player dicts
        if isinstance(entry, dict) and "items" in entry:
            items = entry["items"]
        elif isinstance(entry, dict) and "id" in entry:
            items = [entry]
        else:
            continue

        for p in items:
            try:
                pid = int(p.get("id", 0))
                name = (p.get("fullName", "")
                        or p.get("displayName", "")
                        or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip())
                jersey = str(p.get("jersey", "")).lstrip("0") or "0"

                if pid and name:
                    players.append(NBAPlayer(
                        player_id=pid,
                        full_name=name,
                        team_id=espn_id,
                        jersey=jersey,
                        source="espn",
                    ))
            except (ValueError, TypeError):
                continue

    return players


def _espn_ensure_team_loaded(team_code: str):
    """Load a single team's roster into the player cache."""
    _check_roster_cache()
    if team_code in _roster_loaded:
        return

    players = _espn_load_team_roster(team_code)
    if not players:
        logger.warning("[nba_stats] ESPN roster empty for %s — will try BDL", team_code)
        _roster_loaded.add(team_code)      # don't retry every scan
        return

    for p in players:
        norm_name = _normalize(p.full_name)
        name_parts = norm_name.split()
        if len(name_parts) < 2:
            continue

        first_initial = name_parts[0][0]
        last_name     = name_parts[-1]

        # Primary key: initial-last-jersey  (exact match)
        key1 = f"{first_initial}-{last_name}-{p.jersey}"
        _player_cache[key1] = p

        # Secondary key: initial-last-team  (jersey might differ)
        key2 = f"{first_initial}-{last_name}-{team_code}"
        _player_cache.setdefault(key2, p)

        # Tertiary: full-name-team (for brute-force fallback)
        key3 = f"FULL-{norm_name}-{team_code}"
        _player_cache[key3] = p

    _roster_loaded.add(team_code)
    logger.info("[nba_stats] ESPN roster loaded: %s (%d players)", team_code, len(players))


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAYER LOOKUP  (ESPN primary → BDL fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_player(
    team_id: int,
    first_initial: str,
    last_name: str,
    jersey: str,
    team_code: str = "",
) -> Optional[NBAPlayer]:
    """
    Find an NBA player by initial, last name, jersey, and team.

    Tries ESPN roster cache first (free, bulk-loaded).
    Falls back to BDL search API if ESPN can't match.
    """
    target_first  = first_initial.upper()
    target_last   = _normalize(last_name)
    target_jersey = str(jersey).lstrip("0") or "0"

    # ── 1. ESPN roster lookup ──────────────────────────────────────────────
    if team_code:
        _espn_ensure_team_loaded(team_code)

    # Try exact: initial + last + jersey
    key_exact = f"{target_first}-{target_last}-{target_jersey}"
    if key_exact in _player_cache:
        return _player_cache[key_exact]

    # Try team-based: initial + last + team_code
    if team_code:
        key_team = f"{target_first}-{target_last}-{team_code}"
        if key_team in _player_cache:
            p = _player_cache[key_team]
            _player_cache[key_exact] = p       # cache the jersey key too
            return p

    # Brute-force: scan all loaded players for this team by name similarity
    if team_code:
        espn_tid = _ESPN_TEAM_IDS.get(team_code, 0)
        for ck, p in _player_cache.items():
            if p.team_id != espn_tid:
                continue
            norm = _normalize(p.full_name)
            parts = norm.split()
            if len(parts) < 2:
                continue
            p_first = parts[0][0]
            p_last  = parts[-1]
            if p_first == target_first and p_last == target_last:
                _player_cache[key_exact] = p
                return p

            # Handle compound last names: "GILGEOUS-ALEXANDER" → Kalshi may
            # strip the hyphen and spaces: "GILGEOUSALEXANDER"
            joined = "".join(parts[1:])
            if parts[0][0] == target_first and joined == target_last:
                _player_cache[key_exact] = p
                return p

    # ── 2. BDL fallback ────────────────────────────────────────────────────
    bdl_player = _bdl_lookup_player(team_id, target_first, target_last,
                                     target_jersey, team_code)
    if bdl_player:
        _player_cache[key_exact] = bdl_player
    return bdl_player


def _bdl_lookup_player(
    team_id: int,
    first_initial: str,
    last_name: str,
    jersey: str,
    team_code: str,
) -> Optional[NBAPlayer]:
    """BDL search fallback (requires API key)."""
    data = _bdl_get("players", {"search": last_name, "per_page": 25})
    if not data:
        return None

    players = data.get("data", [])
    if not players:
        return None

    target_jersey = str(jersey).lstrip("0") or "0"
    best: Optional[NBAPlayer] = None
    best_score = -1

    for p in players:
        p_first = _normalize(p.get("first_name", ""))
        p_last  = _normalize(p.get("last_name", ""))
        p_id    = p.get("id")
        p_jersey = str(p.get("jersey_number", "")).lstrip("0") or "0"
        p_team   = p.get("team", {}).get("abbreviation", "")

        if not p_first or not p_id or not p_first.startswith(first_initial):
            continue
        if p_last != _normalize(last_name):
            continue

        score = 10
        if p_jersey == target_jersey:
            score += 5
        if team_code and p_team == team_code:
            score += 3

        if score > best_score:
            best_score = score
            best = NBAPlayer(
                player_id=int(p_id),
                full_name=f"{p.get('first_name', '')} {p.get('last_name', '')}",
                team_id=team_id,
                jersey=p_jersey,
                source="bdl",
            )

    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  SEASON STATS  (ESPN primary → BDL fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _espn_parse_stat_value(raw) -> float:
    """Parse a stat value that might be a string, int, float, or '-'."""
    if raw is None or raw == "-" or raw == "":
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _espn_extract_stats_dict(data: dict) -> dict[str, float]:
    """
    Extract stat name→value from ESPN's various response formats.
    Returns a flat dict like {"GP": 65, "PTS": 25.8, "REB": 5.4, ...}.
    """
    result: dict[str, float] = {}

    # ── Format A: "statistics" object with "splits.categories[].stats[]" ──
    stats_obj = data.get("statistics", {})
    if isinstance(stats_obj, dict):
        splits = stats_obj.get("splits", {})
        if isinstance(splits, dict):
            for cat in splits.get("categories", []):
                for s in cat.get("stats", []):
                    name = s.get("name", "").upper()
                    val  = s.get("value")
                    if name and val is not None:
                        result[name] = _espn_parse_stat_value(val)
        if result:
            return result

    # ── Format B: "statistics" as a list with labels[] + stats[] ──────────
    stats_list = data.get("statistics", [])
    if isinstance(stats_list, list):
        for block in stats_list:
            if not isinstance(block, dict):
                continue
            labels = block.get("labels", block.get("names", []))
            values = block.get("stats", block.get("values", block.get("averages", [])))
            if labels and values and len(labels) == len(values):
                for label, val in zip(labels, values):
                    result[label.upper()] = _espn_parse_stat_value(val)
        if result:
            return result

    # ── Format C: "categories" at top level ───────────────────────────────
    for cat in data.get("categories", []):
        labels = cat.get("labels", [])
        values = cat.get("stats", cat.get("totals", []))
        if labels and values:
            for label, val in zip(labels, values):
                result[label.upper()] = _espn_parse_stat_value(val)

    # ── Format D: "athlete.statistics" embedded ───────────────────────────
    athlete = data.get("athlete", {})
    if isinstance(athlete, dict):
        embedded = athlete.get("statistics", {})
        if embedded:
            return _espn_extract_stats_dict({"statistics": embedded})

    if not result:
        logger.debug("[nba_stats] ESPN stats parse: no recognized format. Keys: %s",
                     list(data.keys())[:12])

    return result


def _espn_fetch_stats(player_id: int) -> Optional[PlayerStats]:
    """Fetch season stats from ESPN."""
    url = f"{_ESPN_BASE}/players/{player_id}/statistics"
    data = _espn_get(url)
    if not data:
        return None

    s = _espn_extract_stats_dict(data)
    if not s:
        logger.info("[nba_stats] ESPN no parseable stats for player %d", player_id)
        return None

    gp = int(s.get("GP", 0))
    if gp == 0:
        return None

    three_pa = s.get("3PA", s.get("FG3A", 0.0))
    three_pm = s.get("3PM", s.get("FG3M", 0.0))

    return PlayerStats(
        player_id=player_id,
        games_played=gp,
        minutes_pg=s.get("MIN", s.get("MINS", 0.0)),
        points_pg=s.get("PTS", 0.0),
        rebounds_pg=s.get("REB", 0.0),
        assists_pg=s.get("AST", 0.0),
        steals_pg=s.get("STL", 0.0),
        blocks_pg=s.get("BLK", 0.0),
        three_pm_pg=three_pm,
        three_pa_pg=three_pa,
        three_pct=three_pm / max(three_pa, 0.01),
        fga_pg=s.get("FGA", 0.0),
        fta_pg=s.get("FTA", 0.0),
        turnovers_pg=s.get("TO", s.get("TOV", 0.0)),
        usage_rate=0.0,
    )


def _bdl_fetch_stats(player_id: int) -> Optional[PlayerStats]:
    """BDL season averages fallback."""
    season = _get_current_season()
    data = _bdl_get("season_averages", {
        "season": season,
        "player_ids": [player_id],
    })
    if not data:
        return None

    averages = data.get("data", [])
    if not averages:
        return None

    avg = averages[0]
    gp = int(avg.get("games_played", 0))
    if gp == 0:
        return None

    minutes = _parse_minutes(avg.get("min", "0"))
    three_pa = float(avg.get("fg3a", 0))
    three_pm = float(avg.get("fg3m", 0))

    return PlayerStats(
        player_id=player_id,
        games_played=gp,
        minutes_pg=minutes,
        points_pg=float(avg.get("pts", 0)),
        rebounds_pg=float(avg.get("reb", 0)),
        assists_pg=float(avg.get("ast", 0)),
        steals_pg=float(avg.get("stl", 0)),
        blocks_pg=float(avg.get("blk", 0)),
        three_pm_pg=three_pm,
        three_pa_pg=three_pa,
        three_pct=three_pm / max(three_pa, 0.01),
        fga_pg=float(avg.get("fga", 0)),
        fta_pg=float(avg.get("fta", 0)),
        turnovers_pg=float(avg.get("turnover", 0)),
        usage_rate=0.0,
    )


def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    """Fetch season averages. ESPN first, BDL fallback."""
    _check_stats_cache()
    if player_id in _stats_cache:
        return _stats_cache[player_id]

    # Determine source from the player object if available
    source = "espn"
    for p in _player_cache.values():
        if p.player_id == player_id:
            source = p.source
            break

    stats: Optional[PlayerStats] = None

    if source == "espn":
        stats = _espn_fetch_stats(player_id)
        if not stats:
            # ESPN failed — don't try BDL with ESPN ID (wrong ID space)
            logger.info("[nba_stats] ESPN stats failed for player %d", player_id)
            return None
    else:
        # BDL player — use BDL stats
        stats = _bdl_fetch_stats(player_id)

    if stats:
        logger.debug("[nba_stats] %d: %dG %.1fppg %.1frpg %.1fapg (via %s)",
                     player_id, stats.games_played, stats.points_pg,
                     stats.rebounds_pg, stats.assists_pg, source)
        _stats_cache[player_id] = stats

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  ROLLING STATS / GAMELOG  (ESPN primary → BDL fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _espn_fetch_gamelog(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch recent game log from ESPN."""
    url = f"{_ESPN_BASE}/players/{player_id}/gamelog"
    data = _espn_get(url)
    if not data:
        return None

    games: list[dict[str, float]] = []

    # ESPN gamelog format: seasonTypes[].categories[].events[]
    # Each event has stats[] aligned with labels[]
    for st in data.get("seasonTypes", []):
        for cat in st.get("categories", []):
            labels = [l.upper() for l in cat.get("labels", [])]
            if not labels:
                continue
            for event in cat.get("events", []):
                raw_stats = event.get("stats", [])
                game: dict[str, float] = {}
                for i, label in enumerate(labels):
                    if i >= len(raw_stats):
                        break
                    val = raw_stats[i]
                    if isinstance(val, str) and "-" in val and not val.startswith("-"):
                        # "10-18" → made/attempted (FG, 3PT, FT)
                        try:
                            parts = val.split("-")
                            game[label] = float(parts[0])           # made
                            game[f"{label}_ATT"] = float(parts[1])  # attempted
                        except (ValueError, IndexError):
                            pass
                    else:
                        try:
                            game[label] = float(val)
                        except (ValueError, TypeError):
                            pass
                if game:
                    games.append(game)

    if not games:
        logger.debug("[nba_stats] ESPN gamelog empty for player %d. Keys: %s",
                     player_id, list(data.keys())[:8])
        return None

    # Take most recent n_games (ESPN returns most recent first)
    recent = games[:n_games]
    n = len(recent)

    pts  = [g.get("PTS", 0.0) for g in recent]
    reb  = [g.get("REB", 0.0) for g in recent]
    ast  = [g.get("AST", 0.0) for g in recent]
    tpm  = [g.get("3PT", g.get("FG3M", g.get("3PM", 0.0))) for g in recent]
    stl  = [g.get("STL", 0.0) for g in recent]
    blk  = [g.get("BLK", 0.0) for g in recent]
    mins = [g.get("MIN", 0.0) for g in recent]

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


def _bdl_fetch_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """BDL game log fallback."""
    season = _get_current_season()
    data = _bdl_get("stats", {
        "player_ids": [player_id],
        "seasons": [season],
        "per_page": n_games,
        "sort": "-game.date",
    })
    if not data:
        return None

    games = data.get("data", [])
    if not games:
        return None

    pts, reb, ast, tpm, stl, blk, mins = [], [], [], [], [], [], []
    for g in games[:n_games]:
        pts.append(float(g.get("pts", 0)))
        reb.append(float(g.get("reb", 0)))
        ast.append(float(g.get("ast", 0)))
        tpm.append(float(g.get("fg3m", 0)))
        stl.append(float(g.get("stl", 0)))
        blk.append(float(g.get("blk", 0)))
        mins.append(_parse_minutes(g.get("min", "0")))

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


def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch last N games rolling stats. ESPN first, BDL fallback."""
    _check_stats_cache()
    cache_key = player_id
    if cache_key in _rolling_cache:
        return _rolling_cache[cache_key]

    source = "espn"
    for p in _player_cache.values():
        if p.player_id == player_id:
            source = p.source
            break

    rolling: Optional[RollingStats] = None

    if source == "espn":
        rolling = _espn_fetch_gamelog(player_id, n_games)
    else:
        rolling = _bdl_fetch_rolling(player_id, n_games)

    if rolling:
        _rolling_cache[cache_key] = rolling

    return rolling