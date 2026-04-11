"""
shared/nba_stats_fetcher.py  (v6 — BallDontLie API)

Purpose-built NBA stats API. Free tier: 30 req/min.
Docs: https://docs.balldontlie.io

Setup:
  1. Sign up at https://app.balldontlie.io
  2. Add BALLDONTLIE_API_KEY to Railway env vars

Endpoints:
  - GET /players?search=LASTNAME → player lookup
  - GET /season_averages?season=2025&player_ids[]=X → season stats
  - GET /stats?player_ids[]=X&seasons[]=2025&per_page=10 → game log
"""

import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_BDL_BASE = "https://api.balldontlie.io/v1"
_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")

if _API_KEY:
    logger.info("[nba_stats] BallDontLie API key loaded (%d chars)", len(_API_KEY))
else:
    logger.warning("[nba_stats] BALLDONTLIE_API_KEY not set — NBA player props will be disabled")

# ── Cache ──────────────────────────────────────────────────────────────────────
_player_cache: dict[str, "NBAPlayer"] = {}
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


def _bdl_get(endpoint: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """Make an authenticated request to BallDontLie API."""
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
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        logger.info("[nba_stats] BDL %d: %s %s → %s", e.code, endpoint, query[:60], body[:100])
        return None
    except Exception as e:
        logger.info("[nba_stats] BDL request failed: %s → %s", endpoint, e)
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


def _get_current_season() -> int:
    """Return NBA season year (e.g. 2025 for the 2025-26 season)."""
    from datetime import datetime
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


# ── Player lookup ──────────────────────────────────────────────────────────────

def lookup_player(
    team_id: int,
    first_initial: str,
    last_name: str,
    jersey: str,
    team_code: str = "",
) -> Optional[NBAPlayer]:
    """Find NBA player via BallDontLie search API."""
    if not _API_KEY:
        return None

    _check_cache()
    cache_key = f"{first_initial}-{last_name}-{jersey}"
    if cache_key in _player_cache:
        return _player_cache[cache_key]

    # Search by last name
    data = _bdl_get("players", {"search": last_name, "per_page": 25})
    if not data:
        return None

    players = data.get("data", [])
    if not players:
        logger.info("[nba_stats] BDL no results for search=%s", last_name)
        return None

    target_first = first_initial.upper()
    target_last = _normalize(last_name)
    target_jersey = str(jersey).lstrip("0") or "0"

    best = None
    best_score = -1

    for p in players:
        p_first = _normalize(p.get("first_name", ""))
        p_last = _normalize(p.get("last_name", ""))
        p_id = p.get("id")
        p_jersey = str(p.get("jersey_number", "")).lstrip("0") or "0"
        p_team = p.get("team", {}).get("abbreviation", "")

        if not p_first or not p_id:
            continue

        if not p_first.startswith(target_first):
            continue

        last_exact = (p_last == target_last)
        if not last_exact:
            continue

        score = 10  # last name match
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
            )

    if best:
        _player_cache[cache_key] = best
        logger.info("[nba_stats] found: %s (BDL ID: %d)", best.full_name, best.player_id)
    else:
        logger.info("[nba_stats] NOT FOUND: %s.%s #%s (searched %d results)",
                     first_initial, last_name, jersey, len(players))

    return best


# ── Season stats ───────────────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    """Fetch season averages from BallDontLie."""
    if not _API_KEY:
        return None

    _check_cache()
    if player_id in _stats_cache:
        return _stats_cache[player_id]

    season = _get_current_season()
    data = _bdl_get("season_averages", {
        "season": season,
        "player_ids": [player_id],
    })

    if not data:
        return None

    averages = data.get("data", [])
    if not averages:
        logger.info("[nba_stats] BDL no season averages for player %d season %d", player_id, season)
        return None

    avg = averages[0]
    gp = int(avg.get("games_played", 0))
    if gp == 0:
        return None

    # Parse minutes string "32:15" → 32.25
    min_str = str(avg.get("min", "0"))
    if ":" in min_str:
        parts = min_str.split(":")
        minutes = float(parts[0]) + float(parts[1]) / 60
    else:
        try:
            minutes = float(min_str)
        except ValueError:
            minutes = 0.0

    three_pa = float(avg.get("fg3a", 0))
    three_pm = float(avg.get("fg3m", 0))

    stats = PlayerStats(
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

    logger.info("[nba_stats] %d: %dG %.1fppg %.1frpg %.1fapg %.1f 3pm",
                 player_id, gp, stats.points_pg, stats.rebounds_pg,
                 stats.assists_pg, stats.three_pm_pg)

    _stats_cache[player_id] = stats
    return stats


# ── Rolling stats ──────────────────────────────────────────────────────────────

def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    """Fetch last N games from BallDontLie stats endpoint."""
    if not _API_KEY:
        return None

    season = _get_current_season()
    data = _bdl_get("stats", {
        "player_ids": [player_id],
        "seasons": [season],
        "per_page": n_games,
        "sort": "-game.date",  # most recent first
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

        min_str = str(g.get("min", "0"))
        if ":" in min_str:
            parts = min_str.split(":")
            mins.append(float(parts[0]) + float(parts[1]) / 60)
        else:
            try:
                mins.append(float(min_str))
            except ValueError:
                mins.append(0.0)

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