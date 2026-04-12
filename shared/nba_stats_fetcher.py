"""
shared/nba_stats_fetcher.py  (v9 — use /overview endpoint)

v9 changes
----------
Use /overview endpoint:
  https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/overview

Response structure:
  {
    "statistics": {
      "labels": ["GP","MIN","FG%","3P%","FT%","REB","AST","BLK","STL","PF","TO","PTS"],
      "splits": [
        {"displayName": "Regular Season", "stats": ["42","31.0","46.8",...,"26.6"]},
        {"displayName": "Career", "stats": [...]}
      ]
    }
  }

splits[0].stats is parallel to labels — both are string arrays.
"""

import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_BDL_BASE = "https://api.balldontlie.io/v1"
_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")

# v9: Use /overview for stats
_ESPN_ROSTER_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_ESPN_STATS_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"

if _API_KEY:
    logger.info("[nba_stats] BDL API key loaded (%d chars) — available as fallback", len(_API_KEY))
else:
    logger.info("[nba_stats] No BDL API key — ESPN-only mode")


# ── ESPN team IDs ──────────────────────────────────────────────────────────────
_ESPN_TEAM_IDS: dict[str, int] = {
    "ATL": 1,  "BOS": 2,  "BKN": 17, "CHA": 30, "CHI": 4,
    "CLE": 5,  "DAL": 6,  "DEN": 7,  "DET": 8,  "GSW": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
    "MIA": 14, "MIL": 15, "MIN": 16, "NOP": 3,  "NYK": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
    "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
}


@dataclass
class NBAPlayer:
    player_id:  int
    full_name:  str
    team_id:    int
    jersey:     str
    source:     str = "espn"


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


_player_cache:   dict[str, NBAPlayer] = {}
_roster_loaded:  set[str] = set()
_roster_ts:      float = 0.0
_ROSTER_TTL      = 21600

_stats_cache:    dict[int, PlayerStats] = {}
_rolling_cache:  dict[int, RollingStats] = {}
_stats_ts:       float = 0.0
_STATS_TTL       = 3600


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


def _espn_get(url: str, timeout: int = 15) -> Optional[dict]:
    req = Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "KalshiFlywheel/1.0")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        logger.debug("[nba_stats] ESPN %d: %s", e.code, url[:100])
        return None
    except Exception as e:
        logger.debug("[nba_stats] ESPN request failed: %s → %s", url[:80], e)
        return None


def _bdl_get(endpoint: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
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
        logger.debug("[nba_stats] BDL %d: %s", e.code, endpoint)
        return None
    except Exception as e:
        logger.debug("[nba_stats] BDL failed: %s → %s", endpoint, e)
        return None


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def _get_current_season() -> int:
    from datetime import datetime
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def _parse_minutes(min_str) -> float:
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


def _espn_load_team_roster(team_code: str) -> list[NBAPlayer]:
    espn_id = _ESPN_TEAM_IDS.get(team_code)
    if not espn_id:
        return []
    url = f"{_ESPN_ROSTER_BASE}/teams/{espn_id}/roster"
    data = _espn_get(url)
    if not data:
        return []
    players: list[NBAPlayer] = []
    athletes = data.get("athletes", [])
    for entry in athletes:
        if isinstance(entry, dict) and "items" in entry:
            items = entry["items"]
        elif isinstance(entry, dict) and "id" in entry:
            items = [entry]
        else:
            continue
        for p in items:
            try:
                pid = int(p.get("id", 0))
                name = (p.get("fullName", "") or p.get("displayName", "") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip())
                jersey = str(p.get("jersey", "")).lstrip("0") or "0"
                if pid and name:
                    players.append(NBAPlayer(player_id=pid, full_name=name, team_id=espn_id, jersey=jersey, source="espn"))
            except (ValueError, TypeError):
                continue
    return players


def _espn_ensure_team_loaded(team_code: str):
    _check_roster_cache()
    if team_code in _roster_loaded:
        return
    players = _espn_load_team_roster(team_code)
    if not players:
        logger.warning("[nba_stats] ESPN roster empty for %s", team_code)
        _roster_loaded.add(team_code)
        return
    for p in players:
        norm_name = _normalize(p.full_name)
        name_parts = norm_name.split()
        if len(name_parts) < 2:
            continue
        first_initial = name_parts[0][0]
        last_name = name_parts[-1]
        _player_cache[f"{first_initial}-{last_name}-{p.jersey}"] = p
        _player_cache.setdefault(f"{first_initial}-{last_name}-{team_code}", p)
        _player_cache[f"FULL-{norm_name}-{team_code}"] = p
    _roster_loaded.add(team_code)
    logger.info("[nba_stats] ESPN roster loaded: %s (%d players)", team_code, len(players))


def lookup_player(team_id: int, first_initial: str, last_name: str, jersey: str, team_code: str = "") -> Optional[NBAPlayer]:
    target_first = first_initial.upper()
    target_last = _normalize(last_name)
    target_jersey = str(jersey).lstrip("0") or "0"
    if team_code:
        _espn_ensure_team_loaded(team_code)
    key_exact = f"{target_first}-{target_last}-{target_jersey}"
    if key_exact in _player_cache:
        return _player_cache[key_exact]
    if team_code:
        key_team = f"{target_first}-{target_last}-{team_code}"
        if key_team in _player_cache:
            p = _player_cache[key_team]
            _player_cache[key_exact] = p
            return p
    if team_code:
        espn_tid = _ESPN_TEAM_IDS.get(team_code, 0)
        for ck, p in _player_cache.items():
            if p.team_id != espn_tid:
                continue
            norm = _normalize(p.full_name)
            parts = norm.split()
            if len(parts) < 2:
                continue
            if parts[0][0] == target_first and parts[-1] == target_last:
                _player_cache[key_exact] = p
                return p
    bdl_player = _bdl_lookup_player(team_id, target_first, target_last, target_jersey, team_code)
    if bdl_player:
        _player_cache[key_exact] = bdl_player
    return bdl_player


def _bdl_lookup_player(team_id: int, first_initial: str, last_name: str, jersey: str, team_code: str) -> Optional[NBAPlayer]:
    data = _bdl_get("players", {"search": last_name, "per_page": 25})
    if not data:
        return None
    players = data.get("data", [])
    target_jersey = str(jersey).lstrip("0") or "0"
    best: Optional[NBAPlayer] = None
    best_score = -1
    for p in players:
        p_first = _normalize(p.get("first_name", ""))
        p_last = _normalize(p.get("last_name", ""))
        p_id = p.get("id")
        p_jersey = str(p.get("jersey_number", "")).lstrip("0") or "0"
        p_team = p.get("team", {}).get("abbreviation", "")
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
            best = NBAPlayer(player_id=int(p_id), full_name=f"{p.get('first_name', '')} {p.get('last_name', '')}", team_id=team_id, jersey=p_jersey, source="bdl")
    return best


def _espn_fetch_overview(player_id: int) -> Optional[PlayerStats]:
    """v9: Fetch stats from /overview endpoint."""
    url = f"{_ESPN_STATS_BASE}/athletes/{player_id}/overview"
    data = _espn_get(url)
    if not data:
        return None
    stats_obj = data.get("statistics")
    if not stats_obj:
        logger.debug("[nba_stats] /overview: no 'statistics' for player %d", player_id)
        return None
    labels = stats_obj.get("labels", [])
    splits = stats_obj.get("splits", [])
    if not labels or not splits:
        logger.debug("[nba_stats] /overview: missing labels/splits for player %d", player_id)
        return None
    stat_values = splits[0].get("stats", [])
    if not stat_values:
        logger.debug("[nba_stats] /overview: no stats in splits[0] for player %d", player_id)
        return None
    s: dict[str, float] = {}
    for i, label in enumerate(labels):
        if i >= len(stat_values):
            break
        key = label.upper().replace("%", "PCT").replace(" ", "")
        try:
            s[key] = float(stat_values[i])
        except (ValueError, TypeError):
            s[key] = 0.0
    gp = int(s.get("GP", 0))
    if gp == 0:
        logger.debug("[nba_stats] /overview: GP=0 for player %d", player_id)
        return None
    pts = s.get("PTS", 0.0)
    reb = s.get("REB", 0.0)
    ast = s.get("AST", 0.0)
    stl = s.get("STL", 0.0)
    blk = s.get("BLK", 0.0)
    mins = s.get("MIN", 0.0)
    to = s.get("TO", 0.0)
    three_pct = s.get("3PCT", s.get("3PPCT", 0.0)) / 100.0
    logger.info("[nba_stats] /overview player %d: %dG %.1fppg %.1frpg %.1fapg", player_id, gp, pts, reb, ast)
    return PlayerStats(player_id=player_id, games_played=gp, minutes_pg=mins, points_pg=pts, rebounds_pg=reb, assists_pg=ast, steals_pg=stl, blocks_pg=blk, three_pm_pg=0.0, three_pa_pg=0.0, three_pct=three_pct, fga_pg=0.0, fta_pg=0.0, turnovers_pg=to, usage_rate=0.0)


def _bdl_fetch_stats(player_id: int) -> Optional[PlayerStats]:
    season = _get_current_season()
    data = _bdl_get("season_averages", {"season": season, "player_ids": [player_id]})
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
    return PlayerStats(player_id=player_id, games_played=gp, minutes_pg=minutes, points_pg=float(avg.get("pts", 0)), rebounds_pg=float(avg.get("reb", 0)), assists_pg=float(avg.get("ast", 0)), steals_pg=float(avg.get("stl", 0)), blocks_pg=float(avg.get("blk", 0)), three_pm_pg=three_pm, three_pa_pg=three_pa, three_pct=three_pm / max(three_pa, 0.01), fga_pg=float(avg.get("fga", 0)), fta_pg=float(avg.get("fta", 0)), turnovers_pg=float(avg.get("turnover", 0)), usage_rate=0.0)


_espn_to_bdl: dict[int, Optional[int]] = {}


def _bridge_espn_to_bdl(espn_player_id: int) -> Optional[int]:
    if espn_player_id in _espn_to_bdl:
        return _espn_to_bdl[espn_player_id]
    if not _API_KEY:
        _espn_to_bdl[espn_player_id] = None
        return None
    player_name = None
    for p in _player_cache.values():
        if p.player_id == espn_player_id:
            player_name = p.full_name
            break
    if not player_name:
        _espn_to_bdl[espn_player_id] = None
        return None
    name_parts = player_name.split()
    if len(name_parts) < 2:
        _espn_to_bdl[espn_player_id] = None
        return None
    last_name = name_parts[-1]
    first_initial = name_parts[0][0].upper()
    data = _bdl_get("players", {"search": last_name, "per_page": 25})
    if not data:
        _espn_to_bdl[espn_player_id] = None
        return None
    for p in data.get("data", []):
        p_first = _normalize(p.get("first_name", ""))
        p_last = _normalize(p.get("last_name", ""))
        if p_first and p_first[0] == first_initial and p_last == _normalize(last_name):
            bdl_id = int(p.get("id", 0))
            if bdl_id:
                _espn_to_bdl[espn_player_id] = bdl_id
                return bdl_id
    _espn_to_bdl[espn_player_id] = None
    return None


def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    _check_stats_cache()
    if player_id in _stats_cache:
        return _stats_cache[player_id]
    source = "espn"
    for p in _player_cache.values():
        if p.player_id == player_id:
            source = p.source
            break
    stats: Optional[PlayerStats] = None
    if source == "espn":
        stats = _espn_fetch_overview(player_id)
        if stats is None and _API_KEY:
            bdl_id = _bridge_espn_to_bdl(player_id)
            if bdl_id:
                stats = _bdl_fetch_stats(bdl_id)
    else:
        stats = _bdl_fetch_stats(player_id)
    if stats:
        _stats_cache[player_id] = stats
    return stats


def _espn_fetch_gamelog(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    url = f"{_ESPN_STATS_BASE}/athletes/{player_id}/gamelog"
    data = _espn_get(url)
    if not data:
        return None
    games: list[dict[str, float]] = []
    for st in data.get("seasonTypes", []):
        for cat in st.get("categories", []):
            labels = cat.get("labels", [])
            if not labels:
                continue
            label_map = [l.upper().replace("%", "PCT") for l in labels]
            for event in cat.get("events", []):
                raw_stats = event.get("stats", [])
                game: dict[str, float] = {}
                for i, label in enumerate(label_map):
                    if i >= len(raw_stats):
                        break
                    val = raw_stats[i]
                    if isinstance(val, str) and "-" in val and not val.startswith("-"):
                        try:
                            game[label] = float(val.split("-")[0])
                        except (ValueError, IndexError):
                            pass
                    else:
                        try:
                            game[label] = float(val) if val else 0.0
                        except (ValueError, TypeError):
                            pass
                if game:
                    games.append(game)
    if not games:
        return None
    recent = games[:n_games]
    n = len(recent)
    pts = [g.get("PTS", 0.0) for g in recent]
    reb = [g.get("REB", 0.0) for g in recent]
    ast = [g.get("AST", 0.0) for g in recent]
    tpm = [g.get("3PT", g.get("3PM", g.get("FG3", 0.0))) for g in recent]
    stl = [g.get("STL", 0.0) for g in recent]
    blk = [g.get("BLK", 0.0) for g in recent]
    mins = [g.get("MIN", 0.0) for g in recent]
    return RollingStats(n_games=n, points_pg=sum(pts)/n, rebounds_pg=sum(reb)/n, assists_pg=sum(ast)/n, three_pm_pg=sum(tpm)/n, steals_pg=sum(stl)/n, blocks_pg=sum(blk)/n, minutes_pg=sum(mins)/n)


def _bdl_fetch_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    season = _get_current_season()
    data = _bdl_get("stats", {"player_ids": [player_id], "seasons": [season], "per_page": n_games, "sort": "-game.date"})
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
    return RollingStats(n_games=n, points_pg=sum(pts)/n, rebounds_pg=sum(reb)/n, assists_pg=sum(ast)/n, three_pm_pg=sum(tpm)/n, steals_pg=sum(stl)/n, blocks_pg=sum(blk)/n, minutes_pg=sum(mins)/n)


def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    _check_stats_cache()
    if player_id in _rolling_cache:
        return _rolling_cache[player_id]
    source = "espn"
    for p in _player_cache.values():
        if p.player_id == player_id:
            source = p.source
            break
    rolling: Optional[RollingStats] = None
    if source == "espn":
        rolling = _espn_fetch_gamelog(player_id, n_games)
        if rolling is None and _API_KEY:
            bdl_id = _bridge_espn_to_bdl(player_id)
            if bdl_id:
                rolling = _bdl_fetch_rolling(bdl_id, n_games)
    else:
        rolling = _bdl_fetch_rolling(player_id, n_games)
    if rolling:
        _rolling_cache[player_id] = rolling
    return rolling
