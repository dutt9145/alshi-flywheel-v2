"""
shared/nba_stats_fetcher.py  (v2 — nba_api + fallback)

Uses the `nba_api` Python package for reliable access to stats.nba.com.
Falls back to direct HTTP if nba_api is not installed.

pip install nba_api --break-system-packages
"""

import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try importing nba_api ──────────────────────────────────────────────────────
_HAS_NBA_API = False
try:
    from nba_api.stats.endpoints import commonteamroster, playerdashboardbygeneralsplits, playergamelog
    _HAS_NBA_API = True
    logger.info("[nba_stats] nba_api package available — using it")
except ImportError:
    logger.warning("[nba_stats] nba_api not installed — falling back to direct HTTP (may fail)")

# ── Cache ──────────────────────────────────────────────────────────────────────
_roster_cache: dict = {}
_player_stats_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600


# ── Data classes ───────────────────────────────────────────────────────────────

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


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_current_season() -> str:
    from datetime import datetime
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[2:]}"


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def _check_cache_freshness():
    global _cache_ts
    now = time.monotonic()
    if now - _cache_ts > _CACHE_TTL:
        _roster_cache.clear()
        _player_stats_cache.clear()
        _cache_ts = now


# ── Roster fetch via nba_api ───────────────────────────────────────────────────

def _fetch_roster_nba_api(team_id: int) -> list[dict]:
    """Fetch roster using nba_api package."""
    _check_cache_freshness()
    if team_id in _roster_cache:
        return _roster_cache[team_id]

    season = _get_current_season()
    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season,
            timeout=30,
        )
        df = roster.get_data_frames()[0]
        rows = df.to_dict('records')
        _roster_cache[team_id] = rows
        return rows
    except Exception as e:
        logger.warning("[nba_stats] nba_api roster failed for team %d: %s", team_id, e)
        return []


def _fetch_roster_http(team_id: int) -> list[dict]:
    """Fetch roster using direct HTTP (fallback)."""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    _check_cache_freshness()
    if team_id in _roster_cache:
        return _roster_cache[team_id]

    season = _get_current_season()
    url = f"https://stats.nba.com/stats/commonteamroster?TeamID={team_id}&Season={season}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Connection": "keep-alive",
    }

    req = Request(url)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hdrs = data["resultSets"][0]["headers"]
        rows_raw = data["resultSets"][0]["rowSet"]
        rows = [dict(zip(hdrs, row)) for row in rows_raw]
        _roster_cache[team_id] = rows
        return rows
    except Exception as e:
        logger.warning("[nba_stats] HTTP roster failed for team %d: %s", team_id, e)
        return []


def _fetch_roster(team_id: int) -> list[dict]:
    if _HAS_NBA_API:
        return _fetch_roster_nba_api(team_id)
    return _fetch_roster_http(team_id)


# ── Player lookup ──────────────────────────────────────────────────────────────

def lookup_player(
    team_id: int,
    first_initial: str,
    last_name: str,
    jersey: str,
) -> Optional[NBAPlayer]:
    roster = _fetch_roster(team_id)
    if not roster:
        return None

    target_last = last_name.upper()
    target_first = first_initial.upper()
    target_jersey = str(jersey).lstrip("0") or "0"

    candidates = []
    for player in roster:
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

        first_match = p_first.startswith(target_first)
        last_exact = (p_last == target_last)
        last_subseq = _is_subsequence(target_last, p_last)
        jersey_match = (p_num == target_jersey)

        if first_match and (last_exact or last_subseq):
            score = 0
            if last_exact: score += 10
            if jersey_match: score += 5
            if last_subseq and not last_exact: score += 3
            candidates.append((score, NBAPlayer(
                player_id=int(p_id), full_name=full_name,
                team_id=team_id, jersey=p_num,
            )))

    if not candidates:
        for player in roster:
            p_num = str(player.get("NUM", "")).strip().lstrip("0") or "0"
            if p_num == target_jersey:
                full_name = player.get("PLAYER", "Unknown")
                p_id = player.get("PLAYER_ID")
                if p_id:
                    return NBAPlayer(player_id=int(p_id), full_name=full_name,
                                     team_id=team_id, jersey=p_num)
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Season stats ───────────────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> Optional[PlayerStats]:
    if player_id in _player_stats_cache:
        return _player_stats_cache[player_id]

    season = _get_current_season()

    if _HAS_NBA_API:
        try:
            dash = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                player_id=player_id,
                season=season,
                measure_type_detailed="Base",
                per_mode_detailed="PerGame",
                season_type_all_star="Regular Season",
                timeout=30,
            )
            df = dash.get_data_frames()[0]
            if df.empty:
                return None
            row = df.iloc[0]
            gp = int(row.get("GP", 0))
            if gp == 0:
                return None
            stats = PlayerStats(
                player_id=player_id, games_played=gp,
                minutes_pg=float(row.get("MIN", 0)),
                points_pg=float(row.get("PTS", 0)),
                rebounds_pg=float(row.get("REB", 0)),
                assists_pg=float(row.get("AST", 0)),
                steals_pg=float(row.get("STL", 0)),
                blocks_pg=float(row.get("BLK", 0)),
                three_pm_pg=float(row.get("FG3M", 0)),
                three_pa_pg=float(row.get("FG3A", 0)),
                three_pct=float(row.get("FG3_PCT", 0)),
                fga_pg=float(row.get("FGA", 0)),
                fta_pg=float(row.get("FTA", 0)),
                turnovers_pg=float(row.get("TOV", 0)),
                usage_rate=0.0,
            )
            _player_stats_cache[player_id] = stats
            return stats
        except Exception as e:
            logger.warning("[nba_stats] nba_api stats failed for player %d: %s", player_id, e)
            return None
    else:
        # Direct HTTP fallback (same structure as v1)
        from urllib.request import Request, urlopen
        url = (f"https://stats.nba.com/stats/playerdashboardbygeneralsplits?"
               f"PlayerID={player_id}&Season={season}&MeasureType=Base"
               f"&PerMode=PerGame&SeasonType=Regular+Season")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
        }
        req = Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hdrs = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]
            if not rows:
                return None
            overall = dict(zip(hdrs, rows[0]))
            gp = int(overall.get("GP", 0))
            if gp == 0:
                return None
            stats = PlayerStats(
                player_id=player_id, games_played=gp,
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
                usage_rate=0.0,
            )
            _player_stats_cache[player_id] = stats
            return stats
        except Exception as e:
            logger.warning("[nba_stats] HTTP stats failed for player %d: %s", player_id, e)
            return None


# ── Rolling stats ──────────────────────────────────────────────────────────────

def fetch_player_rolling(player_id: int, n_games: int = 10) -> Optional[RollingStats]:
    season = _get_current_season()

    if _HAS_NBA_API:
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
                timeout=30,
            )
            df = gl.get_data_frames()[0]
            if df.empty:
                return None
            recent = df.head(n_games)
            n = len(recent)
            return RollingStats(
                n_games=n,
                points_pg=recent["PTS"].mean(),
                rebounds_pg=recent["REB"].mean(),
                assists_pg=recent["AST"].mean(),
                three_pm_pg=recent["FG3M"].mean(),
                steals_pg=recent["STL"].mean(),
                blocks_pg=recent["BLK"].mean(),
                minutes_pg=recent["MIN"].astype(float).mean(),
            )
        except Exception as e:
            logger.warning("[nba_stats] nba_api game log failed for player %d: %s", player_id, e)
            return None
    else:
        from urllib.request import Request, urlopen
        url = (f"https://stats.nba.com/stats/playergamelog?"
               f"PlayerID={player_id}&Season={season}&SeasonType=Regular+Season")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
        }
        req = Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hdrs = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]
            if not rows:
                return None
            recent = [dict(zip(hdrs, row)) for row in rows[:n_games]]
            n = len(recent)
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
        except Exception as e:
            logger.warning("[nba_stats] HTTP game log failed for player %d: %s", player_id, e)
            return None