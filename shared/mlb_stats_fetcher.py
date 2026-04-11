"""
shared/mlb_stats_fetcher.py

Fetches player-level stats from MLB Stats API (free, no auth required) for
feeding into the hit-rate Poisson model.

Given a parsed MLBTicker, this module provides:

  1. lookup_player(team_code, first_initial, last_name, jersey)
       → MLB Stats API player ID + canonical name

  2. fetch_batter_stats(player_id)
       → season batting stats (avg, obp, slg, PA, games)

  3. fetch_batter_gamelog(player_id, n_games=10)
       → last N games hit/AB/HR totals for rolling average

  4. fetch_pitcher_stats(player_id)
       → season pitching stats (k/9, ip, era)

  5. fetch_probable_pitcher(game_date, home_team_id, away_team_id, against_team_id)
       → opposing pitcher ID and hand (L/R)

  6. PARK_FACTORS
       → dict mapping venue_id → multiplicative run/HR factors

Player matching handles Kalshi's diacritic stripping:
  Kalshi:   JRAMREZ11  →  first="J", last="RAMREZ", jersey=11
  MLB API:  "José Ramírez", jerseyNumber="11"
  Match:    normalize("RAMIREZ") == normalize("RAMREZ")?  no — strip 'I' too.

  Strategy: normalize both sides to uppercase-ASCII-only, then also drop
  vowels from Kalshi's version since diacritics sometimes collapse letters.
  Fall back to jersey number as a tiebreaker.

MLB Stats API docs:
  https://statsapi.mlb.com/docs/
  https://github.com/toddrob99/MLB-StatsAPI/wiki

All endpoints are free and require no API key.
"""

import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── HTTP cache (copies the pattern from shared/data_fetchers.py) ──────────────

_cache: dict = {}

def _get(url: str, params: Optional[dict] = None, ttl: int = 300) -> Optional[dict]:
    key = url + str(sorted((params or {}).items()))
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < ttl):
        return entry["data"]
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        _cache[key] = {"ts": time.time(), "data": data}
        return data
    except Exception as e:
        logger.warning("MLB API fetch failed %s: %s", url, e)
        return entry["data"] if entry else None


BASE = "https://statsapi.mlb.com/api/v1"


# ── Name normalization for player matching ───────────────────────────────────

def _normalize_name(s: str) -> str:
    """Strip diacritics and punctuation, uppercase, remove non-letters.

    'José Ramírez Jr.' → 'JOSERAMIREZJR'
    """
    # Decompose unicode, drop combining marks
    nfd = unicodedata.normalize("NFD", s)
    ascii_only = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Keep only letters, uppercase
    return "".join(c for c in ascii_only.upper() if c.isalpha())


def _kalshi_last_name_matches(kalshi_last: str, mlb_last: str) -> bool:
    """Compare Kalshi's stripped-diacritic last name to an MLB full last name.

    Kalshi strips diacritics AND sometimes drops the letter that had the
    accent (RAMÍREZ → RAMREZ, RODRÍGUEZ → RODRGUEZ). We handle both cases.
    """
    k = _normalize_name(kalshi_last)
    m = _normalize_name(mlb_last)

    if k == m:
        return True

    # Kalshi version is often a subsequence of the MLB version (missing
    # the accented vowel). Check: is k a subsequence of m with at most
    # 1 letter missing?
    if len(m) - len(k) == 1:
        # Try dropping each position in m and see if it matches k
        for i in range(len(m)):
            if m[:i] + m[i+1:] == k:
                return True

    return False


# ── Team roster + player lookup ──────────────────────────────────────────────

def fetch_team_roster(team_id: int, season: Optional[int] = None) -> list[dict]:
    """Active roster for a team. Cached 24h — rosters don't change often."""
    if season is None:
        season = date.today().year
    data = _get(
        f"{BASE}/teams/{team_id}/roster",
        {"rosterType": "active", "season": season},
        ttl=86400,
    )
    if not data:
        return []
    return data.get("roster", [])


@dataclass
class PlayerLookupResult:
    player_id:    int
    full_name:    str
    jersey:       Optional[int]
    position:     str
    match_reason: str   # "exact" | "subseq" | "jersey_only"


def lookup_player(
    team_id:       int,
    first_initial: str,
    kalshi_last:   str,
    jersey:        Optional[int] = None,
) -> Optional[PlayerLookupResult]:
    """Find a player on a team's roster by (first_initial, last_name, jersey).

    Returns None if no match.
    """
    roster = fetch_team_roster(team_id)
    if not roster:
        logger.debug("empty roster for team_id=%d", team_id)
        return None

    first_initial = first_initial.upper()
    kalshi_last_norm = _normalize_name(kalshi_last)

    # Pass 1: name match + jersey tiebreaker
    name_matches: list[PlayerLookupResult] = []
    for entry in roster:
        person    = entry.get("person", {})
        full_name = person.get("fullName", "")
        pid       = person.get("id")
        pos       = entry.get("position", {}).get("abbreviation", "")
        jersey_str = entry.get("jerseyNumber", "")
        try:
            j = int(jersey_str) if jersey_str else None
        except ValueError:
            j = None

        if not pid or not full_name:
            continue

        # Split "José Ramírez Jr." into first/last parts
        parts = full_name.split()
        if len(parts) < 2:
            continue
        mlb_first = parts[0]
        mlb_last  = " ".join(parts[1:])  # handles "De La Cruz", "Jr.", etc.

        # First initial must match (after diacritic strip)
        if _normalize_name(mlb_first)[:1] != first_initial:
            continue

        # Last name must match (exact or subseq)
        if not _kalshi_last_name_matches(kalshi_last, mlb_last):
            continue

        reason = "exact" if _normalize_name(mlb_last) == kalshi_last_norm else "subseq"
        name_matches.append(PlayerLookupResult(
            player_id    = pid,
            full_name    = full_name,
            jersey       = j,
            position     = pos,
            match_reason = reason,
        ))

    if len(name_matches) == 1:
        return name_matches[0]

    if len(name_matches) > 1 and jersey is not None:
        # Disambiguate by jersey number
        for m in name_matches:
            if m.jersey == jersey:
                return m
        # No jersey match — return first one with a warning
        logger.warning(
            "ambiguous match for %s. %s on team %d: %d candidates, none with jersey %d",
            first_initial, kalshi_last, team_id, len(name_matches), jersey,
        )
        return name_matches[0]

    if len(name_matches) >= 1:
        return name_matches[0]

    # Pass 2: jersey-only fallback (for cases where name collapsed heavily)
    if jersey is not None:
        for entry in roster:
            jersey_str = entry.get("jerseyNumber", "")
            try:
                j = int(jersey_str) if jersey_str else None
            except ValueError:
                j = None
            if j == jersey:
                person = entry.get("person", {})
                return PlayerLookupResult(
                    player_id    = person.get("id"),
                    full_name    = person.get("fullName", ""),
                    jersey       = j,
                    position     = entry.get("position", {}).get("abbreviation", ""),
                    match_reason = "jersey_only",
                )

    return None


# ── Player stats ─────────────────────────────────────────────────────────────

@dataclass
class BatterSeasonStats:
    player_id:    int
    games:        int
    plate_apps:   int
    at_bats:      int
    hits:         int
    doubles:      int
    triples:      int
    home_runs:    int
    walks:        int
    strikeouts:   int
    avg:          float
    obp:          float
    slg:          float
    ops:          float


def fetch_batter_stats(player_id: int, season: Optional[int] = None) -> Optional[BatterSeasonStats]:
    """Season-to-date hitting stats for a batter."""
    if season is None:
        season = date.today().year
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "season", "group": "hitting", "season": season},
        ttl=3600,
    )
    if not data:
        return None
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return None
        stat = splits[0]["stat"]
        return BatterSeasonStats(
            player_id  = player_id,
            games      = int(stat.get("gamesPlayed", 0)),
            plate_apps = int(stat.get("plateAppearances", 0)),
            at_bats    = int(stat.get("atBats", 0)),
            hits       = int(stat.get("hits", 0)),
            doubles    = int(stat.get("doubles", 0)),
            triples    = int(stat.get("triples", 0)),
            home_runs  = int(stat.get("homeRuns", 0)),
            walks      = int(stat.get("baseOnBalls", 0)),
            strikeouts = int(stat.get("strikeOuts", 0)),
            avg        = float(stat.get("avg") or 0),
            obp        = float(stat.get("obp") or 0),
            slg        = float(stat.get("slg") or 0),
            ops        = float(stat.get("ops") or 0),
        )
    except (KeyError, IndexError, ValueError) as e:
        logger.debug("batter stats parse failed for %d: %s", player_id, e)
        return None


@dataclass
class BatterRollingStats:
    player_id:    int
    n_games:      int
    at_bats:      int
    hits:         int
    home_runs:    int
    total_bases:  int
    avg:          float       # rolling AVG over the window
    iso:          float       # isolated power (SLG - AVG)


def fetch_batter_rolling(player_id: int, n_games: int = 10) -> Optional[BatterRollingStats]:
    """Rolling stats over the last N games."""
    season = date.today().year
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "gameLog", "group": "hitting", "season": season},
        ttl=3600,
    )
    if not data:
        return None
    try:
        splits = data["stats"][0]["splits"]
        recent = splits[-n_games:]  # most recent N games
        if not recent:
            return None

        at_bats = hits = hrs = tb = 0
        for g in recent:
            s = g["stat"]
            ab   = int(s.get("atBats", 0))
            h    = int(s.get("hits", 0))
            hr   = int(s.get("homeRuns", 0))
            d2   = int(s.get("doubles", 0))
            d3   = int(s.get("triples", 0))
            at_bats += ab
            hits    += h
            hrs     += hr
            tb      += (h - d2 - d3 - hr) + 2 * d2 + 3 * d3 + 4 * hr

        avg = hits / at_bats if at_bats > 0 else 0.0
        slg = tb / at_bats if at_bats > 0 else 0.0
        iso = max(0.0, slg - avg)

        return BatterRollingStats(
            player_id   = player_id,
            n_games     = len(recent),
            at_bats     = at_bats,
            hits        = hits,
            home_runs   = hrs,
            total_bases = tb,
            avg         = avg,
            iso         = iso,
        )
    except (KeyError, IndexError, ValueError) as e:
        logger.debug("rolling stats parse failed for %d: %s", player_id, e)
        return None


@dataclass
class PitcherSeasonStats:
    player_id:    int
    games:        int
    innings:      float
    strikeouts:   int
    walks:        int
    hits_allowed: int
    home_runs:    int
    era:          float
    whip:         float
    k_per_9:      float


def fetch_pitcher_stats(player_id: int, season: Optional[int] = None) -> Optional[PitcherSeasonStats]:
    """Season-to-date pitching stats."""
    if season is None:
        season = date.today().year
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
        ttl=3600,
    )
    if not data:
        return None
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return None
        stat = splits[0]["stat"]
        ip = float(stat.get("inningsPitched") or 0)
        ks = int(stat.get("strikeOuts", 0))
        return PitcherSeasonStats(
            player_id    = player_id,
            games        = int(stat.get("gamesPlayed", 0)),
            innings      = ip,
            strikeouts   = ks,
            walks        = int(stat.get("baseOnBalls", 0)),
            hits_allowed = int(stat.get("hits", 0)),
            home_runs    = int(stat.get("homeRuns", 0)),
            era          = float(stat.get("era") or 0),
            whip         = float(stat.get("whip") or 0),
            k_per_9      = (ks * 9 / ip) if ip > 0 else 0.0,
        )
    except (KeyError, IndexError, ValueError) as e:
        logger.debug("pitcher stats parse failed for %d: %s", player_id, e)
        return None


# ── Probable pitcher for a game ──────────────────────────────────────────────

def fetch_probable_pitcher(
    game_date:    date,
    batter_team_id: int,
) -> Optional[dict]:
    """Return {'player_id', 'full_name', 'hand', 'team_id'} for the pitcher
    facing the batter's team on that date. Returns None if game not found or
    probable pitcher not yet posted.
    """
    data = _get(
        f"{BASE}/schedule",
        {"sportId": 1, "date": game_date.isoformat(),
         "hydrate": "probablePitcher"},
        ttl=600,
    )
    if not data:
        return None
    try:
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                teams = game.get("teams", {})
                home_id = teams.get("home", {}).get("team", {}).get("id")
                away_id = teams.get("away", {}).get("team", {}).get("id")
                if batter_team_id not in (home_id, away_id):
                    continue
                # The opposing pitcher is on the other side
                if batter_team_id == home_id:
                    opp_side = teams.get("away", {})
                else:
                    opp_side = teams.get("home", {})
                pp = opp_side.get("probablePitcher")
                if not pp:
                    return None
                pid = pp.get("id")
                if not pid:
                    return None
                # Fetch pitcher hand
                people = _get(f"{BASE}/people/{pid}", {}, ttl=86400)
                hand = "R"
                if people:
                    try:
                        hand = people["people"][0].get("pitchHand", {}).get("code", "R")
                    except (KeyError, IndexError):
                        pass
                return {
                    "player_id": pid,
                    "full_name": pp.get("fullName", ""),
                    "hand":      hand,
                    "team_id":   opp_side.get("team", {}).get("id"),
                }
    except (KeyError, IndexError) as e:
        logger.debug("probable pitcher lookup failed: %s", e)
    return None


# ── Park factors (hitter-friendly = >1.0, pitcher-friendly = <1.0) ───────────

# Approximate 3-year rolling park factors from Baseball Savant / FanGraphs.
# Keyed by MLB Stats API team_id (home team). These are for RUNS; HR factors
# differ slightly but this is close enough for the first-cut model.
PARK_FACTORS = {
    109: 1.05,  # Arizona (Chase Field)
    144: 0.97,  # Atlanta (Truist Park)
    110: 1.08,  # Baltimore (Camden Yards)
    111: 1.07,  # Boston (Fenway)
    112: 1.03,  # Chicago Cubs (Wrigley)
    145: 1.02,  # Chicago White Sox (Guaranteed Rate)
    113: 1.09,  # Cincinnati (GABP)
    114: 1.00,  # Cleveland (Progressive)
    115: 1.14,  # Colorado (Coors) — extreme hitter's park
    116: 0.97,  # Detroit (Comerica)
    117: 0.99,  # Houston (Minute Maid)
    118: 1.04,  # KC (Kauffman)
    108: 0.96,  # LAA (Angel Stadium)
    119: 0.97,  # LAD (Dodger Stadium)
    146: 0.95,  # Miami (loanDepot)
    158: 1.02,  # Milwaukee (American Family)
    142: 0.99,  # Minnesota (Target Field)
    121: 0.92,  # NYM (Citi Field)
    147: 1.01,  # NYY (Yankee Stadium)
    133: 0.92,  # OAK/ATH (Coliseum / Sutter Health)
    143: 1.00,  # Philly (Citizens Bank)
    134: 0.95,  # Pittsburgh (PNC)
    135: 0.93,  # SD (Petco) — pitcher's park
    137: 0.94,  # SF (Oracle) — pitcher's park
    136: 0.94,  # Seattle (T-Mobile) — pitcher's park
    138: 0.99,  # StL (Busch)
    139: 0.97,  # TB (Tropicana)
    140: 1.05,  # Texas (Globe Life)
    141: 1.01,  # Toronto (Rogers)
    120: 0.98,  # Washington (Nationals Park)
}

# Add this function to the END of shared/mlb_stats_fetcher.py
# Also add 'fetch_opposing_pitcher' to any __all__ if present.

def fetch_opposing_pitcher(game_date: str, opponent_team_id: int) -> Optional["PitcherSeasonStats"]:
    """Fetch probable pitcher stats for a given team on a given date.

    Uses MLB Stats API schedule endpoint with probablePitcher hydration.
    game_date: "YYYY-MM-DD" format
    opponent_team_id: MLB team ID of the opposing team (the team PITCHING against our batter)

    Returns PitcherSeasonStats for the opposing starter, or None if unavailable.
    """
    data = _get(
        f"{BASE}/schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
            "teamId": opponent_team_id,
        },
        ttl=1800,  # 30 min cache — probable pitchers don't change often
    )
    if not data:
        return None

    try:
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                away = game.get("teams", {}).get("away", {})
                home = game.get("teams", {}).get("home", {})

                # Find which side has the opponent team → get their pitcher
                if away.get("team", {}).get("id") == opponent_team_id:
                    pp = away.get("probablePitcher")
                elif home.get("team", {}).get("id") == opponent_team_id:
                    pp = home.get("probablePitcher")
                else:
                    continue

                if pp and pp.get("id"):
                    pitcher_stats = fetch_pitcher_stats(pp["id"])
                    if pitcher_stats:
                        logger.info(
                            "[mlb_stats] opposing pitcher: %s (K/9=%.1f, %.0fIP)",
                            pp.get("fullName", "?"),
                            pitcher_stats.k_per_9,
                            pitcher_stats.innings,
                        )
                    return pitcher_stats

        return None
    except Exception as e:
        logger.warning("[mlb_stats] probable pitcher lookup failed: %s", e)
        return None

# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 78)
    print("MLB Stats API fetcher — self-test")
    print("=" * 78)

    # Test cases: top players from Hunter's diagnostic. Format:
    # (team_code, first_initial, kalshi_last, jersey, expected_name_substring)
    TESTS = [
        ("LAA", "N", "SCHANUEL",   18, "Schanuel"),
        ("LAA", "J", "ADELL",      37, "Adell"),
        ("LAA", "M", "TROUT",      27, "Trout"),
        ("CLE", "S", "KWAN",        38, "Kwan"),
        ("SEA", "C", "RALEIGH",    29, "Raleigh"),
        ("CLE", "J", "RAMREZ",     11, "Ramírez"),   # diacritic subseq
        ("SEA", "J", "RODRGUEZ",   44, "Rodríguez"), # diacritic subseq
        ("TOR", "G", "SPRINGER",   4,  "Springer"),
        ("TOR", "V", "GUERRERO",   27, "Guerrero"),  # Vlad Jr.
    ]

    from shared.kalshi_ticker_parser import MLB_TEAMS

    passes = 0
    for team_code, first, last, jersey, expected in TESTS:
        team_id, team_name = MLB_TEAMS[team_code]
        result = lookup_player(team_id, first, last, jersey)
        if result and expected.lower() in result.full_name.lower():
            passes += 1
            print(f"✓ {team_code} {first}. {last} #{jersey}")
            print(f"   → id={result.player_id}  {result.full_name}  "
                  f"(match={result.match_reason})")

            # Also try pulling season stats to verify end-to-end
            stats = fetch_batter_stats(result.player_id)
            if stats:
                print(f"   → 2026 so far: {stats.games}G  "
                      f".{int(stats.avg*1000):03d}/.{int(stats.obp*1000):03d}/"
                      f".{int(stats.slg*1000):03d}  "
                      f"{stats.hits}H {stats.home_runs}HR {stats.plate_apps}PA")
        else:
            print(f"✗ {team_code} {first}. {last} #{jersey}")
            if result:
                print(f"   got: {result.full_name} — expected substring '{expected}'")
            else:
                print(f"   no match found")

    print(f"\n{'=' * 78}")
    print(f"Result: {passes}/{len(TESTS)} players matched successfully")
    print("=" * 78)

    sys.exit(0 if passes >= len(TESTS) - 1 else 1)  # allow 1 miss for early-season rosters

    