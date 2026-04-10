"""
shared/kalshi_ticker_parser.py

Parses Kalshi MLB player prop ticker strings into structured data that can be
fed to the MLB Stats API and a Poisson hit-rate model.

Ticker format (observed from live Kalshi data):

    KX<LEAGUE><PROP>-<YYMMMDD><HHMM><AWAY><HOME>-<TEAM><PLAYER>-<THRESHOLD>

Examples:
    KXMLBHIT-26APR092140COLSD-SDFTATIS23-1
    KXMLBTB-26APR092140COLSD-SDMMACHADO13-5
    KXMLBHR-26APR092140COLSD-SDJMERRILL3-2

Decoded:
    league       = MLB
    prop         = HIT (or TB, HR, HRR)
    game_date    = 2026-04-09
    game_time    = 21:40 (local / ET)
    away_team    = COL (Colorado Rockies)
    home_team    = SD  (San Diego Padres)
    player_team  = SD  (which side the player bats for)
    player_code  = FTATIS23 → first=F, last=TATIS, jersey=23
    threshold    = 1 (≥1 hit)

Supported prop types (v1):
    HIT  — hits (at-bat outcome)
    TB   — total bases (1B=1, 2B=2, 3B=3, HR=4)
    HR   — home runs
    HRR  — home run OR run scored (Kalshi composite)

Extension: other leagues (NBA, NHL, NFL) follow similar patterns and can be
added by writing analogous parsers next to parse_mlb_ticker().
"""

import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, date, time
from typing import Optional

logger = logging.getLogger(__name__)


# ── MLB team abbreviation map ─────────────────────────────────────────────────
# Maps Kalshi's 2-3 letter codes to (team_id, full_name) for MLB Stats API.
# MLB Stats API team IDs are stable; full names used for display only.

MLB_TEAMS = {
    "ARI": (109, "Arizona Diamondbacks"),
    "AZ":  (109, "Arizona Diamondbacks"),  # Kalshi uses AZ, not ARI
    "ATL": (144, "Atlanta Braves"),
    "BAL": (110, "Baltimore Orioles"),
    "BOS": (111, "Boston Red Sox"),
    "CHC": (112, "Chicago Cubs"),
    "CWS": (145, "Chicago White Sox"),
    "CHW": (145, "Chicago White Sox"),  # alt
    "CIN": (113, "Cincinnati Reds"),
    "CLE": (114, "Cleveland Guardians"),
    "COL": (115, "Colorado Rockies"),
    "DET": (116, "Detroit Tigers"),
    "HOU": (117, "Houston Astros"),
    "KC":  (118, "Kansas City Royals"),
    "KCR": (118, "Kansas City Royals"),  # alt
    "LAA": (108, "Los Angeles Angels"),
    "LAD": (119, "Los Angeles Dodgers"),
    "MIA": (146, "Miami Marlins"),
    "MIL": (158, "Milwaukee Brewers"),
    "MIN": (142, "Minnesota Twins"),
    "NYM": (121, "New York Mets"),
    "NYY": (147, "New York Yankees"),
    "OAK": (133, "Oakland Athletics"),
    "ATH": (133, "Athletics"),           # 2025+ Sacramento rebrand
    "PHI": (143, "Philadelphia Phillies"),
    "PIT": (134, "Pittsburgh Pirates"),
    "SD":  (135, "San Diego Padres"),
    "SDP": (135, "San Diego Padres"),    # alt
    "SF":  (137, "San Francisco Giants"),
    "SFG": (137, "San Francisco Giants"),  # alt
    "SEA": (136, "Seattle Mariners"),
    "STL": (138, "St. Louis Cardinals"),
    "TB":  (139, "Tampa Bay Rays"),
    "TBR": (139, "Tampa Bay Rays"),      # alt
    "TEX": (140, "Texas Rangers"),
    "TOR": (141, "Toronto Blue Jays"),
    "WSH": (120, "Washington Nationals"),
    "WAS": (120, "Washington Nationals"),  # alt
}

# Teams sorted longest-first so we match "CWS" before "CW"
_TEAM_CODES_SORTED = sorted(MLB_TEAMS.keys(), key=len, reverse=True)


# ── Prop type map ─────────────────────────────────────────────────────────────
# Maps Kalshi prop suffix → canonical prop name used by the stats fetcher.

MLB_PROPS = {
    "HIT":  "hits",
    "TB":   "total_bases",
    "HR":   "home_runs",
    "HRR":  "hr_or_run",       # Kalshi composite: HR scored or run scored
    "RBI":  "rbis",
    "KS":   "strikeouts_pitcher",  # pitcher Ks
    "SO":   "strikeouts_batter",   # batter Ks
    "SB":   "stolen_bases",
    "R":    "runs_scored",
    "BB":   "walks",
}

# Non-player-prop MLB market prefixes — parser intentionally returns None for
# these. Includes game-level markets (totals, spreads, F5), season-long futures
# (season HR totals, ROTY), and entertainment markets (broadcast mentions).
MLB_NON_PLAYER_PROP_PREFIXES = (
    # ── Game-level markets ──────────────────────────────────────────
    "KXMLBTOTAL",      # game total runs (over/under)
    "KXMLBTEAMTOTAL",  # team total runs (over/under, per-team)
    "KXMLBSPREAD",     # game run line
    "KXMLBF5TOTAL",    # first 5 innings total
    "KXMLBF5SPREAD",   # first 5 innings spread
    "KXMLBF5",         # first 5 innings moneyline
    "KXMLBML",         # game moneyline
    "KXMLBGAME",       # generic game outcome
    "KXMLBRFI",        # run in 1st inning (game-level)
    # ── Season-long futures ─────────────────────────────────────────
    "KXMLBSEASONHR",   # player season home run totals
    "KXMLBSEASONH",    # (catch-all for other season player props)
    "KXMLBNLROTY",     # NL Rookie of the Year
    "KXMLBALROTY",     # AL Rookie of the Year
    "KXMLBMVP",        # MVP futures
    "KXMLBCY",         # Cy Young futures
    "KXMLBWS",         # World Series winner
    # ── Entertainment / meta markets ────────────────────────────────
    "KXMLBMENTION",    # broadcast mention markets
)

# Legacy alias for backward compat
MLB_GAME_LEVEL_PREFIXES = MLB_NON_PLAYER_PROP_PREFIXES


def is_mlb_non_player_prop_market(ticker: str) -> bool:
    """True if this is a non-player-prop MLB market (game-level, futures, meta).

    Player-prop models should skip these cleanly rather than report parse failures.
    """
    t = ticker.strip().upper()
    return any(t.startswith(p + "-") for p in MLB_NON_PLAYER_PROP_PREFIXES)


# Legacy alias
def is_mlb_game_level_market(ticker: str) -> bool:
    """Legacy alias for is_mlb_non_player_prop_market."""
    return is_mlb_non_player_prop_market(ticker)

# Sorted longest-first so "HRR" matches before "HR" before "R"
_PROP_CODES_SORTED = sorted(MLB_PROPS.keys(), key=len, reverse=True)


@dataclass
class MLBTicker:
    """Parsed representation of an MLB player prop ticker."""

    raw_ticker:       str
    league:           str           # "MLB"
    prop_code:        str           # "HIT", "TB", "HR", ...
    prop_name:        str           # "hits", "total_bases", ...
    game_date:        date
    game_time:        Optional[time]
    away_team_code:   str
    home_team_code:   str
    away_team_id:     Optional[int]
    home_team_id:     Optional[int]
    away_team_name:   str
    home_team_name:   str
    player_team_code: str
    player_first:     str           # single letter
    player_last:      str           # uppercase last name
    player_jersey:    Optional[int]
    threshold:        int           # ≥N outcome

    def as_dict(self) -> dict:
        d = asdict(self)
        d["game_date"] = self.game_date.isoformat()
        d["game_time"] = self.game_time.isoformat() if self.game_time else None
        return d

    @property
    def player_display(self) -> str:
        """Best-effort display name: 'F. Tatis #23'"""
        last = self.player_last.title()
        jersey = f" #{self.player_jersey}" if self.player_jersey is not None else ""
        return f"{self.player_first}. {last}{jersey}"


# ── Main parser ───────────────────────────────────────────────────────────────

# Matches: KXMLB<PROP>-<DATE><TIME?><AWAY><HOME>-<TEAM><PLAYER>-<THRESHOLD>
# Examples:
#   KXMLBHIT-26APR092140COLSD-SDFTATIS23-1
#   KXMLBTB-26APR092140COLSD-SDJMERRILL3-5
_MLB_TICKER_RE = re.compile(
    r"""
    ^KXMLB(?P<prop>[A-Z]+?)         # prop suffix (HIT/TB/HR/HRR/etc.)
    -
    (?P<yy>\d{2})                   # year (last 2 digits)
    (?P<mon>[A-Z]{3})               # month (3-letter uppercase)
    (?P<day>\d{2})                  # day
    (?P<hhmm>\d{4})?                # optional game time HHMM
    (?P<matchup>[A-Z]+)             # AWAYHOME concatenated
    -
    (?P<player_blob>[A-Z]+\d*)      # team + first + last + jersey (split in code)
    -
    (?P<threshold>\d+)              # ≥N threshold
    $
    """,
    re.VERBOSE,
)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _split_team_codes(matchup: str) -> tuple[str, str]:
    """Split a concatenated matchup string (e.g. 'COLSD') into (away, home).

    We greedy-match the *longest* valid team code at the start, then verify
    the remainder is also a valid team code. If the first split fails, we
    try the second-longest, etc.
    """
    for i, away in enumerate(_TEAM_CODES_SORTED):
        if not matchup.startswith(away):
            continue
        rest = matchup[len(away):]
        if rest in MLB_TEAMS:
            return away, rest
    raise ValueError(f"cannot split matchup into valid teams: {matchup}")


def _split_prop_code(prop_and_suffix: str) -> str:
    """Greedy-match longest valid prop code at the start."""
    for code in _PROP_CODES_SORTED:
        if prop_and_suffix.startswith(code):
            return code
    raise ValueError(f"unknown prop code: {prop_and_suffix}")


def _split_player_code(code: str) -> tuple[str, str, Optional[int]]:
    """Parse 'FTATIS23' → ('F', 'TATIS', 23), 'JMERRILL3' → ('J', 'MERRILL', 3).

    Structure: <FIRST_INITIAL><LAST_NAME_UPPERCASE><JERSEY_DIGITS>
    """
    if len(code) < 2:
        raise ValueError(f"player code too short: {code}")

    # Jersey number: trailing digits
    m = re.match(r"^([A-Z])([A-Z]+)(\d+)?$", code)
    if not m:
        raise ValueError(f"cannot parse player code: {code}")

    first_initial = m.group(1)
    last_name     = m.group(2)
    jersey        = int(m.group(3)) if m.group(3) else None
    return first_initial, last_name, jersey


def _split_player_blob(blob: str) -> tuple[str, str, str, Optional[int]]:
    """Split 'SDFTATIS23' → ('SD', 'F', 'TATIS', 23).

    Greedy-matches the longest valid team code at the start, then parses
    the remainder as the player code. Must be called AFTER the regex has
    validated the overall ticker shape.
    """
    for team_code in _TEAM_CODES_SORTED:
        if blob.startswith(team_code):
            remainder = blob[len(team_code):]
            if not remainder:
                continue
            try:
                first, last, jersey = _split_player_code(remainder)
                return team_code, first, last, jersey
            except ValueError:
                continue
    raise ValueError(f"cannot split player blob: {blob}")


def parse_mlb_ticker(ticker: str) -> Optional[MLBTicker]:
    """Parse a Kalshi MLB player prop ticker into a structured MLBTicker.

    Returns None if the ticker does not match the expected MLB player prop
    format. Raises no exceptions — logs a debug message and returns None.
    """
    ticker = ticker.strip().upper()

    match = _MLB_TICKER_RE.match(ticker)
    if not match:
        logger.debug("ticker does not match MLB player prop format: %s", ticker)
        return None

    try:
        prop_raw  = match.group("prop")
        prop_code = _split_prop_code(prop_raw)
        prop_name = MLB_PROPS[prop_code]

        yy  = int(match.group("yy"))
        mon = _MONTHS[match.group("mon")]
        day = int(match.group("day"))
        game_date = date(2000 + yy, mon, day)

        game_time: Optional[time] = None
        if match.group("hhmm"):
            hh = int(match.group("hhmm")[:2])
            mm = int(match.group("hhmm")[2:])
            try:
                game_time = time(hh, mm)
            except ValueError:
                # Sometimes the 4 digits after date are not a time but
                # an extension of the matchup — fall through.
                pass

        matchup = match.group("matchup")
        away_code, home_code = _split_team_codes(matchup)

        player_blob = match.group("player_blob")
        player_team, first, last, jersey = _split_player_blob(player_blob)

        if player_team not in MLB_TEAMS:
            raise ValueError(f"invalid player team code: {player_team}")

        threshold = int(match.group("threshold"))

        away_id, away_name = MLB_TEAMS[away_code]
        home_id, home_name = MLB_TEAMS[home_code]

        return MLBTicker(
            raw_ticker       = ticker,
            league           = "MLB",
            prop_code        = prop_code,
            prop_name        = prop_name,
            game_date        = game_date,
            game_time        = game_time,
            away_team_code   = away_code,
            home_team_code   = home_code,
            away_team_id     = away_id,
            home_team_id     = home_id,
            away_team_name   = away_name,
            home_team_name   = home_name,
            player_team_code = player_team,
            player_first     = first,
            player_last      = last,
            player_jersey    = jersey,
            threshold        = threshold,
        )

    except (KeyError, ValueError) as e:
        logger.debug("parse failed for %s: %s", ticker, e)
        return None


# ── Self-test when run directly ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Known examples from the Railway logs
    SAMPLES = [
        "KXMLBHIT-26APR092140COLSD-SDFTATIS23-1",
        "KXMLBHIT-26APR092140COLSD-SDMMACHADO13-2",
        "KXMLBHIT-26APR092140COLSD-SDJMERRILL3-3",
        "KXMLBTB-26APR092140COLSD-SDMMACHADO13-5",
        "KXMLBTB-26APR092140COLSD-SDJMERRILL3-4",
        "KXMLBHR-26APR092140COLSD-SDFTATIS23-1",
        "KXMLBHR-26APR092140COLSD-SDJMERRILL3-2",
        "KXMLBHRR-26APR092140COLSD-SDMMACHADO13-1",
    ]

    print("=" * 78)
    print("Kalshi MLB ticker parser — self-test")
    print("=" * 78)

    n_pass = 0
    for t in SAMPLES:
        result = parse_mlb_ticker(t)
        if result:
            n_pass += 1
            print(f"\n✓ {t}")
            print(f"  → {result.player_display} ({result.player_team_code}) "
                  f"| {result.prop_name} ≥ {result.threshold}")
            print(f"  → {result.away_team_name} @ {result.home_team_name} "
                  f"on {result.game_date}"
                  f"{(' ' + result.game_time.isoformat()) if result.game_time else ''}")
        else:
            print(f"\n✗ {t}  — FAILED TO PARSE")

    print(f"\n{'=' * 78}")
    print(f"Result: {n_pass}/{len(SAMPLES)} parsed successfully")
    print("=" * 78)

    sys.exit(0 if n_pass == len(SAMPLES) else 1)