"""
shared/nba_ticker_parser.py

Parses Kalshi NBA player prop tickers into structured data.
Mirrors shared/kalshi_ticker_parser.py for MLB.

Ticker format (inferred from KXMLB pattern):
  KXNBAPTS-26APR102000BOSCLE-BOSJTATUM0-25
  ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^
  prop     date+time+teams  team+player   threshold

Prop codes:
  KXNBAPTS  → points
  KXNBAREB  → rebounds
  KXNBAAST  → assists
  KXNBA3PT  → three_pointers
  KXNBASTL  → steals
  KXNBABLK  → blocks

Non-player-prop prefixes (game-level markets, skip):
  KXNBASPREAD, KXNBATOTAL, KXNBATEAMTOTAL, KXNBAGAME,
  KXNBA1HSPREAD, KXNBA1HTOTAL, KXNBA1HWINNER,
  KXNBA2HWINNER, KXNBA2D, KXNBA3D, KXNBAMENTION
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── NBA team codes → (nba_api team_id, full_name) ─────────────────────────────
NBA_TEAMS = {
    "ATL": (1610612737, "Atlanta Hawks"),
    "BOS": (1610612738, "Boston Celtics"),
    "BKN": (1610612751, "Brooklyn Nets"),
    "CHA": (1610612766, "Charlotte Hornets"),
    "CHI": (1610612741, "Chicago Bulls"),
    "CLE": (1610612739, "Cleveland Cavaliers"),
    "DAL": (1610612742, "Dallas Mavericks"),
    "DEN": (1610612743, "Denver Nuggets"),
    "DET": (1610612765, "Detroit Pistons"),
    "GSW": (1610612744, "Golden State Warriors"),
    "HOU": (1610612745, "Houston Rockets"),
    "IND": (1610612754, "Indiana Pacers"),
    "LAC": (1610612746, "LA Clippers"),
    "LAL": (1610612747, "Los Angeles Lakers"),
    "MEM": (1610612763, "Memphis Grizzlies"),
    "MIA": (1610612748, "Miami Heat"),
    "MIL": (1610612749, "Milwaukee Bucks"),
    "MIN": (1610612750, "Minnesota Timberwolves"),
    "NOP": (1610612740, "New Orleans Pelicans"),
    "NYK": (1610612752, "New York Knicks"),
    "OKC": (1610612760, "Oklahoma City Thunder"),
    "ORL": (1610612753, "Orlando Magic"),
    "PHI": (1610612755, "Philadelphia 76ers"),
    "PHX": (1610612756, "Phoenix Suns"),
    "POR": (1610612757, "Portland Trail Blazers"),
    "SAC": (1610612758, "Sacramento Kings"),
    "SAS": (1610612759, "San Antonio Spurs"),
    "TOR": (1610612761, "Toronto Raptors"),
    "UTA": (1610612762, "Utah Jazz"),
    "WAS": (1610612764, "Washington Wizards"),
}

# Aliases for team codes that Kalshi might use differently
_TEAM_ALIASES = {
    "GS":  "GSW",
    "NO":  "NOP",
    "NY":  "NYK",
    "SA":  "SAS",
    "PHO": "PHX",
    "UTAH": "UTA",
    "BRK": "BKN",
    "CHA": "CHA",
    "WSH": "WAS",
}

# ── Prop code mapping ──────────────────────────────────────────────────────────
PROP_MAP = {
    "KXNBAPTS": ("PTS", "points"),
    "KXNBAREB": ("REB", "rebounds"),
    "KXNBAAST": ("AST", "assists"),
    "KXNBA3PT": ("3PT", "three_pointers"),
    "KXNBASTL": ("STL", "steals"),
    "KXNBABLK": ("BLK", "blocks"),
}

# ── Non-player-prop prefixes (game-level markets) ──────────────────────────────
NBA_NON_PLAYER_PROP_PREFIXES = (
    "KXNBATOTAL",
    "KXNBATEAMTOTAL",
    "KXNBASPREAD",
    "KXNBAGAME",
    "KXNBA1H",
    "KXNBA2H",
    "KXNBA2D",
    "KXNBA3D",
    "KXNBAMENTION",
)


@dataclass
class NBATicker:
    """Parsed NBA player prop ticker."""
    raw_ticker:       str
    prop_prefix:      str       # e.g. "KXNBAPTS"
    prop_code:        str       # e.g. "PTS"
    prop_name:        str       # e.g. "points"
    game_date:        str       # e.g. "2026-04-10"
    away_team_code:   str       # e.g. "BOS"
    home_team_code:   str       # e.g. "CLE"
    player_team_code: str       # e.g. "BOS"
    player_first:     str       # first initial, e.g. "J"
    player_last:      str       # e.g. "TATUM"
    player_jersey:    str       # e.g. "0"
    threshold:        int       # e.g. 25


def is_nba_non_player_prop_market(ticker: str) -> bool:
    """Return True if this is a game-level NBA market, not a player prop."""
    upper = ticker.upper()
    return any(upper.startswith(p) for p in NBA_NON_PLAYER_PROP_PREFIXES)


def _resolve_team_code(code: str) -> str:
    """Normalize team code aliases."""
    upper = code.upper()
    return _TEAM_ALIASES.get(upper, upper)


def parse_nba_ticker(ticker: str) -> Optional[NBATicker]:
    """
    Parse a Kalshi NBA player prop ticker into structured data.

    Expected format (mirroring MLB):
      KXNBAPTS-26APR102000BOSCLE-BOSJTATUM0-25

    Returns None if the ticker doesn't match the expected format.
    """
    upper = ticker.upper()

    # Determine prop type
    prop_prefix = None
    prop_code = None
    prop_name = None
    for prefix, (code, name) in PROP_MAP.items():
        if upper.startswith(prefix):
            prop_prefix = prefix
            prop_code = code
            prop_name = name
            break

    if not prop_prefix:
        return None

    # Strip prefix and split by dash
    rest = upper[len(prop_prefix):]
    if not rest.startswith("-"):
        return None
    rest = rest[1:]  # remove leading dash

    # Split into parts: DATE+TIME+TEAMS, TEAM+PLAYER+JERSEY, THRESHOLD
    parts = rest.split("-")
    if len(parts) < 3:
        return None

    # Part 1: date + time + teams
    # e.g. "26APR102000BOSCLE"
    date_teams = parts[0]

    # Parse date: first 7 chars = "26APR10" (YY+MON+DD)
    date_match = re.match(r'^(\d{2})([A-Z]{3})(\d{2})', date_teams)
    if not date_match:
        return None

    yy = int(date_match.group(1))
    mon_str = date_match.group(2)
    dd = int(date_match.group(3))

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    mm = month_map.get(mon_str)
    if mm is None:
        return None

    year = 2000 + yy
    game_date = f"{year}-{mm:02d}-{dd:02d}"

    # After the date, there's a 4-digit time then team codes
    # e.g. "2000BOSCLE" → time=2000, teams=BOSCLE
    after_date = date_teams[7:]

    # Try to extract 4-digit time
    time_match = re.match(r'^(\d{4})(.*)', after_date)
    if time_match:
        teams_str = time_match.group(2)
    else:
        teams_str = after_date

    # Teams: 6 chars = 3+3 (e.g. "BOSCLE")
    # But some teams have 2-char codes, so try 3+3 first, then 2+3, 3+2, 2+2
    away_code = None
    home_code = None

    for away_len in [3, 2]:
        for home_len in [3, 2]:
            if away_len + home_len <= len(teams_str):
                candidate_away = _resolve_team_code(teams_str[:away_len])
                candidate_home = _resolve_team_code(teams_str[away_len:away_len + home_len])
                if candidate_away in NBA_TEAMS and candidate_home in NBA_TEAMS:
                    away_code = candidate_away
                    home_code = candidate_home
                    break
        if away_code:
            break

    if not away_code or not home_code:
        logger.debug("[nba_parser] can't extract teams from: %s", teams_str)
        return None

    # Part 2: TEAM + PLAYER + JERSEY
    # e.g. "BOSJTATUM0" → team=BOS, first=J, last=TATUM, jersey=0
    player_part = parts[1]

    # Try to match team prefix (2 or 3 chars)
    player_team = None
    player_rest = None
    for tlen in [3, 2]:
        candidate = _resolve_team_code(player_part[:tlen])
        if candidate in NBA_TEAMS:
            player_team = candidate
            player_rest = player_part[tlen:]
            break

    if not player_team or not player_rest:
        logger.debug("[nba_parser] can't extract player team from: %s", player_part)
        return None

    # Player rest: first initial + last name + jersey number
    # e.g. "JTATUM0" → J, TATUM, 0
    # e.g. "AANTETOKOUNMPO34" → A, ANTETOKOUNMPO, 34
    player_match = re.match(r'^([A-Z])([A-Z]+?)(\d+)$', player_rest)
    if not player_match:
        logger.debug("[nba_parser] can't parse player from: %s", player_rest)
        return None

    first_initial = player_match.group(1)
    last_name = player_match.group(2)
    jersey = player_match.group(3)

    # Part 3+: threshold (last part)
    threshold_str = parts[-1]
    try:
        threshold = int(threshold_str)
    except ValueError:
        # Might be a half-integer like "25" for 25.5, but Kalshi uses ints
        logger.debug("[nba_parser] can't parse threshold: %s", threshold_str)
        return None

    return NBATicker(
        raw_ticker=ticker,
        prop_prefix=prop_prefix,
        prop_code=prop_code,
        prop_name=prop_name,
        game_date=game_date,
        away_team_code=away_code,
        home_team_code=home_code,
        player_team_code=player_team,
        player_first=first_initial,
        player_last=last_name,
        player_jersey=jersey,
        threshold=threshold,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("NBA ticker parser — self-test")
    print("=" * 78)

    # Synthetic test tickers (format matches MLB pattern)
    test_tickers = [
        "KXNBAPTS-26APR102000BOSCLE-BOSJTATUM0-25",
        "KXNBAPTS-26APR102000BOSCLE-BOSJTATUM0-30",
        "KXNBAREB-26APR102000BOSCLE-CLEDGARLAND10-8",
        "KXNBAAST-26APR101930LALMIA-LALLJAMES23-7",
        "KXNBA3PT-26APR101930LALMIA-MIATHERRO14-3",
        "KXNBASTL-26APR101930LALMIA-LALLJAMES23-2",
        "KXNBABLK-26APR102000BOSCLE-BOSAHORFORD42-2",
    ]

    parsed_ok = 0
    for t in test_tickers:
        result = parse_nba_ticker(t)
        if result:
            parsed_ok += 1
            print(f"\n✓ {t}")
            print(f"  → {result.player_first}. {result.player_last} #{result.player_jersey} ({result.player_team_code})")
            print(f"  → {result.prop_name} ≥ {result.threshold}")
            print(f"  → {result.away_team_code} @ {result.home_team_code} on {result.game_date}")
        else:
            print(f"\n✗ FAILED: {t}")

    # Test non-player-prop detection
    game_tickers = [
        "KXNBATOTAL-26APR102000BOSCLE-9",
        "KXNBASPREAD-26APR102000BOSCLE-BOS3",
        "KXNBATEAMTOTAL-26APR102000BOSCLE-BOS5",
        "KXNBAGAME-26APR102000BOSCLE-BOS",
        "KXNBA1HTOTAL-26APR102000BOSCLE-4",
        "KXNBAMENTION-26APR10-LEBRON",
    ]

    print("\n\n── Non-player-prop detection ──")
    for t in game_tickers:
        skip = is_nba_non_player_prop_market(t)
        print(f"  {'⊘ SKIP' if skip else '→ parse'}  {t}")

    print(f"\n{'=' * 78}")
    print(f"Parsed: {parsed_ok}/{len(test_tickers)} synthetic tickers")
    print(f"{'=' * 78}")