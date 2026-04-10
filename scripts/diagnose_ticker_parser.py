"""
scripts/diagnose_ticker_parser.py

Runs the MLB ticker parser against every KXMLB* signal in Supabase and
reports parse success rate, failure examples, and a breakdown by prop type.

Usage:
    python3 scripts/diagnose_ticker_parser.py

Requires:
    DATABASE_URL environment variable (same one Railway uses)
    psycopg2-binary installed
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from shared.kalshi_ticker_parser import (
    parse_mlb_ticker,
    is_mlb_game_level_market,
    MLB_PROPS,
)


DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Load your .env first:")
    print("       set -a; source .env; set +a")
    print("       python3 scripts/diagnose_ticker_parser.py")
    sys.exit(1)


def main() -> int:
    print("=" * 78)
    print("Kalshi MLB ticker parser — Supabase diagnostic")
    print("=" * 78)

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    # Pull all distinct MLB tickers
    cur.execute("""
        SELECT DISTINCT ticker
        FROM signals
        WHERE ticker LIKE 'KXMLB%'
        ORDER BY ticker
    """)
    tickers = [row[0] for row in cur.fetchall()]

    print(f"\nFound {len(tickers):,} distinct KXMLB* tickers in signals table\n")

    # Categorize each one
    parsed_count    = 0
    game_level_count = 0
    failed_tickers  = []
    prop_counter    = Counter()
    team_counter    = Counter()
    player_counter  = Counter()
    game_level_counter = Counter()
    failures_by_prefix: dict[str, list[str]] = defaultdict(list)

    for ticker in tickers:
        # First: is this a game-level market we intentionally skip?
        if is_mlb_game_level_market(ticker):
            game_level_count += 1
            # Track which game-level prefix
            prefix = ticker.split("-", 1)[0]
            game_level_counter[prefix] += 1
            continue

        # Otherwise: try to parse as player prop
        result = parse_mlb_ticker(ticker)
        if result:
            parsed_count += 1
            prop_counter[result.prop_code] += 1
            team_counter[result.player_team_code] += 1
            player_counter[f"{result.player_first}. {result.player_last.title()}"] += 1
        else:
            failed_tickers.append(ticker)
            prefix = ticker[:12] if len(ticker) >= 12 else ticker
            failures_by_prefix[prefix].append(ticker)

    # ── Report ───────────────────────────────────────────────────────────────
    player_prop_total = parsed_count + len(failed_tickers)
    success_rate = (parsed_count / player_prop_total * 100) if player_prop_total else 0

    print(f"Total tickers:           {len(tickers):,}")
    print(f"  Game-level (skipped):  {game_level_count:,}  ← not player props, intentional")
    print(f"  Player props:          {player_prop_total:,}")
    print(f"    ✓ Parsed:            {parsed_count:,} ({success_rate:.1f}%)")
    print(f"    ✗ Failed:            {len(failed_tickers):,}")

    print("\n── Game-level markets (skipped) ────────────────────────────────────")
    for prefix, count in game_level_counter.most_common():
        print(f"  {prefix:18s} {count:6,}")

    print("\n── Player prop type breakdown ──────────────────────────────────────")
    for code, count in prop_counter.most_common():
        name = MLB_PROPS.get(code, "?")
        print(f"  {code:6s} ({name:22s}) {count:6,}")

    print("\n── Top 15 player teams ─────────────────────────────────────────────")
    for team, count in team_counter.most_common(15):
        print(f"  {team:4s} {count:6,}")

    print("\n── Top 15 players ──────────────────────────────────────────────────")
    for player, count in player_counter.most_common(15):
        print(f"  {player:28s} {count:6,}")

    if failed_tickers:
        print("\n── True parse failures (player props that didn't parse) ────────────")
        pattern_counts = sorted(
            failures_by_prefix.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )
        for prefix, examples in pattern_counts[:10]:
            print(f"\n  {prefix}* — {len(examples):,} failures")
            for ex in examples[:3]:
                print(f"    • {ex}")

    cur.close()
    conn.close()

    print("\n" + "=" * 78)
    if success_rate >= 95:
        print("✓ Parser ready — >95% success rate on player props.")
        print("  Safe to build stats fetcher.")
        return 0
    elif success_rate >= 80:
        print("⚠ Parser mostly works — review failure patterns above and patch.")
        return 0
    else:
        print("✗ Parser needs work — significant failures to investigate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())