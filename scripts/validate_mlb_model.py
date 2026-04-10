"""
scripts/validate_mlb_model.py

Pulls every resolved MLB player prop signal from Supabase, runs the new
MLB hit-rate model on it, and compares Brier score + accuracy to what the
old flat-prior bot predicted.

Output:
  - Old Brier (from signals.our_prob)
  - New Brier (from model prediction)
  - Sample size
  - Per-prop breakdown (HIT / TB / HR / HRR / KS)
  - Worst misses on each side

Usage:
    export DATABASE_URL='...'
    python3 -m scripts.validate_mlb_model

Limitations:
  - Uses CURRENT season stats, not stats-as-of-game-date. For very old
    games the current stats may differ materially from what was available
    when the market was live. Since this is 2026 early-season data, the
    delta is small.
  - No probable pitcher data — we use None for pitcher stats. Adding
    pitcher context is the biggest lever for further Brier improvement.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from shared.kalshi_ticker_parser import (
    parse_mlb_ticker,
    is_mlb_non_player_prop_market,
    MLB_TEAMS,
)
from shared.mlb_stats_fetcher import (
    lookup_player,
    fetch_batter_stats,
    fetch_batter_rolling,
    fetch_pitcher_stats,
)
from shared.mlb_hit_model import predict_mlb_prop


DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Run: set -a; source .env; set +a")
    sys.exit(1)


def brier(prob: float, outcome: int) -> float:
    """Single-sample Brier score. outcome is 0 or 1."""
    return (prob - float(outcome)) ** 2


def main() -> int:
    print("=" * 78)
    print("MLB hit-rate model — historical validation")
    print("=" * 78)

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    # Pull all resolved MLB player-prop signals
    cur.execute("""
        SELECT ticker, our_prob, outcome, direction, market_prob
        FROM signals
        WHERE ticker LIKE 'KXMLB%'
          AND outcome IS NOT NULL
          AND sector = 'sports'
        ORDER BY created_at ASC
    """)
    rows = cur.fetchall()
    print(f"\nPulled {len(rows):,} resolved MLB signals from Supabase\n")

    if not rows:
        print("No resolved MLB signals found. Validation needs historical data.")
        return 1

    # Player ID cache to avoid re-hitting the API for the same player
    player_cache: dict[tuple, int] = {}

    def cached_lookup(team_code, first, last, jersey):
        key = (team_code, first, last, jersey)
        if key in player_cache:
            return player_cache[key]
        team_id = MLB_TEAMS.get(team_code, (None, None))[0]
        if not team_id:
            player_cache[key] = None
            return None
        result = lookup_player(team_id, first, last, jersey)
        player_cache[key] = result.player_id if result else None
        return result.player_id if result else None

    # Stats cache
    batter_season_cache:  dict[int, object] = {}
    batter_rolling_cache: dict[int, object] = {}

    def cached_batter_stats(pid):
        if pid not in batter_season_cache:
            batter_season_cache[pid] = fetch_batter_stats(pid)
        return batter_season_cache[pid]

    def cached_batter_rolling(pid):
        if pid not in batter_rolling_cache:
            batter_rolling_cache[pid] = fetch_batter_rolling(pid, n_games=10)
        return batter_rolling_cache[pid]

    # Run the new model on every resolved ticket
    old_brier_total = 0.0
    new_brier_total = 0.0
    old_correct     = 0
    new_correct     = 0
    evaluated       = 0
    skipped_reason  = defaultdict(int)

    per_prop = defaultdict(lambda: {
        "n": 0, "old_brier": 0.0, "new_brier": 0.0,
        "old_correct": 0, "new_correct": 0,
    })

    worst_old: list[tuple] = []  # (brier, ticker, our_prob, outcome)
    worst_new: list[tuple] = []
    biggest_flips: list[tuple] = []  # (|old - new|, ticker, old, new, outcome)

    t_start = time.time()
    for i, (ticker, our_prob, outcome, direction, market_prob) in enumerate(rows, 1):
        if i % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  progress: {i}/{len(rows)}  ({elapsed:.1f}s elapsed)")

        # Skip non-player-prop markets (game-level, futures, etc.)
        if is_mlb_non_player_prop_market(ticker):
            skipped_reason["non_player_prop"] += 1
            continue

        parsed = parse_mlb_ticker(ticker)
        if not parsed:
            skipped_reason["parse_failed"] += 1
            continue

        pid = cached_lookup(
            parsed.player_team_code, parsed.player_first,
            parsed.player_last, parsed.player_jersey,
        )
        if not pid:
            skipped_reason["player_not_found"] += 1
            continue

        if parsed.prop_code == "KS":
            # Pitcher prop — different stats path
            if pid not in batter_season_cache:
                batter_season_cache[pid] = fetch_pitcher_stats(pid)
            pitcher_stats = batter_season_cache[pid]
            pred = predict_mlb_prop(parsed, None, None, pitcher_stats)
        else:
            season  = cached_batter_stats(pid)
            rolling = cached_batter_rolling(pid)
            if not season:
                skipped_reason["no_season_stats"] += 1
                continue
            pred = predict_mlb_prop(parsed, season, rolling, None)

        if not pred:
            skipped_reason["no_model"] += 1
            continue

        # Compute scores for both old and new
        outcome_int = int(outcome)
        old_b = brier(float(our_prob), outcome_int)
        new_b = brier(pred.prob_yes, outcome_int)

        old_brier_total += old_b
        new_brier_total += new_b
        if (float(our_prob) > 0.5) == bool(outcome_int):
            old_correct += 1
        if (pred.prob_yes > 0.5) == bool(outcome_int):
            new_correct += 1
        evaluated += 1

        # Per-prop tracking
        pp = per_prop[parsed.prop_code]
        pp["n"] += 1
        pp["old_brier"] += old_b
        pp["new_brier"] += new_b
        if (float(our_prob) > 0.5) == bool(outcome_int):
            pp["old_correct"] += 1
        if (pred.prob_yes > 0.5) == bool(outcome_int):
            pp["new_correct"] += 1

        worst_old.append((old_b, ticker, float(our_prob), outcome_int))
        worst_new.append((new_b, ticker, pred.prob_yes, outcome_int))
        biggest_flips.append((
            abs(pred.prob_yes - float(our_prob)), ticker,
            float(our_prob), pred.prob_yes, outcome_int,
        ))

    cur.close()
    conn.close()

    # ── Report ───────────────────────────────────────────────────────────────
    if evaluated == 0:
        print("\nNo signals evaluated. Skipped reasons:")
        for reason, count in skipped_reason.items():
            print(f"  {reason}: {count}")
        return 1

    old_brier_avg = old_brier_total / evaluated
    new_brier_avg = new_brier_total / evaluated
    old_acc       = old_correct / evaluated
    new_acc       = new_correct / evaluated
    improvement   = (old_brier_avg - new_brier_avg) / old_brier_avg * 100

    print(f"\n{'=' * 78}")
    print("RESULTS")
    print("=" * 78)
    print(f"Evaluated:         {evaluated:,} signals")
    print(f"Skipped:           {sum(skipped_reason.values()):,}")
    for reason, count in sorted(skipped_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason:25s} {count:,}")

    print(f"\n{'Metric':<20} {'Old (flat prior)':<22} {'New (model)':<22} {'Δ':<10}")
    print("─" * 78)
    print(f"{'Brier score':<20} {old_brier_avg:<22.4f} {new_brier_avg:<22.4f} "
          f"{new_brier_avg - old_brier_avg:+.4f}")
    print(f"{'Accuracy':<20} {old_acc:<22.1%} {new_acc:<22.1%} "
          f"{(new_acc - old_acc) * 100:+.1f}pp")
    print(f"\nBrier improvement: {improvement:+.1f}%")

    if new_brier_avg < old_brier_avg:
        print("✓ New model is better — deploy with confidence")
    else:
        print("✗ New model is WORSE — investigate before deploying")

    print("\n── Per-prop breakdown ──────────────────────────────────────────────")
    print(f"{'Prop':<6} {'n':>5}  {'Old Brier':>10}  {'New Brier':>10}  "
          f"{'Old Acc':>8}  {'New Acc':>8}  {'Δ Brier':>10}")
    for code in ("HIT", "TB", "HR", "HRR", "KS"):
        pp = per_prop.get(code)
        if not pp or pp["n"] == 0:
            continue
        print(
            f"{code:<6} {pp['n']:>5}  "
            f"{pp['old_brier']/pp['n']:>10.4f}  "
            f"{pp['new_brier']/pp['n']:>10.4f}  "
            f"{pp['old_correct']/pp['n']:>8.1%}  "
            f"{pp['new_correct']/pp['n']:>8.1%}  "
            f"{(pp['new_brier'] - pp['old_brier'])/pp['n']:>+10.4f}"
        )

    # Show biggest flips — where the new model most strongly disagrees with old
    biggest_flips.sort(reverse=True)
    print("\n── 10 biggest prediction flips (|new - old|) ───────────────────────")
    for delta, t, old_p, new_p, outc in biggest_flips[:10]:
        winner = "NEW ✓" if brier(new_p, outc) < brier(old_p, outc) else "old ✓"
        print(f"  {t}")
        print(f"    old={old_p:.3f}  new={new_p:.3f}  outcome={outc}  → {winner}")

    return 0 if new_brier_avg < old_brier_avg else 1


if __name__ == "__main__":
    sys.exit(main())