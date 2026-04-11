"""
scripts/cleanup_signals.py

Cleans up misclassified and garbage signals from Supabase.

Removes:
  1. Sports tickers claimed by wrong sector (weather/crypto/tech/politics/econ)
  2. Entertainment tickers claimed by crypto
  3. Flat-prior noise (our_p ≈ 0.435-0.455) from the broken BayesianPolyModel

Usage:
    # DRY RUN (preview only — no deletes):
    python3 -m scripts.cleanup_signals

    # EXECUTE (actually delete):
    python3 -m scripts.cleanup_signals --execute
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Run: set -a; source .env; set +a")
    sys.exit(1)

EXECUTE = "--execute" in sys.argv


# ── Sports prefixes — any signal with these prefixes in a non-sports sector
#    is misclassified and should be deleted. ────────────────────────────────

SPORTS_PREFIXES = [
    "KXMVE", "KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXMLS",
    "KXUFC", "KXNCAA", "KXCBB", "KXCFB",
    "KXATP", "KXWTA", "KXTENNIS",
    "KXGOLF", "KXPGA", "KXOWGA",
    "KXEPL", "KXSOCCER",
    "KXARG", "KXLALI", "KXBUND", "KXSERI", "KXLIGA", "KXLIGU",
    "KXBRAS", "KXSWIS", "KXBELG", "KXECUL", "KXPSLG", "KXSAUD",
    "KXJLEA", "KXUCL", "KXFIFA", "KXINTL", "KXALEA",
    "KXCBA", "KXCRICKET",
    "KXFIBA", "KXACBG", "KXVTBG", "KXBALG", "KXBBSE", "KXNPBG",
    "KXAHLG", "KXDIMA",
    "KXBOXING", "KXWWE",
    "KXF1", "KXNASCAR",
    "KXRUGBY", "KXOLYMPIC", "KXTHAIL", "KXSL",
    "KXOW", "KXVALORANT", "KXLOL", "KXLEAGUE",
    "KXRL", "KXROCKETLEAGUE",
    "KXCSGO", "KXCS2", "KXDOTA", "KXINTLF",
    "KXAPEX", "KXFORT", "KXHALO", "KXSC2",
    "KXESPORT", "KXEGYPT", "KXVENF", "KXVENFUTVE",
    "KXR6G", "KXSURV",
    # v11.7 additions
    "KXCONMEBOL", "KXCHLL", "KXEUROLEAGUE", "KXEKSTRAKLASA",
    "KXSHL", "KXURYPD", "KXSUPERLIG", "KXEGYPL",
    "KXITF", "KXIPL", "KXSCOTTISHPREM", "KXUEL",
    "KXEREDIVISIE", "KXMOTOGP", "KXBBL", "KXKF",
    "KXALLSVENSKAN", "KXARGPREMDIV", "KXNEXTAG",
    "KXUFL",
]

ENTERTAINMENT_PREFIXES = [
    "KXSPOTIFY", "KXSPOTSTREAM", "KXBOX", "KXNETFLIX",
    "KXHBO", "KXGRAMMY", "KXOSCAR", "KXEMMY",
    "KXGOLDENGLOBE", "KXBILLBOARD",
]


def main() -> int:
    mode = "EXECUTE" if EXECUTE else "DRY RUN"
    print("=" * 78)
    print(f"Signal Cleanup — {mode}")
    print("=" * 78)

    if not EXECUTE:
        print("\n  ⚠  DRY RUN — no data will be deleted.")
        print("  Run with --execute to actually delete.\n")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # ── Get before-state ───────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM signals")
    total_before = cur.fetchone()[0]

    cur.execute("""
        SELECT sector, COUNT(*), 
               COUNT(*) FILTER (WHERE outcome IS NOT NULL),
               AVG((our_prob - COALESCE(outcome::float, 0.5))^2) 
                   FILTER (WHERE outcome IS NOT NULL)
        FROM signals 
        GROUP BY sector 
        ORDER BY sector
    """)
    print(f"── Before Cleanup ({total_before:,} total signals) ──────────────────────")
    print(f"{'Sector':<16} {'Total':>8} {'Resolved':>10} {'Brier':>8}")
    print("─" * 46)
    for sector, total, resolved, brier in cur.fetchall():
        brier_str = f"{brier:.4f}" if brier else "—"
        print(f"{sector:<16} {total:>8,} {resolved:>10,} {brier_str:>8}")

    # ── 1. Sports tickers in wrong sectors ─────────────────────────────────
    sports_conditions = " OR ".join(
        f"ticker LIKE '{p}%'" for p in SPORTS_PREFIXES
    )
    cur.execute(f"""
        SELECT id, ticker, sector, our_prob, outcome
        FROM signals
        WHERE sector != 'sports'
          AND ({sports_conditions})
    """)
    sports_bleed = cur.fetchall()

    # ── 2. Entertainment tickers in wrong sectors ──────────────────────────
    ent_conditions = " OR ".join(
        f"ticker LIKE '{p}%'" for p in ENTERTAINMENT_PREFIXES
    )
    cur.execute(f"""
        SELECT id, ticker, sector, our_prob, outcome
        FROM signals
        WHERE sector != 'entertainment'
          AND ({ent_conditions})
    """)
    ent_bleed = cur.fetchall()

    # ── 3. Flat-prior noise from sports ────────────────────────────────────
    # The broken BayesianPolyModel outputs constants near 0.435-0.455.
    # These are garbage signals from unmodeled sports that pollute Brier.
    cur.execute("""
        SELECT id, ticker, sector, our_prob, outcome
        FROM signals
        WHERE sector = 'sports'
          AND our_prob BETWEEN 0.430 AND 0.460
          AND ticker NOT LIKE 'KXMLBHIT%'
          AND ticker NOT LIKE 'KXMLBTB%'
          AND ticker NOT LIKE 'KXMLBHR%'
          AND ticker NOT LIKE 'KXMLBHRR%'
          AND ticker NOT LIKE 'KXMLBKS%'
    """)
    flat_prior = cur.fetchall()

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n── Signals to Delete ────────────────────────────────────────────")

    # Sports bleed breakdown
    bleed_by_sector = defaultdict(int)
    for _, _, sector, _, _ in sports_bleed:
        bleed_by_sector[sector] += 1
    print(f"\n  1. Sports tickers in wrong sectors: {len(sports_bleed):,}")
    for sector, count in sorted(bleed_by_sector.items(), key=lambda x: -x[1]):
        print(f"     → {sector}: {count:,}")

    print(f"\n  2. Entertainment tickers in wrong sectors: {len(ent_bleed):,}")

    print(f"\n  3. Flat-prior noise (our_p ≈ 0.435-0.455, non-MLB): {len(flat_prior):,}")

    # Count resolved in each category
    sports_resolved = sum(1 for _, _, _, _, o in sports_bleed if o is not None)
    ent_resolved = sum(1 for _, _, _, _, o in ent_bleed if o is not None)
    flat_resolved = sum(1 for _, _, _, _, o in flat_prior if o is not None)

    total_delete = len(sports_bleed) + len(ent_bleed) + len(flat_prior)
    total_resolved_delete = sports_resolved + ent_resolved + flat_resolved

    print(f"\n  TOTAL to delete: {total_delete:,} signals "
          f"({total_resolved_delete:,} resolved)")
    print(f"  Remaining after: {total_before - total_delete:,} signals")

    if not EXECUTE:
        print(f"\n  ⚠  DRY RUN — nothing deleted. Run with --execute to proceed.")
        cur.close()
        conn.close()
        return 0

    # ── Execute deletes ────────────────────────────────────────────────────
    print(f"\n── Executing Deletes ────────────────────────────────────────────")

    all_ids = (
        [row[0] for row in sports_bleed] +
        [row[0] for row in ent_bleed] +
        [row[0] for row in flat_prior]
    )

    if not all_ids:
        print("  Nothing to delete.")
        cur.close()
        conn.close()
        return 0

    # Delete in batches of 1000
    deleted = 0
    for i in range(0, len(all_ids), 1000):
        batch = all_ids[i:i + 1000]
        placeholders = ",".join(["%s"] * len(batch))
        cur.execute(f"DELETE FROM signals WHERE id IN ({placeholders})", batch)
        deleted += cur.rowcount
        print(f"  Deleted batch {i // 1000 + 1}: {cur.rowcount:,} rows")

    conn.commit()
    print(f"\n  ✓ Total deleted: {deleted:,} signals")

    # ── After-state ────────────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM signals")
    total_after = cur.fetchone()[0]

    cur.execute("""
        SELECT sector, COUNT(*),
               COUNT(*) FILTER (WHERE outcome IS NOT NULL),
               AVG((our_prob - COALESCE(outcome::float, 0.5))^2)
                   FILTER (WHERE outcome IS NOT NULL)
        FROM signals
        GROUP BY sector
        ORDER BY sector
    """)
    print(f"\n── After Cleanup ({total_after:,} total signals) ───────────────────────")
    print(f"{'Sector':<16} {'Total':>8} {'Resolved':>10} {'Brier':>8}")
    print("─" * 46)
    for sector, total, resolved, brier in cur.fetchall():
        brier_str = f"{brier:.4f}" if brier else "—"
        print(f"{sector:<16} {total:>8,} {resolved:>10,} {brier_str:>8}")

    cur.close()
    conn.close()

    print(f"\n{'=' * 78}")
    print(f"✓ Cleaned {deleted:,} garbage signals. Brier scores are now accurate.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())