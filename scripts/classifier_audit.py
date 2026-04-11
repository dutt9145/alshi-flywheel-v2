"""
scripts/classifier_audit.py

Pulls all signals from Supabase and audits sector classification.
Finds: cross-sector bleed, wrong-sector claims, unclaimed markets.

Usage:
    export DATABASE_URL='...'
    python3 -m scripts.classifier_audit

Output:
    - Per-sector signal counts
    - Misclassified tickers (ticker prefix doesn't match claimed sector)
    - Top offenders by volume
    - Specific known bugs (Chilean soccer in weather, etc.)
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


# ── Expected sector mapping by ticker prefix ────────────────────────────────
# Each prefix maps to the sector that SHOULD claim it.

PREFIX_TO_EXPECTED_SECTOR = {}

_SPORTS_PREFIXES = [
    "kxmve", "kxnba", "kxnfl", "kxmlb", "kxnhl", "kxmls",
    "kxufc", "kxncaa", "kxcbb", "kxcfb",
    "kxatp", "kxwta", "kxtennis",
    "kxgolf", "kxpga", "kxowga",
    "kxepl", "kxsoccer",
    "kxarg", "kxlali", "kxbund", "kxseri", "kxliga", "kxligu",
    "kxbras", "kxswis", "kxbelg", "kxecul", "kxpslg", "kxsaud",
    "kxjlea", "kxuclt", "kxfifa", "kxintl", "kxalea",
    "kxcba", "kxcricket",
    "kxfiba", "kxacbg", "kxvtbg", "kxbalg", "kxbbse", "kxnpbg",
    "kxahlg", "kxdima",
    "kxboxing", "kxwwe",
    "kxf1", "kxnascar",
    "kxrugby", "kxolympic", "kxthail", "kxsl",
    "kxow", "kxvalorant", "kxlol", "kxleague",
    "kxrl", "kxrocketleague",
    "kxcsgo", "kxcs2", "kxdota", "kxintlf",
    "kxapex", "kxfort", "kxhalo", "kxsc2",
    "kxesport", "kxegypt", "kxvenf", "kxr6g",
    "kxsurv",
    # Known problem prefixes — Chilean/Colombian/etc. soccer
    "kxchll", "kxdel", "kxconmebol", "kxcol",
    "kxnba3pt", "kxnbapts", "kxnbareb", "kxnbaast",
    "kxnbastl", "kxnbablk",
]
for p in _SPORTS_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "sports"

_CRYPTO_PREFIXES = [
    "kxbtc", "kxeth", "kxsol", "kxxrp", "kxcrypto", "kxdoge",
    "kxbnb", "kxavax", "kxlink", "kxcoin", "kxmatic", "kxada",
    "kxdot", "kxatom", "kxnear", "kxfil", "kxapt", "kxsui",
    "kxshib", "kxnetf",
]
for p in _CRYPTO_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "crypto"

_ECON_PREFIXES = [
    "kxcpi", "kxfed", "kxfomc", "kxgdp", "kxjobs", "kxunrate",
    "kxpce", "kxnfp", "kxyield", "kxrate", "kxinfl",
    "kxhousing", "kxretail", "kxdebt", "kxtrade", "kxconsumer",
    "kxoil", "kxgold", "kxcommodity", "kxdow", "kxspx", "kxvix",
    "kxwti", "kxjble", "kxekst", "kxapfd",
]
for p in _ECON_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "economics"

_WEATHER_PREFIXES = [
    "kxwthr", "kxhurr", "kxsnow", "kxtmp", "kxrain",
    "kxstorm", "kxtornado", "kxblizz", "kxfrost",
    "kxcyclone", "kxtyphoon", "kxmonsoon",
    "kxlowt", "kxhigh", "kxtemp", "kxdens",
]
for p in _WEATHER_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "weather"

_TECH_PREFIXES = [
    "kxaapl", "kxgoog", "kxmsft", "kxamzn", "kxmeta", "kxnvda",
    "kxtsla", "kxearnings", "kxtech", "kxai", "kxipo",
    "kxopenai", "kxanthropic", "kxnasdaq", "kxsemiconductor",
    "kxintc", "kxamd", "kxqcom", "kxtsm", "kxsamsng",
]
for p in _TECH_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "tech"

_POLITICS_PREFIXES = [
    "kxpres", "kxsenate", "kxhouse", "kxgov", "kxelect",
    "kxpol", "kxcong", "kxsupct", "kxadmin",
    "kxun", "kxnato", "kxeu",
]
for p in _POLITICS_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "politics"

_ENTERTAINMENT_PREFIXES = [
    "kxspotify", "kxbox", "kxnetflix", "kxhbo",
    "kxgrammy", "kxoscar", "kxemmy", "kxgoldenglobe",
    "kxbillboard",
]
for p in _ENTERTAINMENT_PREFIXES:
    PREFIX_TO_EXPECTED_SECTOR[p] = "entertainment"


def get_expected_sector(ticker: str) -> str:
    """Determine what sector a ticker SHOULD belong to based on prefix."""
    tk = ticker.lower()
    # Try longest prefix first (more specific)
    best_match = ""
    best_sector = "unknown"
    for prefix, sector in PREFIX_TO_EXPECTED_SECTOR.items():
        if tk.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_sector = sector
    return best_sector


def main() -> int:
    print("=" * 78)
    print("Classifier Audit — Cross-Sector Bleed Report")
    print("=" * 78)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Pull all signals
    cur.execute("""
        SELECT ticker, sector, our_prob, outcome, created_at
        FROM signals
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\nTotal signals: {len(rows):,}\n")

    # ── Per-sector counts ──────────────────────────────────────────────────
    sector_counts = defaultdict(int)
    sector_resolved = defaultdict(int)
    misclassified = []
    unknown_prefix = defaultdict(int)

    for ticker, sector, our_prob, outcome, created_at in rows:
        sector_counts[sector] += 1
        if outcome is not None:
            sector_resolved[sector] += 1

        expected = get_expected_sector(ticker)

        if expected == "unknown":
            # Extract prefix for reporting
            tk = ticker.lower()
            prefix = tk.split("-")[0] if "-" in tk else tk[:10]
            unknown_prefix[prefix] += 1
        elif expected != sector:
            misclassified.append({
                "ticker": ticker,
                "claimed_by": sector,
                "expected": expected,
                "our_prob": our_prob,
                "outcome": outcome,
            })

    # ── Report: sector summary ─────────────────────────────────────────────
    print("── Sector Signal Counts ─────────────────────────────────────────")
    print(f"{'Sector':<16} {'Total':>8} {'Resolved':>10} {'Pct':>6}")
    print("─" * 44)
    for sector in sorted(sector_counts.keys()):
        total = sector_counts[sector]
        resolved = sector_resolved.get(sector, 0)
        pct = f"{resolved/total*100:.1f}%" if total > 0 else "—"
        print(f"{sector:<16} {total:>8,} {resolved:>10,} {pct:>6}")
    print(f"{'TOTAL':<16} {sum(sector_counts.values()):>8,} "
          f"{sum(sector_resolved.values()):>10,}")

    # ── Report: misclassifications ─────────────────────────────────────────
    print(f"\n── Misclassified Signals ({len(misclassified):,} total) ───────────────────────")

    if not misclassified:
        print("  None found! All signals match expected sectors.")
    else:
        # Group by (claimed, expected) pair
        bleed_pairs = defaultdict(list)
        for m in misclassified:
            key = (m["claimed_by"], m["expected"])
            bleed_pairs[key].append(m["ticker"])

        print(f"{'Claimed By':<14} {'Should Be':<14} {'Count':>6}  Sample Tickers")
        print("─" * 78)
        for (claimed, expected), tickers in sorted(
            bleed_pairs.items(), key=lambda x: -len(x[1])
        ):
            samples = ", ".join(sorted(set(t[:35] for t in tickers[:3])))
            print(f"{claimed:<14} {expected:<14} {len(tickers):>6}  {samples}")

        # Top 20 worst individual tickers
        ticker_bleed_count = defaultdict(int)
        for m in misclassified:
            # Use event-level ticker (strip threshold)
            base = "-".join(m["ticker"].split("-")[:2])
            ticker_bleed_count[(base, m["claimed_by"], m["expected"])] += 1

        print(f"\n── Top 20 Bleeding Ticker Groups ───────────────────────────────")
        print(f"{'Ticker Base':<40} {'Claimed':<12} {'Expected':<12} {'Count':>5}")
        print("─" * 72)
        for (base, claimed, expected), count in sorted(
            ticker_bleed_count.items(), key=lambda x: -x[1]
        )[:20]:
            print(f"{base:<40} {claimed:<12} {expected:<12} {count:>5}")

    # ── Report: unknown prefix tickers ─────────────────────────────────────
    if unknown_prefix:
        print(f"\n── Unknown Prefix Tickers ({sum(unknown_prefix.values()):,} signals) ──────")
        print("  These don't match any known sector prefix.")
        print(f"  {'Prefix':<25} {'Count':>6}")
        print("  " + "─" * 35)
        for prefix, count in sorted(unknown_prefix.items(), key=lambda x: -x[1])[:30]:
            print(f"  {prefix:<25} {count:>6}")

    # ── Known problem patterns ─────────────────────────────────────────────
    print(f"\n── Known Bug Check ─────────────────────────────────────────────")
    known_bugs = [
        ("kxchll", "weather", "Chilean soccer prefix 'kxchll' matching weather 'kxchll'"),
        ("kxdel", "crypto", "KXDELGAME leaking into crypto"),
        ("kxdel", "weather", "KXDELGAME leaking into weather"),
        ("kxconmebol", "weather", "CONMEBOL (South American soccer) in weather"),
        ("kxspot", "crypto", "KXSPOTIFY matching crypto's 'kxspot' prefix"),
    ]

    found_bugs = 0
    for prefix, wrong_sector, description in known_bugs:
        count = sum(1 for m in misclassified
                    if m["ticker"].lower().startswith(prefix)
                    and m["claimed_by"] == wrong_sector)
        status = f"FOUND ({count} signals)" if count > 0 else "clean"
        icon = "⚠" if count > 0 else "✓"
        print(f"  {icon} {description}: {status}")
        if count > 0:
            found_bugs += 1

    print(f"\n{'=' * 78}")
    if not misclassified and found_bugs == 0:
        print("✓ All clear — no cross-sector bleed detected.")
    else:
        print(f"⚠ Found {len(misclassified):,} misclassified signals across "
              f"{len(bleed_pairs)} bleed patterns.")
        if found_bugs > 0:
            print(f"  {found_bugs} known bugs still present.")
    print("=" * 78)

    return 0 if not misclassified else 1


if __name__ == "__main__":
    sys.exit(main())