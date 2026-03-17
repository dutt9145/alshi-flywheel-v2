"""
dashboard.py  (v2 — Supabase / PostgreSQL + SQLite fallback)

LOCAL:    python3 dashboard.py          → reads flywheel.db
RAILWAY:  set DATABASE_URL env var      → reads Supabase postgres

Open http://localhost:5555
"""

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH      = os.getenv("DB_PATH", "flywheel.db")
USE_POSTGRES = bool(DATABASE_URL)
PORT         = int(os.getenv("PORT", "5555"))

SECTORS      = ["economics", "crypto", "politics", "weather", "tech", "sports"]
SECTOR_ICONS = {
    "economics": "📈", "crypto": "₿", "politics": "🏛",
    "weather": "🌤",  "tech": "💻",  "sports": "🏆",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def query(sql, params=()):
    try:
        if USE_POSTGRES:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            cur  = conn.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
            return rows
        else:
            import sqlite3
            if not os.path.exists(DB_PATH):
                return []
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            conn.close()
            return rows
    except Exception as e:
        print(f"Query error: {e}")
        return []


def db_ready():
    if USE_POSTGRES:
        return True
    return os.path.exists(DB_PATH)


# ── Data queries ──────────────────────────────────────────────────────────────

def get_overview():
    t = query("SELECT COUNT(*) as n, SUM(dollars_risked) as total FROM trades")
    s = query("SELECT COUNT(*) as n FROM signals")
    a = query("SELECT COUNT(*) as n FROM arb_opportunities")
    o = query("SELECT COUNT(*) as n, SUM(pnl_usd) as pnl FROM outcomes")
    return {
        "trades":   (t[0]["n"]     if t else 0) or 0,
        "signals":  (s[0]["n"]     if s else 0) or 0,
        "arbs":     (a[0]["n"]     if a else 0) or 0,
        "resolved": (o[0]["n"]     if o else 0) or 0,
        "pnl":      round((o[0]["pnl"] if o else 0) or 0, 2),
        "volume":   round((t[0]["total"] if t else 0) or 0, 2),
    }


def get_sector_stats():
    sigs = query("""
        SELECT sector,
               COUNT(*)                                        AS signals,
               AVG(edge)                                       AS avg_edge,
               AVG(confidence)                                 AS avg_conf,
               SUM(CASE WHEN direction='YES' THEN 1 ELSE 0 END) AS yes_count,
               SUM(CASE WHEN direction='NO'  THEN 1 ELSE 0 END) AS no_count,
               AVG(brier_score)                                AS brier
        FROM signals GROUP BY sector
    """)
    sig_map = {r["sector"]: r for r in sigs}

    trd = query("""
        SELECT s.sector, COUNT(t.id) AS trades,
               SUM(t.dollars_risked) AS dollars
        FROM trades t JOIN signals s ON t.ticker = s.ticker
        GROUP BY s.sector
    """)
    trd_map = {r["sector"]: r for r in trd}

    pnl = query("""
        SELECT s.sector, SUM(o.pnl_usd) AS pnl, COUNT(*) AS resolved
        FROM outcomes o JOIN signals s ON o.ticker = s.ticker
        GROUP BY s.sector
    """)
    pnl_map = {r["sector"]: r for r in pnl}

    result = []
    for sec in SECTORS:
        b = sig_map.get(sec, {})
        t = trd_map.get(sec, {})
        p = pnl_map.get(sec, {})
        result.append({
            "sector":   sec,
            "icon":     SECTOR_ICONS[sec],
            "signals":  b.get("signals", 0) or 0,
            "trades":   t.get("trades", 0) or 0,
            "dollars":  round((t.get("dollars") or 0), 2),
            "avg_edge": round(((b.get("avg_edge") or 0)) * 100, 2),
            "avg_conf": round(((b.get("avg_conf") or 0)) * 100, 1),
            "brier":    round((b.get("brier") or 0), 4),
            "yes":      b.get("yes_count", 0) or 0,
            "no":       b.get("no_count", 0) or 0,
            "pnl":      round((p.get("pnl") or 0), 2),
            "resolved": p.get("resolved", 0) or 0,
        })
    return result


def get_recent_trades(limit=20):
    return query(f"""
        SELECT ticker, direction, contracts, yes_price_cents,
               dollars_risked, avg_edge, avg_confidence, demo_mode, created_at
        FROM trades ORDER BY created_at DESC LIMIT {limit}
    """)


def get_recent_arbs(limit=10):
    try:
        return query(f"""
            SELECT ticker, kalshi_prob, poly_prob, sharp_prob,
                   spread, mode, notes, detected_at
            FROM arb_opportunities ORDER BY detected_at DESC LIMIT {limit}
        """)
    except Exception:
        return []


def get_pnl_series():
    rows = query("""
        SELECT DATE(logged_at) as day, SUM(pnl_usd) as daily_pnl
        FROM outcomes WHERE pnl_usd IS NOT NULL
        GROUP BY DATE(logged_at) ORDER BY day
    """)
    cumulative, total = [], 0
    for r in rows:
        total += r["daily_pnl"] or 0
        cumulative.append({"day": str(r["day"]), "pnl": round(total, 2)})
    return cumulative


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(overview, sectors, trades, arbs, pnl_series):

    sector_cards = ""
    for s in sectors:
        ec = "#00ff88" if s["avg_edge"] > 0 else "#ff4444"
        pc = "#00ff88" if s["pnl"] >= 0 else "#ff4444"
        bb = min(int(s["brier"] * 1000), 100)
        sector_cards += f"""
        <div class="sector-card">
          <div class="sector-header">
            <span class="sector-icon">{s["icon"]}</span>
            <span class="sector-name">{s["sector"].upper()}</span>
            <span class="sector-pnl" style="color:{pc}">${s["pnl"]:+.2f}</span>
          </div>
          <div class="stat-row">
            <div class="stat"><div class="stat-val">{s["signals"]}</div><div class="stat-lbl">signals</div></div>
            <div class="stat"><div class="stat-val">{s["trades"]}</div><div class="stat-lbl">trades</div></div>
            <div class="stat"><div class="stat-val" style="color:{ec}">{s["avg_edge"]:+.1f}%</div><div class="stat-lbl">avg edge</div></div>
            <div class="stat"><div class="stat-val">{s["avg_conf"]:.0f}%</div><div class="stat-lbl">conf</div></div>
          </div>
          <div class="brier-row">
            <span class="brier-lbl">Brier {s["brier"]:.4f}</span>
            <div class="brier-track"><div class="brier-fill" style="width:{bb}%"></div></div>
          </div>
          <div class="yes-no-row">
            <span class="yes-badge">YES {s["yes"]}</span>
            <span class="no-badge">NO {s["no"]}</span>
            <span class="resolved-lbl">{s["resolved"]} resolved</span>
          </div>
        </div>"""

    trade_rows = ""
    for t in trades:
        demo    = "DEMO" if t["demo_mode"] else "LIVE"
        dcls    = "badge-demo" if t["demo_mode"] else "badge-live"
        dircls  = "dir-yes" if t["direction"] == "YES" else "dir-no"
        ec      = "#00ff88" if (t["avg_edge"] or 0) > 0 else "#ff4444"
        ts      = str(t["created_at"] or "")[:16].replace("T", " ")
        trade_rows += f"""<tr>
          <td>{ts}</td><td class="ticker-cell">{t["ticker"]}</td>
          <td><span class="{dircls}">{t["direction"]}</span></td>
          <td>{t["contracts"]}×@{t["yes_price_cents"]}¢</td>
          <td>${t["dollars_risked"]:.2f}</td>
          <td style="color:{ec}">{(t["avg_edge"] or 0)*100:+.1f}%</td>
          <td>{(t["avg_confidence"] or 0)*100:.0f}%</td>
          <td><span class="{dcls}">{demo}</span></td>
        </tr>"""

    arb_rows = ""
    for a in arbs:
        sc  = "#00ff88" if (a["spread"] or 0) >= 0.05 else "#ffaa00"
        ts  = str(a.get("detected_at") or "")[:16].replace("T", " ")
        arb_rows += f"""<tr>
          <td>{ts}</td><td class="ticker-cell">{a["ticker"]}</td>
          <td>{(a["kalshi_prob"] or 0)*100:.1f}%</td>
          <td>{(a["poly_prob"] or 0)*100:.1f}%</td>
          <td style="color:{sc}">{(a["spread"] or 0)*100:.1f}%</td>
          <td>{a["mode"]}</td>
        </tr>"""

    pnl_labels = json.dumps([p["day"] for p in pnl_series])
    pnl_data   = json.dumps([p["pnl"] for p in pnl_series])
    total_pnl  = overview["pnl"]
    pnl_color  = "#00ff88" if total_pnl >= 0 else "#ff4444"
    db_mode    = "Supabase PostgreSQL" if USE_POSTGRES else f"SQLite · {DB_PATH}"
    no_data    = "" if db_ready() else """<div class="no-data-banner">
      ⚡ No database found — run python3 orchestrator.py first</div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kalshi Flywheel Oracle</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');
:root{{--bg:#080c14;--surface:#0d1420;--card:#111b2e;--border:#1e2d47;
--green:#00ff88;--red:#ff4444;--amber:#ffaa00;--blue:#4488ff;
--text:#e2e8f0;--muted:#4a6080;
--mono:'JetBrains Mono',monospace;--ui:'Space Grotesk',sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--ui);min-height:100vh}}
.topbar{{background:var(--surface);border-bottom:1px solid var(--border);
padding:0 32px;display:flex;align-items:center;justify-content:space-between;
height:60px;position:sticky;top:0;z-index:100}}
.logo{{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--green);letter-spacing:2px}}
.logo span{{color:var(--muted)}}
.db-badge{{font-family:var(--mono);font-size:10px;color:var(--muted);
background:var(--border);padding:4px 10px;border-radius:4px;letter-spacing:1px}}
.live-dot{{width:8px;height:8px;border-radius:50%;background:var(--green);
animation:pulse 2s infinite;display:inline-block;margin-right:8px}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(1.3)}}}}
.refresh-btn{{font-family:var(--mono);font-size:11px;color:var(--muted);
background:var(--border);border:none;padding:6px 14px;border-radius:4px;
cursor:pointer;letter-spacing:1px}}
.refresh-btn:hover{{color:var(--text)}}
.main{{padding:28px 32px;max-width:1400px;margin:0 auto}}
.no-data-banner{{background:#1a1200;border:1px solid var(--amber);color:var(--amber);
padding:14px 20px;border-radius:8px;font-family:var(--mono);font-size:13px;margin-bottom:24px}}
.overview{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:28px}}
.ov-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center}}
.ov-val{{font-family:var(--mono);font-size:22px;font-weight:700;line-height:1;margin-bottom:6px}}
.ov-lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}}
.section-title{{font-family:var(--mono);font-size:11px;color:var(--muted);
letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;margin-top:28px}}
.sectors-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}}
.sector-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;
padding:18px;transition:border-color .2s}}
.sector-card:hover{{border-color:var(--blue)}}
.sector-header{{display:flex;align-items:center;gap:10px;margin-bottom:14px}}
.sector-icon{{font-size:20px}}.sector-name{{font-family:var(--mono);font-size:12px;
font-weight:700;letter-spacing:2px;flex:1}}
.sector-pnl{{font-family:var(--mono);font-size:14px;font-weight:700}}
.stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px}}
.stat{{text-align:center}}
.stat-val{{font-family:var(--mono);font-size:16px;font-weight:700;line-height:1;margin-bottom:3px}}
.stat-lbl{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
.brier-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.brier-lbl{{font-family:var(--mono);font-size:10px;color:var(--muted);white-space:nowrap}}
.brier-track{{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}}
.brier-fill{{height:100%;background:linear-gradient(90deg,var(--green),var(--amber));
border-radius:2px;transition:width 1s ease}}
.yes-no-row{{display:flex;align-items:center;gap:8px}}
.yes-badge{{background:rgba(0,255,136,.12);color:var(--green);font-family:var(--mono);
font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid rgba(0,255,136,.2)}}
.no-badge{{background:rgba(255,68,68,.12);color:var(--red);font-family:var(--mono);
font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid rgba(255,68,68,.2)}}
.resolved-lbl{{font-size:10px;color:var(--muted);margin-left:auto}}
.chart-container{{background:var(--card);border:1px solid var(--border);
border-radius:12px;padding:20px;margin-bottom:28px}}
.table-wrap{{background:var(--card);border:1px solid var(--border);
border-radius:12px;overflow:hidden;margin-bottom:28px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead tr{{background:var(--surface);border-bottom:1px solid var(--border)}}
th{{padding:10px 14px;text-align:left;font-family:var(--mono);font-size:10px;
color:var(--muted);letter-spacing:1px;text-transform:uppercase;font-weight:400}}
td{{padding:10px 14px;border-bottom:1px solid rgba(30,45,71,.5);
font-family:var(--mono);font-size:12px}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.ticker-cell{{color:var(--blue)}}
.dir-yes{{color:var(--green);font-weight:700}}
.dir-no{{color:var(--red);font-weight:700}}
.badge-demo{{background:rgba(255,170,0,.12);color:var(--amber);padding:2px 8px;
border-radius:4px;font-size:10px;border:1px solid rgba(255,170,0,.2)}}
.badge-live{{background:rgba(0,255,136,.12);color:var(--green);padding:2px 8px;
border-radius:4px;font-size:10px;border:1px solid rgba(0,255,136,.2)}}
.empty-state{{text-align:center;padding:40px;color:var(--muted);
font-family:var(--mono);font-size:12px}}
footer{{text-align:center;padding:24px;font-family:var(--mono);font-size:11px;
color:var(--muted);border-top:1px solid var(--border);margin-top:20px}}
</style></head><body>

<div class="topbar">
  <div class="logo"><span class="live-dot"></span>KALSHI<span>//</span>FLYWHEEL <span>ORACLE</span></div>
  <span class="db-badge">⬡ {db_mode}</span>
  <button class="refresh-btn" onclick="location.reload()">⟳ REFRESH</button>
</div>

<div class="main">
{no_data}
<div class="section-title">// portfolio overview</div>
<div class="overview">
  <div class="ov-card"><div class="ov-val">{overview["signals"]}</div><div class="ov-lbl">Signals</div></div>
  <div class="ov-card"><div class="ov-val">{overview["trades"]}</div><div class="ov-lbl">Trades</div></div>
  <div class="ov-card"><div class="ov-val">{overview["arbs"]}</div><div class="ov-lbl">Arbs Found</div></div>
  <div class="ov-card"><div class="ov-val">{overview["resolved"]}</div><div class="ov-lbl">Resolved</div></div>
  <div class="ov-card"><div class="ov-val">${overview["volume"]:.0f}</div><div class="ov-lbl">Vol Deployed</div></div>
  <div class="ov-card"><div class="ov-val" style="color:{pnl_color}">${total_pnl:+.2f}</div><div class="ov-lbl">Total P&amp;L</div></div>
</div>

<div class="section-title">// sector breakdown</div>
<div class="sectors-grid">{sector_cards}</div>

<div class="section-title">// cumulative p&l</div>
<div class="chart-container"><canvas id="pnlChart" height="80"></canvas></div>

<div class="section-title">// recent trades</div>
<div class="table-wrap">{"<table><thead><tr><th>Time</th><th>Ticker</th><th>Dir</th><th>Size</th><th>Risked</th><th>Edge</th><th>Conf</th><th>Mode</th></tr></thead><tbody>"+trade_rows+"</tbody></table>" if trades else '<div class="empty-state">No trades yet — bot is scanning markets</div>'}</div>

<div class="section-title">// arb opportunities</div>
<div class="table-wrap">{"<table><thead><tr><th>Time</th><th>Ticker</th><th>Kalshi</th><th>Poly</th><th>Spread</th><th>Mode</th></tr></thead><tbody>"+arb_rows+"</tbody></table>" if arbs else '<div class="empty-state">No arb opportunities logged yet</div>'}</div>
</div>

<footer>KALSHI FLYWHEEL ORACLE · AUTO-REFRESH 60s · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</footer>

<script>
const labels={pnl_labels},data={pnl_data};
const ctx=document.getElementById('pnlChart').getContext('2d');
const pos=data.length&&data[data.length-1]>=0;
new Chart(ctx,{{type:'line',data:{{labels:labels.length?labels:['No data'],
datasets:[{{label:'Cumulative P&L ($)',data:data.length?data:[0],
borderColor:pos?'#00ff88':'#ff4444',
backgroundColor:pos?'rgba(0,255,136,0.05)':'rgba(255,68,68,0.05)',
borderWidth:2,pointRadius:3,pointBackgroundColor:'#00ff88',fill:true,tension:0.4}}]}},
options:{{responsive:true,plugins:{{legend:{{display:false}}}},
scales:{{x:{{ticks:{{color:'#4a6080',font:{{family:'JetBrains Mono',size:10}}}},
grid:{{color:'#1e2d47'}}}},
y:{{ticks:{{color:'#4a6080',font:{{family:'JetBrains Mono',size:10}},callback:v=>'$'+v}},
grid:{{color:'#1e2d47'}}}}}}}}}}});
setTimeout(()=>location.reload(),60000);
</script></body></html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html(
            get_overview(), get_sector_stats(),
            get_recent_trades(), get_recent_arbs(), get_pnl_series()
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n  ██ KALSHI FLYWHEEL ORACLE")
    print(f"  http://localhost:{PORT}")
    print(f"  DB: {'Supabase PostgreSQL' if USE_POSTGRES else DB_PATH}")
    print(f"  Ctrl+C to stop\n")
    server.serve_forever()