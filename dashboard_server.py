"""
Live Trading Bot Dashboard — Flask HTTP Server
================================================
Run alongside the bot:
  python3 dashboard_server.py

Open in browser: http://127.0.0.1:5001

The bot writes data/status.json every scan cycle.
This server reads it and serves a live-updating HTML dashboard.

No WebSockets needed — JavaScript polls /api/status every 5 seconds.
"""

import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, render_template_string

from config import DASHBOARD_PORT, DASHBOARD_HOST, STATUS_FILE, DB_PATH

app = Flask(__name__)

# ─── HTML Template ────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Options Bot — Live Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body { background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; }
    .card { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; }
    .green { color:#22c55e; } .red { color:#ef4444; } .yellow { color:#eab308; }
    .blue  { color:#60a5fa; } .gray { color:#94a3b8; }
    .badge-buy  { background:#166534; color:#bbf7d0; padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-sell { background:#7f1d1d; color:#fecaca; padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-abs  { background:#1e3a5f; color:#bae6fd; padding:2px 8px; border-radius:99px; font-size:11px; }
    .layer-ok   { background:#14532d; border:1px solid #22c55e; }
    .layer-warn { background:#713f12; border:1px solid #eab308; }
    .layer-off  { background:#1c1917; border:1px solid #44403c; }
    .voter-card { border-radius:8px; padding:10px; margin:4px; min-width:140px; flex:1; }
    .voter-buy  { background:#14532d; border:1px solid #22c55e; }
    .voter-sell { background:#7f1d1d; border:1px solid #ef4444; }
    .voter-abs  { background:#1e293b; border:1px solid #475569; }
    .pulse { animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
    .scroll-log { max-height:260px; overflow-y:auto; }
    .scroll-log::-webkit-scrollbar { width:4px; }
    .scroll-log::-webkit-scrollbar-thumb { background:#475569; border-radius:2px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th { color:#94a3b8; font-weight:600; padding:6px 10px; text-align:left;
         border-bottom:1px solid #334155; }
    td { padding:6px 10px; border-bottom:1px solid #1e293b; }
    tr:hover td { background:#1e293b; }
    .conf-bar { height:6px; border-radius:3px; background:#1e293b; overflow:hidden; }
    .conf-fill { height:100%; border-radius:3px; transition:width 0.3s; }
  </style>
</head>
<body class="p-4">

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<div class="flex items-center justify-between mb-4">
  <div class="flex items-center gap-3">
    <div class="text-2xl font-bold text-white">🤖 AI Options Bot</div>
    <div id="bot-status" class="px-3 py-1 rounded-full text-sm font-semibold bg-green-900 text-green-300 pulse">
      ● LIVE
    </div>
  </div>
  <div class="text-right">
    <div id="clock" class="text-lg font-mono text-blue-400"></div>
    <div id="last-update" class="text-xs text-gray-500">Updating...</div>
  </div>
</div>

<!-- ── KPI Cards ──────────────────────────────────────────────────────────── -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">CAPITAL</div>
    <div id="capital" class="text-2xl font-bold text-white">—</div>
    <div class="text-xs text-gray-500">Starting ₹39,430</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">DAY P&L</div>
    <div id="day-pnl" class="text-2xl font-bold">—</div>
    <div id="pnl-pct" class="text-xs text-gray-500">—</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">OPEN TRADES</div>
    <div id="open-count" class="text-2xl font-bold text-yellow-400">—</div>
    <div class="text-xs text-gray-500">Max 1 per underlying</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">WIN RATE</div>
    <div id="win-rate" class="text-2xl font-bold">—</div>
    <div id="trade-count" class="text-xs text-gray-500">—</div>
  </div>
</div>

<!-- ── 7 Architecture Layers ─────────────────────────────────────────────── -->
<div class="card mb-4">
  <div class="text-sm font-semibold text-gray-300 mb-3">⚙️ 7-Layer Architecture</div>
  <div class="grid grid-cols-7 gap-2" id="layers">
    <div class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-green-400">L1</div>
      <div class="text-xs text-gray-300">Indicators</div>
    </div>
    <div class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-blue-400">L2</div>
      <div class="text-xs text-gray-300">Kronos</div>
    </div>
    <div class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-purple-400">L3</div>
      <div class="text-xs text-gray-300">ML Ensemble</div>
    </div>
    <div id="layer4" class="layer-warn rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-yellow-400">L4</div>
      <div class="text-xs text-gray-300">Consensus</div>
      <div id="votes-display" class="text-xs font-mono text-yellow-300">—/5</div>
    </div>
    <div id="layer5" class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-green-400">L5</div>
      <div class="text-xs text-gray-300">Mkt Filter</div>
      <div id="filter-status" class="text-xs">—</div>
    </div>
    <div class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-orange-400">L6</div>
      <div class="text-xs text-gray-300">Sizing</div>
      <div id="position-pct" class="text-xs font-mono text-orange-300">—%</div>
    </div>
    <div class="layer-ok rounded-lg p-2 text-center">
      <div class="text-xs font-bold text-red-400">L7</div>
      <div class="text-xs text-gray-300">ATR Risk</div>
    </div>
  </div>
</div>

<!-- ── 5 Voters Panel ─────────────────────────────────────────────────────── -->
<div class="card mb-4">
  <div class="text-sm font-semibold text-gray-300 mb-3">🗳️ 5 Voters — Current Reading</div>
  <div class="flex flex-wrap gap-2" id="voters-panel">
    <div class="voter-card voter-abs"><div class="text-xs font-bold text-gray-400">Trend</div><div class="text-xs text-gray-500">Loading...</div></div>
    <div class="voter-card voter-abs"><div class="text-xs font-bold text-gray-400">Reversion</div><div class="text-xs text-gray-500">Loading...</div></div>
    <div class="voter-card voter-abs"><div class="text-xs font-bold text-gray-400">Breakout</div><div class="text-xs text-gray-500">Loading...</div></div>
    <div class="voter-card voter-abs"><div class="text-xs font-bold text-gray-400">Kronos</div><div class="text-xs text-gray-500">Loading...</div></div>
    <div class="voter-card voter-abs"><div class="text-xs font-bold text-gray-400">ML</div><div class="text-xs text-gray-500">Loading...</div></div>
  </div>
</div>

<!-- ── Open Positions + Signal Log ───────────────────────────────────────── -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
  <div class="card">
    <div class="text-sm font-semibold text-gray-300 mb-3">📈 Open Positions</div>
    <div id="open-positions">
      <div class="text-center text-gray-500 text-sm py-4">No open positions</div>
    </div>
  </div>
  <div class="card">
    <div class="text-sm font-semibold text-gray-300 mb-3">📡 Signal Log</div>
    <div id="signal-log" class="scroll-log">
      <div class="text-center text-gray-500 text-sm py-4">Waiting for signals...</div>
    </div>
  </div>
</div>

<!-- ── Trade History ──────────────────────────────────────────────────────── -->
<div class="card mb-4">
  <div class="text-sm font-semibold text-gray-300 mb-3">📋 Trade History</div>
  <div style="overflow-x:auto">
    <table id="trades-table">
      <thead><tr>
        <th>Date</th><th>Symbol</th><th>Entry</th><th>Exit</th>
        <th>P&L</th><th>Exit Reason</th><th>Signal</th>
      </tr></thead>
      <tbody id="trades-body"><tr><td colspan="7" class="text-center text-gray-500">No trades yet</td></tr></tbody>
    </table>
  </div>
</div>

<!-- ── Market Filter Detail ───────────────────────────────────────────────── -->
<div class="card mb-4">
  <div class="text-sm font-semibold text-gray-300 mb-3">🔍 Layer 5 — Market Filter Detail</div>
  <div class="grid grid-cols-3 gap-3" id="filter-detail">
    <div class="text-center">
      <div class="text-xs text-gray-400">ATR Ratio</div>
      <div id="atr-ratio-val" class="text-xl font-mono">—</div>
      <div class="text-xs text-gray-500">Min 0.80</div>
    </div>
    <div class="text-center">
      <div class="text-xs text-gray-400">ADX</div>
      <div id="adx-val" class="text-xl font-mono">—</div>
      <div class="text-xs text-gray-500">Min 20</div>
    </div>
    <div class="text-center">
      <div class="text-xs text-gray-400">Vol Ratio</div>
      <div id="vol-ratio-val" class="text-xl font-mono">—</div>
      <div class="text-xs text-gray-500">Min 1.0</div>
    </div>
  </div>
</div>

<div class="text-center text-xs text-gray-600 mt-4">
  AI Options Bot — Paper Trade Mode | Auto-refreshes every 5s
</div>

<!-- ── JavaScript ─────────────────────────────────────────────────────────── -->
<script>
const REFRESH_MS = 5000;
let signalLog = [];

function fmt(n, prefix='₹') {
  if (n == null) return '—';
  return prefix + n.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
}

function pnlClass(n) {
  return n > 0 ? 'green' : n < 0 ? 'red' : 'gray';
}

function voterClass(vote) {
  return vote === 'BUY' ? 'voter-buy' : vote === 'SELL' ? 'voter-sell' : 'voter-abs';
}

function badgeClass(vote) {
  return vote === 'BUY' ? 'badge-buy' : vote === 'SELL' ? 'badge-sell' : 'badge-abs';
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-IN', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const d   = await res.json();
    renderDashboard(d);
    document.getElementById('last-update').textContent =
      'Updated: ' + new Date().toLocaleTimeString('en-IN');
  } catch(e) {
    document.getElementById('bot-status').textContent = '● OFFLINE';
    document.getElementById('bot-status').className =
      'px-3 py-1 rounded-full text-sm font-semibold bg-red-900 text-red-300';
  }
}

function renderDashboard(d) {
  // KPI
  const cap = d.capital || 0;
  document.getElementById('capital').textContent = '₹' + cap.toLocaleString('en-IN');

  const pnl = d.day_pnl || 0;
  const pnlEl = document.getElementById('day-pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + fmt(pnl);
  pnlEl.className   = 'text-2xl font-bold ' + pnlClass(pnl);
  const pnlPct = cap > 0 ? (pnl / cap * 100).toFixed(2) : '0.00';
  document.getElementById('pnl-pct').textContent = pnlPct + '% of capital';

  document.getElementById('open-count').textContent = d.open_trades || 0;

  const wr = d.win_rate != null ? (d.win_rate * 100).toFixed(1) + '%' : '—';
  const wrEl = document.getElementById('win-rate');
  wrEl.textContent = wr;
  wrEl.className = 'text-2xl font-bold ' + (d.win_rate > 0.4 ? 'green' : d.win_rate > 0.25 ? 'yellow' : 'red');
  document.getElementById('trade-count').textContent =
    (d.total_trades || 0) + ' trades (' + (d.wins || 0) + 'W/' + (d.losses || 0) + 'L)';

  // Layer 4 — Consensus
  const votes = d.last_consensus || {};
  const vFor  = votes.votes_for || 0;
  document.getElementById('votes-display').textContent = vFor + '/5';
  const l4 = document.getElementById('layer4');
  l4.className = (vFor >= 4 ? 'layer-ok' : vFor >= 2 ? 'layer-warn' : 'layer-off') +
                 ' rounded-lg p-2 text-center';

  // Layer 5 — Market filter
  const flt = d.market_filter || {};
  const fEl = document.getElementById('filter-status');
  fEl.textContent = flt.allowed ? '✓' : '✗';
  fEl.className   = 'text-xs ' + (flt.allowed ? 'text-green-400' : 'text-red-400');
  document.getElementById('layer5').className =
    (flt.allowed ? 'layer-ok' : 'layer-warn') + ' rounded-lg p-2 text-center';

  // Layer 6 — Position sizing
  document.getElementById('position-pct').textContent =
    (votes.position_pct || 0) + '%';

  // ATR/ADX/Vol details
  document.getElementById('atr-ratio-val').textContent = (flt.atr_ratio || '—');
  document.getElementById('atr-ratio-val').className =
    'text-xl font-mono ' + ((flt.atr_ratio || 0) >= 0.8 ? 'green' : 'red');
  document.getElementById('adx-val').textContent = (flt.adx || '—');
  document.getElementById('adx-val').className =
    'text-xl font-mono ' + ((flt.adx || 0) >= 20 ? 'green' : 'red');
  document.getElementById('vol-ratio-val').textContent = (flt.vol_ratio || '—');
  document.getElementById('vol-ratio-val').className =
    'text-xl font-mono ' + ((flt.vol_ratio || 0) >= 1.0 ? 'green' : 'red');

  // 5 Voters
  const voters = d.last_voters || [];
  const vPanel = document.getElementById('voters-panel');
  if (voters.length > 0) {
    vPanel.innerHTML = voters.map(v => `
      <div class="voter-card ${voterClass(v.vote)}">
        <div class="flex justify-between items-center mb-1">
          <span class="text-xs font-bold text-gray-300">${v.name}</span>
          <span class="${badgeClass(v.vote)}">${v.vote}</span>
        </div>
        <div class="conf-bar mb-1">
          <div class="conf-fill ${v.vote==='BUY'?'bg-green-500':v.vote==='SELL'?'bg-red-500':'bg-gray-500'}"
               style="width:${Math.round((v.confidence||0)*100)}%"></div>
        </div>
        <div class="text-xs text-gray-400">${Math.round((v.confidence||0)*100)}% conf</div>
        <div class="text-xs text-gray-500 mt-1" style="font-size:10px;">${(v.reason||'').slice(0,40)}</div>
      </div>
    `).join('');
  }

  // Open positions
  const positions = d.open_positions || [];
  const posDiv = document.getElementById('open-positions');
  if (positions.length === 0) {
    posDiv.innerHTML = '<div class="text-center text-gray-500 text-sm py-4">No open positions</div>';
  } else {
    posDiv.innerHTML = '<table>' +
      '<thead><tr><th>Symbol</th><th>Entry</th><th>Curr</th><th>P&L</th><th>SL</th></tr></thead>' +
      '<tbody>' + positions.map(p => {
        const pnlAmt = ((p.current||p.entry) - p.entry) * p.qty;
        return `<tr>
          <td class="text-xs">${p.symbol}</td>
          <td class="text-xs">₹${p.entry}</td>
          <td class="text-xs">₹${p.current||'—'}</td>
          <td class="text-xs ${pnlClass(pnlAmt)}">${pnlAmt>=0?'+':''}₹${pnlAmt.toFixed(0)}</td>
          <td class="text-xs text-red-400">₹${p.sl}</td>
        </tr>`;
      }).join('') + '</tbody></table>';
  }

  // Signal log
  if (d.signal_log && d.signal_log.length > 0) {
    signalLog = d.signal_log;
    document.getElementById('signal-log').innerHTML =
      [...signalLog].reverse().map(s => `
        <div class="flex gap-2 text-xs py-1 border-b border-gray-800">
          <span class="text-gray-500">${s.time}</span>
          <span class="${s.action==='BUY'?'green':s.action==='SELL'?'red':'gray'}">${s.action}</span>
          <span class="text-gray-300">${s.symbol}</span>
          <span class="text-gray-500">${s.reason||''}</span>
        </div>
      `).join('');
  }

  // Trade history
  const trades = d.trade_history || [];
  const tbody  = document.getElementById('trades-body');
  if (trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500">No trades yet</td></tr>';
  } else {
    tbody.innerHTML = [...trades].reverse().map(t => {
      const pnlCls = pnlClass(t.net_pnl);
      const pnlStr = t.net_pnl != null
        ? (t.net_pnl >= 0 ? '+' : '') + '₹' + Math.abs(t.net_pnl).toLocaleString('en-IN')
        : '—';
      return `<tr>
        <td class="text-xs text-gray-400">${t.date||'—'}</td>
        <td class="text-xs">${t.symbol||'—'}</td>
        <td class="text-xs">₹${t.entry_price||'—'}</td>
        <td class="text-xs">₹${t.exit_price||'—'}</td>
        <td class="text-xs font-semibold ${pnlCls}">${pnlStr}</td>
        <td class="text-xs text-gray-400">${t.exit_reason||'—'}</td>
        <td class="text-xs"><span class="${badgeClass(t.action)}">${t.action||'—'}</span></td>
      </tr>`;
    }).join('');
  }
}

setInterval(updateClock, 1000);
setInterval(fetchStatus, REFRESH_MS);
updateClock();
fetchStatus();
</script>
</body>
</html>
"""


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    """Read status.json written by the bot + pull trade history from SQLite."""
    data = {}

    # Bot live status
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                data = json.load(f)
        except Exception:
            pass

    # Trade history from DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute("""
            SELECT symbol, action, entry_price, exit_price,
                   net_pnl, exit_reason, profitable, trade_date
            FROM trades ORDER BY entry_time DESC LIMIT 20
        """)
        trades = [dict(r) for r in cur.fetchall()]
        data["trade_history"] = [
            {
                "symbol":       t["symbol"],
                "action":       t["action"],
                "entry_price":  t["entry_price"],
                "exit_price":   t["exit_price"],
                "net_pnl":      t["net_pnl"],
                "exit_reason":  t["exit_reason"],
                "date":         t["trade_date"],
            }
            for t in trades if t["exit_price"] is not None
        ]

        # Win rate stats
        cur.execute("SELECT COUNT(*) as total, SUM(profitable) as wins FROM trades WHERE exit_price IS NOT NULL")
        row = dict(cur.fetchone())
        total = row["total"] or 0
        wins  = int(row["wins"] or 0)
        data["total_trades"] = total
        data["wins"]         = wins
        data["losses"]       = total - wins
        data["win_rate"]     = round(wins / total, 4) if total > 0 else 0.0
        conn.close()
    except Exception as e:
        data.setdefault("trade_history", [])

    data["server_time"] = datetime.now().isoformat()
    return jsonify(data)


# ─── Status writer (called by bot) ───────────────────────────────────────────

def write_status(payload: dict):
    """Bot calls this to update the dashboard. Thread-safe via atomic write."""
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    tmp = STATUS_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, default=str)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        pass


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Trading Bot Dashboard")
    print(f"  http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"{'='*50}\n")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)
