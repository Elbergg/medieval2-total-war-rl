#!/usr/bin/env python3
"""Live Dashboard for Original Medieval II: Total War 100-Battle Trial (Port 8766)."""
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORKSPACE = os.environ.get("WORKSPACE", "/home/tomek/medieval2-docker/workspace")
PORT = int(os.environ.get("PORT", "8766"))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Original M2TW 100-Battle Trial Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 12px; flex-wrap: wrap; gap: 10px; }
  .title { font-size: 21px; font-weight: 700; color: #f59e0b; }
  .badges { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .badge { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 13px; }
  .badge span { font-weight: 700; color: #f8fafc; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
  .card.full { grid-column: 1 / -1; }
  .card h3 { margin: 0 0 12px 0; font-size: 15px; color: #cbd5e1; display: flex; justify-content: space-between; align-items: center; }
  .chart-box { position: relative; height: 280px; width: 100%; }
  .bar-bg { background: #0f172a; border: 1px solid #334155; border-radius: 8px; height: 14px; overflow: hidden; margin-bottom: 14px; }
  .bar-fill { background: linear-gradient(90deg, #f59e0b, #22c55e); height: 100%; width: 0%; transition: width 0.4s; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; text-align: left; }
  th { padding: 7px 8px; border-bottom: 1px solid #334155; color: #94a3b8; position: sticky; top: 0; background: #1e293b; }
  td { padding: 6px 8px; border-bottom: 1px solid #0f172a; }
  .btn { background: #0f172a; color: #38bdf8; border: 1px solid #38bdf8; border-radius: 6px; padding: 4px 9px; cursor: pointer; font-size: 11.5px; font-weight: 700; }
  .btn:hover { background: #1e3a5f; }
  a.sim-link { color: #38bdf8; text-decoration: none; font-size: 12.5px; border: 1px solid #334155; padding: 6px 10px; border-radius: 8px; background: #1e293b; }
</style>
</head>
<body>
  <div class="header">
    <div class="title">🏰 Original Medieval II: Total War — 100-Battle Frozen Trial</div>
    <div class="badges">
      <div class="badge">Progress: <span id="st-prog">-</span></div>
      <div class="badge">Win Rate: <span id="st-wr">-</span></div>
      <div class="badge">W / L / D: <span id="st-wld">-</span></div>
      <div class="badge">Avg Enemy Dmg: <span id="st-dmg">-</span></div>
      <div class="badge">Avg Wall Time: <span id="st-dur">-</span></div>
      <div class="badge">Elapsed: <span id="st-elap">-</span></div>
      <button id="trendToggle" onclick="toggleTrendMode()" style="background:#1e293b;color:#38bdf8;border:1px solid #38bdf8;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px;font-weight:700;">📈 View: Raw Points</button>
      <a class="sim-link" href="http://localhost:8765" target="_blank">⚔️ Sim Dashboard (:8765)</a>
    </div>
  </div>

  <div class="bar-bg"><div class="bar-fill" id="progBar"></div></div>

  <div class="grid">
    <div class="card">
      <h3>
        <span>1. Win Rate Over Trial Battles (%)</span>
        <span style="font-size:12px;color:#94a3b8;">Cumulative &amp; Rolling-10 (50% = Parity)</span>
      </h3>
      <div class="chart-box"><canvas id="wrChart"></canvas></div>
    </div>

    <div class="card">
      <h3>
        <span>2. Enemy Casualties Inflicted (%)</span>
        <span style="font-size:12px;color:#94a3b8;">Per-Battle &amp; Cumulative Mean</span>
      </h3>
      <div class="chart-box"><canvas id="dmgChart"></canvas></div>
    </div>

    <div class="card">
      <h3>
        <span>3. Completed Real-Engine Battles</span>
        <span style="font-size:12px;color:#94a3b8;" id="tblCount">0 battles</span>
      </h3>
      <div style="max-height:360px;overflow-y:auto;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Battle ID</th>
              <th>Worker</th>
              <th>Result</th>
              <th>Enemy Dmg</th>
              <th>Wall Time</th>
              <th>Replay</th>
            </tr>
          </thead>
          <tbody id="battleTbody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h3>
        <span>🎬 4. 2D Replay Viewer (Real M2TW Engine)</span>
        <span id="vidLabel" style="font-size:12px;color:#38bdf8;">Click "Watch" on any battle</span>
      </h3>
      <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;display:flex;justify-content:center;align-items:center;min-height:330px;">
        <video id="vidPlayer" controls loop style="max-width:100%;max-height:330px;border-radius:6px;display:none;"></video>
        <div id="vidPlaceholder" style="color:#64748b;font-size:13px;">Select any completed battle on the left to render &amp; play its 2D replay.</div>
      </div>
    </div>
  </div>

<script>
const commonOpts = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { mode: 'index', intersect: false },
  scales: {
    x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
    y: { min: 0, max: 100, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
  },
  plugins: { legend: { labels: { color: '#e2e8f0' } } }
};

const wrChart = new Chart(document.getElementById('wrChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Cumulative Win Rate (%)', data: [], borderColor: '#facc15', borderWidth: 2.8, pointRadius: 2, tension: 0.2 },
    { label: 'Rolling 10 Win Rate (%)', data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.12)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.25 },
    { label: '50% Parity Line', data: [], borderColor: '#64748b', borderDash: [5,5], borderWidth: 1.4, pointRadius: 0 }
  ]},
  options: commonOpts
});

const dmgChart = new Chart(document.getElementById('dmgChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Per-Battle Enemy Casualties (%)', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.12)', fill: true, borderWidth: 1.8, pointRadius: 3.5, pointBackgroundColor: [], tension: 0.2 },
    { label: 'Cumulative Avg Casualties (%)', data: [], borderColor: '#f59e0b', borderWidth: 2.6, pointRadius: 0, tension: 0.25 }
  ]},
  options: commonOpts
});

let trendMode = localStorage.getItem('m2tw_trial_trend') === '1';
let lastData = null;

function updateTrendBtn() {
  const btn = document.getElementById('trendToggle');
  if (!btn) return;
  if (trendMode) {
    btn.textContent = '📉 View: Trend Lines';
    btn.style.background = '#0284c7';
    btn.style.color = '#ffffff';
  } else {
    btn.textContent = '📈 View: Raw Points';
    btn.style.background = '#1e293b';
    btn.style.color = '#38bdf8';
  }
}
function toggleTrendMode() {
  trendMode = !trendMode;
  localStorage.setItem('m2tw_trial_trend', trendMode ? '1' : '0');
  updateTrendBtn();
  if (lastData) renderCharts(lastData);
}
function smoothSeries(arr, win) {
  if (!arr.length) return [];
  const w = Math.max(3, Math.min(win, Math.max(3, Math.floor(arr.length / 3))));
  const out = [], buf = [];
  let s = 0;
  for (const v of arr) {
    buf.push(v); s += v;
    if (buf.length > w) s -= buf.shift();
    out.push(Number((s / buf.length).toFixed(2)));
  }
  return out;
}

function renderCharts(d) {
  const bs = d.battles || [];
  const labels = bs.map((_, i) => '#' + (i + 1));
  const cumWr = [], roll10 = [], dmg = [], cumDmg = [], ptCols = [], parity = [];
  let wAcc = 0, dAcc = 0;
  for (let i = 0; i < bs.length; i++) {
    const b = bs[i];
    if (b.result === 1) wAcc++;
    dAcc += (b.enemy_dmg || 0) * 100;
    cumWr.push(Number((wAcc / (i + 1) * 100).toFixed(1)));
    const slice = bs.slice(Math.max(0, i - 9), i + 1);
    const rw = slice.filter(x => x.result === 1).length / slice.length * 100;
    roll10.push(Number(rw.toFixed(1)));
    dmg.push(Number(((b.enemy_dmg || 0) * 100).toFixed(1)));
    cumDmg.push(Number((dAcc / (i + 1)).toFixed(1)));
    ptCols.push(b.result === 1 ? '#22c55e' : (b.result === -1 ? '#f43f5e' : '#94a3b8'));
    parity.push(50);
  }
  const tens = trendMode ? 0.35 : 0.2;
  wrChart.data.labels = labels;
  wrChart.data.datasets[0].data = trendMode ? smoothSeries(cumWr, 8) : cumWr;
  wrChart.data.datasets[0].pointRadius = trendMode ? 0 : 2;
  wrChart.data.datasets[0].tension = tens;
  wrChart.data.datasets[1].data = trendMode ? smoothSeries(roll10, 8) : roll10;
  wrChart.data.datasets[1].tension = tens;
  wrChart.data.datasets[2].data = parity;
  wrChart.update();

  dmgChart.data.labels = labels;
  dmgChart.data.datasets[0].data = trendMode ? smoothSeries(dmg, 10) : dmg;
  dmgChart.data.datasets[0].pointRadius = trendMode ? 0 : 3.5;
  dmgChart.data.datasets[0].pointBackgroundColor = ptCols;
  dmgChart.data.datasets[0].tension = tens;
  dmgChart.data.datasets[1].data = trendMode ? smoothSeries(cumDmg, 8) : cumDmg;
  dmgChart.data.datasets[1].tension = tens;
  dmgChart.update();
}
updateTrendBtn();

async function watchReplay(bid) {
  const lbl = document.getElementById('vidLabel');
  const vid = document.getElementById('vidPlayer');
  const ph = document.getElementById('vidPlaceholder');
  lbl.textContent = `Rendering replay_${bid}.mp4...`;
  try {
    const res = await fetch(`/api/render?bid=${bid}`);
    const j = await res.json();
    if (!j.ok) {
      lbl.textContent = `Replay #${bid} unavailable: ${j.error}`;
      return;
    }
    ph.style.display = 'none';
    vid.style.display = 'block';
    vid.src = `/replays/replay_${bid}.mp4?t=${Date.now()}`;
    lbl.innerHTML = `Playing <b>Battle #${bid}</b> — <a href="/replays/replay_${bid}.mp4" target="_blank" style="color:#38bdf8;">Open .mp4</a>`;
    vid.play().catch(() => {});
  } catch (e) {
    lbl.textContent = `Error loading replay #${bid}`;
  }
}

function fmtSec(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}m ${r}s`;
}

async function refresh() {
  try {
    const res = await fetch('/api/trial');
    const d = await res.json();
    lastData = d;
    const n = d.completed || 0, tgt = d.target_battles || 100;
    document.getElementById('st-prog').textContent = `${n} / ${tgt}`;
    const wrEl = document.getElementById('st-wr');
    wrEl.textContent = `${(d.win_rate_pct || 0).toFixed(1)}%`;
    wrEl.style.color = (d.win_rate_pct || 0) >= 50 ? '#22c55e' : '#f43f5e';
    document.getElementById('st-wld').textContent = `${d.wins || 0}W / ${d.losses || 0}L / ${d.draws || 0}D`;
    document.getElementById('st-dmg').textContent = `${(d.avg_enemy_dmg_pct || 0).toFixed(1)}%`;
    document.getElementById('st-dur').textContent = `${(d.avg_duration_sec || 0).toFixed(1)}s`;
    document.getElementById('st-elap').textContent = fmtSec(d.elapsed_wall_sec || 0);
    document.getElementById('progBar').style.width = `${Math.min(100, (n / tgt) * 100)}%`;
    document.getElementById('tblCount').textContent = `${n} battles completed`;

    renderCharts(d);

    const bs = (d.battles || []).slice().reverse();
    document.getElementById('battleTbody').innerHTML = bs.map((b, idx) => {
      const num = (d.battles || []).length - idx;
      const col = b.result === 1 ? '#22c55e' : (b.result === -1 ? '#f43f5e' : '#94a3b8');
      const resTxt = (b.result === 1 ? 'WIN' : (b.result === -1 ? 'LOSS' : 'DRAW')) + (b.decisive_win || b.decisive_loss ? ' (DEC)' : '');
      const wShort = (b.container || '').replace('medieval2-', '');
      return `<tr>
        <td style="color:#64748b;">${num}</td>
        <td style="font-weight:700;color:#38bdf8;">#${b.battle_id}</td>
        <td style="color:#94a3b8;">${wShort}</td>
        <td style="font-weight:700;color:${col};">${resTxt}</td>
        <td>${((b.enemy_dmg || 0) * 100).toFixed(1)}%</td>
        <td style="color:#94a3b8;">${(b.duration_sec || 0).toFixed(1)}s</td>
        <td><button class="btn" onclick="watchReplay(${b.battle_id})">🎬 Watch</button></td>
      </tr>`;
    }).join('');
  } catch (e) { console.error(e); }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class TrialHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
        elif self.path.startswith("/api/trial"):
            p = os.path.join(WORKSPACE, "m2tw_trial_summary.json")
            data = json.load(open(p)) if os.path.exists(p) else {"target_battles": 100, "completed": 0, "battles": []}
            self._send(200, "application/json", json.dumps(data).encode("utf-8"))
        elif self.path.startswith("/api/render"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            bid = qs.get("bid", [""])[0]
            if not bid.isdigit():
                self._send(400, "application/json", b'{"ok":false,"error":"bad bid"}')
                return
            rjson = os.path.join(WORKSPACE, "replays", f"replay_{bid}.json")
            rmp4 = os.path.join(WORKSPACE, "replays", f"replay_{bid}.mp4")
            if not os.path.exists(rmp4):
                if not os.path.exists(rjson):
                    self._send(404, "application/json", b'{"ok":false,"error":"replay json not found"}')
                    return
                subprocess.run([sys.executable, os.path.join(WORKSPACE, "render_replay_video.py"), rjson, rmp4],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(rmp4):
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self._send(500, "application/json", b'{"ok":false,"error":"render failed"}')
        elif self.path.startswith("/replays/"):
            fname = os.path.basename(self.path.split("?")[0])
            fpath = os.path.join(WORKSPACE, "replays", fname)
            if os.path.exists(fpath):
                self._send(200, "video/mp4", open(fpath, "rb").read())
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    print(f"[+] Original M2TW Trial Monitor running at http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), TrialHandler).serve_forever()
