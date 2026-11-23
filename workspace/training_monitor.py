#!/usr/bin/env python3
"""
Live Self-Updating Training Monitor for Medieval II RL AI.
Serves an interactive dashboard at http://localhost:8765 that auto-refreshes every 5 seconds,
plotting:
  1. LLM Judge Penalty (total penalty & sub-penalties) + Judge 0..10 Scores & Reasoning
  2. Per-Battle Model Penalty / Reward Breakdown (casualty penalty, cohesion penalty/bonus, terminal reward)
  3. Rolling Win Rate (20 & 50 battles) and Enemy Casualties %
"""
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

WORKSPACE = os.environ.get("WORKSPACE", "/home/tomek/medieval2-docker/workspace")
PORT = int(os.environ.get("MONITOR_PORT", "8765"))

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>M2TW RL Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 12px; }
  .title { font-size: 22px; font-weight: 700; color: #f8fafc; }
  .badges { display: flex; gap: 12px; flex-wrap: wrap; }
  .badge { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 14px; font-size: 14px; }
  .badge span { font-weight: 700; color: #38bdf8; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
  .card.full { grid-column: 1 / -1; }
  .card h3 { margin: 0 0 12px 0; font-size: 16px; color: #cbd5e1; display: flex; justify-content: space-between; }
  .chart-box { position: relative; height: 290px; width: 100%; }
  .verdicts { max-height: 220px; overflow-y: auto; font-size: 13px; border-top: 1px solid #334155; margin-top: 10px; padding-top: 8px; }
  .v-row { padding: 6px 0; border-bottom: 1px solid #0f172a; }
  .v-row b { color: #f43f5e; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <div class="header">
    <div class="title">⚔️ M2TW RL Training Monitor (Auto-Updates Every 5s)</div>
    <div class="badges">
      <div class="badge">Phase: <span id="st-phase">-</span></div>
      <div class="badge">Latest Battle: <span id="st-battle">-</span></div>
      <div class="badge">Rolling 50 WR: <span id="st-wr50">-</span></div>
      <div class="badge">Rolling 500 WR: <span id="st-wr500">-</span></div>
      <div class="badge">Rolling 50 Enemy Dmg: <span id="st-dmg50">-</span></div>
      <div class="badge">Latest Judge Penalty: <span id="st-pen">-</span></div>
      <button id="trendToggle" onclick="toggleTrendMode()" style="background:#1e293b;color:#38bdf8;border:1px solid #38bdf8;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px;font-weight:700;">📈 View: Raw Points</button>
      <a href="/m2tw" style="background:#78350f;color:#fde68a;border:1px solid #f59e0b;border-radius:8px;padding:6px 12px;text-decoration:none;font-size:13px;font-weight:700;">🏰 Original M2TW Trial</a>
    </div>
  </div>

  <div class="grid">
    <div class="card full">
      <h3>
        <span>1. LLM Judge Penalty & Total Episode Penalty/Reward Over Battles</span>
        <span style="font-size:12px;color:#94a3b8;">Closer to 0 = Less Penalty</span>
      </h3>
      <div class="chart-box"><canvas id="penaltyChart"></canvas></div>
      <div id="battleDetail" style="margin-top:10px;padding:10px 14px;background:#0f172a;border:1px solid #334155;border-radius:8px;font-size:13px;display:none;"></div>
      <div class="verdicts" id="verdictList"></div>
    </div>

    <div class="card">
      <h3><span>2. LLM Judge Scores (0–10: Fairness, Historical, Reasonability)</span></h3>
      <div class="chart-box"><canvas id="judgeScoreChart"></canvas></div>
    </div>

    <div class="card">
      <h3><span>3. Rolling 50 & 500 Win Rate (%) + Enemy Casualties (%)</span></h3>
      <div class="chart-box"><canvas id="wrChart"></canvas></div>
    </div>

    <div class="card">
      <h3><span>3b. Imitation Warm-Start: Cross-Entropy vs Vanilla (lower = closer to vanilla)</span></h3>
      <div class="chart-box"><canvas id="ceChart"></canvas></div>
    </div>

    <div class="card">
      <h3>
        <span>3c. Benchmark Evaluations (Sim N=500 + Real M2TW N=20)</span>
        <span style="font-size:12px;color:#94a3b8;">Every 3k Sim Battles</span>
      </h3>
      <div id="evalHistoryList" style="max-height:290px;overflow-y:auto;"></div>
    </div>

    <div class="card full">
      <h3>
        <span>🔬 Phase 0: Learned M2TW Physics World Models — DamageNet &amp; MoraleNet (Self-Attention)</span>
        <span id="physicsMetaLabel" style="font-size:12px;color:#38bdf8;">Target: &lt; 1.0% full-battle rollout error on held-out real M2TW battles</span>
      </h3>
      <div id="physicsSummaryBar" style="display:flex;gap:14px;flex-wrap:wrap;padding:6px 10px;margin-bottom:8px;background:#0f172a;border:1px solid #334155;border-radius:6px;font-size:12px;color:#94a3b8;">Waiting for Phase 0 physics training rounds...</div>
      <div class="chart-box"><canvas id="physicsChart"></canvas></div>
    </div>

    <div class="card full">
      <h3>
        <span>🎬 4. Recorded 2D Tactical Battle Replays (.mp4)</span>
        <span id="curVideoLabel" style="font-size:13px;color:#38bdf8;">Select a battle below to watch</span>
      </h3>
      <div id="videoList" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;"></div>
      <div style="display:flex;justify-content:center;background:#0f172a;padding:12px;border-radius:8px;border:1px solid #334155;">
        <video id="mainVideoPlayer" controls loop style="max-width:960px;width:100%;border-radius:6px;"></video>
      </div>
    </div>

    <div class="card full">
      <h3>
        <span>🧠 5. Interactive 2-Layer Neural Network Snapshot Inspector (Saved Every 50 Battles)</span>
        <span id="nnMetaLabel" style="font-size:13px;color:#38bdf8;">Click any node (Rule, Top Neuron, or Action) to inspect weights</span>
      </h3>
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:10px;">
        <div id="nnRoleTabs" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
        <div id="nnIterList" style="display:flex;gap:6px;flex-wrap:wrap;max-height:70px;overflow-y:auto;"></div>
      </div>
      <div id="nnStatsBar" style="display:flex;gap:18px;flex-wrap:wrap;background:#0f172a;padding:8px 14px;border-radius:8px;border:1px solid #334155;font-size:13px;margin-bottom:10px;"></div>
      <div style="display:grid;grid-template-columns:2fr 1fr;gap:14px;">
        <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px;display:flex;justify-content:center;">
          <canvas id="nnCanvas" width="1040" height="560" style="width:100%;max-width:1040px;height:560px;cursor:pointer;"></canvas>
        </div>
        <div id="nnNodeDetail" style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;font-size:13px;overflow-y:auto;max-height:520px;">
          <b style="color:#facc15;">Click any neuron or action on the network diagram</b> to inspect its incoming & outgoing weights, KBANN rule couplings, recurrent memory, and learned delta since Battle #0.
        </div>
      </div>
      <div style="margin-top:12px;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
          <span style="font-size:13px;font-weight:700;color:#38bdf8;">📥 Input Features & Learned Sensitivity (Sorted by Layer-2 Weight Norm)</span>
          <input id="nnFeatFilter" type="text" oninput="renderNNStatsAndCanvas()" placeholder="Filter canvas & table (e.g. terrain, general, morale, g[17], grid)..." style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:5px 10px;font-size:12px;min-width:300px;" />
        </div>
        <div id="nnFeatTable" style="max-height:240px;overflow-y:auto;font-size:12px;"></div>
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
    y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
  },
  plugins: { legend: { labels: { color: '#e2e8f0' } } }
};

let latestBreakdowns = [];
function showBreakdown(idx) {
  const b = latestBreakdowns[idx];
  if (!b) return;
  const box = document.getElementById('battleDetail');
  box.style.display = 'block';
  const mag = Math.abs(b.term_base) + Math.abs(b.dec_bonus) + Math.abs(b.kills) + Math.abs(b.losses) + Math.abs(b.coh) + Math.abs(b.llm) || 1;
  const pct = (v) => (Math.abs(v) / mag * 100).toFixed(1) + '%';
  const resTag = b.win ? '<span style="color:#22c55e;font-weight:700;">WIN</span>' : '<span style="color:#f43f5e;font-weight:700;">LOSS</span>';
  const decTag = b.decisive ? ' (Decisive Rout)' : '';
  const llmStr = b.llm !== 0 ? `<span style="color:${b.llm >= 0 ? '#22c55e' : '#f43f5e'};">${b.llm >= 0 ? '+' : ''}${b.llm.toFixed(1)}</span> (${pct(b.llm)}, raw ${b.llm_raw >= 0 ? '+' : ''}${b.llm_raw.toFixed(1)} × 85)` : '0.0 (0.0%)';
  const repBase = b.src === 'm2tw' ? '/m2tw_replays' : '/replays';
  const vidHtml = b.has_replay
    ? `<div style="margin-top:8px;"><a href="${repBase}/replay_${b.bid}.mp4" target="_blank" style="color:#38bdf8;font-weight:700;text-decoration:none;">🎬 Open Replay Video (${b.label || ('#' + b.bid)}.mp4)</a>` +
      `<div style="margin-top:6px;"><video controls autoplay loop style="max-width:680px;width:100%;border-radius:6px;border:1px solid #334155;" src="${repBase}/replay_${b.bid}.mp4"></video></div></div>`
    : '';
  box.innerHTML =
    `<div style="display:flex;justify-content:space-between;margin-bottom:6px;">
       <span><b>Battle #${b.label || b.bid}</b> — ${resTag}${decTag} | Enemy Damage: <b>${b.dmg.toFixed(1)}%</b></span>
       <span><b>Combined Return / Unit: <span style="color:#facc15;">${b.total.toFixed(2)}</span></b></span>
     </div>
     <div style="display:flex;gap:16px;flex-wrap:wrap;color:#cbd5e1;">
       <span>Win/Loss Base: <b>${b.term_base.toFixed(1)}</b> (${pct(b.term_base)})</span>
       <span>Decisive Rout: <b>${b.dec_bonus.toFixed(1)}</b> (${pct(b.dec_bonus)})</span>
       <span>Unit Kills: <b style="color:#38bdf8;">+${b.kills.toFixed(2)}</b> (${pct(b.kills)})</span>
       <span>Unit Losses: <b style="color:#fb923c;">${b.losses.toFixed(2)}</b> (${pct(b.losses)})</span>
       <span>Cohesion: <b style="color:#34d399;">${b.coh >= 0 ? '+' : ''}${b.coh.toFixed(2)}</b> (${pct(b.coh)})</span>
       <span>LLM Judge: <b>${llmStr}</b></span>
     </div>` + (b.reasoning ? `<div style="margin-top:6px;color:#94a3b8;">Judge: ${b.reasoning}</div>` : '') + vidHtml;
}

const penaltyChart = new Chart(document.getElementById('penaltyChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'LLM Judge Score [-15..+15] (Judged Battles)', data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244,63,94,0.15)', borderWidth: 2.5, spanGaps: true, pointRadius: 4, tension: 0.2 },
    { label: 'In-Battle Cohesion + Casualty (All Battles)', data: [], borderColor: '#fb923c', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
    { label: 'Net Step Shaping Reward (Kills + Cohesion - Losses)', data: [], borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
    { label: 'Combined Episode Return / Unit (All Battles)', data: [], borderColor: '#facc15', borderWidth: 1.6, pointRadius: 3, pointHoverRadius: 6, tension: 0.2 },
    { label: 'Rolling 20 Combined Return', data: [], borderColor: '#22c55e', borderWidth: 2.5, pointRadius: 0, tension: 0.25 }
  ]},
  options: {
    ...commonOpts,
    onClick: (evt, elements) => {
      if (elements && elements.length > 0) {
        showBreakdown(elements[0].index);
      }
    }
  }
});

const judgeScoreChart = new Chart(document.getElementById('judgeScoreChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Fairness (0-10)', data: [], borderColor: '#34d399', borderWidth: 2, tension: 0.2 },
    { label: 'Historical Accuracy (0-10)', data: [], borderColor: '#60a5fa', borderWidth: 2, tension: 0.2 },
    { label: 'Reasonability (0-10)', data: [], borderColor: '#c084fc', borderWidth: 2, tension: 0.2 }
  ]},
  options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, max: 10, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } } }
});

const wrChart = new Chart(document.getElementById('wrChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Rolling 50 Win Rate (%)', data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.12)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.2 },
    { label: 'Rolling 500 Win Rate (%)', data: [], borderColor: '#facc15', borderWidth: 2.8, pointRadius: 0, tension: 0.25 },
    { label: 'Rolling 50 Enemy Casualties (%)', data: [], borderColor: '#38bdf8', borderDash: [4,4], borderWidth: 2, pointRadius: 0, tension: 0.2 }
  ]},
  options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, max: 100, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } } }
});

const physicsChart = new Chart(document.getElementById('physicsChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'DamageNet Rollout Err (%)', data: [], borderColor: '#f43f5e', borderWidth: 2.5, pointRadius: 3, tension: 0.2 },
    { label: 'DamageNet Tick Err (%)', data: [], borderColor: '#fb923c', borderDash: [4,3], borderWidth: 2, pointRadius: 2, tension: 0.2 },
    { label: 'MoraleNet Rollout Err (%)', data: [], borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 3, tension: 0.2 },
    { label: 'MoraleNet Tick Err (%)', data: [], borderColor: '#a78bfa', borderDash: [4,3], borderWidth: 2, pointRadius: 2, tension: 0.2 },
    { label: 'Target (1.0%)', data: [], borderColor: '#22c55e', borderDash: [6,6], borderWidth: 2.2, pointRadius: 3 }
  ]},
  options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } } }
});

const ceChart = new Chart(document.getElementById('ceChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Sim Imitation CE (per ~50 battles)', data: [], borderColor: '#f472b6', backgroundColor: 'rgba(244,114,182,0.12)', fill: true, borderWidth: 2.4, pointRadius: 2, tension: 0.2 },
    { label: 'M2TW Vanilla AI Imitation CE', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.15)', fill: false, borderWidth: 2.6, pointRadius: 3, tension: 0.2 }
  ]},
  options: { ...commonOpts, scales: { ...commonOpts.scales, y: { beginAtZero: false, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } } }
});

let trendMode = localStorage.getItem('m2tw_trend_mode') === '1';
let lastMetricsData = null;

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
  localStorage.setItem('m2tw_trend_mode', trendMode ? '1' : '0');
  updateTrendBtn();
  if (lastMetricsData) renderAllCharts(lastMetricsData);
}

function smoothSeries(arr, maxWin) {
  if (!arr || !arr.length) return [];
  let valid = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] !== null && arr[i] !== undefined && !Number.isNaN(arr[i])) valid++;
  }
  const w = Math.max(3, Math.min(maxWin, Math.max(3, Math.floor(valid / 5))));
  const out = [];
  const buf = [];
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v === null || v === undefined || Number.isNaN(v)) {
      out.push(null);
      continue;
    }
    buf.push(v);
    sum += v;
    if (buf.length > w) sum -= buf.shift();
    out.push(Number((sum / buf.length).toFixed(3)));
  }
  return out;
}

function renderAllCharts(d) {
  const prep = (arr, win) => trendMode ? smoothSeries(arr || [], win) : (arr || []);
  const tens = trendMode ? 0.35 : 0.2;

  const ce = d.imitation_ce || [];
  const m2ce = d.m2tw_imitation_ce || [];
  const ceLabels = ce.map(e => 'Sim#' + e.battle).concat(m2ce.map(e => 'M2#' + e.battle));
  const simCeVals = prep(ce.map(e => e.ce), 12).concat(m2ce.map(() => null));
  const m2CeVals = ce.map(() => null).concat(prep(m2ce.map(e => e.ce), 8));
  ceChart.data.labels = ceLabels;
  ceChart.data.datasets[0].data = simCeVals;
  ceChart.data.datasets[0].pointRadius = trendMode ? 0 : 2;
  ceChart.data.datasets[0].tension = tens;
  ceChart.data.datasets[1].data = m2CeVals;
  ceChart.data.datasets[1].pointRadius = trendMode ? 0 : 3;
  ceChart.data.datasets[1].tension = tens;
  ceChart.update();

  const rawPenRadii = [4, 0, 0, 3, 0];
  penaltyChart.data.labels = d.battles.map(b => '#' + b);
  penaltyChart.data.datasets[0].data = prep(d.llm_penalty_series, 15);
  penaltyChart.data.datasets[1].data = prep(d.step_penalty_series, 60);
  penaltyChart.data.datasets[2].data = prep(d.step_net_series, 60);
  penaltyChart.data.datasets[3].data = prep(d.combined_series, 60);
  penaltyChart.data.datasets[4].data = prep(d.combined_avg20_series, 60);
  penaltyChart.data.datasets.forEach((ds, i) => {
    ds.pointRadius = trendMode ? 0 : rawPenRadii[i];
    ds.tension = tens;
  });
  penaltyChart.update();

  judgeScoreChart.data.labels = d.judge_battles.map(b => '#' + b);
  judgeScoreChart.data.datasets[0].data = prep(d.judge_fair, 15);
  judgeScoreChart.data.datasets[1].data = prep(d.judge_hist, 15);
  judgeScoreChart.data.datasets[2].data = prep(d.judge_reas, 15);
  judgeScoreChart.data.datasets.forEach(ds => {
    ds.pointRadius = trendMode ? 0 : 3;
    ds.tension = tens;
  });
  judgeScoreChart.update();

  wrChart.data.labels = d.battles.map(b => '#' + b);
  wrChart.data.datasets[0].data = prep(d.wr50_series, 60);
  wrChart.data.datasets[1].data = prep(d.wr500_series || [], 60);
  wrChart.data.datasets[2].data = prep(d.dmg50_series, 60);
  wrChart.data.datasets.forEach(ds => {
    ds.pointRadius = 0;
    ds.tension = tens;
  });
  wrChart.update();
}
updateTrendBtn();

async function refresh() {
  try {
    const res = await fetch('/api/metrics');
    const d = await res.json();
    lastMetricsData = d;
    document.getElementById('st-phase').textContent = d.phase;
    document.getElementById('st-battle').textContent = '#' + d.latest_battle;
    document.getElementById('st-wr50').textContent = d.wr50.toFixed(1) + '%';
    document.getElementById('st-wr500').textContent = (d.wr500 || 0).toFixed(1) + '%';
    document.getElementById('st-dmg50').textContent = d.dmg50.toFixed(1) + '%';
    document.getElementById('st-pen').textContent = d.latest_judge_penalty !== null ? d.latest_judge_penalty.toFixed(1) : '-';

    const ph = d.physics_history || [];
    const phBar = document.getElementById('physicsSummaryBar');
    if (phBar && ph.length) {
      const lp = ph[ph.length - 1];
      const tgt = lp.target_pct || 1.0;
      if (!lp.converged) document.getElementById('st-phase').textContent = '0 (Physics)';
      const dmgOk = lp.dmg_cum_err_pct < tgt;
      const morOk = lp.mor_final_err_pct < tgt;
      document.getElementById('physicsMetaLabel').textContent = `Target: < ${tgt.toFixed(1)}% full-battle rollout error on held-out real M2TW battles`;
      phBar.innerHTML = `
        <span><b>Phase 0 Round:</b> <b style="color:#f8fafc;">#${lp.round}</b></span>
        <span><b>Real M2TW Battles:</b> <b style="color:#38bdf8;">${lp.battles}</b> (${lp.train_ticks} train / ${lp.val_ticks} val ticks)</span>
        <span><b>DamageNet:</b> rollout <b style="color:${dmgOk ? '#22c55e' : '#f43f5e'};">${lp.dmg_cum_err_pct.toFixed(2)}%</b> | tick <b style="color:#fb923c;">${lp.dmg_val_mae_pct.toFixed(2)}%</b></span>
        <span><b>MoraleNet:</b> rollout <b style="color:${morOk ? '#22c55e' : '#38bdf8'};">${lp.mor_final_err_pct.toFixed(2)}%</b> | tick <b style="color:#a78bfa;">${lp.mor_val_mae_pct.toFixed(2)}%</b></span>
        <span><b>Status:</b> <b style="color:${lp.converged ? '#22c55e' : '#facc15'};">${lp.converged ? '✅ CONVERGED (<' + tgt.toFixed(1) + '%)' : '⏳ Training...'}</b></span>`;
      physicsChart.data.labels = ph.map(r => `R${r.round} (${r.battles}b)`);
      physicsChart.data.datasets[0].data = ph.map(r => r.dmg_cum_err_pct);
      physicsChart.data.datasets[1].data = ph.map(r => r.dmg_val_mae_pct);
      physicsChart.data.datasets[2].data = ph.map(r => r.mor_final_err_pct);
      physicsChart.data.datasets[3].data = ph.map(r => r.mor_val_mae_pct);
      physicsChart.data.datasets[4].label = `Target (${tgt.toFixed(1)}%)`;
      physicsChart.data.datasets[4].pointRadius = ph.length < 2 ? 4 : 0;
      physicsChart.data.datasets[4].data = ph.map(r => r.target_pct || 1.0);
      physicsChart.update('none');
    }

    const ev = d.eval_history || [];
    const evBox = document.getElementById('evalHistoryList');
    if (evBox) {
      if (!ev.length) {
        evBox.innerHTML = '<div style="color:#64748b;font-size:13px;padding:12px;">No benchmark evaluations recorded yet (runs every 3,000 sim battles).</div>';
      } else {
        const latest = ev[ev.length - 1];
        const m2Entries = ev.filter(h => h.m2tw_wr !== undefined);
        const latestM2 = m2Entries.length ? m2Entries[m2Entries.length - 1] : null;
        const bestSim = Math.max(...ev.map(h => h.overall_wr || 0));
        const bestM2 = m2Entries.length ? Math.max(...m2Entries.map(h => h.m2tw_wr || 0)) : null;
        const genWr = latestM2
          ? ((latest.overall_wr + latestM2.m2tw_wr) / 2.0).toFixed(1) + '%'
          : latest.overall_wr.toFixed(1) + '%';
        const sumBar = `
          <div style="display:flex;gap:14px;flex-wrap:wrap;padding:6px 8px;margin-bottom:8px;background:#0f172a;border:1px solid #334155;border-radius:6px;font-size:12px;">
            <span><b>General (Avg):</b> <b style="color:#38bdf8;">${genWr}</b></span>
            <span><b>Sim (Vanilla):</b> <b style="color:${latest.overall_wr >= 50 ? '#22c55e' : '#facc15'};">${latest.overall_wr.toFixed(1)}%</b> <span style="color:#64748b;">(best ${bestSim.toFixed(1)}%)</span></span>
            <span><b>Real M2TW (20):</b> <b style="color:${latestM2 && latestM2.m2tw_wr >= 50 ? '#22c55e' : '#fbbf24'};">${latestM2 ? latestM2.m2tw_wr.toFixed(1) + '% (' + (latestM2.m2tw_wins ?? 0) + 'W/' + (latestM2.m2tw_losses ?? 0) + 'L)' : 'pending #3k'}</b>${bestM2 !== null ? ' <span style="color:#64748b;">(best ' + bestM2.toFixed(1) + '%)</span>' : ''}</span>
          </div>`;
        evBox.innerHTML = sumBar + `
          <table style="width:100%;border-collapse:collapse;font-size:12px;text-align:left;">
            <thead>
              <tr style="border-bottom:1px solid #334155;color:#94a3b8;">
                <th style="padding:6px;">Battle</th>
                <th style="padding:6px;color:#38bdf8;">General WR</th>
                <th style="padding:6px;color:#34d399;">Sim WR / Dmg</th>
                <th style="padding:6px;color:#fbbf24;">M2TW WR / Dmg (20)</th>
                <th style="padding:6px;">Field</th>
                <th style="padding:6px;">River</th>
                <th style="padding:6px;">Assault</th>
                <th style="padding:6px;">Defense</th>
              </tr>
            </thead>
            <tbody>
              ${ev.map(h => {
                const hasM2 = h.m2tw_wr !== undefined;
                const combWr = hasM2 ? ((h.overall_wr + h.m2tw_wr) / 2.0) : h.overall_wr;
                const m2Cell = hasM2
                  ? `<b style="color:${h.m2tw_wr >= 50 ? '#22c55e' : '#fbbf24'};">${h.m2tw_wr.toFixed(1)}%</b> <span style="color:#94a3b8;">(${h.m2tw_wins ?? 0}W/${h.m2tw_losses ?? 0}L · ${(h.m2tw_dmg ?? 0).toFixed(0)}%)</span>`
                  : '<span style="color:#64748b;">-</span>';
                return `
                <tr style="border-bottom:1px solid #1e293b;">
                  <td style="padding:6px;font-weight:700;color:#38bdf8;">#${h.battle}</td>
                  <td style="padding:6px;font-weight:700;color:${combWr >= 50 ? '#22c55e' : '#e2e8f0'};">${combWr.toFixed(1)}%</td>
                  <td style="padding:6px;"><b style="color:${h.overall_wr >= 50 ? '#22c55e' : '#facc15'};">${h.overall_wr.toFixed(1)}%</b> <span style="color:#94a3b8;">(${h.overall_dmg.toFixed(0)}%)</span></td>
                  <td style="padding:6px;">${m2Cell}</td>
                  <td style="padding:6px;color:#94a3b8;">${h.by_terrain?.open_field?.wr ?? '-'}%</td>
                  <td style="padding:6px;color:#94a3b8;">${h.by_terrain?.river_crossing?.wr ?? '-'}%</td>
                  <td style="padding:6px;color:#94a3b8;">${h.by_terrain?.siege_assault?.wr ?? '-'}%</td>
                  <td style="padding:6px;color:#94a3b8;">${h.by_terrain?.siege_defense?.wr ?? '-'}%</td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        `;
      }
    }

    latestBreakdowns = d.breakdowns || [];
    renderAllCharts(d);

    const vBox = document.getElementById('verdictList');
    vBox.innerHTML = d.recent_verdicts.map(v =>
      `<div class="v-row">#${v.bid} — <b>Penalty: ${v.total}</b> (Fair: ${v.fair}/10, Hist: ${v.hist}/10, Reas: ${v.reas}/10) — ${v.reasoning || ''}</div>`
    ).join('');

    const vidList = document.getElementById('videoList');
    const allRep = (d.breakdowns || []).filter(b => b.has_replay);
    const simRep = allRep.filter(b => (b.src || 'sim') !== 'm2tw').slice(-20);
    const m2Rep = allRep.filter(b => b.src === 'm2tw').slice(-20);
    const repBids = simRep.concat(m2Rep);
    const btypeTag = (bt) => bt === 'm2tw' ? '🎮 M2TW' : (bt === 'river_crossing' ? '🌊 RIVER' : (bt === 'siege_assault' ? '🏰 SIEGE ATK' : (bt === 'siege_defense' ? '🛡️ SIEGE DEF' : '⚔️ FIELD')));
    vidList.innerHTML = repBids.map(b => {
      const col = b.win ? '#22c55e' : '#f43f5e';
      return `<button onclick="playReplay(${b.bid}, ${b.win}, ${b.dmg}, '${b.btype || 'open_field'}', '${b.src || 'sim'}')" style="background:#0f172a;color:${col};border:1px solid ${col};border-radius:6px;padding:6px 10px;cursor:pointer;font-weight:700;font-size:12px;">🎬 #${b.label || b.bid} ${btypeTag(b.btype)} (${b.win ? 'W' : 'L'} ${b.dmg}%)</button>`;
    }).join('');
    if (repBids.length > 0 && !window._loadedInitialVid) {
      window._loadedInitialVid = true;
      const last = repBids[repBids.length - 1];
      playReplay(last.bid, last.win, last.dmg, last.btype || 'open_field', last.src || 'sim');
    }
    updateNNControls(d.nn_iters || []);
  } catch (e) { console.error(e); }
}

const NN_ROLES = [
  { id: '0', name: 'line_inf' },
  { id: '1', name: 'spear_inf' },
  { id: '2', name: 'missile_foot' },
  { id: '3', name: 'shock_cav' },
  { id: '4', name: 'missile_cav' },
  { id: '5', name: 'general' },
  { id: '6', name: 'artillery' },
  { id: 'commander', name: 'Commander RNN' }
];
let curNNRole = '0';
let curNNIter = 'latest';
let curNNData = null;
let selectedNNNode = null;
let nnNodes = [];

function updateNNControls(iters) {
  const roleBox = document.getElementById('nnRoleTabs');
  roleBox.innerHTML = NN_ROLES.map(r => {
    const active = curNNRole === r.id;
    return `<button onclick="selectNNRole('${r.id}')" style="background:${active ? '#0284c7' : '#0f172a'};color:#f8fafc;border:1px solid ${active ? '#38bdf8' : '#334155'};border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px;font-weight:700;">${r.name}</button>`;
  }).join('');

  const allIters = ['latest', ...iters.slice(-30).reverse()];
  const iterBox = document.getElementById('nnIterList');
  iterBox.innerHTML = allIters.map(it => {
    const active = String(curNNIter) === String(it);
    const lbl = it === 'latest' ? '⚡ Latest' : `#${it}`;
    return `<button onclick="selectNNIter('${it}')" style="background:${active ? '#ca8a04' : '#0f172a'};color:#f8fafc;border:1px solid ${active ? '#facc15' : '#334155'};border-radius:6px;padding:4px 9px;cursor:pointer;font-size:12px;font-weight:600;">${lbl}</button>`;
  }).join('');

  if (!window._nnLoadedOnce || curNNIter === 'latest') {
    window._nnLoadedOnce = true;
    loadNNSnapshot();
  }
}

function selectNNRole(r) { curNNRole = r; selectedNNNode = null; loadNNSnapshot(); }
function selectNNIter(it) { curNNIter = it; loadNNSnapshot(); }

async function loadNNSnapshot() {
  try {
    const res = await fetch(`/api/nn_snapshot?iter=${curNNIter}&role=${curNNRole}`);
    curNNData = await res.json();
    renderNNStatsAndCanvas();
  } catch (e) { console.error(e); }
}

function renderNNStatsAndCanvas() {
  if (!curNNData || !curNNData.ok) return;
  const st = document.getElementById('nnStatsBar');
  const kbTag = curNNData.kb_unfrozen
    ? `<span style="color:#22c55e;font-weight:700;">UNFROZEN (Δmax=${curNNData.kb_delta_max.toFixed(4)})</span>`
    : `<span style="color:#38bdf8;font-weight:700;">FROZEN (Δmax=${curNNData.kb_delta_max.toFixed(5)})</span>`;
  const adaptTag = curNNData.has_adapter
    ? `<span>Layer 3 (M2TW Adapter) max|W_adapt_out|: <b style="color:#fbbf24;">${(curNNData.adapt_out_max || 0).toFixed(4)}</b> | Swing (∑|W_adapt_out|): <b style="color:#f59e0b;">${(curNNData.adapt_l1_max || 0).toFixed(2)}</b> (mean: <b>${(curNNData.adapt_l1_mean || 0).toFixed(2)}</b>)</span>`
    : '';
  st.innerHTML =
    `<span>Role: <b style="color:#38bdf8;">${curNNData.role_name}</b></span>` +
    `<span>Snapshot: <b style="color:#facc15;">${curNNData.iter_label}</b></span>` +
    `<span>Layer 1 (KBANN Rules): ${kbTag}</span>` +
    `<span>Layer 1 -> Layer 2 ||W_top_kb|| max: <b style="color:#c084fc;">${curNNData.w_top_kb_max.toFixed(3)}</b> (Δmax: <b>${curNNData.top_kb_delta_max.toFixed(3)}</b>)</span>` +
    `<span>Layer 2 -> Output Logit Swing (∑|W2|): <b style="color:#34d399;">${curNNData.top_l1_max.toFixed(2)}</b> (mean: <b>${curNNData.top_l1_mean.toFixed(2)}</b>)</span>` +
    adaptTag;

  const cv = document.getElementById('nnCanvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);

  const allFeats = curNNData.input_features || [];
  const q = (document.getElementById('nnFeatFilter')?.value || '').trim().toLowerCase();
  const matchedFeats = q
    ? allFeats.filter(f => f.name.toLowerCase().includes(q) || String(f.idx).includes(q) || f.group.toLowerCase().includes(q))
    : allFeats;
  const shownFeats = matchedFeats.slice(0, 24);

  const hasAdapt = !!curNNData.has_adapter && (curNNData.adapt_neurons || []).length > 0;
  const nIn = shownFeats.length;
  const nKb = curNNData.kb_rules.length;
  const nTop = curNNData.top_neurons.length;
  const nAdapt = hasAdapt ? curNNData.adapt_neurons.length : 0;
  const nAct = curNNData.actions.length;

  const xIn = hasAdapt ? 170 : 185;
  const xKb = hasAdapt ? 365 : 435;
  const xTop = hasAdapt ? 545 : 675;
  const xAdapt = 715;
  const xAct = hasAdapt ? 875 : 875;
  nnNodes = [];
  const inNodes = [], kbNodes = [], topNodes = [], adaptNodes = [], actNodes = [];

  for (let k = 0; k < nIn; k++) {
    const y = 34 + k * ((H - 54) / Math.max(1, nIn - 1));
    const nd = { col: 'in', idx: shownFeats[k].idx, pos: k, x: xIn, y: y, r: 6.5, data: shownFeats[k] };
    inNodes.push(nd); nnNodes.push(nd);
  }
  for (let i = 0; i < nKb; i++) {
    const y = 34 + i * ((H - 54) / Math.max(1, nKb - 1));
    const nd = { col: 'kb', idx: i, x: xKb, y: y, r: 8.5, data: curNNData.kb_rules[i] };
    kbNodes.push(nd); nnNodes.push(nd);
  }
  for (let j = 0; j < nTop; j++) {
    const y = 28 + j * ((H - 44) / Math.max(1, nTop - 1));
    const nd = { col: 'top', idx: j, x: xTop, y: y, r: 6.0, data: curNNData.top_neurons[j] };
    topNodes.push(nd); nnNodes.push(nd);
  }
  for (let m = 0; m < nAdapt; m++) {
    const y = 34 + m * ((H - 54) / Math.max(1, nAdapt - 1));
    const nd = { col: 'adapt', idx: m, x: xAdapt, y: y, r: 6.8, data: curNNData.adapt_neurons[m] };
    adaptNodes.push(nd); nnNodes.push(nd);
  }
  for (let a = 0; a < nAct; a++) {
    const y = 40 + a * ((H - 68) / Math.max(1, nAct - 1));
    const nd = { col: 'act', idx: a, x: xAct, y: y, r: 9.5, data: curNNData.actions[a] };
    actNodes.push(nd); nnNodes.push(nd);
  }

  // Column headers
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 12px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(`Input Features (${nIn}/${allFeats.length})`, xIn - 25, 16);
  ctx.fillText(`L1: KBANN (${nKb})`, xKb, 16);
  ctx.fillText(`L2: 85k RNN (${nTop})`, xTop, 16);
  if (hasAdapt) {
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(`L3: Adapter (${nAdapt})`, xAdapt, 16);
    ctx.fillStyle = '#94a3b8';
  }
  ctx.fillText(`Output Actions (${nAct})`, xAct + 30, 16);
  ctx.textAlign = 'left';

  const W_kb = curNNData.W_kb || [];           // [nKb][nAllIn]
  const W_top_x = curNNData.W_top_x || [];     // [nTop][nTopIn]
  const W_top_kb = curNNData.W_top_kb;         // [nTop][nKb]
  const W2 = curNNData.W2;                     // [nAct][nTop]
  const W_adapt_h = curNNData.W_adapt_h || []; // [nAdapt][nTop]
  const W_adapt_out = curNNData.W_adapt_out || []; // [nAct][nAdapt]

  // Edges: Input Features -> Layer 1 KBANN (W_kb)
  for (let k = 0; k < nIn; k++) {
    const fIdx = inNodes[k].idx;
    for (let i = 0; i < nKb; i++) {
      const w = (W_kb[i] && W_kb[i][fIdx]) || 0;
      const isSel = selectedNNNode && ((selectedNNNode.col === 'in' && selectedNNNode.idx === fIdx) || (selectedNNNode.col === 'kb' && selectedNNNode.idx === i));
      if (!isSel && Math.abs(w) < 1.2) continue;
      if (selectedNNNode && !isSel) continue;
      ctx.beginPath();
      ctx.moveTo(xIn, inNodes[k].y);
      ctx.lineTo(xKb, kbNodes[i].y);
      const alpha = isSel ? Math.min(0.95, 0.25 + Math.abs(w) * 0.25) : Math.min(0.35, Math.abs(w) * 0.08);
      ctx.strokeStyle = w >= 0 ? `rgba(244,114,182,${alpha})` : `rgba(251,113,133,${alpha})`;
      ctx.lineWidth = isSel ? Math.min(3.0, 0.8 + Math.abs(w) * 0.25) : 0.9;
      ctx.stroke();
    }
  }

  // Edges: Direct Input Features -> Layer 2 Skip Connections (W_top_x)
  for (let k = 0; k < nIn; k++) {
    const fIdx = inNodes[k].idx;
    for (let j = 0; j < nTop; j++) {
      const w = (W_top_x[j] && fIdx < W_top_x[j].length) ? W_top_x[j][fIdx] : 0;
      const isSel = selectedNNNode && ((selectedNNNode.col === 'in' && selectedNNNode.idx === fIdx) || (selectedNNNode.col === 'top' && selectedNNNode.idx === j));
      if (!isSel && Math.abs(w) < 0.35) continue;
      if (selectedNNNode && !isSel) continue;
      ctx.beginPath();
      ctx.moveTo(xIn, inNodes[k].y);
      ctx.quadraticCurveTo((xIn + xTop) * 0.5, (inNodes[k].y + topNodes[j].y) * 0.5 - 24, xTop, topNodes[j].y);
      const alpha = isSel ? Math.min(0.9, 0.22 + Math.abs(w) * 1.4) : Math.min(0.22, Math.abs(w) * 0.35);
      ctx.strokeStyle = w >= 0 ? `rgba(192,132,252,${alpha})` : `rgba(251,146,60,${alpha})`;
      ctx.lineWidth = isSel ? Math.min(2.8, 0.7 + Math.abs(w) * 2.5) : 0.75;
      ctx.stroke();
    }
  }

  // Edges: Layer 1 -> Layer 2 (W_top_kb)
  for (let j = 0; j < nTop; j++) {
    for (let i = 0; i < nKb; i++) {
      const w = W_top_kb[j][i];
      const isSel = selectedNNNode && ((selectedNNNode.col === 'kb' && selectedNNNode.idx === i) || (selectedNNNode.col === 'top' && selectedNNNode.idx === j));
      if (!isSel && Math.abs(w) < 0.18) continue;
      if (selectedNNNode && !isSel) continue;
      ctx.beginPath();
      ctx.moveTo(xKb, kbNodes[i].y);
      ctx.lineTo(xTop, topNodes[j].y);
      const alpha = isSel ? Math.min(0.95, 0.25 + Math.abs(w) * 1.2) : Math.min(0.45, Math.abs(w) * 0.6);
      ctx.strokeStyle = w >= 0 ? `rgba(52,211,153,${alpha})` : `rgba(244,63,94,${alpha})`;
      ctx.lineWidth = isSel ? Math.min(3.5, 0.8 + Math.abs(w) * 2.0) : 1.0;
      ctx.stroke();
    }
  }

  // Edges: Layer 2 -> Output Actions (W2)
  for (let a = 0; a < nAct; a++) {
    for (let j = 0; j < nTop; j++) {
      const w = W2[a][j];
      const isSel = selectedNNNode && ((selectedNNNode.col === 'top' && selectedNNNode.idx === j) || (selectedNNNode.col === 'act' && selectedNNNode.idx === a));
      if (!isSel && Math.abs(w) < (hasAdapt ? 0.08 : 0.025)) continue;
      if (selectedNNNode && !isSel) continue;
      ctx.beginPath();
      ctx.moveTo(xTop, topNodes[j].y);
      if (hasAdapt) {
        ctx.quadraticCurveTo((xTop + xAct) * 0.5, (topNodes[j].y + actNodes[a].y) * 0.5 - 26, xAct, actNodes[a].y);
      } else {
        ctx.lineTo(xAct, actNodes[a].y);
      }
      const alpha = isSel ? Math.min(0.95, 0.3 + Math.abs(w) * 3.5) : Math.min(hasAdapt ? 0.28 : 0.55, 0.08 + Math.abs(w) * 1.8);
      ctx.strokeStyle = w >= 0 ? `rgba(56,189,248,${alpha})` : `rgba(251,146,60,${alpha})`;
      ctx.lineWidth = isSel ? Math.min(4.0, 1.0 + Math.abs(w) * 6.0) : Math.min(2.0, 0.5 + Math.abs(w) * 3.0);
      ctx.stroke();
    }
  }

  // Edges: Layer 2 -> Layer 3 Adapter (W_adapt_h) & Layer 3 -> Output Actions (W_adapt_out)
  if (hasAdapt) {
    for (let m = 0; m < nAdapt; m++) {
      for (let j = 0; j < nTop; j++) {
        const w = (W_adapt_h[m] && W_adapt_h[m][j]) || 0;
        const isSel = selectedNNNode && ((selectedNNNode.col === 'top' && selectedNNNode.idx === j) || (selectedNNNode.col === 'adapt' && selectedNNNode.idx === m));
        if (!isSel && Math.abs(w) < 0.14) continue;
        if (selectedNNNode && !isSel) continue;
        ctx.beginPath();
        ctx.moveTo(xTop, topNodes[j].y);
        ctx.lineTo(xAdapt, adaptNodes[m].y);
        const alpha = isSel ? Math.min(0.9, 0.25 + Math.abs(w) * 2.5) : Math.min(0.28, Math.abs(w) * 0.9);
        ctx.strokeStyle = w >= 0 ? `rgba(251,191,36,${alpha})` : `rgba(244,63,94,${alpha})`;
        ctx.lineWidth = isSel ? Math.min(3.2, 0.8 + Math.abs(w) * 4.0) : 0.8;
        ctx.stroke();
      }
    }
    for (let a = 0; a < nAct; a++) {
      for (let m = 0; m < nAdapt; m++) {
        const w = (W_adapt_out[a] && W_adapt_out[a][m]) || 0;
        const isSel = selectedNNNode && ((selectedNNNode.col === 'adapt' && selectedNNNode.idx === m) || (selectedNNNode.col === 'act' && selectedNNNode.idx === a));
        if (!isSel && Math.abs(w) < 0.004) continue;
        if (selectedNNNode && !isSel) continue;
        ctx.beginPath();
        ctx.moveTo(xAdapt, adaptNodes[m].y);
        ctx.lineTo(xAct, actNodes[a].y);
        const alpha = isSel ? Math.min(0.95, 0.35 + Math.abs(w) * 8.0) : Math.min(0.75, 0.18 + Math.abs(w) * 6.0);
        ctx.strokeStyle = w >= 0 ? `rgba(250,204,21,${alpha})` : `rgba(244,63,94,${alpha})`;
        ctx.lineWidth = isSel ? Math.min(4.2, 1.0 + Math.abs(w) * 10.0) : Math.min(2.8, 0.7 + Math.abs(w) * 8.0);
        ctx.stroke();
      }
    }
  }

  // Draw nodes & labels
  for (const nd of nnNodes) {
    const isSel = selectedNNNode && selectedNNNode.col === nd.col && selectedNNNode.idx === nd.idx;
    ctx.beginPath();
    ctx.arc(nd.x, nd.y, isSel ? nd.r + 3 : nd.r, 0, Math.PI * 2);
    if (nd.col === 'in') {
      ctx.fillStyle = nd.data.group === 'Terrain' ? '#f59e0b' : '#db2777';
      ctx.strokeStyle = isSel ? '#facc15' : '#f9a8d4';
    } else if (nd.col === 'kb') {
      ctx.fillStyle = '#0284c7';
      ctx.strokeStyle = isSel ? '#facc15' : '#38bdf8';
    } else if (nd.col === 'top') {
      const mag = Math.min(1, (nd.data.out_norm || 0) * 3.0);
      ctx.fillStyle = `rgb(${Math.round(45 + 140 * mag)}, ${Math.round(70 + 130 * mag)}, 235)`;
      ctx.strokeStyle = isSel ? '#facc15' : '#93c5fd';
    } else if (nd.col === 'adapt') {
      const mag = Math.min(1, (nd.data.out_norm || 0) * 8.0);
      ctx.fillStyle = `rgb(${Math.round(120 + 130 * mag)}, ${Math.round(80 + 120 * mag)}, ${Math.round(20 + 20 * mag)})`;
      ctx.strokeStyle = isSel ? '#ffffff' : '#fbbf24';
    } else {
      const push = nd.data.res_norm || 0;
      ctx.fillStyle = push > 0.12 ? '#16a34a' : '#334155';
      ctx.strokeStyle = isSel ? '#facc15' : '#cbd5e1';
    }
    ctx.lineWidth = isSel ? 3 : 1.5;
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isSel ? '#facc15' : '#e2e8f0';
    ctx.font = isSel ? 'bold 10.5px system-ui' : '10.5px system-ui';
    if (nd.col === 'in') {
      ctx.textAlign = 'right';
      const shortName = nd.data.name.length > 22 ? nd.data.name.slice(0, 21) + '…' : nd.data.name;
      ctx.fillText(shortName, nd.x - 10, nd.y + 3.5);
    } else if (nd.col === 'kb') {
      ctx.textAlign = 'right';
      const shortLbl = nd.data.label.length > 18 ? nd.data.label.slice(0, 17) + '…' : nd.data.label;
      ctx.fillText(shortLbl, nd.x - 11, nd.y + 3.5);
    } else if (nd.col === 'adapt') {
      if (nd.idx % 2 === 0 || isSel) {
        ctx.textAlign = 'left';
        ctx.fillStyle = isSel ? '#facc15' : '#fbbf24';
        ctx.font = '10px system-ui';
        ctx.fillText(`A${nd.idx}`, nd.x + 9, nd.y + 3);
      }
    } else if (nd.col === 'act') {
      ctx.textAlign = 'left';
      const adStr = hasAdapt ? ` | Δ${(nd.data.adapt_l1_swing || 0).toFixed(2)}` : '';
      ctx.fillText(`${nd.data.label} (${nd.data.top_l1_swing.toFixed(2)}${adStr})`, nd.x + 13, nd.y + 3.5);
    } else if (nd.idx % 2 === 0 || isSel) {
      ctx.textAlign = 'left';
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px system-ui';
      ctx.fillText(`T${nd.idx}`, nd.x + 9, nd.y + 3);
    }
  }
  ctx.textAlign = 'left';
  if (selectedNNNode) showNNNodeDetail(selectedNNNode);
  renderNNFeatures();
}

function renderNNFeatures() {
  const box = document.getElementById('nnFeatTable');
  if (!box || !curNNData) return;
  const feats = curNNData.input_features || [];
  const q = (document.getElementById('nnFeatFilter')?.value || '').trim().toLowerCase();
  const filtered = q ? feats.filter(f => f.name.toLowerCase().includes(q) || String(f.idx).includes(q) || f.group.toLowerCase().includes(q)) : feats;
  const rows = filtered.slice(0, 80);
  if (!rows.length) {
    box.innerHTML = '<div style="color:#64748b;padding:8px;">No matching input features.</div>';
    return;
  }
  box.innerHTML = `
    <table style="width:100%;border-collapse:collapse;text-align:left;">
      <thead>
        <tr style="border-bottom:1px solid #334155;color:#94a3b8;">
          <th style="padding:5px 6px;">Idx</th>
          <th style="padding:5px 6px;">Group</th>
          <th style="padding:5px 6px;">Feature Name</th>
          <th style="padding:5px 6px;">L2 Direct Norm ||W_top_x[:,f]||</th>
          <th style="padding:5px 6px;">L1 KBANN Norm ||W_kb[:,f]||</th>
          <th style="padding:5px 6px;">L1 Learned Δ</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(f => `
          <tr style="border-bottom:1px solid #1e293b;">
            <td style="padding:4px 6px;color:#38bdf8;font-family:monospace;">[${f.idx}]</td>
            <td style="padding:4px 6px;color:#94a3b8;">${f.group}</td>
            <td style="padding:4px 6px;color:#e2e8f0;font-weight:600;">${f.name}</td>
            <td style="padding:4px 6px;color:#34d399;font-family:monospace;">${f.top_norm.toFixed(3)}</td>
            <td style="padding:4px 6px;color:#cbd5e1;font-family:monospace;">${f.kb_norm.toFixed(3)}</td>
            <td style="padding:4px 6px;color:#facc15;font-family:monospace;">${f.kb_delta.toFixed(3)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
    ${filtered.length > 80 ? `<div style="color:#64748b;padding:6px;">Showing top 80 of ${filtered.length} matching features. Type in the filter box to narrow down.</div>` : ''}
  `;
}

document.addEventListener('DOMContentLoaded', () => {
  const cv = document.getElementById('nnCanvas');
  if (!cv) return;
  cv.addEventListener('click', (evt) => {
    const rect = cv.getBoundingClientRect();
    const sx = (evt.clientX - rect.left) * (cv.width / rect.width);
    const sy = (evt.clientY - rect.top) * (cv.height / rect.height);
    let best = null, bestD = 22;
    for (const nd of nnNodes) {
      const d = Math.hypot(nd.x - sx, nd.y - sy);
      if (d < bestD) { bestD = d; best = nd; }
    }
    selectedNNNode = best;
    renderNNStatsAndCanvas();
  });
});

function showNNNodeDetail(nd) {
  const box = document.getElementById('nnNodeDetail');
  const d = nd.data;
  if (nd.col === 'in') {
    const fIdx = nd.idx;
    const W_kb = curNNData.W_kb || [];
    const W_top_x = curNNData.W_top_x || [];
    const kbLinks = [];
    for (let i = 0; i < W_kb.length; i++) {
      const w = (W_kb[i] && W_kb[i][fIdx]) || 0;
      if (Math.abs(w) > 1e-3) kbLinks.push({ rule: curNNData.kb_rules[i].label, w });
    }
    kbLinks.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const topLinks = [];
    for (let j = 0; j < W_top_x.length; j++) {
      const w = (W_top_x[j] && fIdx < W_top_x[j].length) ? W_top_x[j][fIdx] : 0;
      if (Math.abs(w) > 1e-3) topLinks.push({ idx: j, w });
    }
    topLinks.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    box.innerHTML =
      `<div style="font-size:15px;font-weight:700;color:#f472b6;margin-bottom:6px;">Input Feature [${fIdx}] — ${d.name}</div>` +
      `<div style="color:#cbd5e1;margin-bottom:8px;">Group: <b>${d.group}</b> | L2 Direct Norm: <b style="color:#34d399;">${d.top_norm.toFixed(3)}</b> | L1 KBANN Norm: <b>${d.kb_norm.toFixed(3)}</b> (Δ=${d.kb_delta.toFixed(3)})</div>` +
      `<div style="margin-bottom:8px;"><b style="color:#c084fc;">Strongest Direct Connections to Layer 2 (W_top_x[:, ${fIdx}]):</b><br>${topLinks.slice(0, 8).map(t => `• <b>T${t.idx}</b>: <span style="color:${t.w >= 0 ? '#22c55e' : '#f43f5e'};">${t.w >= 0 ? '+' : ''}${t.w.toFixed(3)}</span>`).join('<br>') || '<span style="color:#64748b;">None (grid features enter via Layer 1 only)</span>'}</div>` +
      `<div><b style="color:#38bdf8;">Strongest Connections to Layer 1 KBANN Rules (W_kb[:, ${fIdx}]):</b><br>${kbLinks.slice(0, 8).map(k => `• <b>${k.rule}</b>: <span style="color:${k.w >= 0 ? '#22c55e' : '#f43f5e'};">${k.w >= 0 ? '+' : ''}${k.w.toFixed(3)}</span>`).join('<br>') || '<span style="color:#64748b;">None</span>'}</div>`;
  } else if (nd.col === 'kb') {
    const fin = (d.feat_in || []).map(f => `• <code>[${f.idx}]</code> <b>${f.name}</b>: <span style="color:${f.w >= 0 ? '#22c55e' : '#f43f5e'};">${f.w >= 0 ? '+' : ''}${f.w.toFixed(3)}</span>`).join('<br>');
    box.innerHTML =
      `<div style="font-size:15px;font-weight:700;color:#38bdf8;margin-bottom:6px;">Layer 1 KBANN Rule — ${d.label}</div>` +
      `<div style="color:#cbd5e1;margin-bottom:8px;">Bias <code>b_kb = ${d.bias.toFixed(2)}</code> | Input L2 norm: <b>${d.in_norm.toFixed(2)}</b> | Learned ΔW_kb: <b>${d.delta_norm.toFixed(4)}</b></div>` +
      (fin ? `<div style="margin-bottom:8px;"><b style="color:#f472b6;">Top Input Features Driving Rule (W_kb[${nd.idx}, :]):</b><br>${fin}</div>` : '') +
      `<div style="margin-bottom:8px;"><b style="color:#34d399;">Strongest Couplings to Layer 2 Neurons (W_top_kb[:, ${nd.idx}]):</b><br>${d.top_links.map(t => `• <b>T${t.idx}</b>: ${t.w >= 0 ? '+' : ''}${t.w.toFixed(3)} (Δ=${t.dw >= 0 ? '+' : ''}${t.dw.toFixed(3)})`).join('<br>')}</div>` +
      `<div><b style="color:#facc15;">Effective Multi-Hop Rule -> Action Influence (W2 @ W_top_kb[:, ${nd.idx}]):</b><br>${d.eff_outs.map(o => `• <b>${o.act}</b>: <span style="color:${o.w >= 0 ? '#22c55e' : '#f43f5e'};">${o.w >= 0 ? '+' : ''}${o.w.toFixed(3)}</span>`).join('<br>')}</div>`;
  } else if (nd.col === 'top') {
    const fin = (d.feat_in || []).map(f => `• <code>[${f.idx}]</code> <b>${f.name}</b>: <span style="color:${f.w >= 0 ? '#22c55e' : '#f43f5e'};">${f.w >= 0 ? '+' : ''}${f.w.toFixed(3)}</span>`).join('<br>');
    box.innerHTML =
      `<div style="font-size:15px;font-weight:700;color:#c084fc;margin-bottom:6px;">Layer 2 Homogeneous Recurrent Neuron — T${nd.idx}</div>` +
      `<div style="color:#cbd5e1;margin-bottom:8px;">Bias <code>b_top = ${d.bias.toFixed(3)}</code> | Self-Memory <code>Wh[${nd.idx},${nd.idx}] = ${d.self_wh.toFixed(3)}</code> | Context ||W_top_x||: <b>${d.ctx_norm.toFixed(3)}</b> | Output ||W2[:,${nd.idx}]||: <b style="color:#34d399;">${d.out_norm.toFixed(4)}</b></div>` +
      (fin ? `<div style="margin-bottom:8px;"><b style="color:#f472b6;">Top Direct Input Features (W_top_x[${nd.idx}, :]):</b><br>${fin}</div>` : '') +
      `<div style="margin-bottom:8px;"><b style="color:#38bdf8;">Top Incoming Layer 1 KBANN Rules (W_top_kb[${nd.idx}, :]):</b><br>${d.kb_in.map(k => `• <b>${k.rule}</b>: ${k.w >= 0 ? '+' : ''}${k.w.toFixed(3)} (Δ=${k.dw >= 0 ? '+' : ''}${k.dw.toFixed(3)})`).join('<br>')}</div>` +
      `<div><b style="color:#facc15;">Layer 2 -> Output Action Weights (W2[:, ${nd.idx}]):</b><br>${d.act_out.map(a => `• <b>${a.act}</b>: <span style="color:${a.w >= 0 ? '#22c55e' : '#f43f5e'};">${a.w >= 0 ? '+' : ''}${a.w.toFixed(4)}</span>`).join('<br>')}</div>`;
  } else if (nd.col === 'adapt') {
    const fin = (d.feat_in || []).map(f => `• <code>[${f.idx}]</code> <b>${f.name}</b>: <span style="color:${f.w >= 0 ? '#22c55e' : '#f43f5e'};">${f.w >= 0 ? '+' : ''}${f.w.toFixed(3)}</span>`).join('<br>');
    box.innerHTML =
      `<div style="font-size:15px;font-weight:700;color:#fbbf24;margin-bottom:6px;">Layer 3 Sim-to-Real Adapter Neuron — A${nd.idx}</div>` +
      `<div style="color:#cbd5e1;margin-bottom:8px;">Bias <code>b_adapt = ${d.bias.toFixed(4)}</code> | ||W_adapt_h||: <b>${d.h_norm.toFixed(3)}</b> | ||W_adapt_x||: <b>${d.x_norm.toFixed(3)}</b> | Out ||W_adapt_out[:,${nd.idx}]||: <b style="color:#facc15;">${d.out_norm.toFixed(4)}</b></div>` +
      (fin ? `<div style="margin-bottom:8px;"><b style="color:#f472b6;">Top Context Input Features (W_adapt_x[${nd.idx}, :]):</b><br>${fin}</div>` : '') +
      `<div style="margin-bottom:8px;"><b style="color:#c084fc;">Top Incoming Layer 2 (85k RNN) Neurons (W_adapt_h[${nd.idx}, :]):</b><br>${(d.top_in || []).map(t => `• <b>T${t.idx}</b>: <span style="color:${t.w >= 0 ? '#22c55e' : '#f43f5e'};">${t.w >= 0 ? '+' : ''}${t.w.toFixed(3)}</span>`).join('<br>')}</div>` +
      `<div><b style="color:#facc15;">Layer 3 -> Output Action Residual Weights (W_adapt_out[:, ${nd.idx}]):</b><br>${(d.act_out || []).map(a => `• <b>${a.act}</b>: <span style="color:${a.w >= 0 ? '#22c55e' : '#f43f5e'};">${a.w >= 0 ? '+' : ''}${a.w.toFixed(4)}</span>`).join('<br>')}</div>`;
  } else {
    const adBlock = curNNData.has_adapter
      ? `<div>• <b>Layer 3 Adapter L1 Swing</b> (<code>∑|W_adapt_out[a, :]|</code>): <b style="color:#fbbf24;">${(d.adapt_l1_swing || 0).toFixed(3)}</b> (max: <b>${(d.adapt_max || 0).toFixed(4)}</b>, clamped ±2.0)</div>`
      : '';
    const adDrivers = (curNNData.has_adapter && d.adapt_drivers && d.adapt_drivers.length)
      ? `<div style="margin-bottom:8px;"><b style="color:#fbbf24;">Strongest Layer 3 Adapter Neurons Driving ${d.label} (W_adapt_out[${nd.idx}, :]):</b><br>${d.adapt_drivers.map(a => `• <b>A${a.idx}</b>: <span style="color:${a.w >= 0 ? '#22c55e' : '#f43f5e'};">${a.w >= 0 ? '+' : ''}${a.w.toFixed(4)}</span>`).join('<br>')}</div>`
      : '';
    box.innerHTML =
      `<div style="font-size:15px;font-weight:700;color:#22c55e;margin-bottom:6px;">Output Action — ${d.label}</div>` +
      `<div style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px 10px;margin-bottom:10px;font-size:12px;">` +
      `<div>• <b>Layer 2 Total L1 Max Logit Swing</b> (<code>∑|W2[a, :]|</code>): <b style="color:#34d399;">${d.top_l1_swing.toFixed(2)}</b></div>` +
      `<div>• <b>Layer 2 Typical RMS Logit Swing</b> (<code>0.6·||W2[a, :]||₂</code>): <b style="color:#38bdf8;">${d.top_rms_swing.toFixed(2)}</b> | Action Bias <code>b2 = ${d.b2.toFixed(3)}</code></div>` +
      adBlock +
      `</div>` +
      adDrivers +
      `<div style="margin-bottom:8px;"><b style="color:#38bdf8;">Strongest Layer 2 Neurons Driving ${d.label} (W2[${nd.idx}, :]):</b><br>${d.top_drivers.map(t => `• <b>T${t.idx}</b>: <span style="color:${t.w >= 0 ? '#22c55e' : '#f43f5e'};">${t.w >= 0 ? '+' : ''}${t.w.toFixed(4)}</span>`).join('<br>')}</div>` +
      `<div><b style="color:#facc15;">Effective Layer 1 KBANN Rule Influence via Layer 2 ((W2 @ W_top_kb)[${nd.idx}, :]):</b><br>${d.kb_drivers.map(k => `• <b>${k.rule}</b>: <span style="color:${k.w >= 0 ? '#22c55e' : '#f43f5e'};">${k.w >= 0 ? '+' : ''}${k.w.toFixed(3)}</span>`).join('<br>')}</div>`;
  }
}

async function playReplay(bid, win, dmg, btype, src) {
  const player = document.getElementById('mainVideoPlayer');
  const lbl = document.getElementById('curVideoLabel');
  const base = src === 'm2tw' ? '/m2tw_replays' : '/replays';
  const tag = src === 'm2tw' ? `M2-${bid}` : `${bid}`;
  if (src === 'm2tw') {
    lbl.innerHTML = `Rendering <b>Battle #${tag} [M2TW]</b>...`;
    try {
      const rr = await fetch(`/api/m2tw_render?bid=${bid}`);
      const rj = await rr.json();
      if (!rj.ok) {
        lbl.innerHTML = `<span style="color:#f43f5e;">Render failed for #${tag}: ${rj.error || 'unknown'}</span>`;
        return;
      }
    } catch (e) {
      lbl.innerHTML = `<span style="color:#f43f5e;">Render request failed for #${tag}</span>`;
      return;
    }
  }
  lbl.innerHTML = `Playing <b>Battle #${tag} [${(btype || 'open_field').toUpperCase().replace('_', ' ')}]</b> (${win ? '<span style="color:#22c55e;">WIN</span>' : '<span style="color:#f43f5e;">LOSS</span>'}, Enemy Casualties: ${dmg}%) — <a href="${base}/replay_${bid}.mp4" target="_blank" style="color:#38bdf8;">Open raw .mp4</a>`;
  player.src = `${base}/replay_${bid}.mp4`;
  player.play().catch(() => {});
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

CACHE = {"episodes": {}, "verdicts": {}}


def collect_metrics():
    state_path = os.path.join(WORKSPACE, "state.json")
    phase = 2
    if os.path.exists(state_path):
        try:
            phase = json.load(open(state_path)).get("phase", 2)
        except Exception:
            pass

    vfiles = glob.glob(os.path.join(WORKSPACE, "processed", "verdict_*.json"))
    live_vbids = set()
    for vpath in vfiles:
        try:
            bid = int(os.path.basename(vpath).split("_")[1].split(".")[0])
            live_vbids.add(bid)
            mtime = os.path.getmtime(vpath)
            if bid not in CACHE["verdicts"] or CACHE["verdicts"][bid][0] != mtime:
                v = json.load(open(vpath))
                CACHE["verdicts"][bid] = (mtime, {
                    "bid": bid,
                    "fair": float(v.get("fairness", 0)),
                    "hist": float(v.get("historical_accuracy", 0)),
                    "reas": float(v.get("reasonability", 0)),
                    "total": float(v.get("penalties", {}).get("total", 0.0)),
                    "reasoning": v.get("reasoning", ""),
                })
        except Exception:
            pass
    for k in list(CACHE["verdicts"].keys()):
        if k not in live_vbids:
            del CACHE["verdicts"][k]

    m2tw_ws = "/home/tomek/medieval2-docker/workspace"
    ep_sources = [("sim", os.path.join(WORKSPACE, "processed", "episode_*.json"))]
    if os.path.abspath(WORKSPACE) != os.path.abspath(m2tw_ws):
        ep_sources.append(("m2tw", os.path.join(m2tw_ws, "processed", "episode_*.json")))

    live_ebids = set()
    for src_tag, pat in ep_sources:
        for epath in glob.glob(pat):
            try:
                bid = int(os.path.basename(epath).split("_")[1].split(".")[0])
                key = (src_tag, bid)
                live_ebids.add(key)
                mtime = os.path.getmtime(epath)
                if key not in CACHE["episodes"] or CACHE["episodes"][key][0] != mtime:
                    ep = json.load(open(epath))
                    step_pen = 0.0
                    step_net = 0.0
                    tot_kills = 0.0
                    tot_losses = 0.0
                    tot_coh = 0.0
                    unit_cnt = 0
                    for u in ep.get("units", []):
                        if u.get("role") == "commander" or not u.get("steps"):
                            continue
                        unit_cnt += 1
                        for s in u.get("steps", []):
                            dk_term = 22.5 * float(s.get("dk", 0.0))
                            dl_pen = -2.2 * float(s.get("dl", 0.0))
                            coh = float(s.get("coh", 0.0))
                            coh_term = 4.25 * coh
                            tot_kills += dk_term
                            tot_losses += dl_pen
                            tot_coh += coh_term
                            step_pen += (dl_pen + coh_term)
                            step_net += (dk_term + dl_pen + coh_term)
                    res = ep.get("result", 0)
                    dmg_frac = float(ep.get("enemy_damage_frac", 0.0))
                    is_dec = bool(ep.get("decisive_win"))
                    is_dec_loss = bool(ep.get("decisive_loss"))
                    term_base = (100.0 * 0.25) if (res > 0 and not is_dec) else (100.0 * res)
                    if res > 0 and is_dec:
                        dec_bonus = 340.0 * max(0.5, dmg_frac)
                    elif res <= 0 and is_dec_loss:
                        dec_bonus = -170.0
                    else:
                        dec_bonus = 0.0
                    n_u = max(1, unit_cnt)
                    CACHE["episodes"][key] = (mtime, {
                        "bid": bid,
                        "src": src_tag,
                        "label": f"M2-{bid}" if src_tag == "m2tw" else str(bid),
                        "mode": ep.get("mode", "rl_vs_vanilla"),
                        "phase": int(ep.get("phase", 2)),
                        "btype": "m2tw" if src_tag == "m2tw" else ep.get("battle_type", "open_field"),
                        "win": 1 if res > 0 else 0,
                        "decisive": is_dec,
                        "dmg": dmg_frac * 100.0,
                        "step_pen": round(step_pen, 2),
                        "step_net": round(step_net, 2),
                        "term_base": round(term_base, 2),
                        "dec_bonus": round(dec_bonus, 2),
                        "kills": round(tot_kills / n_u, 2),
                        "losses": round(tot_losses / n_u, 2),
                        "coh": round(tot_coh / n_u, 2),
                        "base_ret": round((step_net / n_u) + term_base + dec_bonus, 2),
                    })
            except Exception:
                pass
    for k in list(CACHE["episodes"].keys()):
        if k not in live_ebids:
            del CACHE["episodes"][k]

    # Filter to Phase 2+ from this training run (use earliest Sim Phase 2 mtime as run start)
    def _is_p2(e):
        return e.get("mode") != "vanilla_vs_vanilla" and int(e.get("phase", 2)) >= 2

    sim_p2_mtimes = [
        mt for (src_tag, _), (mt, e) in CACHE["episodes"].items()
        if src_tag == "sim" and _is_p2(e)
    ]
    min_run_mtime = min(sim_p2_mtimes) if sim_p2_mtimes else 0.0
    ordered_keys = sorted(
        [
            k for k, (mt, e) in CACHE["episodes"].items()
            if _is_p2(e) and (k[0] == "sim" or mt >= min_run_mtime)
        ],
        key=lambda k: (CACHE["episodes"][k][0], 1 if k[0] == "m2tw" else 0, k[1]),
    )
    battles, llm_pen, step_pen_s, step_net_s, comb_s, comb20_s, wr50_s, wr500_s, dmg50_s, breakdowns = [], [], [], [], [], [], [], [], [], []
    wins_hist, dmg_hist, comb_hist = [], [], []
    for k in ordered_keys:
        e = CACHE["episodes"][k][1]
        b = e["bid"]
        wins_hist.append(e["win"])
        dmg_hist.append(e["dmg"])
        battles.append(e["label"])
        v = CACHE["verdicts"].get(b) if e["src"] == "sim" else None
        llm_raw = v[1]["total"] if v else None
        llm_pen.append(llm_raw)
        step_pen_s.append(e["step_pen"])
        step_net_s.append(e["step_net"])
        llm_eff = round(85.0 * llm_raw, 2) if llm_raw is not None else 0.0
        comb_val = round(e["base_ret"] + llm_eff, 2)
        comb_s.append(comb_val)
        comb_hist.append(comb_val)
        c20 = comb_hist[-20:]
        comb20_s.append(round(sum(c20) / len(c20), 2))
        rep_dir = os.path.join(m2tw_ws, "replays") if e["src"] == "m2tw" else os.path.join(WORKSPACE, "replays")
        has_rep = (
            os.path.exists(os.path.join(rep_dir, f"replay_{b}.mp4"))
            or (e["src"] == "m2tw" and os.path.exists(os.path.join(rep_dir, f"replay_{b}.json")))
        )
        breakdowns.append({
            "bid": b,
            "src": e["src"],
            "label": e["label"],
            "btype": e.get("btype", "open_field"),
            "win": e["win"],
            "decisive": e["decisive"],
            "dmg": round(e["dmg"], 1),
            "term_base": e["term_base"],
            "dec_bonus": e["dec_bonus"],
            "kills": e["kills"],
            "losses": e["losses"],
            "coh": e["coh"],
            "llm_raw": llm_raw if llm_raw is not None else 0.0,
            "llm": llm_eff,
            "total": comb_val,
            "reasoning": v[1]["reasoning"] if v else "",
            "has_replay": has_rep,
        })
        w50 = wins_hist[-50:]
        w500 = wins_hist[-500:]
        d50 = dmg_hist[-50:]
        wr50_s.append(round(sum(w50) / len(w50) * 100.0, 1))
        wr500_s.append(round(sum(w500) / len(w500) * 100.0, 1))
        dmg50_s.append(round(sum(d50) / len(d50), 1))

    jbids = sorted(CACHE["verdicts"].keys())
    j_fair = [CACHE["verdicts"][b][1]["fair"] for b in jbids]
    j_hist = [CACHE["verdicts"][b][1]["hist"] for b in jbids]
    j_reas = [CACHE["verdicts"][b][1]["reas"] for b in jbids]
    recent_v = [CACHE["verdicts"][b][1] for b in reversed(jbids[-8:])]

    ck_files = glob.glob(os.path.join(WORKSPACE, "checkpoints", "weights_iter_*.json"))
    nn_iters = sorted(int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in ck_files if os.path.basename(p).split("_")[-1].split(".")[0].isdigit())

    return {
        "phase": phase,
        "latest_battle": ordered_keys[-1][1] if ordered_keys else 0,
        "wr50": wr50_s[-1] if wr50_s else 0.0,
        "wr500": wr500_s[-1] if wr500_s else 0.0,
        "dmg50": dmg50_s[-1] if dmg50_s else 0.0,
        "latest_judge_penalty": CACHE["verdicts"][jbids[-1]][1]["total"] if jbids else None,
        "battles": battles,
        "llm_penalty_series": llm_pen,
        "step_penalty_series": step_pen_s,
        "step_net_series": step_net_s,
        "combined_series": comb_s,
        "combined_avg20_series": comb20_s,
        "breakdowns": breakdowns,
        "wr50_series": wr50_s,
        "wr500_series": wr500_s,
        "dmg50_series": dmg50_s,
        "judge_battles": jbids,
        "judge_fair": j_fair,
        "judge_hist": j_hist,
        "judge_reas": j_reas,
        "recent_verdicts": recent_v,
        "nn_iters": nn_iters,
        "imitation_ce": _read_ce(),
        "m2tw_imitation_ce": _read_m2tw_ce(),
        "eval_history": _read_eval_history(),
        "physics_history": _read_physics_history(),
    }


def _read_physics_history():
    try:
        p = os.path.join(WORKSPACE, "physics_model_history.json")
        if os.path.exists(p):
            return json.load(open(p))
    except Exception:
        pass
    return []


def _read_eval_history():
    try:
        with open(os.path.join(WORKSPACE, "eval_history.json")) as f:
            return json.load(f)
    except Exception:
        return []


def _read_ce():
    try:
        with open(os.path.join(WORKSPACE, "imitation_ce.json")) as f:
            return json.load(f)
    except Exception:
        return []


def _read_m2tw_ce():
    try:
        with open("/home/tomek/medieval2-docker/workspace/m2tw_imitation_ce.json") as f:
            return json.load(f)
    except Exception:
        return []


UNIT_RULE_LABELS = [
    "R0: Default Doctrine", "R1: Enemy Front", "R2: Enemy Left", "R3: Enemy Right",
    "R4: Enemy Rear", "R5: Far Right Coh", "R6: Far Left Coh", "R7: Lagging Coh",
    "R8: Skirmish/Melee Lock", "R9: Flank/Overext", "R10: Cmd Order 0", "R11: Cmd Order 1",
    "R12: Cmd Order 2", "R13: Cmd Order 3", "R14: Cmd Order 4", "R15: Cmd Order 5",
    "R16: March to Contact", "R17: Charge/Standoff", "R18: Flank Right/Gen Behind", "R19: Flank Left",
    "R20: Siege Staging", "R21: Storm Breach", "R22: Defense Hold", "R23: Counter-Charge",
]
UNIT_ACT_LABELS = [
    "A0: Move Fwd", "A1: Fwd-Left", "A2: Fwd-Right", "A3: Move Left", "A4: Move Right",
    "A5: Back-Left", "A6: Back-Right", "A7: Move Back", "A8: Hold/Fire",
    "A9: Atk Front", "A10: Atk Left", "A11: Atk Right", "A12: Atk Rear",
]
CMD_RULE_LABELS = [
    "R0: Ranged Shootout", "R1: Advance Line", "R2: Engaged Press", "R3: Left Flank",
    "R4: Right Flank", "R5: Outnumbered Regroup", "R6: Siege Assault", "R7: Siege Defense",
    "R8: River Crossing", "R9: Enemy Wavering",
    "R10: Cav Superiority", "R11: Clearly Ahead", "R12: Morale Low", "R13: Early Form-Up",
    "R14: Left Wing Weak", "R15: Right Wing Weak", "R16: Assault Breached", "R17: Defense Breached",
    "R18: Enemy General Dead",
]
CMD_ACT_LABELS = [f"Order {i}" for i in range(10)]
ROLE_NAMES = ["line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general", "artillery"]
_INIT_CACHE = {}

_GLOBAL_NAMES = [
    "g[0] Bias (1.0)", "g[1] Battle Time", "g[2] Own Alive Frac", "g[3] Enemy Alive Frac",
    "g[4] Casualty Advantage", "g[5] Center Balance", "g[6] Left Flank Balance", "g[7] Right Flank Balance",
    "g[8] Own Rear Reserve Frac", "g[9] Missile Advantage", "g[10] Cavalry Advantage", "g[11] Engaged Frac",
    "g[12] Own General Alive", "g[13] Enemy General Alive", "g[14] Own Morale Frac", "g[15] Enemy Morale Frac",
    "g[16] Army Strength Share", "g[17] Terrain/Siege Side (×3)", "g[18] Breach / River Prox (×3)", "g[19] Siege Plaza Timer (×3)",
]
_LOCAL_NAMES = [
    "is_inf", "is_cav", "is_heavy", "is_light_melee", "is_missile", "is_spear", "l[6] unused", "can_phalanx",
    "l[8] unused", "melee_atk", "charge_bonus", "missile_atk", "missile_range", "total_defense",
    "move_speed", "unit_mass", "rel_x (flank)", "rel_y (depth)", "hp_frac", "morale_level",
    "fatigue", "ammo_frac", "is_engaged", "is_under_fire", "nearby_friends_80m", "nearby_enemies_80m",
    "inf_centroid_dx", "inf_centroid_dy",
]
_GRID_CH_NAMES = ["Own", "Enemy", "Hold", "Attack", "Fire", "Rout"]


def _feat_meta(idx, is_cmd):
    if idx < 20:
        return ("Terrain" if idx >= 17 else "Global"), _GLOBAL_NAMES[idx]
    if is_cmd:
        if idx < 92:
            rel = idx - 20
            cell, ch = rel // 2, rel % 2
            r, c = cell // 6, cell % 6
            return "Map6x6", f"Map6x6(r={r},c={c}) {'Own' if ch == 0 else 'Enemy'}"
        if idx < 172:
            rel = idx - 92
            u, k = rel // 4, rel % 4
            return "OwnUnit", f"OwnUnit#{u} {['rx', 'ry', 'hp', 'role'][k]}"
        if idx < 252:
            rel = idx - 172
            u, k = rel // 4, rel % 4
            return "EnUnit", f"EnemyUnit#{u} {['rx', 'ry', 'hp', 'role'][k]}"
        if idx == 252:
            return "CmdCtx", "Step0 GlobalOrder Flag"
        if 253 <= idx <= 262:
            return "CmdCtx", f"GlobalOrder={idx - 253}"
        if 263 <= idx <= 272:
            return "CmdCtx", f"PrevStepOrder={idx - 263}"
        if 273 <= idx <= 282:
            return "CmdCtx", f"PrevSquadOrder={idx - 273}"
        if 283 <= idx <= 289:
            return "CmdCtx", f"SquadLeadRole={ROLE_NAMES[idx - 283]}"
        if idx == 290:
            return "CmdCtx", "Squad rel_x"
        if idx == 291:
            return "CmdCtx", "Squad rel_y"
        if idx == 292:
            return "CmdCtx", "Squad hp_frac"
        return "CmdCtx", f"x[{idx}]"
    else:
        if idx < 30:
            return "CmdOrder", f"Global Order {idx - 20} (×3)"
        if idx < 40:
            return "CmdOrder", f"Squad Order {idx - 30} (×3)"
        if idx < 50:
            return "CmdOrder", f"Lead Inf Order {idx - 40} (×3)"
        if idx < 78:
            return "Local", f"l[{idx - 50}] {_LOCAL_NAMES[idx - 50]}"
        rel = idx - 78
        cell, ch = rel // 6, rel % 6
        r, c = cell // 8, cell % 8
        return "Grid8x8", f"Grid(r={r},c={c}) {_GRID_CH_NAMES[ch]}"


def get_nn_snapshot(iter_str="latest", role_str="0"):
    import sys
    import numpy as np
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer")
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    import sarsa_trainer as st

    if "unit0" not in _INIT_CACHE:
        _INIT_CACHE["unit0"] = st.new_weights()
        _INIT_CACHE["cmd0"] = st.new_commander_rnn()

    is_cmd = (role_str == "commander")
    if is_cmd:
        path = (
            os.path.join(WORKSPACE, "commander_rnn.json")
            if iter_str == "latest"
            else os.path.join(WORKSPACE, "checkpoints", f"commander_rnn_iter_{iter_str}.json")
        )
        net = st.load_commander_rnn(path)
        net0 = _INIT_CACHE["cmd0"]
        W_kb, b_kb, W_kb_out, b_kb_out = net["W_kb"], net["b_kb"], net["W_kb_out"], net["b_kb_out"]
        W_top_kb, W_top_x, Wh, b_top, W2 = net["W_top_kb"], net["Wx"], net["Wh"], net["bh"], net["Wo"]
        W_kb0, W_top_kb0 = net0["W_kb"], net0["W_top_kb"]
        r_labels, a_labels = CMD_RULE_LABELS, CMD_ACT_LABELS
        role_name = "Commander RNN (Purely Additive: 293 -> 19 KBANN + 48 Zero-Init Top -> 10 Orders)"
    else:
        ri = max(0, min(6, int(role_str)))
        path = (
            os.path.join(WORKSPACE, "weights_latest.json")
            if iter_str == "latest"
            else os.path.join(WORKSPACE, "checkpoints", f"weights_iter_{iter_str}.json")
        )
        all_w = st.load_weights(path)
        net = all_w[ri]
        net0 = _INIT_CACHE["unit0"][ri]
        W_kb, b_kb, W_kb_out, b_kb_out = net["W_kb"], net["b_kb"], net["W_kb_out"], net["b_kb_out"]
        W_top_kb, W_top_x, Wh, b_top, W2 = net["W_top_kb"], net["W_top_x"], net["Wh"], net["b_top"], net["W2"]
        W_kb0, W_top_kb0 = net0["W_kb"], net0["W_top_kb"]
        r_labels, a_labels = UNIT_RULE_LABELS, UNIT_ACT_LABELS
        role_name = f"{ROLE_NAMES[ri]} (Purely Additive: 462 -> 24 KBANN + 32 Zero-Init Top -> 13 Actions)"

    n_kb = len(r_labels)
    kb_delta = np.abs(W_kb - W_kb0)
    top_kb_delta = W_top_kb - W_top_kb0
    kb_delta_max = float(np.max(kb_delta))
    eff_kb_to_act = W_kb_out + (W2 @ W_top_kb)

    kb_rules = []
    for i in range(n_kb):
        eff_outs = sorted(
            [{"act": a_labels[a], "w": float(eff_kb_to_act[a, i])} for a in range(len(a_labels))],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:5]
        t_links = sorted(
            [{"idx": j, "w": float(W_top_kb[j, i]), "dw": float(top_kb_delta[j, i])} for j in range(W_top_kb.shape[0])],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:6]
        top_f_idx = np.argsort(np.abs(W_kb[i]))[::-1][:6]
        feat_in = [
            {"idx": int(fi), "name": _feat_meta(int(fi), is_cmd)[1], "w": float(W_kb[i, fi])}
            for fi in top_f_idx if abs(W_kb[i, fi]) > 1e-3
        ]
        kb_rules.append({
            "label": r_labels[i],
            "bias": float(b_kb[i]),
            "in_norm": float(np.linalg.norm(W_kb[i])),
            "delta_norm": float(np.linalg.norm(kb_delta[i])),
            "feat_in": feat_in,
            "eff_outs": eff_outs,
            "top_links": t_links,
        })

    top_neurons = []
    for j in range(W_top_kb.shape[0]):
        kb_in = sorted(
            [{"rule": r_labels[i], "w": float(W_top_kb[j, i]), "dw": float(top_kb_delta[j, i])} for i in range(n_kb)],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:5]
        act_out = sorted(
            [{"act": a_labels[a], "w": float(W2[a, j])} for a in range(len(a_labels))],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:5]
        top_fx_idx = np.argsort(np.abs(W_top_x[j]))[::-1][:6]
        feat_in = [
            {"idx": int(fi), "name": _feat_meta(int(fi), is_cmd)[1], "w": float(W_top_x[j, fi])}
            for fi in top_fx_idx if abs(W_top_x[j, fi]) > 1e-3
        ]
        top_neurons.append({
            "bias": float(b_top[j]),
            "self_wh": float(Wh[j, j]),
            "ctx_norm": float(np.linalg.norm(W_top_x[j])),
            "out_norm": float(np.linalg.norm(W2[:, j])),
            "feat_in": feat_in,
            "kb_in": kb_in,
            "act_out": act_out,
        })

    n_in = W_kb.shape[1]
    n_top_x = W_top_x.shape[1]
    kb_col_norms = np.linalg.norm(W_kb, axis=0)
    kb_col_deltas = np.linalg.norm(kb_delta, axis=0)
    top_col_norms = np.linalg.norm(W_top_x, axis=0)
    input_features = []
    for fi in range(n_in):
        grp, fname = _feat_meta(fi, is_cmd)
        input_features.append({
            "idx": fi,
            "group": grp,
            "name": fname,
            "top_norm": float(top_col_norms[fi]) if fi < n_top_x else 0.0,
            "kb_norm": float(kb_col_norms[fi]),
            "kb_delta": float(kb_col_deltas[fi]),
        })
    input_features.sort(key=lambda x: (x["top_norm"], x["kb_norm"]), reverse=True)

    has_adapter = all(k in net for k in ("W_adapt_h", "W_adapt_x", "b_adapt", "W_adapt_out"))
    adapt_neurons = []
    adapt_l1_swings = []
    W_adapt_h = net["W_adapt_h"] if has_adapter else None
    W_adapt_x = net["W_adapt_x"] if has_adapter else None
    b_adapt = net["b_adapt"] if has_adapter else None
    W_adapt_out = net["W_adapt_out"] if has_adapter else None
    if has_adapter:
        for m in range(W_adapt_h.shape[0]):
            top_in = sorted(
                [{"idx": j, "w": float(W_adapt_h[m, j])} for j in range(W_adapt_h.shape[1])],
                key=lambda x: abs(x["w"]), reverse=True,
            )[:5]
            top_fx_idx = np.argsort(np.abs(W_adapt_x[m]))[::-1][:6]
            feat_in = [
                {"idx": int(fi), "name": _feat_meta(int(fi), is_cmd)[1], "w": float(W_adapt_x[m, fi])}
                for fi in top_fx_idx if abs(W_adapt_x[m, fi]) > 1e-3
            ]
            act_out = sorted(
                [{"act": a_labels[a], "w": float(W_adapt_out[a, m])} for a in range(len(a_labels))],
                key=lambda x: abs(x["w"]), reverse=True,
            )[:5]
            adapt_neurons.append({
                "bias": float(b_adapt[m]),
                "h_norm": float(np.linalg.norm(W_adapt_h[m])),
                "x_norm": float(np.linalg.norm(W_adapt_x[m])),
                "out_norm": float(np.linalg.norm(W_adapt_out[:, m])),
                "top_in": top_in,
                "feat_in": feat_in,
                "act_out": act_out,
            })

    actions = []
    top_l1_swings = []
    for a in range(len(a_labels)):
        kb_drv = sorted(
            [{"rule": r_labels[i], "w": float(W_kb_out[a, i])} for i in range(n_kb) if abs(W_kb_out[a, i]) > 0.05],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:5]
        top_drv = sorted(
            [{"idx": j, "w": float(W2[a, j])} for j in range(W2.shape[1])],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:6]
        top_l1 = float(np.sum(np.abs(W2[a])))
        top_rms = float(0.6 * np.linalg.norm(W2[a]))
        top_l1_swings.append(top_l1)
        ad_l1 = float(np.sum(np.abs(W_adapt_out[a]))) if has_adapter else 0.0
        ad_max = float(np.max(np.abs(W_adapt_out[a]))) if has_adapter else 0.0
        ad_drv = sorted(
            [{"idx": m, "w": float(W_adapt_out[a, m])} for m in range(W_adapt_out.shape[1])],
            key=lambda x: abs(x["w"]), reverse=True,
        )[:6] if has_adapter else []
        if has_adapter:
            adapt_l1_swings.append(ad_l1)
        actions.append({
            "label": a_labels[a],
            "b2": float(b_kb_out[a]),
            "res_norm": float(np.linalg.norm(W2[a])),
            "top_l1_swing": top_l1,
            "top_rms_swing": top_rms,
            "adapt_l1_swing": ad_l1,
            "adapt_max": ad_max,
            "adapt_drivers": ad_drv,
            "kb_drivers": kb_drv,
            "top_drivers": top_drv,
        })

    return {
        "ok": True,
        "role_name": role_name,
        "iter_label": "Latest Live Weights" if iter_str == "latest" else f"Checkpoint Battle #{iter_str}",
        "kb_unfrozen": kb_delta_max > 1e-4,
        "kb_delta_max": kb_delta_max,
        "w2_max": float(np.max(np.abs(W2))),
        "w2_mean": float(np.mean(np.abs(W2))),
        "w_top_kb_max": float(np.max(np.abs(W_top_kb))),
        "top_l1_max": float(max(top_l1_swings)),
        "top_l1_mean": float(np.mean(top_l1_swings)),
        "top_kb_delta_max": float(np.max(np.abs(top_kb_delta))),
        "has_adapter": has_adapter,
        "adapt_out_max": float(np.max(np.abs(W_adapt_out))) if has_adapter else 0.0,
        "adapt_l1_max": float(max(adapt_l1_swings)) if adapt_l1_swings else 0.0,
        "adapt_l1_mean": float(np.mean(adapt_l1_swings)) if adapt_l1_swings else 0.0,
        "kb_rules": kb_rules,
        "top_neurons": top_neurons,
        "adapt_neurons": adapt_neurons,
        "actions": actions,
        "input_features": input_features,
        "W_kb": np.round(W_kb, 3).tolist(),
        "W_top_x": np.round(W_top_x, 3).tolist(),
        "W_top_kb": np.round(W_top_kb, 4).tolist(),
        "W2": np.round(W2, 4).tolist(),
        "W_adapt_h": np.round(W_adapt_h, 4).tolist() if has_adapter else [],
        "W_adapt_out": np.round(W_adapt_out, 4).tolist() if has_adapter else [],
    }


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            body = json.dumps(collect_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/nn_snapshot"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            it = qs.get("iter", ["latest"])[0]
            rl = qs.get("role", ["0"])[0]
            body = json.dumps(get_nn_snapshot(it, rl)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/replays/replay_") and self.path.endswith(".mp4"):
            fname = os.path.basename(self.path.split("?")[0])
            mp4_path = os.path.join(WORKSPACE, "replays", fname)
            json_path = mp4_path.replace(".mp4", ".json")
            if not os.path.exists(mp4_path) and os.path.exists(json_path):
                import render_replay_video as rrv
                rrv.render_replay(json_path, mp4_path)
            if os.path.exists(mp4_path):
                data = open(mp4_path, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path == "/m2tw" or self.path.startswith("/m2tw?"):
            import m2tw_trial_monitor as mtm
            html = (
                mtm.HTML
                .replace("'/api/trial'", "'/api/m2tw_trial'")
                .replace("`/api/render?bid=${bid}`", "`/api/m2tw_render?bid=${bid}`")
                .replace("`/replays/replay_${bid}.mp4", "`/m2tw_replays/replay_${bid}.mp4")
                .replace('href="/replays/replay_${bid}.mp4"', 'href="/m2tw_replays/replay_${bid}.mp4"')
                .replace('href="http://localhost:8765" target="_blank"', 'href="/"')
                .replace("⚔️ Sim Dashboard (:8765)", "⬅️ Back to Sim Dashboard")
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/m2tw_trial"):
            p = "/home/tomek/medieval2-docker/workspace/m2tw_trial_summary.json"
            data = json.load(open(p)) if os.path.exists(p) else {"target_battles": 100, "completed": 0, "battles": []}
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/m2tw_render"):
            import subprocess, sys
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            bid = qs.get("bid", [""])[0]
            real_ws = "/home/tomek/medieval2-docker/workspace"
            rjson = os.path.join(real_ws, "replays", f"replay_{bid}.json")
            rmp4 = os.path.join(real_ws, "replays", f"replay_{bid}.mp4")
            if bid.isdigit() and not os.path.exists(rmp4) and os.path.exists(rjson):
                subprocess.run([sys.executable, os.path.join(real_ws, "render_replay_video.py"), rjson, rmp4],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = bid.isdigit() and os.path.exists(rmp4)
            body = b'{"ok":true}' if ok else b'{"ok":false,"error":"replay unavailable"}'
            self.send_response(200 if ok else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/m2tw_replays/"):
            fname = os.path.basename(self.path.split("?")[0])
            fpath = os.path.join("/home/tomek/medieval2-docker/workspace/replays", fname)
            if os.path.exists(fpath):
                data = open(fpath, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    print(f"[+] Training Monitor running at http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), MonitorHandler).serve_forever()
