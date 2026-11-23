"""Benchmark evaluator for trained policy against Vanilla AI.

Runs N battles under super-low temperature (default inv_temp=15.0, T=0.067)
to measure deterministic tactical performance without exploration noise.
"""
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SIM_DIR)
for p in (SIM_DIR, os.path.join(ROOT_DIR, "workspace", "trainer")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rl_env
import sim_train as T

LOW_INV_TEMP = 15.0
orig_softmax = rl_env._softmax_sample


def _low_temp_softmax(logits, rng, inv_temp=None):
    return orig_softmax(logits, rng, inv_temp=LOW_INV_TEMP)


def _init_eval_worker():
    T._init_worker()
    rl_env._softmax_sample = _low_temp_softmax


def _run_eval_lite(task):
    ep, _ = T._run_one(task)
    return {
        "result": ep["result"],
        "enemy_damage_frac": ep["enemy_damage_frac"],
        "battle_type": ep["battle_type"],
    }


def run_benchmark(w, rnn, battle_n: int, ws: str, n_battles: int = 1000, max_workers: int = 10, inv_temp: float = 15.0) -> dict:
    global LOW_INV_TEMP
    LOW_INV_TEMP = inv_temp

    t0 = time.time()
    bids = list(range(100000 + battle_n * 10, 100000 + battle_n * 10 + n_battles))
    tasks = [(b, "rl_vs_vanilla", w, rnn, None, None, False, 2, ws) for b in bids]

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_eval_worker) as pool:
        eps = list(pool.map(_run_eval_lite, tasks, chunksize=10))

    elapsed = max(1e-3, time.time() - t0)
    by = collections.defaultdict(list)
    for e in eps:
        by[e["battle_type"]].append(e)

    tot = [e["result"] for e in eps]
    wins = sum(r == 1 for r in tot)
    draws = sum(r == 0 for r in tot)
    losses = sum(r == -1 for r in tot)
    wr = wins / len(tot)
    avg_dmg = sum(e["enemy_damage_frac"] for e in eps) / len(eps)

    terrain_stats = {}
    print("\n" + "=" * 70, flush=True)
    print(f"== [EVALUATION @ BATTLE #{battle_n}] SUPER LOW TEMP (inv_temp={inv_temp}) vs VANILLA AI", flush=True)
    print(f"== (N={len(eps)} battles in {elapsed:.1f}s | {len(eps)/elapsed:.1f} bat/s)", flush=True)
    print(f"== OVERALL WIN RATE: {wr:.1%} ({wins}W / {losses}L / {draws}D) | AVG ENEMY CASUALTIES: {avg_dmg:.1%}", flush=True)
    print("-" * 70, flush=True)
    for k, v in sorted(by.items()):
        w_k = sum(e["result"] == 1 for e in v)
        dmg_k = sum(e["enemy_damage_frac"] for e in v) / len(v)
        terrain_stats[k] = {
            "n": len(v),
            "wins": w_k,
            "wr": round(w_k / len(v) * 100.0, 1),
            "dmg": round(dmg_k * 100.0, 1),
        }
        print(f"   {k:16s} n={len(v):3d} | WR = {w_k/len(v):.1%} ({w_k}/{len(v)}) | Avg Dmg = {dmg_k:.1%}", flush=True)
    print("=" * 70 + "\n", flush=True)

    summary = {
        "battle": battle_n,
        "n_battles": n_battles,
        "elapsed_sec": round(elapsed, 1),
        "overall_wr": round(wr * 100.0, 1),
        "overall_dmg": round(avg_dmg * 100.0, 1),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "by_terrain": terrain_stats,
    }

    hist_path = os.path.join(ws, "eval_history.json")
    history = []
    if os.path.exists(hist_path):
        try:
            history = json.load(open(hist_path))
        except Exception:
            history = []
    # Avoid duplicate entry for the exact same battle number
    history = [h for h in history if h.get("battle") != battle_n]
    history.append(summary)
    history.sort(key=lambda x: x.get("battle", 0))

    tmp = hist_path + ".tmp"
    json.dump(history, open(tmp, "w"), indent=2)
    os.replace(tmp, hist_path)

    return summary
