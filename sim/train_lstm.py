#!/usr/bin/env python3
"""Large-batch Full-Army LSTM Simulator Training Loop (A2C / Policy Gradient, NO PPO)."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import os
import random
import shutil
import sys
import time
import torch

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SIM_DIR)
TRAINER_DIR = os.path.join(ROOT_DIR, "workspace", "trainer")
LLM_DIR = os.path.join(ROOT_DIR, "llm")
for p in (SIM_DIR, TRAINER_DIR, LLM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

env_file = os.path.join(ROOT_DIR, ".env")
if os.path.exists(env_file):
    for _line in open(env_file):
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

from combat import BattleSim  # noqa: E402
from edu_parser import build_faction_pools, parse_edu, parse_projectiles  # noqa: E402
from lstm_env import run_lstm_battle  # noqa: E402
from lstm_model import FullArmyLSTM, load_lstm, save_lstm, update_batch, update_imitation_batch  # noqa: E402
import scenario_generator as sg  # noqa: E402

_G_PROJS = None
_G_POOLS = None
_G_MODEL = None


def _init_worker():
    global _G_PROJS, _G_POOLS, _G_MODEL
    torch.set_num_threads(1)
    _G_PROJS = parse_projectiles()
    units = parse_edu()
    _G_POOLS = build_faction_pools(units)
    _G_MODEL = FullArmyLSTM()
    _G_MODEL.eval()


def _pick_army(faction, counts, rng):
    out = []
    for role, n in counts.items():
        pool = _G_POOLS[faction][role]
        for _ in range(n):
            out.append(rng.choice(pool))
    return out


def _run_one_lstm(args):
    bid, state_dict, inv_temp, greedy, rec_rep, ws, phase = args
    if state_dict is not None:
        _G_MODEL.load_state_dict(state_dict)
    _G_MODEL.eval()
    sc = sg.make_scenario(bid)
    rng = random.Random(bid)
    s0 = _pick_army(sc["sides"][0]["faction"], sc["sides"][0]["army"], rng)
    s1 = _pick_army(sc["sides"][1]["faction"], sc["sides"][1]["army"], rng)
    sim = BattleSim(
        s0, s1, _G_PROJS, weather=sc["weather"], seed=bid,
        deploy0=sc["sides"][0].get("deploy"), deploy1=sc["sides"][1].get("deploy"),
        terrain=sc.get("terrain"),
    )
    ep, rep = run_lstm_battle(sim, bid, _G_MODEL, inv_temp=inv_temp, greedy=greedy, record_replay=rec_rep, phase=phase)
    if rep is not None:
        rpath = os.path.join(ws, "replays", f"replay_{bid}.json")
        rep_str = json.dumps(rep)
        open(rpath, "w").write(rep_str)
        if phase >= 2 and bid % 32 == 0:
            try:
                import server as llm_srv
                verdict = llm_srv.evaluate_replay(rep_str)
                if verdict is not None:
                    json.dump(verdict, open(os.path.join(ws, "processed", f"verdict_{bid}.json"), "w"), indent=1)
                    ep["llm_terminal_penalty"] = float(verdict["penalties"]["total"])
                    ep["weight_mult"] = 3.0
                    print(
                        f"  [LLM b{bid}] fair={verdict['fairness']}/10 hist={verdict['historical_accuracy']}/10 "
                        f"reas={verdict['reasonability']}/10 pen={verdict['penalties']['total']}",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [LLM b{bid}] failed: {e}", flush=True)
        if bid % 64 == 0 or bid == 1:
            try:
                import subprocess
                subprocess.Popen(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, sys.argv[1]); import render_replay_video as r; r.render_replay(sys.argv[2])",
                     os.path.join(ROOT_DIR, "workspace"), rpath],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                )
            except Exception:
                pass
    return ep


def run_sim_eval(model: FullArmyLSTM, battle_n: int, ws: str, pool: ProcessPoolExecutor, n_battles: int = 200) -> dict:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tasks = [(900000 + i, sd, 1.0, True, False, ws, 2) for i in range(n_battles)]
    eps = list(pool.map(_run_one_lstm, tasks))
    wins = sum(1 for e in eps if e["result"] > 0)
    dmg = sum(e["enemy_damage_frac"] for e in eps) / max(1, len(eps))
    rec = {
        "battle": battle_n,
        "n_battles": len(eps),
        "overall_wr": round(100.0 * wins / max(1, len(eps)), 1),
        "overall_dmg": round(100.0 * dmg, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    eh_path = os.path.join(ws, "eval_history.json")
    eh = json.load(open(eh_path)) if os.path.exists(eh_path) else []
    if eh and eh[-1].get("battle") == battle_n:
        eh[-1] = rec
    else:
        eh.append(rec)
    tmp = eh_path + ".tmp"
    json.dump(eh, open(tmp, "w"), indent=2)
    os.replace(tmp, eh_path)
    print(f"  [LSTM Sim Eval #{battle_n}] N={len(eps)} | WR={rec['overall_wr']:.1f}% | AvgDmg={rec['overall_dmg']:.1f}%", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.path.join(ROOT_DIR, "workspace_sim"))
    ap.add_argument("--battles", type=int, default=2000)
    ap.add_argument("--phase1-battles", type=int, default=2048, help="Phase-1 Vanilla AI imitation battles before Phase-2 RL")
    ap.add_argument("--batch", type=int, default=64, help="Large batch size (battles per gradient update)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--eval-interval", type=int, default=0)
    ap.add_argument("--eval-battles", type=int, default=200)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    try:
        os.nice(15)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ws = args.workspace
    if args.fresh and os.path.exists(ws):
        shutil.rmtree(ws)
    for d in ("processed", "replays", "checkpoints"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    P = lambda *a: os.path.join(ws, *a)
    state_p = P("state.json")
    if os.path.exists(state_p) and not args.fresh:
        s = json.load(open(state_p))
        s["recent"] = deque(s.get("recent", []), maxlen=500)
    else:
        s = {"phase": 1 if args.phase1_battles > 0 else 2, "battle": 0, "arch": "lstm", "recent": deque(maxlen=500)}

    ce_path = P("imitation_ce.json")
    ce_log = json.load(open(ce_path)) if (os.path.exists(ce_path) and not args.fresh) else []

    weights_p = P("weights_lstm.pt")
    model, opt = load_lstm(weights_p, lr=args.lr)
    save_lstm(model, weights_p, opt=opt)

    torch.set_num_threads(2)
    target_end = s["battle"] + args.battles
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        while s["battle"] < target_end:
            cur_phase = int(s.get("phase", 2))
            bsz = min(args.batch, target_end - s["battle"])
            if cur_phase == 1 and s["battle"] < args.phase1_battles:
                bsz = min(bsz, args.phase1_battles - s["battle"])
            sd = None if cur_phase == 1 else {k: v.detach().cpu() for k, v in model.state_dict().items()}
            start_b = s["battle"]
            tasks = [
                (bid, sd, 1.0, False, (cur_phase >= 2 and (bid % 32 == 0 or bid == 1)), ws, cur_phase)
                for bid in range(start_b + 1, start_b + bsz + 1)
            ]
            t0 = time.time()
            eps = list(pool.map(_run_one_lstm, tasks))
            t_sim = time.time() - t0

            t1 = time.time()
            if cur_phase == 1:
                stats = update_imitation_batch(model, opt, eps)
            else:
                stats = update_batch(model, opt, eps, gamma=0.97, ent_coef=0.015, vf_coef=0.5)
            t_upd = time.time() - t1

            save_lstm(model, weights_p, opt=opt)

            for ep in eps:
                n = ep["battle_id"]
                s["battle"] = n
                if cur_phase >= 2:
                    s["recent"].append(ep["result"])
                # Dashboard-compatible episode summary
                ep_lite = {
                    "battle_id": n,
                    "phase": cur_phase,
                    "mode": "vanilla_vs_vanilla" if cur_phase == 1 else "rl_vs_vanilla",
                    "arch": "lstm",
                    "battle_type": ep["battle_type"],
                    "duration_sec": ep["duration_sec"],
                    "result": ep["result"],
                    "enemy_damage_frac": ep["enemy_damage_frac"],
                    "own_loss_frac": ep["own_loss_frac"],
                    "decisive_win": ep["decisive_win"],
                    "decisive_loss": ep["decisive_loss"],
                    "units": [{
                        "role": "army",
                        "steps": [{"dk": st_s["dk"], "dl": st_s["dl"], "coh": st_s["coh"]} for st_s in ep["steps"]],
                    }],
                }
                json.dump(ep_lite, open(P("processed", f"episode_{n}.json"), "w"))

            if cur_phase == 1:
                ce_log.append({"battle": s["battle"], "ce": stats["ce_loss"], "acc": stats["acc"]})
                tmp_ce = ce_path + ".tmp"
                json.dump(ce_log, open(tmp_ce, "w"))
                os.replace(tmp_ce, ce_path)
                if s["battle"] >= args.phase1_battles:
                    s["phase"] = 2
                    s["recent"].clear()
                    print(f"[+] Phase-1 Imitation complete @ #{s['battle']} (CE={stats['ce_loss']:.4f}, Acc={stats['acc']:.1f}%) -> Entering Phase-2 RL!", flush=True)

            s_out = {
                "phase": s["phase"],
                "arch": "lstm",
                "battle": s["battle"],
                "recent": list(s["recent"]),
            }
            tmp_s = state_p + ".tmp"
            json.dump(s_out, open(tmp_s, "w"))
            os.replace(tmp_s, state_p)

            if cur_phase == 1:
                print(
                    f"[LSTM P1 Imit #{s['battle']:5d}/{args.phase1_battles}] batch={len(eps)} "
                    f"CE={stats['ce_loss']:.4f} Acc={stats['acc']:5.1f}% | sim={t_sim:.1f}s upd={t_upd:.2f}s",
                    flush=True,
                )
            else:
                b_wins = sum(1 for e in eps if e["result"] > 0)
                b_dmg = 100.0 * sum(e["enemy_damage_frac"] for e in eps) / max(1, len(eps))
                wr500 = 100.0 * sum(1 for r in s["recent"] if r > 0) / max(1, len(s["recent"]))
                print(
                    f"[LSTM P2 Sim  #{s['battle']:5d}] batch={len(eps)} WR={100*b_wins/len(eps):5.1f}% "
                    f"(WR500={wr500:5.1f}%) Dmg={b_dmg:5.1f}% | "
                    f"pg={stats['pg_loss']:+.3f} vf={stats['vf_loss']:.3f} ent={stats['entropy']:.2f} | "
                    f"sim={t_sim:.1f}s upd={t_upd:.2f}s",
                    flush=True,
                )

            if args.eval_interval > 0 and (s["battle"] % args.eval_interval == 0):
                run_sim_eval(model, s["battle"], ws, pool, n_battles=args.eval_battles)

    print(f"[+] LSTM Sim stage complete ({args.battles} battles in {time.time() - t_start:.1f}s).", flush=True)


if __name__ == "__main__":
    main()
