"""Fast multi-phase Medieval II simulator training loop (Phases 1 -> 2 -> 3 -> 4)."""
import argparse
import copy
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import os
import random
import shutil
import sys
import time

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

from edu_parser import build_faction_pools, parse_edu, parse_projectiles  # noqa: E402
from combat import BattleSim  # noqa: E402
from rl_env import run_battle  # noqa: E402
import scenario_generator as sg  # noqa: E402
import torch  # noqa: E402
import sarsa_trainer as st  # noqa: E402
import eval_benchmark as eb  # noqa: E402

MODES = {
    1: "vanilla_vs_vanilla",
    2: "rl_vs_vanilla",
    3: "rl_vs_vanilla_disadvantaged",
    4: "rl_vs_frozen",
}

# Worker globals (initialized once per worker process)
_G_PROJS = None
_G_POOLS = None


def _init_worker():
    global _G_PROJS, _G_POOLS
    _G_PROJS = parse_projectiles()
    units = parse_edu()
    _G_POOLS = build_faction_pools(units)


def _pick_army(faction, counts, rng, scale=1.0):
    out = []
    for role, n in counts.items():
        pool = _G_POOLS[faction][role]
        for _ in range(n):
            out.append(rng.choice(pool))
    return out


def _run_one(args):
    bid, mode, w_rl, rnn_rl, w_opp, rnn_opp, rec_rep, phase, ws = args
    sc = sg.make_scenario(bid)
    rng = random.Random(bid)
    s0 = _pick_army(sc["sides"][0]["faction"], sc["sides"][0]["army"], rng)
    s1 = _pick_army(sc["sides"][1]["faction"], sc["sides"][1]["army"], rng)
    sim = BattleSim(
        s0, s1, _G_PROJS, weather=sc["weather"], seed=bid,
        deploy0=sc["sides"][0].get("deploy"), deploy1=sc["sides"][1].get("deploy"),
        terrain=sc.get("terrain"),
    )
    if mode == "rl_vs_vanilla_disadvantaged":
        # 15-30% troop disadvantage on RL side (Phase 3)
        keep = rng.uniform(0.70, 0.85)
        for u in sim.sides[0]:
            u.alive = max(5, int(round(u.alive * keep)))
            u.start = u.alive
            u.prev_alive = u.alive
            u.hp_pool = float(u.alive * u.tpl.hp)
        sim.own_start = sum(u.start for u in sim.sides[0])
    ep, rep = run_battle(sim, bid, mode, w_rl, rnn_rl, w_opp=w_opp, rnn_opp=rnn_opp, seed=bid, record_replay=rec_rep)
    verdict = None
    if rep is not None:
        rpath = os.path.join(ws, "replays", f"replay_{bid}.json")
        rep_str = json.dumps(rep)
        open(rpath, "w").write(rep_str)
        if bid % 25 == 0 and phase >= 2:
            try:
                import server as llm_srv
                verdict = llm_srv.evaluate_replay(rep_str)
                if verdict is not None:
                    json.dump(verdict, open(os.path.join(ws, "processed", f"verdict_{bid}.json"), "w"), indent=1)
                    print(
                        f"  [LLM b{bid}] fair={verdict['fairness']}/10 hist={verdict['historical_accuracy']}/10 "
                        f"reas={verdict['reasonability']}/10 pen={verdict['penalties']['total']}",
                        flush=True,
                    )
            except Exception as e:
                print(f"llm_judge b{bid} failed: {e}", flush=True)
        if bid % 25 == 0 or bid == 1:
            try:
                # Render in a detached background process so the batch doesn't wait on the slowest worker
                import subprocess
                subprocess.Popen(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, sys.argv[1]); import render_replay_video as r; r.render_replay(sys.argv[2])",
                     os.path.join(ROOT_DIR, "workspace"), rpath],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                )
            except Exception as e:
                print(f"video render b{bid} failed: {e}", flush=True)
    return ep, verdict


def winrate(q):
    return sum(1 for r in q if r > 0) / len(q) if q else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.path.join(ROOT_DIR, "workspace_sim"))
    ap.add_argument("--battles", type=int, default=2000)
    ap.add_argument("--phase1-battles", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--kbann-lr-scale", type=float, default=0.0, help="Learning rate scale for Layer 1 KBANN neurons (0.0=frozen, >0=unfrozen)")
    ap.add_argument("--unfreeze-after", type=int, default=1500, help="Automatically unfreeze KBANN Layer 1 (scale=0.15) after N battles")
    ap.add_argument("--eval-interval", type=int, default=5000, help="Run 1,000-battle low-temp benchmark every N battles")
    ap.add_argument("--eval-battles", type=int, default=1000, help="Number of battles in the benchmark evaluation")
    ap.add_argument("--hold-phase2", action="store_true", help="Stay in Phase 2 even if WR500 > 75% (e.g. before real-engine trial run)")
    ap.add_argument("--lora", action="store_true", help="3-layer KBANN -> Phase-1 Imitation -> Phase-2+ LoRA training")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    ws = args.workspace
    if args.fresh and os.path.exists(ws):
        shutil.rmtree(ws)
    for d in ("processed", "replays", "checkpoints"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    P = lambda *a: os.path.join(ws, *a)

    if os.path.exists(P("state.json")) and not args.fresh:
        s = json.load(open(P("state.json")))
        s["recent"] = deque(s.get("recent", []), maxlen=500)
        s["recent_frozen"] = deque(s.get("recent_frozen", []), maxlen=500)
    else:
        s = {"phase": 1 if args.phase1_battles > 0 else 2, "battle": 0, "recent": deque(maxlen=500), "recent_frozen": deque(maxlen=500)}

    w = st.load_weights(P("weights_latest.json"))
    rnn = st.load_commander_rnn(P("commander_rnn.json"))
    w_opp = st.load_weights(P("weights_frozen.json")) if os.path.exists(P("weights_frozen.json")) else None
    rnn_opp = st.load_commander_rnn(P("commander_rnn_frozen.json")) if os.path.exists(P("commander_rnn_frozen.json")) else None

    st.save_weights(w, P("weights_latest.json"))
    st.save_commander_rnn(rnn, P("commander_rnn.json"))

    ce_path = P("imitation_ce.json")
    ce_log = json.load(open(ce_path)) if (os.path.exists(ce_path) and not args.fresh) else []
    t_start = time.time()
    tot_sim_sec = 0.0
    target_end = s["battle"] + args.battles

    torch.set_num_threads(2)  # keep the main (training) process from fighting the sim workers for cores
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:

        def plan(sub, phase):
            bsz = min(args.batch, target_end - sub)
            if phase == 1 and sub < args.phase1_battles:
                bsz = min(bsz, args.phase1_battles - sub)
            return max(0, bsz)

        def submit_batch(start, bsz, phase):
            # Snapshot merged weights (excluding _lora metadata) so workers stay light
            w_s = [{k: v.copy() for k, v in rw.items() if not k.startswith("_")} for rw in w]
            rnn_s = {k: v.copy() for k, v in rnn.items() if not k.startswith("_")}
            mode = MODES[phase]
            tasks = [(bid, mode, w_s, rnn_s, w_opp, rnn_opp, (bid % 10 == 0 or bid % 25 == 0 or bid == 1), phase, ws)
                     for bid in range(start + 1, start + bsz + 1)]
            return phase, [pool.submit(_run_one, t) for t in tasks]

        submitted = s["battle"]
        eval_hist_path = P("eval_history.json")
        evaluated_battles = set()
        if os.path.exists(eval_hist_path):
            try:
                evaluated_battles = {h.get("battle") for h in json.load(open(eval_hist_path))}
            except Exception:
                pass
        last_eval_milestone = (
            (s["battle"] // args.eval_interval)
            if (args.eval_interval > 0 and s["battle"] in evaluated_battles)
            else (((s["battle"] - 1) // args.eval_interval) if args.eval_interval > 0 else 0)
        )
        pending = None
        b0 = plan(submitted, s["phase"])
        if b0:
            pending = submit_batch(submitted, b0, s["phase"])
        while pending is not None:
            batch_phase, futs = pending
            results = [f.result() for f in futs]
            submitted += len(results)
            pending = None

            # Suppress pipelining if this batch crosses an evaluation milestone so eval workers have full resources
            crossing_eval = (
                args.eval_interval > 0
                and ((s["battle"] + len(results)) // args.eval_interval > last_eval_milestone)
            )

            # Option B: start simulating the NEXT batch (with current weights) while we train on this one
            if not crossing_eval and not (batch_phase == 1 and submitted >= args.phase1_battles):
                b1 = plan(submitted, batch_phase)
                if b1:
                    pending = submit_batch(submitted, b1, batch_phase)
            role_grads = st.init_role_grads(w)
            cmd_grads = st.init_cmd_grads(rnn)
            use_lora = bool(args.lora and batch_phase >= 2)
            p1_train_all = bool(args.lora and batch_phase == 1)
            kb_lr = (1.0 if p1_train_all else 0.0) if args.lora else (args.kbann_lr_scale if args.kbann_lr_scale > 0.0 else (0.15 if (args.unfreeze_after > 0 and s["battle"] >= args.unfreeze_after) else 0.0))
            for ep, verdict in results:
                phase = batch_phase
                llm_terminal = float(verdict["penalties"]["total"]) if verdict else 0.0
                weight_mult = 3.0 if verdict else 1.0
                if phase == 1:
                    st.update_phase1(w, ep, kbann_lr_scale=kb_lr, train_all=p1_train_all, use_lora=False)
                else:
                    st.update_from_episode(w, ep, phase, None, llm_terminal, weight_mult, apply_in_place=False, role_grads=role_grads, kbann_lr_scale=kb_lr, use_lora=use_lora)
                    st.update_commander_rnn(rnn, ep, phase, llm_terminal, weight_mult, apply_in_place=False, cmd_grads=cmd_grads, kbann_lr_scale=kb_lr, use_lora=use_lora)

            if batch_phase >= 2:
                st.apply_role_grads(w, role_grads, kbann_lr_scale=kb_lr, use_lora=use_lora)
                st.apply_cmd_grads(rnn, cmd_grads, kbann_lr_scale=kb_lr, use_lora=use_lora)

            for ep, verdict in results:
                n = ep["battle_id"]
                s["battle"] = n
                tot_sim_sec += ep["duration_sec"]
                phase = batch_phase

                # Save episode summary WITH per-step (dk, dl, coh) so training_monitor shows full reward breakdown
                ep_lite = {k: v for k, v in ep.items() if k not in ("units", "commander_steps")}
                ep_lite["units"] = [
                    {
                        "role": u["role"],
                        "steps": [
                            {"dk": round(st_s.get("dk", 0.0), 4), "dl": round(st_s.get("dl", 0.0), 4), "coh": round(st_s.get("coh", 0.0), 4)}
                            for st_s in u.get("steps", [])
                        ],
                    }
                    for u in ep.get("units", [])
                ]
                json.dump(ep_lite, open(P("processed", f"episode_{n}.json"), "w"))

                if phase >= 2:
                    (s["recent_frozen"] if phase == 4 else s["recent"]).append(ep["result"])

                # Phase transitions gated on Rolling 500 Winrate
                if phase == 1 and n >= args.phase1_battles:
                    s["phase"] = 2
                    s["recent"].clear()
                    print(f"[b{n}] Phase 1 complete -> Phase 2 (rl_vs_vanilla)", flush=True)
                elif phase == 2 and not args.hold_phase2 and len(s["recent"]) >= 500 and winrate(s["recent"]) > 0.75:
                    s["phase"] = 3
                    s["recent"].clear()
                    print(f"[b{n}] Phase 2 WR(500) > 75% -> Phase 3 (rl_vs_vanilla_disadvantaged)", flush=True)
                elif phase == 3 and len(s["recent"]) >= 500 and winrate(s["recent"]) > 0.60:
                    s["phase"] = 4
                    shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
                    shutil.copy(P("commander_rnn.json"), P("commander_rnn_frozen.json"))
                    w_opp = st.load_weights(P("weights_frozen.json"))
                    rnn_opp = st.load_commander_rnn(P("commander_rnn_frozen.json"))
                    print(f"[b{n}] Phase 3 WR(500) > 60% -> Phase 4 (rl_vs_frozen)", flush=True)
                elif phase == 4 and n % 500 == 0 and len(s["recent_frozen"]) >= 500 and winrate(s["recent_frozen"]) > 0.55:
                    shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
                    shutil.copy(P("commander_rnn.json"), P("commander_rnn_frozen.json"))
                    w_opp = st.load_weights(P("weights_frozen.json"))
                    rnn_opp = st.load_commander_rnn(P("commander_rnn_frozen.json"))
                    print(f"[b{n}] Phase 4 promoted frozen weights (WR500={winrate(s['recent_frozen']):.1%})", flush=True)

                if n % 10 == 0:
                    json.dump(dict(s, recent=list(s["recent"]), recent_frozen=list(s["recent_frozen"])), open(P("state.json"), "w"))

                if n % 50 == 0:
                    st.save_weights(w, P("weights_latest.json"))
                    st.save_commander_rnn(rnn, P("commander_rnn.json"))
                    shutil.copy(P("weights_latest.json"), P("checkpoints", f"weights_iter_{n}.json"))
                    shutil.copy(P("commander_rnn.json"), P("checkpoints", f"commander_rnn_iter_{n}.json"))

                    p1_ce = None
                    if phase == 1 and st.P1_STATS["steps"] > 0:
                        p1_ce = st.P1_STATS["loss"] / st.P1_STATS["steps"]
                        ce_log.append({"battle": n, "ce": round(p1_ce, 5), "steps": st.P1_STATS["steps"]})
                        json.dump(ce_log, open(P("imitation_ce.json"), "w"))
                        st.P1_STATS["loss"], st.P1_STATS["steps"] = 0.0, 0   # windowed: CE over the last ~50 battles
                    wall = max(1e-3, time.time() - t_start)
                    q = list(s["recent_frozen"] if s["phase"] == 4 else s["recent"])
                    q50 = q[-50:]
                    print(
                        f"b{n:4d} | P{s['phase']} ({MODES[s['phase']]}) | "
                        f"WR500({len(q)})={winrate(q):.1%} | WR50={winrate(q50):.1%} | "
                        f"dur={ep['duration_sec']:5.1f}s | "
                        f"speed={tot_sim_sec/wall:,.0f}x ({(n/wall):.1f} bat/s)"
                        + (f" | imitation CE={p1_ce:.4f}" if phase == 1 else ""),
                        flush=True,
                    )

            # Benchmark evaluation trigger every N battles
            if args.eval_interval > 0 and (s["battle"] // args.eval_interval > last_eval_milestone):
                last_eval_milestone = s["battle"] // args.eval_interval
                st.save_weights(w, P("weights_latest.json"))
                st.save_commander_rnn(rnn, P("commander_rnn.json"))
                json.dump(dict(s, recent=list(s["recent"]), recent_frozen=list(s["recent_frozen"])), open(P("state.json"), "w"))
                eb.run_benchmark(
                    w, rnn, s["battle"], ws,
                    n_battles=args.eval_battles,
                    max_workers=min(args.workers, 10),
                    inv_temp=15.0,
                )

            if pending is not None and pending[0] != s["phase"]:
                for f in pending[1]:
                    f.cancel()
                pending = None
            if pending is None and submitted < target_end:
                b2 = plan(submitted, s["phase"])
                if b2:
                    pending = submit_batch(submitted, b2, s["phase"])


    st.save_weights(w, P("weights_latest.json"))
    st.save_commander_rnn(rnn, P("commander_rnn.json"))
    json.dump(dict(s, recent=list(s["recent"]), recent_frozen=list(s["recent_frozen"])), open(P("state.json"), "w"))
    wall = max(1e-3, time.time() - t_start)
    print(f"Done {s['battle']} battles in {wall:.1f}s | avg speed = {tot_sim_sec/wall:,.0f}x real-time", flush=True)


if __name__ == "__main__":
    main()
