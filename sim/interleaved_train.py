#!/usr/bin/env python3
"""
Interleaved Sim + Real Medieval II Training Loop:
  Each cycle runs:
    1) 1,000 Simulator battles (variable unit sizes, 10-battle gradient accumulation)
    2) Syncs Layer 1/2 weights to workspace/ (preserving Layer-3 adapter weights)
    3) 100 Real Medieval II battles across 3 Docker workers (Layer-3 adapter, 10-battle gradient accumulation)
    4) Syncs updated weights back to workspace_sim/ and records cycle metrics
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

BASE = "/home/tomek/medieval2-docker"
WS_SIM = os.path.join(BASE, "workspace_sim")
WS_M2TW = os.path.join(BASE, "workspace")
PY = os.path.join(BASE, ".venv", "bin", "python")


ADAPT_KEYS = ("W_adapt_h", "W_adapt_x", "b_adapt", "W_adapt_out")


def sync_sim_to_m2tw(preserve_adapter=True):
    """Copy updated Layer 1/2 weights from workspace_sim/ -> workspace/, preserving M2TW Layer-3 adapter."""
    # Unit weights (list of role dicts)
    sim_w = json.load(open(os.path.join(WS_SIM, "weights_latest.json")))
    m2_w_path = os.path.join(WS_M2TW, "weights_latest.json")
    if preserve_adapter and os.path.exists(m2_w_path):
        try:
            old_w = json.load(open(m2_w_path))
            if isinstance(old_w, list) and len(old_w) == len(sim_w):
                for sw, ow in zip(sim_w, old_w):
                    for k in ADAPT_KEYS:
                        if k in ow:
                            sw[k] = ow[k]
        except Exception:
            pass
    json.dump(sim_w, open(m2_w_path + ".tmp", "w"))
    os.replace(m2_w_path + ".tmp", m2_w_path)

    # Commander RNN (single dict)
    sim_c = json.load(open(os.path.join(WS_SIM, "commander_rnn.json")))
    m2_c_path = os.path.join(WS_M2TW, "commander_rnn.json")
    if preserve_adapter and os.path.exists(m2_c_path):
        try:
            old_c = json.load(open(m2_c_path))
            if isinstance(old_c, dict):
                for k in ADAPT_KEYS:
                    if k in old_c:
                        sim_c[k] = old_c[k]
        except Exception:
            pass
    json.dump(sim_c, open(m2_c_path + ".tmp", "w"))
    os.replace(m2_c_path + ".tmp", m2_c_path)


def _sync_sidecar(src_json, dst_json):
    src_sc = src_json + ".lora.npz"
    dst_sc = dst_json + ".lora.npz"
    if os.path.exists(src_sc):
        shutil.copy2(src_sc, dst_sc + ".tmp.npz")
        os.replace(dst_sc + ".tmp.npz", dst_sc)
    elif os.path.exists(dst_sc):
        os.remove(dst_sc)


def sync_full(src_ws, dst_ws):
    """Copy full 3-layer merged weights + LoRA sidecars between workspace_sim/ and workspace/."""
    for fname in ("weights_latest.json", "commander_rnn.json"):
        s = os.path.join(src_ws, fname)
        d = os.path.join(dst_ws, fname)
        if os.path.exists(s):
            shutil.copy2(s, d + ".tmp")
            os.replace(d + ".tmp", d)
            _sync_sidecar(s, d)


def sync_m2tw_to_sim():
    """Copy updated Layer 1/2 weights from workspace/ -> workspace_sim/ after full-network M2TW training."""
    m2_w = json.load(open(os.path.join(WS_M2TW, "weights_latest.json")))
    sim_w_path = os.path.join(WS_SIM, "weights_latest.json")
    sim_w = json.load(open(sim_w_path)) if os.path.exists(sim_w_path) else None
    if isinstance(sim_w, list) and len(sim_w) == len(m2_w):
        for sw, mw in zip(sim_w, m2_w):
            for k, v in mw.items():
                if k not in ADAPT_KEYS:
                    sw[k] = v
        out_w = sim_w
    else:
        out_w = m2_w
    json.dump(out_w, open(sim_w_path + ".tmp", "w"))
    os.replace(sim_w_path + ".tmp", sim_w_path)

    m2_c = json.load(open(os.path.join(WS_M2TW, "commander_rnn.json")))
    sim_c_path = os.path.join(WS_SIM, "commander_rnn.json")
    sim_c = json.load(open(sim_c_path)) if os.path.exists(sim_c_path) else None
    if isinstance(sim_c, dict):
        for k, v in m2_c.items():
            if k not in ADAPT_KEYS:
                sim_c[k] = v
        out_c = sim_c
    else:
        out_c = m2_c
    json.dump(out_c, open(sim_c_path + ".tmp", "w"))
    os.replace(sim_c_path + ".tmp", sim_c_path)


def run_combined_benchmark(sim_battle, args):
    """Run deterministic Sim benchmark (N=500) + frozen Real M2TW benchmark (N=20) and append to eval_history.json."""
    print(f"\n{'*'*76}", flush=True)
    print(f"** COMBINED BENCHMARK @ SIM BATTLE #{sim_battle}: Sim N={args.eval_sim_battles} + Real M2TW N={args.eval_m2tw_battles}", flush=True)
    print(f"{'*'*76}", flush=True)

    sys.path.insert(0, os.path.join(BASE, "sim"))
    sys.path.insert(0, os.path.join(WS_M2TW, "trainer"))
    if args.arch == "lstm":
        from concurrent.futures import ProcessPoolExecutor
        import train_lstm as tl
        import lstm_model as lm
        model, _ = lm.load_lstm(os.path.join(WS_SIM, "weights_lstm.pt"))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=tl._init_worker) as pool:
            rec = tl.run_sim_eval(model, sim_battle, WS_SIM, pool, n_battles=args.eval_sim_battles)
        m2tw_eval_cmd = [
            PY, "-u", os.path.join(WS_M2TW, "run_m2tw_trial.py"),
            "--arch", "lstm",
            "--battles", str(args.eval_m2tw_battles),
            "--mode", "rl_vs_vanilla",
            "--containers", *args.containers,
        ]
    else:
        import sarsa_trainer as st
        import eval_benchmark as eb
        w_sim = st.load_weights(os.path.join(WS_SIM, "weights_latest.json"))
        rnn_sim = st.load_commander_rnn(os.path.join(WS_SIM, "commander_rnn.json"))
        rec = eb.run_benchmark(
            w_sim, rnn_sim, battle_n=sim_battle, ws=WS_SIM,
            n_battles=args.eval_sim_battles, max_workers=args.workers, inv_temp=15.0,
        )
        m2tw_eval_cmd = [
            PY, "-u", os.path.join(WS_M2TW, "run_m2tw_trial.py"),
            "--battles", str(args.eval_m2tw_battles),
            "--mode", "rl_vs_vanilla",
            "--containers", *args.containers,
        ]
    subprocess.run(m2tw_eval_cmd)
    summary_path = os.path.join(WS_M2TW, "m2tw_trial_summary.json")
    if os.path.exists(summary_path):
        ms = json.load(open(summary_path))
        rec["m2tw_wr"] = float(ms.get("win_rate_pct", 0.0))
        rec["m2tw_dmg"] = float(ms.get("avg_enemy_dmg_pct", 0.0))
        rec["m2tw_wins"] = int(ms.get("wins", 0))
        rec["m2tw_losses"] = int(ms.get("losses", 0))
        rec["m2tw_n"] = int(ms.get("completed", 0))

    eval_hist_path = os.path.join(WS_SIM, "eval_history.json")
    eval_hist = json.load(open(eval_hist_path)) if os.path.exists(eval_hist_path) else []
    # eb.run_benchmark() already appended the Sim-only record for sim_battle; replace it in-place with the combined record
    if eval_hist and eval_hist[-1].get("battle") == sim_battle:
        eval_hist[-1] = rec
    else:
        eval_hist.append(rec)
    tmp = eval_hist_path + ".tmp"
    json.dump(eval_hist, open(tmp, "w"), indent=2)
    os.replace(tmp, eval_hist_path)
    print(
        f"[+] Combined Benchmark #{sim_battle} saved: Sim WR={rec.get('overall_wr', 0):.1f}% | "
        f"M2TW WR={rec.get('m2tw_wr', 0):.1f}% ({rec.get('m2tw_wins', 0)}W/{rec.get('m2tw_losses', 0)}L)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-battles", type=int, default=1000)
    ap.add_argument("--m2tw-battles", type=int, default=100)
    ap.add_argument("--m2tw-min-sim-wr", type=float, default=30.0, help="Stay in Phase 2 (Sim-only) until Sim WR500 >= this %%, then unlock Phase 2.5 (Hybrid Sim+M2TW)")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--m2tw-batch", type=int, default=0, help="Override batch size for M2TW stage (0 = use --batch)")
    ap.add_argument("--m2tw-train-full", action="store_true", help="Train the entire network (L1+L2+L3) on M2TW battles and sync L1/L2 back to Sim")
    ap.add_argument("--lora", action="store_true", help="3-layer KBANN -> Phase-1 Imitation -> Phase-2+ LoRA across both Sim and M2TW")
    ap.add_argument("--arch", default="legacy", choices=["legacy", "lstm"], help="Model architecture (legacy or full-army lstm)")
    ap.add_argument("--m2tw-uniform-lr", type=float, default=0.0, help="Uniform LR across all layers during M2TW stage (0 = default)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=3000)
    ap.add_argument("--eval-sim-battles", type=int, default=500)
    ap.add_argument("--eval-m2tw-battles", type=int, default=20)
    ap.add_argument("--run-eval-first", action="store_true", help="Run combined benchmark immediately before Cycle 1")
    ap.add_argument("--fresh", action="store_true", help="Start from fresh KBANN weights on Cycle 1")
    ap.add_argument("--containers", nargs="+", default=["medieval2-ubuntu", "medieval2-worker2", "medieval2-worker3"])
    args = ap.parse_args()

    # Keep training low-priority and single-threaded per worker so the host stays responsive
    try:
        os.nice(15)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    hist_path = os.path.join(WS_SIM, "m2tw_cycle_history.json")
    history = [] if args.fresh else (json.load(open(hist_path)) if os.path.exists(hist_path) else [])

    if args.run_eval_first:
        st_now = json.load(open(os.path.join(WS_SIM, "state.json"))) if os.path.exists(os.path.join(WS_SIM, "state.json")) else {}
        run_combined_benchmark(int(st_now.get("battle", 0)), args)

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'='*76}", flush=True)
        print(f"== CYCLE {cycle}/{args.cycles}: {args.sim_battles} SIM BATTLES (batch={args.batch}) -> {args.m2tw_battles} M2TW BATTLES (batch={args.batch})", flush=True)
        print(f"{'='*76}", flush=True)

        # Stage A: Sim battles
        if args.arch == "lstm":
            sim_cmd = [
                PY, "-u", os.path.join(BASE, "sim", "train_lstm.py"),
                "--battles", str(args.sim_battles),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--eval-interval", "0",
                "--workspace", WS_SIM,
            ]
        else:
            sim_cmd = [
                PY, "-u", os.path.join(BASE, "sim", "sim_train.py"),
                "--battles", str(args.sim_battles),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--hold-phase2",
                "--eval-interval", "0",
                "--workspace", WS_SIM,
            ]
            if args.lora:
                sim_cmd.append("--lora")
        if args.fresh and cycle == 1:
            sim_cmd.append("--fresh")
        t0 = time.time()
        rc = subprocess.run(sim_cmd).returncode
        if rc != 0:
            print(f"[!] Sim trainer exited with code {rc}, aborting.", flush=True)
            sys.exit(rc)
        print(f"[+] Cycle {cycle} Sim stage finished in {time.time()-t0:.1f}s.", flush=True)

        # Sync weights from Sim -> M2TW
        if args.arch == "lstm":
            src_pt = os.path.join(WS_SIM, "weights_lstm.pt")
            dst_pt = os.path.join(WS_M2TW, "weights_lstm.pt")
            shutil.copy2(src_pt, dst_pt + ".tmp")
            os.replace(dst_pt + ".tmp", dst_pt)
        elif args.lora:
            sync_full(WS_SIM, WS_M2TW)
        else:
            sync_sim_to_m2tw(preserve_adapter=not (args.fresh and cycle == 1))

        # Stage B: Real Medieval II battles (Phase 2.5 Hybrid — gated on Sim WR >= --m2tw-min-sim-wr)
        state_path = os.path.join(WS_SIM, "state.json")
        sim_state_now = json.load(open(state_path)) if os.path.exists(state_path) else {}
        if args.arch == "lstm" and float(sim_state_now.get("phase", 2)) < 2.0:
            print(f"[*] Cycle {cycle} still in LSTM Phase-1 Imitation (battle #{sim_state_now.get('battle', 0)}); continuing Sim imitation before RL.", flush=True)
            continue

        cur_phase_val = float(sim_state_now.get("phase", 2))
        rec_list = sim_state_now.get("recent", [])
        sim_wr500 = (100.0 * sum(1 for r in rec_list if r > 0) / len(rec_list)) if rec_list else 0.0
        hybrid_unlocked = (cur_phase_val >= 2.5)
        if not hybrid_unlocked and sim_wr500 >= args.m2tw_min_sim_wr:
            hybrid_unlocked = True
            sim_state_now["phase"] = 2.5
            tmp_s = state_path + ".tmp"
            json.dump(sim_state_now, open(tmp_s, "w"))
            os.replace(tmp_s, state_path)
            print(
                f"\n[+] SIM WR500 REACHED {sim_wr500:.1f}% (>= {args.m2tw_min_sim_wr:.1f}%) -> UNLOCKING PHASE 2.5 HYBRID (3k Sim + 100 M2TW)!\n",
                flush=True,
            )

        if args.arch == "lstm" and not hybrid_unlocked:
            print(
                f"[*] Phase 2 (Sim-only): Sim WR500={sim_wr500:.1f}% < {args.m2tw_min_sim_wr:.1f}% target — "
                f"skipping M2TW & evals until Phase 2.5 (30% Sim WR) is unlocked.",
                flush=True,
            )
            continue

        m2tw_mode = "rl_vs_vanilla" if args.arch == "lstm" else ("vanilla_vs_vanilla" if sim_state_now.get("phase", 2) == 1 else "rl_vs_vanilla")
        m2tw_bs = args.m2tw_batch if args.m2tw_batch > 0 else args.batch
        print(
            f"[*] Cycle {cycle} M2TW stage starting (arch={args.arch}, mode={m2tw_mode}, batch={m2tw_bs})...",
            flush=True,
        )
        if args.arch == "lstm":
            m2tw_cmd = [
                PY, "-u", os.path.join(WS_M2TW, "run_m2tw_trial.py"),
                "--arch", "lstm",
                "--train-lstm",
                "--battles", str(args.m2tw_battles),
                "--mode", "rl_vs_vanilla",
                "--batch-size", str(m2tw_bs),
                "--containers", *args.containers,
            ]
        else:
            train_flag = "--train-lora" if args.lora else ("--train-full" if args.m2tw_train_full else "--train-adapter")
            m2tw_cmd = [
                PY, "-u", os.path.join(WS_M2TW, "run_m2tw_trial.py"),
                "--battles", str(args.m2tw_battles),
                "--mode", m2tw_mode,
                train_flag,
                "--batch-size", str(m2tw_bs),
                "--containers", *args.containers,
            ]
            if args.m2tw_uniform_lr > 0.0:
                m2tw_cmd += ["--uniform-lr", str(args.m2tw_uniform_lr)]
        t1 = time.time()
        rc = subprocess.run(m2tw_cmd).returncode
        if rc != 0:
            print(f"[!] run_m2tw_trial.py exited with code {rc}, continuing to next cycle.", flush=True)
        elif args.arch == "lstm":
            src_pt = os.path.join(WS_M2TW, "weights_lstm.pt")
            dst_pt = os.path.join(WS_SIM, "weights_lstm.pt")
            shutil.copy2(src_pt, dst_pt + ".tmp")
            os.replace(dst_pt + ".tmp", dst_pt)
            print(f"[+] Synced M2TW Full-Army LSTM weights back to workspace_sim/.", flush=True)
        elif args.lora:
            sync_full(WS_M2TW, WS_SIM)
            print(f"[+] Synced M2TW LoRA/3-layer weights back to workspace_sim/.", flush=True)
        elif args.m2tw_train_full:
            sync_m2tw_to_sim()
            print(f"[+] Synced M2TW-trained L1/L2 weights back to workspace_sim/.", flush=True)
        print(f"[+] Cycle {cycle} M2TW stage finished in {time.time()-t1:.1f}s.", flush=True)

        # Record cycle summary
        summary_path = os.path.join(WS_M2TW, "m2tw_trial_summary.json")
        if os.path.exists(summary_path):
            ms = json.load(open(summary_path))
            sim_state = json.load(open(os.path.join(WS_SIM, "state.json"))) if os.path.exists(os.path.join(WS_SIM, "state.json")) else {}
            history.append({
                "cycle": cycle,
                "sim_battle": sim_state.get("battle", cycle * args.sim_battles),
                "m2tw_completed": ms.get("completed", 0),
                "m2tw_wr_pct": ms.get("win_rate_pct", 0.0),
                "m2tw_wins": ms.get("wins", 0),
                "m2tw_losses": ms.get("losses", 0),
                "m2tw_avg_dmg_pct": ms.get("avg_enemy_dmg_pct", 0.0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            json.dump(history, open(hist_path, "w"), indent=2)

        # Run combined Sim (N=200) + Real M2TW (N=20) benchmark every eval_interval Phase-2 sim battles
        sim_state_end = json.load(open(os.path.join(WS_SIM, "state.json"))) if os.path.exists(os.path.join(WS_SIM, "state.json")) else {}
        cur_sim_b = int(sim_state_end.get("battle", cycle * args.sim_battles))
        if args.eval_interval > 0 and ((cycle * args.sim_battles) % args.eval_interval == 0):
            run_combined_benchmark(cur_sim_b, args)


if __name__ == "__main__":
    main()
