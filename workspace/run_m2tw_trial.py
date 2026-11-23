#!/usr/bin/env python3
"""Run a frozen-weight N-battle evaluation trial in original Medieval II: Total War."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import fcntl

WORKSPACE = "/home/tomek/medieval2-docker/workspace"
BASE_DIR = "/home/tomek/medieval2-docker"


def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_wid(container):
    if "worker2" in container:
        return "2"
    if "worker3" in container:
        return "3"
    return "1"


def is_game_running(container):
    return run_cmd(f"docker exec {container} pgrep -f medieval2.exe").returncode == 0


def kill_game(container):
    run_cmd(f"docker exec {container} pkill -9 -f medieval2.exe")
    run_cmd(f"docker exec {container} pkill -9 -f m2tweop_starter.exe")
    run_cmd(f"docker exec {container} wineserver -k")
    run_cmd(f"docker exec {container} pkill -9 -f Xvfb")
    time.sleep(0.5)


def start_game(container):
    wid = get_wid(container)
    run_cmd(f"docker exec {container} rm -f /home/tomek/game/logs/system.log.txt /home/tomek/workspace/rl_heartbeat_{wid}.txt")
    if wid == "1":
        run_cmd(f"docker exec {container} rm -f /home/tomek/workspace/rl_heartbeat.txt")
    run_cmd(
        f"docker exec -d {container} bash -c "
        f"'cd /home/tomek/game && xvfb-run -a -s \"-screen 0 1024x768x24\" "
        f"wine m2tweop_starter.exe . > /home/tomek/workspace/m2tweop_starter_{container}.log 2>&1'"
    )
    for _ in range(30):
        time.sleep(1)
        if is_game_running(container):
            return True
    return False


def claim_battle_id(container):
    wid = get_wid(container)
    lock_path = os.path.join(WORKSPACE, ".battle_id.lock")
    counter_path = os.path.join(WORKSPACE, "next_battle_id.txt")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            bid = int(open(counter_path).read().strip()) if os.path.exists(counter_path) else 1
            open(counter_path, "w").write(str(bid + 1))
            open(os.path.join(WORKSPACE, f"worker_battle_id_{wid}.txt"), "w").write(str(bid))
            return bid
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def wait_for_episode(container, target_id, start_time, timeout=240):
    start = time.time()
    ep_path = os.path.join(WORKSPACE, "episodes", f"episode_{target_id}.json")
    while time.time() - start < timeout:
        if os.path.exists(ep_path):
            try:
                if os.path.getmtime(ep_path) >= start_time - 2.0 and os.path.getsize(ep_path) > 50:
                    ep = json.load(open(ep_path))
                    return True, ep
            except Exception:
                pass
        if not is_game_running(container):
            return False, None
        time.sleep(3)
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=100)
    ap.add_argument("--containers", nargs="+", default=["medieval2-ubuntu", "medieval2-worker2", "medieval2-worker3"])
    ap.add_argument("--train-adapter", action="store_true", help="Train Layer-3 adapter online with gradient accumulation")
    ap.add_argument("--train-full", action="store_true", help="Train the entire network (L1 KBANN + L2 Top RNN + L3 Adapter) online")
    ap.add_argument("--train-lora", action="store_true", help="Train rank-4 LoRA adapters across all 3 layers (base weights frozen)")
    ap.add_argument("--arch", default="legacy", choices=["legacy", "lstm"], help="Model architecture (legacy or full-army lstm)")
    ap.add_argument("--train-lstm", action="store_true", help="Train Full-Army LSTM with large-batch A2C Policy Gradient (no PPO)")
    ap.add_argument("--lstm-lr", type=float, default=5.0e-4, help="Learning rate for Full-Army LSTM on M2TW")
    ap.add_argument("--kbann-lr-scale", type=float, default=0.15, help="Layer-1 KBANN LR multiplier when --train-full is set (default 0.15)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="Global LR multiplier applied to LR_BASE, ADAPT_LR, LORA_LR (default 1.0)")
    ap.add_argument("--uniform-lr", type=float, default=0.0, help="If > 0, use this exact same LR for active layers / LoRA")
    ap.add_argument("--batch-size", type=int, default=10, help="Battles per gradient accumulation step")
    ap.add_argument("--mode", default="rl_vs_vanilla", choices=["rl_vs_vanilla", "vanilla_vs_vanilla"], help="Battle mode (vanilla_vs_vanilla = Phase 1 imitation of real M2TW Vanilla AI)")
    args = ap.parse_args()
    is_lstm = (args.arch == "lstm" or args.train_lstm)
    do_train = (not is_lstm) and (args.train_adapter or args.train_full or args.train_lora)
    use_lora = bool(args.train_lora and args.mode != "vanilla_vs_vanilla")
    train_all = bool(args.train_full or (args.train_lora and args.mode == "vanilla_vs_vanilla"))
    adapter_only = bool(args.train_adapter and not train_all and not use_lora)
    kb_lr_scale = 1.0 if (args.train_lora and args.mode == "vanilla_vs_vanilla") else (args.kbann_lr_scale if train_all else 0.0)

    # Update workspace/scenario.json mode & arch so rl_battle_ai.lua runs in the requested mode
    sc_path = os.path.join(WORKSPACE, "scenario.json")
    if os.path.exists(sc_path):
        try:
            sc_data = json.load(open(sc_path))
            sc_data["mode"] = args.mode
            sc_data["arch"] = "lstm" if is_lstm else "legacy"
            json.dump(sc_data, open(sc_path + ".tmp", "w"), indent=1)
            os.replace(sc_path + ".tmp", sc_path)
        except Exception:
            pass

    os.chdir(BASE_DIR)
    os.makedirs(os.path.join(WORKSPACE, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE, "processed"), exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE, "replays"), exist_ok=True)

    st = None
    lstm_mod = lstm_model = lstm_opt = None
    lstm_ep_buf = []
    w_tr = rnn_tr = role_grads = cmd_grads = None
    batch_cnt = 0
    m2tw_ce_path = os.path.join(WORKSPACE, "m2tw_imitation_ce.json")
    m2tw_ce_log = json.load(open(m2tw_ce_path)) if os.path.exists(m2tw_ce_path) else []
    if is_lstm and args.train_lstm:
        sys.path.insert(0, os.path.join(WORKSPACE, "trainer"))
        import lstm_model as lm
        lstm_mod = lm
        lstm_model, lstm_opt = lm.load_lstm(os.path.join(WORKSPACE, "weights_lstm.pt"), lr=args.lstm_lr)
        print(f"[*] Training Full-Army LSTM (lr={args.lstm_lr:.2e}, batch={args.batch_size}, NO PPO)", flush=True)
    elif do_train:
        sys.path.insert(0, os.path.join(WORKSPACE, "trainer"))
        import sarsa_trainer as st
        if args.uniform_lr > 0.0:
            st.LR_BASE = args.uniform_lr
            st.ADAPT_LR = args.uniform_lr
            st.LORA_LR = args.uniform_lr
            if train_all and not args.train_lora:
                kb_lr_scale = 1.0
        else:
            st.LR_BASE *= args.lr_scale
            st.ADAPT_LR *= args.lr_scale
            st.LORA_LR *= args.lr_scale
        if use_lora:
            print(f"[*] Training LoRA (rank={st.LORA_RANK}, lr={st.LORA_LR:.2e}, base weights frozen)", flush=True)
        else:
            print(
                f"[*] Training LRs: "
                f"L1={st.LR_BASE * kb_lr_scale:.2e}, L2={st.LR_BASE if train_all else 0.0:.2e}, L3={st.ADAPT_LR:.2e}",
                flush=True,
            )
        w_tr = st.load_weights(os.path.join(WORKSPACE, "weights_latest.json"))
        rnn_tr = st.load_commander_rnn(os.path.join(WORKSPACE, "commander_rnn.json"))
        role_grads = st.init_role_grads(w_tr)
        cmd_grads = st.init_cmd_grads(rnn_tr)

    lock = threading.Lock()
    results = []
    t_start = time.time()
    summary_path = os.path.join(WORKSPACE, "m2tw_trial_summary.json")

    def save_summary():
        n = len(results)
        wins = sum(1 for r in results if r["result"] == 1)
        losses = sum(1 for r in results if r["result"] == -1)
        draws = sum(1 for r in results if r["result"] == 0)
        avg_dmg = (sum(r["enemy_dmg"] for r in results) / n) if n else 0.0
        avg_own = (sum(r["own_loss"] for r in results) / n) if n else 0.0
        avg_dur = (sum(r["duration_sec"] for r in results) / n) if n else 0.0
        out = {
            "target_battles": args.battles,
            "completed": n,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate_pct": round(100.0 * wins / n, 1) if n else 0.0,
            "avg_enemy_dmg_pct": round(100.0 * avg_dmg, 1),
            "avg_own_loss_pct": round(100.0 * avg_own, 1),
            "avg_duration_sec": round(avg_dur, 1),
            "elapsed_wall_sec": round(time.time() - t_start, 1),
            "battles": results,
        }
        tmp = summary_path + ".tmp"
        json.dump(out, open(tmp, "w"), indent=2)
        os.replace(tmp, summary_path)
        return out

    def worker_loop(container):
        wid = get_wid(container)
        pending_id = None
        while True:
            with lock:
                if len(results) >= args.battles:
                    break
            kill_game(container)
            if not start_game(container):
                print(f"[{container}] Failed to boot game, retrying in 5s...", flush=True)
                time.sleep(5)
                continue

            if pending_id is not None:
                curr_id = pending_id
                open(os.path.join(WORKSPACE, f"worker_battle_id_{wid}.txt"), "w").write(str(curr_id))
            else:
                curr_id = claim_battle_id(container)

            t0 = time.time()
            res = subprocess.run(
                [sys.executable, os.path.join(WORKSPACE, "launch_scenario_battle.py"), "--container", container],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if res.returncode != 0:
                print(f"[{container}] Launch failed for #{curr_id}, retrying...", flush=True)
                pending_id = curr_id
                continue

            ok, ep = wait_for_episode(container, curr_id, t0, timeout=480)
            if ok and ep is not None:
                pending_id = None
                r_val = int(ep.get("result", 0))
                en_dmg = float(ep.get("enemy_damage_frac", 0.0))
                dec_w = bool(ep.get("decisive_win", False))
                dec_l = bool(ep.get("decisive_loss", False))
                wall_s = time.time() - t0
                with lock:
                    if len(results) < args.battles:
                        results.append({
                            "battle_id": curr_id,
                            "container": container,
                            "result": r_val,
                            "enemy_dmg": round(en_dmg, 4),
                            "decisive_win": dec_w,
                            "decisive_loss": dec_l,
                            "own_loss": 0.0,
                            "duration_sec": round(wall_s, 1),
                        })
                        s = save_summary()
                        tag = "WIN" if r_val == 1 else ("LOSS" if r_val == -1 else "DRAW")
                        dec_tag = " [DEC]" if (dec_w or dec_l) else ""
                        print(
                            f"[M2TW Trial {s['completed']:3d}/{args.battles}] #{curr_id} ({container}) | "
                            f"{tag:4s}{dec_tag:6s} | en_dmg={en_dmg*100:5.1f}% | wall={wall_s:4.1f}s | "
                            f"Running WR={s['win_rate_pct']:5.1f}% ({s['wins']}W/{s['losses']}L/{s['draws']}D) | AvgDmg={s['avg_enemy_dmg_pct']:5.1f}%",
                            flush=True,
                        )
                        if is_lstm and args.train_lstm and lstm_mod is not None:
                            lstm_ep_buf.append(ep)
                            if len(lstm_ep_buf) >= args.batch_size:
                                stats = lstm_mod.update_batch(lstm_model, lstm_opt, lstm_ep_buf)
                                lstm_mod.save_lstm(lstm_model, os.path.join(WORKSPACE, "weights_lstm.pt"), opt=lstm_opt)
                                print(
                                    f"  [LSTM Batch Update @ M2TW {s['completed']}/{args.battles}] "
                                    f"batch={len(lstm_ep_buf)} pg={stats['pg_loss']:+.3f} vf={stats['vf_loss']:.3f} ent={stats['entropy']:.2f}",
                                    flush=True,
                                )
                                lstm_ep_buf.clear()
                            ep_src = os.path.join(WORKSPACE, "episodes", f"episode_{curr_id}.json")
                            if os.path.exists(ep_src):
                                os.replace(ep_src, os.path.join(WORKSPACE, "processed", f"episode_{curr_id}.json"))
                        elif do_train and st is not None:
                            nonlocal batch_cnt, role_grads, cmd_grads
                            tr_phase = 1 if args.mode == "vanilla_vs_vanilla" else 2
                            p1_loss_0, p1_steps_0 = st.P1_STATS["loss"], st.P1_STATS["steps"]
                            st.update_from_episode(
                                w_tr, ep, phase=tr_phase, penalties=None, llm_terminal_penalty=0.0,
                                weight_mult=1.0, apply_in_place=False, role_grads=role_grads,
                                kbann_lr_scale=kb_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora,
                            )
                            if tr_phase == 1:
                                d_steps = st.P1_STATS["steps"] - p1_steps_0
                                if d_steps > 0:
                                    ep_ce = (st.P1_STATS["loss"] - p1_loss_0) / d_steps
                                    m2tw_ce_log.append({"battle": curr_id, "ce": round(float(ep_ce), 4)})
                                    tmp_ce = m2tw_ce_path + ".tmp"
                                    json.dump(m2tw_ce_log, open(tmp_ce, "w"))
                                    os.replace(tmp_ce, m2tw_ce_path)
                                    print(f"  [M2TW P1 Imitation] #{curr_id} CE={ep_ce:.4f} ({d_steps} steps)", flush=True)
                            if tr_phase >= 2:
                                st.update_commander_rnn(
                                    rnn_tr, ep, phase=tr_phase, llm_terminal_penalty=0.0,
                                    weight_mult=1.0, apply_in_place=False, cmd_grads=cmd_grads,
                                    kbann_lr_scale=kb_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora,
                                )
                            batch_cnt += 1
                            if batch_cnt >= args.batch_size:
                                st.apply_role_grads(w_tr, role_grads, kbann_lr_scale=kb_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
                                if tr_phase >= 2:
                                    st.apply_cmd_grads(rnn_tr, cmd_grads, kbann_lr_scale=kb_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
                                st.save_weights(w_tr, os.path.join(WORKSPACE, "weights_latest.json"))
                                st.save_commander_rnn(rnn_tr, os.path.join(WORKSPACE, "commander_rnn.json"))
                                u_norm = max(float(abs(rw["W_adapt_out"]).max()) for rw in w_tr)
                                c_norm = float(abs(rnn_tr["W_adapt_out"]).max())
                                mode_lbl = "LoRA" if use_lora else ("FullNet" if train_all else "Adapter")
                                print(
                                    f"  [{mode_lbl} Batch Update P{tr_phase} @ M2TW {s['completed']}/{args.battles}] "
                                    f"max|W_adapt_out| unit={u_norm:.4f} cmd={c_norm:.4f}",
                                    flush=True,
                                )
                                role_grads = st.init_role_grads(w_tr)
                                cmd_grads = st.init_cmd_grads(rnn_tr)
                                batch_cnt = 0
                            ep_src = os.path.join(WORKSPACE, "episodes", f"episode_{curr_id}.json")
                            if os.path.exists(ep_src):
                                os.replace(ep_src, os.path.join(WORKSPACE, "processed", f"episode_{curr_id}.json"))
            else:
                print(f"[{container}] Episode #{curr_id} timed out or crashed, retrying...", flush=True)
                pending_id = curr_id
            kill_game(container)

    print(f"[*] Starting {args.battles}-battle original Medieval II trial across {args.containers}...", flush=True)
    save_summary()
    threads = [threading.Thread(target=worker_loop, args=(c,), daemon=True) for c in args.containers]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    final = save_summary()
    print("\n" + "=" * 72, flush=True)
    print(f"== ORIGINAL MEDIEVAL II TRIAL COMPLETE ({final['completed']} BATTLES in {final['elapsed_wall_sec']:.1f}s)", flush=True)
    print(f"== WIN RATE: {final['win_rate_pct']:.1f}% ({final['wins']}W / {final['losses']}L / {final['draws']}D)", flush=True)
    print(f"== AVG ENEMY CASUALTIES: {final['avg_enemy_dmg_pct']:.1f}% | AVG OWN LOSSES: {final['avg_own_loss_pct']:.1f}%", flush=True)
    print("=" * 72 + "\n", flush=True)


if __name__ == "__main__":
    main()
