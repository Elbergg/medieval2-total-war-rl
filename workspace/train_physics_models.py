#!/usr/bin/env python3
"""Collect real Medieval II battles & train Self-Attention DamageNet + MoraleNet until < 5% error.

Runs real M2TW battles across all 3 Docker containers (arch=lstm so all 48 units' full
state is logged every 4.5s tick), splits 80/20 by battle into train/val, trains DamageNet
and MoraleNet after every batch, logs live curves to workspace_sim/physics_model_history.json
for the dashboard, and stops once BOTH validation errors are < 5.0%.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import torch

BASE = "/home/tomek/medieval2-docker"
WS_M2TW = os.path.join(BASE, "workspace")
WS_SIM = os.path.join(BASE, "workspace_sim")
PY = os.path.join(BASE, ".venv", "bin", "python")

sys.path.insert(0, os.path.join(WS_M2TW, "trainer"))
import physics_world_model as pwm  # noqa: E402


from lstm_model import OBS_DIM  # noqa: E402


def load_m2tw_lstm_episodes():
    """Load ONLY real M2TW episodes that have all 26 real unit attributes (obs_len == OBS_DIM = 1260)."""
    paths = sorted(
        glob.glob(os.path.join(WS_M2TW, "processed", "episode_*.json"))
        + glob.glob(os.path.join(WS_M2TW, "episodes", "episode_*.json")),
        key=os.path.getmtime,
    )
    seen = set()
    eps = []
    for p in paths:
        bid = os.path.basename(p)
        if bid in seen:
            continue
        seen.add(bid)
        try:
            ep = json.load(open(p))
            steps = ep.get("steps") or []
            if ep.get("arch") == "lstm" and len(steps) >= 4 and len(steps[0].get("obs") or []) == OBS_DIM:
                eps.append(ep)
        except Exception:
            pass
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-battles", type=int, default=15, help="Real M2TW battles to collect per round")
    ap.add_argument("--min-battles", type=int, default=45, help="Minimum real M2TW battles before allowing <5% early stop")
    ap.add_argument("--max-battles", type=int, default=450, help="Hard cap on total M2TW battles")
    ap.add_argument("--target-err", type=float, default=5.0, help="Stop when BOTH DamageNet & MoraleNet val error < target (%)")
    ap.add_argument("--epochs-per-round", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--containers", nargs="+", default=["medieval2-ubuntu", "medieval2-worker2", "medieval2-worker3"])
    args = ap.parse_args()

    try:
        os.nice(15)
    except Exception:
        pass
    torch.set_num_threads(2)
    os.makedirs(WS_SIM, exist_ok=True)

    dmg_path = os.path.join(WS_M2TW, "damage_net.pt")
    mor_path = os.path.join(WS_M2TW, "morale_net.pt")
    hist_path = os.path.join(WS_SIM, "physics_model_history.json")

    if args.fresh:
        for p in (dmg_path, mor_path, hist_path):
            if os.path.exists(p):
                os.remove(p)

    dmg_net = pwm.load_physics_net(dmg_path, target_kind="damage")
    mor_net = pwm.load_physics_net(mor_path, target_kind="morale")
    dmg_opt = torch.optim.AdamW(dmg_net.parameters(), lr=args.lr, weight_decay=1e-4)
    mor_opt = torch.optim.AdamW(mor_net.parameters(), lr=args.lr, weight_decay=1e-4)
    hist = json.load(open(hist_path)) if (os.path.exists(hist_path) and not args.fresh) else []

    n_params = sum(p.numel() for p in dmg_net.parameters())
    print(f"[*] Physics World Model Trainer: DamageNet ({n_params:,} params) + MoraleNet ({n_params:,} params)", flush=True)
    print(f"[*] Target: BOTH val errors < {args.target_err:.1f}% (min {args.min_battles} real M2TW battles)", flush=True)

    rnd = len(hist)
    while True:
        eps = load_m2tw_lstm_episodes()
        n_eps = len(eps)

        # Always collect a fresh batch of real M2TW battles if we need more data
        if n_eps < args.min_battles or rnd > 0:
            if n_eps >= args.max_battles:
                print(f"[!] Reached max_battles={args.max_battles}, stopping.", flush=True)
                break
            print(f"\n[Round {rnd + 1}] Collecting {args.batch_battles} real M2TW battles (have {n_eps})...", flush=True)
            cmd = [
                PY, "-u", os.path.join(WS_M2TW, "run_m2tw_trial.py"),
                "--arch", "lstm",
                "--battles", str(args.batch_battles),
                "--mode", "rl_vs_vanilla",
                "--containers", *args.containers,
            ]
            subprocess.run(cmd)
            # Move newly collected episodes into processed/ so they're archived cleanly
            for ep_p in glob.glob(os.path.join(WS_M2TW, "episodes", "episode_*.json")):
                try:
                    os.replace(ep_p, os.path.join(WS_M2TW, "processed", os.path.basename(ep_p)))
                except OSError:
                    pass
            eps = load_m2tw_lstm_episodes()
            n_eps = len(eps)

        if n_eps < 5:
            print("[!] Need at least 5 episodes to split train/val, retrying...", flush=True)
            continue

        # Deterministic 80/20 battle-level split (every 5th battle is held-out validation)
        train_eps = [e for i, e in enumerate(eps) if (i % 5) != 4]
        val_eps = [e for i, e in enumerate(eps) if (i % 5) == 4]

        X_tr, M_tr, Yd_tr, Ym_tr = pwm.extract_transitions(train_eps)
        if X_tr is None or len(X_tr) < 10:
            continue

        dmg_loss = pwm.train_physics_epoch(dmg_net, dmg_opt, X_tr, M_tr, Yd_tr, batch_size=64, epochs=args.epochs_per_round)
        mor_loss = pwm.train_physics_epoch(mor_net, mor_opt, X_tr, M_tr, Ym_tr, batch_size=64, epochs=args.epochs_per_round)

        pwm.save_physics_net(dmg_net, dmg_path)
        pwm.save_physics_net(mor_net, mor_path)

        dmg_eval = pwm.evaluate_physics_net(dmg_net, val_eps)
        mor_eval = pwm.evaluate_physics_net(mor_net, val_eps)

        rnd += 1
        rec = {
            "round": rnd,
            "battles": n_eps,
            "train_ticks": int(len(X_tr)),
            "val_ticks": dmg_eval["n_ticks"],
            "dmg_val_mae_pct": dmg_eval["mae_pct"],
            "dmg_cum_err_pct": dmg_eval["cum_err_pct"],
            "mor_val_mae_pct": mor_eval["mae_pct"],
            "mor_final_err_pct": mor_eval["cum_err_pct"],
            "dmg_loss": dmg_loss,
            "mor_loss": mor_loss,
            "target_pct": args.target_err,
            "converged": (n_eps >= args.min_battles and dmg_eval["cum_err_pct"] < args.target_err and mor_eval["cum_err_pct"] < args.target_err),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        hist.append(rec)
        tmp = hist_path + ".tmp"
        json.dump(hist, open(tmp, "w"), indent=2)
        os.replace(tmp, hist_path)

        print(
            f"[Physics Round {rnd:2d} | {n_eps:3d} M2TW battles | {len(X_tr)} train ticks] "
            f"DamageNet: tick_MAE={dmg_eval['mae_pct']:4.2f}% battle_err={dmg_eval['cum_err_pct']:5.2f}% | "
            f"MoraleNet: tick_MAE={mor_eval['mae_pct']:4.2f}% final_err={mor_eval['cum_err_pct']:5.2f}% "
            f"(target < {args.target_err:.1f}%)",
            flush=True,
        )

        if rec["converged"]:
            print(f"\n[+] CONVERGED! Both DamageNet ({dmg_eval['cum_err_pct']:.2f}%) and MoraleNet ({mor_eval['mae_pct']:.2f}%) are < {args.target_err:.1f}% error!", flush=True)
            break


if __name__ == "__main__":
    main()
