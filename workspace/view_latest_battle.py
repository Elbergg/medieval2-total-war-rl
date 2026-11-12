#!/usr/bin/env python3
"""
View Latest Battle Recording:
Fetches the latest recorded battle replay, clearly displays which side is our
Reinforcement Learning AI vs the Default Total War AI, shows tactical stats,
and renders/opens the visual tactical map video.

Usage:
  python3 workspace/view_latest_battle.py [--id <battle_id>] [--open]
"""
import os
import sys
import glob
import json
import argparse
import subprocess

WORKSPACE = "/home/tomek/medieval2-docker/workspace"
REPLAYS_DIR = os.path.join(WORKSPACE, "replays")

# ANSI Color codes for rich terminal display
CYAN = "\033[96m"
BLUE = "\033[94m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

MODE_DESCRIPTIONS = {
    "vanilla_vs_vanilla": (
        "Phase 1: Imitation Learning Baseline",
        f"{YELLOW}Default Vanilla AI (Observed for Warm-Start){RESET}",
        f"{RED}Default Vanilla Engine AI{RESET}"
    ),
    "rl_vs_vanilla": (
        "Phase 2: Active RL vs Game Engine",
        f"{CYAN}{BOLD}OUR RL AI (Linear SARSA(λ) Learner){RESET}",
        f"{RED}{BOLD}DEFAULT ENGINE AI (Native Vanilla C++ AI){RESET}"
    ),
    "rl_vs_frozen": (
        "Phase 3: Self-Play Against Frozen Checkpoint",
        f"{CYAN}{BOLD}OUR RL AI (Active Learner){RESET}",
        f"{BLUE}{BOLD}FROZEN OPPONENT (Past RL AI Generation){RESET}"
    )
}

def get_latest_replay_file(target_id=None):
    if target_id is not None:
        p = os.path.join(REPLAYS_DIR, f"replay_{target_id}.json")
        if os.path.exists(p):
            return p
    files = sorted(glob.glob(os.path.join(REPLAYS_DIR, "replay_*.json")), key=os.path.getmtime)
    return files[-1] if files else None

def parse_replay(path):
    with open(path, "r") as f:
        return json.load(f)

def print_summary(data, replay_path):
    battle_id = data.get("battle_id", 1)
    mode = data.get("mode")
    
    # If mode not in replay, infer from scenario.json or state.json
    if not mode:
        sc_path = os.path.join(WORKSPACE, "scenario.json")
        if os.path.exists(sc_path):
            try:
                sc = json.load(open(sc_path))
                mode = sc.get("mode", "rl_vs_vanilla")
            except Exception:
                mode = "rl_vs_vanilla"
        else:
            mode = "rl_vs_vanilla"
            
    mode_title, side0_label, side1_label = MODE_DESCRIPTIONS.get(
        mode, ("RL Evaluation", f"{CYAN}OUR RL AI{RESET}", f"{RED}DEFAULT ENGINE AI{RESET}")
    )
    
    res_code = data.get("result", 0)
    if res_code > 0:
        res_str = f"{GREEN}{BOLD}VICTORY (+1){RESET}"
    elif res_code < 0:
        res_str = f"{RED}{BOLD}DEFEAT (-1){RESET}"
    else:
        res_str = f"{YELLOW}{BOLD}DRAW / STALEMATE (0){RESET}"
        
    frames = data.get("frames", [])
    total_t = frames[-1].get("t", 0.0) if frames else 0.0
    roster = data.get("roster", [])
    
    # Calculate initial army strengths
    side0_start, side1_start = 0, 0
    side0_units, side1_units = [], []
    for r in roster:
        if r.get("side") == 0:
            side0_start += r.get("start", 0)
            side0_units.append(r.get("role", "inf"))
        else:
            side1_start += r.get("start", 0)
            side1_units.append(r.get("role", "inf"))
            
    # Calculate surviving army strengths from last frame
    side0_end_hp, side1_end_hp = 0, 0
    side0_survivors, side1_survivors = 0, 0
    if frames:
        last_units = frames[-1].get("units", [])
        for u in last_units:
            side, hp = u[0], u[4]
            if side == 1:
                side0_end_hp += hp
                if hp > 0: side0_survivors += 1
            else:
                side1_end_hp += hp
                if hp > 0: side1_survivors += 1

    print("\n" + "="*80)
    print(f" {BOLD}MEDIEVAL II: TOTAL WAR — BATTLE REPLAY #{battle_id}{RESET}")
    print("="*80)
    print(f" {BOLD}Curriculum Mode:{RESET} {mode_title}")
    print(f" {BOLD}Battle Outcome :{RESET} {res_str}")
    print(f" {BOLD}Battle Duration:{RESET} {total_t:.1f}s in-game ({len(frames)} recorded tactical frames)")
    print("-" * 80)
    
    print(f" {BOLD}SIDE 0:{RESET} {side0_label}")
    print(f"   • Control Type   : {'Decentralized SARSA(λ) Learner' if 'rl' in mode else 'Vanilla Engine (Passive Obs)'}")
    print(f"   • Units Deployed : {len(side0_units)} units ({side0_start} men)")
    print(f"   • Surviving Units: {side0_survivors}/{len(side0_units)} ({side0_end_hp} total HP remaining)")
    print(f"   • Visual Color   : {CYAN}{BOLD}BLUE / CYAN [I, S, A, C, G]{RESET}")
    print()
    
    print(f" {BOLD}SIDE 1:{RESET} {side1_label}")
    print(f"   • Control Type   : {'Total War Native Tactical AI (Rome/Medieval C++)' if 'frozen' not in mode else 'Frozen RL Model'}")
    print(f"   • Units Deployed : {len(side1_units)} units ({side1_start} men)")
    print(f"   • Surviving Units: {side1_survivors}/{len(side1_units)} ({side1_end_hp} total HP remaining)")
    print(f"   • Visual Color   : {RED}{BOLD}RED / CRIMSON [i, s, a, c, g]{RESET}")
    print("="*80)
    
    # Display Middle Frame ASCII Map preview
    if frames:
        mid_idx = len(frames) // 2
        mid_fr = frames[mid_idx]
        grid = mid_fr.get("grid", [])
        if grid:
            print(f"\n {BOLD}TACTICAL ASCII SNAPSHOT AT T={mid_fr.get('t', 0):.1f}s:{RESET}")
            print(f" (Uppercase {CYAN}BLUE{RESET} = Our AI, Lowercase {RED}RED{RESET} = Default AI | H=Hold, A=Atk, F=Fire, R=Rout)")
            print(" ┌" + "──" * 24 + "┐")
            for row in grid[::2]:  # show alternate rows for compact height
                row_colored = ""
                tokens = row.split()
                for tok in tokens:
                    if tok == "..":
                        row_colored += f"{DIM}· {RESET}"
                    elif tok[0].isupper():
                        row_colored += f"{CYAN}{BOLD}{tok}{RESET}"
                    else:
                        row_colored += f"{RED}{BOLD}{tok}{RESET}"
                print(" │ " + row_colored + " │")
            print(" └" + "──" * 24 + "┘")

def main():
    parser = argparse.ArgumentParser(description="View latest battle replay and distinguish AI versions.")
    parser.add_argument("--id", type=int, default=None, help="Specific battle ID to view")
    parser.add_argument("--open", action="store_true", help="Open the rendered video with desktop video player")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering MP4 video")
    args = parser.parse_args()
    
    replay_file = get_latest_replay_file(args.id)
    if not replay_file:
        print(f"[-] No replay files found in {REPLAYS_DIR} yet.")
        print("    A battle is currently executing. Check back once the episode finishes.")
        sys.exit(1)
        
    data = parse_replay(replay_file)
    print_summary(data, replay_file)
    
    mp4_file = replay_file.replace(".json", ".mp4")
    if not args.no_render:
        if not os.path.exists(mp4_file) or os.path.getmtime(replay_file) > os.path.getmtime(mp4_file):
            print(f"\n[*] Rendering 2D tactical map video for Replay #{data.get('battle_id', 1)}...")
            render_script = os.path.join(WORKSPACE, "render_replay_video.py")
            subprocess.run([sys.executable, render_script, replay_file])
            
        if os.path.exists(mp4_file):
            print(f"\n[+] Video Available: {GREEN}{mp4_file}{RESET}")
            print(f"    File Link: file://{os.path.abspath(mp4_file)}")
            if args.open:
                subprocess.Popen(["xdg-open", mp4_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
