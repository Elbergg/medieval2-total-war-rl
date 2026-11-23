#!/usr/bin/env python3
"""
Continuous RL Battle Supervisor for Medieval II: Total War.
Manages the end-to-end loop:
  1. Boot game container / run_headless.sh
  2. Navigate menus -> Start Battle
  3. Wait for battle completion and episode dump
  4. Wait for trainer to update weights
  5. Repeat for next episode
"""
import os
import sys
import time
import glob
import json
import fcntl
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--container", default="medieval2-ubuntu", help="Target container name")
args, _ = parser.parse_known_args()
CONTAINER = args.container

WORKSPACE = "/home/tomek/medieval2-docker/workspace"
BASE_DIR = "/home/tomek/medieval2-docker"

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def is_game_running():
    res = run_cmd(f"docker exec {CONTAINER} pgrep -f medieval2.exe")
    return res.returncode == 0

def kill_game():
    run_cmd(f"docker exec {CONTAINER} pkill -9 -f medieval2.exe")
    run_cmd(f"docker exec {CONTAINER} pkill -9 -f m2tweop_starter.exe")
    run_cmd(f"docker exec {CONTAINER} wineserver -k")
    run_cmd(f"docker exec {CONTAINER} pkill -9 -f Xvfb")
    time.sleep(0.5)

def start_game():
    wid = "1"
    if "worker2" in CONTAINER:
        wid = "2"
    elif "worker3" in CONTAINER:
        wid = "3"
    print(f"[*] Starting game instance on {CONTAINER} via run_headless.sh...")
    # Clean previous run logs and heartbeat for this specific worker
    run_cmd(f"docker exec {CONTAINER} rm -f /home/tomek/game/logs/system.log.txt /home/tomek/workspace/rl_heartbeat_{wid}.txt")
    if wid == "1":
        run_cmd(f"docker exec {CONTAINER} rm -f /home/tomek/workspace/rl_heartbeat.txt")
    run_cmd(f"docker exec -d {CONTAINER} bash -c 'cd /home/tomek/game && xvfb-run -a -s \"-screen 0 1024x768x24\" wine m2tweop_starter.exe . > /home/tomek/workspace/m2tweop_starter_{CONTAINER}.log 2>&1'")
    
    # Wait for window
    for _ in range(30):
        time.sleep(1)
        res = run_cmd(f"docker exec {CONTAINER} pgrep -f medieval2.exe")
        if res.returncode == 0:
            print(f"[+] medieval2.exe running on {CONTAINER}!")
            return True
    return False

def run_battle():
    # launch_scenario_battle.py waits for M2TWEOP heartbeat before clicking
    print(f"[*] Launching battle scenario on {CONTAINER}...")
    res = subprocess.run([sys.executable, os.path.join(WORKSPACE, "launch_scenario_battle.py"), "--container", CONTAINER])
    return res.returncode == 0

def get_wid():
    if "worker2" in CONTAINER:
        return "2"
    if "worker3" in CONTAINER:
        return "3"
    return "1"


def claim_battle_id():
    """Atomically claim the next sequential battle ID using a file mutex."""
    wid = get_wid()
    lock_path = os.path.join(WORKSPACE, ".battle_id.lock")
    counter_path = os.path.join(WORKSPACE, "next_battle_id.txt")
    state_path = os.path.join(WORKSPACE, "state.json")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if os.path.exists(counter_path):
                with open(counter_path) as cf:
                    bid = int(cf.read().strip())
            elif os.path.exists(state_path):
                with open(state_path) as sf:
                    bid = int(json.load(sf).get("battle", 0)) + 1
            else:
                bid = 1
            with open(counter_path, "w") as cf:
                cf.write(str(bid + 1))
            with open(os.path.join(WORKSPACE, f"worker_battle_id_{wid}.txt"), "w") as wf:
                wf.write(str(bid))
            return bid
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def wait_for_episode(target_id, start_time, timeout=180):
    start = time.time()
    wid = get_wid()

    print(f"[*] Waiting for episode #{target_id} (worker {wid}, timeout {timeout}s)...")
    while time.time() - start < timeout:
        candidates = [
            os.path.join(WORKSPACE, "episodes", f"episode_{target_id}.json"),
            os.path.join(WORKSPACE, "processed", f"episode_{target_id}.json"),
        ]
        for m in candidates:
            if os.path.exists(m):
                try:
                    if os.path.getmtime(m) >= start_time - 2.0:
                        print(f"[+] Episode {os.path.basename(m)} produced!")
                        return True, str(target_id)
                except Exception:
                    pass
        # Also check if game died unexpectedly
        if not is_game_running():
            print("[-] Game process terminated before episode dump.")
            return False, None
        time.sleep(4)
    print(f"[-] Timed out waiting for episode.")
    return False, None

def main():
    os.chdir(BASE_DIR)
    print(f"[+] Starting RL Battle Supervisor loop for {CONTAINER}...")
    pending_id = None
    while True:
        kill_game()
        if not start_game():
            print("[-] Failed to start game! Retrying in 10s...")
            time.sleep(10)
            continue

        if pending_id is not None:
            curr_id = pending_id
            with open(os.path.join(WORKSPACE, f"worker_battle_id_{get_wid()}.txt"), "w") as wf:
                wf.write(str(curr_id))
        else:
            curr_id = claim_battle_id()
        print(f"\n==========================================")
        print(f"   STARTING BATTLE ON {CONTAINER} (ID {curr_id})")
        print(f"==========================================")

        start_battle_time = time.time()
        if not run_battle():
            print(f"[-] Battle #{curr_id} launch failed, retrying same ID...")
            pending_id = curr_id
            continue
            
        ep_timeout = 300 if (curr_id % 5 == 0) else 180
        success, tag = wait_for_episode(curr_id, start_battle_time, timeout=ep_timeout)
        if success and tag:
            pending_id = None
            print(f"[+] Battle episode on {CONTAINER} completed successfully (tag={tag})!")
            # Render MP4 automatically on judged battles (every 5th); others render on demand via view_latest.sh
            if int(tag) % 5 == 0:
                rep_file = os.path.join(WORKSPACE, "replays", f"replay_{tag}.json")
                mp4_file = os.path.join(WORKSPACE, "replays", f"replay_{tag}.mp4")
                if os.path.exists(rep_file) and not os.path.exists(mp4_file):
                    print(f"[*] Auto-rendering video for Replay #{tag}...")
                    subprocess.Popen([sys.executable, os.path.join(WORKSPACE, "render_replay_video.py"), rep_file, mp4_file])
            time.sleep(0.5)
        else:
            print(f"[-] Episode #{curr_id} on {CONTAINER} failed or timed out; will retry #{curr_id}.")
            pending_id = curr_id
            
        kill_game()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
