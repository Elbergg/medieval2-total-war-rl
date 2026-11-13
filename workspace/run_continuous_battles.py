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
    time.sleep(2)

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
    time.sleep(2)
    print(f"[*] Launching battle scenario on {CONTAINER}...")
    res = subprocess.run([sys.executable, os.path.join(WORKSPACE, "launch_scenario_battle.py"), "--container", CONTAINER])
    return res.returncode == 0

def wait_for_episode(target_id, start_time, timeout=120):
    start = time.time()
    wid = "1"
    if "worker2" in CONTAINER:
        wid = "2"
    elif "worker3" in CONTAINER:
        wid = "3"

    print(f"[*] Waiting for episode (worker {wid}, timeout {timeout}s)...")
    while time.time() - start < timeout:
        pat = "episode_[0-9]*.json" if wid == "1" else f"episode_w{wid}_*.json"
        pattern_ep = os.path.join(WORKSPACE, "episodes", pat)
        pattern_proc = os.path.join(WORKSPACE, "processed", pat)
        
        matches = glob.glob(pattern_ep) + glob.glob(pattern_proc)
        for m in matches:
            try:
                if os.path.getmtime(m) >= start_time - 2.0:
                    tag = os.path.basename(m).replace("episode_", "").replace(".json", "")
                    print(f"[+] Episode {os.path.basename(m)} produced!")
                    return True, tag
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
    while True:
        # Determine current battle ID from processed + episodes
        existing = glob.glob(os.path.join(WORKSPACE, "processed", "episode_*.json")) + \
                   glob.glob(os.path.join(WORKSPACE, "episodes", "episode_*.json"))
        curr_id = len(existing) + 1
        print(f"\n==========================================")
        print(f"   STARTING BATTLE ON {CONTAINER} (ID est. {curr_id})")
        print(f"==========================================")
        
        kill_game()
        if not start_game():
            print("[-] Failed to start game! Retrying in 10s...")
            time.sleep(10)
            continue
            
        start_battle_time = time.time()
        if not run_battle():
            print("[-] Battle launch failed, restarting...")
            continue
            
        success, tag = wait_for_episode(curr_id, start_battle_time)
        if success and tag:
            print(f"[+] Battle episode on {CONTAINER} completed successfully (tag={tag})!")
            # Trigger background video render for instant playback
            rep_file = os.path.join(WORKSPACE, "replays", f"replay_{tag}.json")
            mp4_file = os.path.join(WORKSPACE, "replays", f"replay_{tag}.mp4")
            if os.path.exists(rep_file) and not os.path.exists(mp4_file):
                print(f"[*] Auto-rendering video for Replay #{tag}...")
                subprocess.Popen([sys.executable, os.path.join(WORKSPACE, "render_replay_video.py"), rep_file, mp4_file])
            # Wait a few seconds for trainer to process
            time.sleep(4)
        else:
            print(f"[-] Episode on {CONTAINER} failed or timed out.")
            
        kill_game()
        time.sleep(3)

if __name__ == "__main__":
    main()
