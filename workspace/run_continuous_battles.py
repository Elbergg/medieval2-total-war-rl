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

WORKSPACE = "/home/tomek/medieval2-docker/workspace"
BASE_DIR = "/home/tomek/medieval2-docker"

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def is_game_running():
    res = run_cmd("docker exec medieval2-ubuntu pgrep -f medieval2.exe")
    return res.returncode == 0

def kill_game():
    run_cmd("docker exec medieval2-ubuntu pkill -9 -f medieval2.exe")
    run_cmd("docker exec medieval2-ubuntu pkill -9 -f m2tweop_starter.exe")
    run_cmd("docker exec medieval2-ubuntu wineserver -k")
    run_cmd("docker exec medieval2-ubuntu pkill -9 -f Xvfb")
    time.sleep(2)

def start_game():
    print("[*] Starting game instance via run_headless.sh...")
    # Clean previous run logs and heartbeat
    run_cmd("docker exec medieval2-ubuntu rm -f /home/tomek/game/logs/system.log.txt /home/tomek/workspace/rl_heartbeat.txt")
    run_cmd("docker exec -d medieval2-ubuntu bash -c 'cd /home/tomek/game && xvfb-run -a -s \"-screen 0 1024x768x24\" wine m2tweop_starter.exe . > /home/tomek/workspace/m2tweop_starter.log 2>&1'")
    
    # Wait for window
    for _ in range(30):
        time.sleep(1)
        res = run_cmd("docker exec medieval2-ubuntu pgrep -f medieval2.exe")
        if res.returncode == 0:
            print("[+] medieval2.exe running!")
            return True
    return False

def run_battle():
    # Wait for menu to become clickable
    time.sleep(10)
    print("[*] Launching battle scenario...")
    res = subprocess.run([sys.executable, os.path.join(WORKSPACE, "launch_scenario_battle.py")])
    return res.returncode == 0

def wait_for_episode(target_id, timeout=120):
    start = time.time()
    print(f"[*] Waiting for episode_{target_id}.json (timeout {timeout}s)...")
    while time.time() - start < timeout:
        ep_file = os.path.join(WORKSPACE, "episodes", f"episode_{target_id}.json")
        proc_file = os.path.join(WORKSPACE, "processed", f"episode_{target_id}.json")
        if os.path.exists(ep_file) or os.path.exists(proc_file):
            print(f"[+] Episode {target_id} produced!")
            return True
        # Also check if game died unexpectedly
        if not is_game_running():
            print("[-] Game process terminated before episode dump.")
            return False
        time.sleep(5)
    print(f"[-] Timed out waiting for episode {target_id}.")
    return False

def main():
    os.chdir(BASE_DIR)
    print("[+] Starting RL Battle Supervisor loop...")
    while True:
        # Determine current battle ID from processed + episodes
        existing = glob.glob(os.path.join(WORKSPACE, "processed", "episode_*.json")) + \
                   glob.glob(os.path.join(WORKSPACE, "episodes", "episode_*.json"))
        curr_id = len(existing) + 1
        print(f"\n==========================================")
        print(f"   STARTING BATTLE EPISODE {curr_id}")
        print(f"==========================================")
        
        kill_game()
        if not start_game():
            print("[-] Failed to start game! Retrying in 10s...")
            time.sleep(10)
            continue
            
        if not run_battle():
            print("[-] Battle launch failed, restarting...")
            continue
            
        success = wait_for_episode(curr_id)
        if success:
            print(f"[+] Battle episode {curr_id} completed successfully!")
            # Trigger background video render for instant playback
            rep_file = os.path.join(WORKSPACE, "replays", f"replay_{curr_id}.json")
            mp4_file = os.path.join(WORKSPACE, "replays", f"replay_{curr_id}.mp4")
            if os.path.exists(rep_file) and not os.path.exists(mp4_file):
                print(f"[*] Auto-rendering video for Replay #{curr_id}...")
                subprocess.Popen([sys.executable, os.path.join(WORKSPACE, "render_replay_video.py"), rep_file, mp4_file])
            # Wait a few seconds for trainer to process
            time.sleep(5)
        else:
            print(f"[-] Episode {curr_id} failed or timed out.")
            
        kill_game()
        time.sleep(3)

if __name__ == "__main__":
    main()
