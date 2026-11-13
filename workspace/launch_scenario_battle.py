#!/usr/bin/env python3
"""
Automates battle execution in headless Medieval 2: Total War via X11 / xdotool.
Navigates Single Player -> Quick Battle (or Custom Battle), triggers Start Battle,
and monitors completion.
"""
import os
import subprocess
import time
import glob
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--container", default="medieval2-ubuntu", help="Target container name")
args, _ = parser.parse_known_args()
CONTAINER = args.container

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def get_x11_env():
    # Find active Xvfb display and Xauthority
    res = run_cmd(f"docker exec {CONTAINER} ps aux | grep -i Xvfb")
    for line in res.stdout.splitlines():
        if "Xvfb :" in line:
            parts = line.split("Xvfb :")[1].split()
            disp = ":" + parts[0]
            auth_res = run_cmd(f"docker exec {CONTAINER} bash -c 'ls -t /tmp/xvfb-run.*/Xauthority 2>/dev/null | head -1'")
            auth = auth_res.stdout.strip()
            return disp, auth
    return ":99", "/tmp/Xauthority"

def exec_in_x11(cmd_str):
    disp, auth = get_x11_env()
    full_cmd = f"docker exec -e DISPLAY={disp} -e XAUTHORITY={auth} {CONTAINER} bash -c '{cmd_str}'"
    return run_cmd(full_cmd)

def wait_for_window(name="Medieval 2", timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        res = exec_in_x11(f"xdotool search --name '{name}' | head -1")
        wid = res.stdout.strip()
        if wid and wid.isdigit():
            return int(wid)
        time.sleep(1)
    return None

def click_coords(wid, x, y, delay=0.5):
    exec_in_x11(f"xdotool mousemove --window {wid} {x} {y} click 1")
    time.sleep(delay)

def launch_quick_battle():
    wid = wait_for_window("Medieval 2", timeout=30)
    if not wid:
        print("[-] Medieval 2 window not found!")
        return False
    print(f"[+] Found Medieval 2 window ID: {wid}")

    # 1. Click 'Single Player' (x=239, y=233)
    print("[*] Clicking Single Player...")
    click_coords(wid, 239, 233, delay=1.5)

    # 2. Click 'Quick Battle' (x=239, y=395)
    print("[*] Clicking Quick Battle (randomized factions & armies)...")
    click_coords(wid, 239, 395, delay=2.0)

    # 3. Wait for tactical battlefield loading
    print(f"[*] Waiting for tactical battle to load on {CONTAINER}...")
    loaded = False
    wid_tag = "1"
    if "worker2" in CONTAINER:
        wid_tag = "2"
    elif "worker3" in CONTAINER:
        wid_tag = "3"
        
    heartbeat_paths = [f"/home/tomek/medieval2-docker/workspace/rl_heartbeat_{wid_tag}.txt"]
    if wid_tag == "1":
        heartbeat_paths.append("/home/tomek/medieval2-docker/workspace/rl_heartbeat.txt")
    for i in range(60):
        time.sleep(1)
        for hb_p in heartbeat_paths:
            if os.path.exists(hb_p):
                try:
                    with open(hb_p) as f:
                        content = f.read()
                        if "STATE=" in content and "STATE=0" not in content:
                            loaded = True
                            print(f"[+] Battlefield loaded on {CONTAINER} after {i+1}s ({content.strip()})!")
                            break
                except Exception:
                    pass
        if loaded:
            break
    
    if not loaded:
        print("[-] Battlefield load timed out.")
        return False

    # 4. Click 'Start Battle' and ensure conflict starts
    print("[*] Clicking 'Start Battle' / confirming conflict start...")
    for _ in range(8):
        click_coords(wid, 40, 590, delay=0.5)
        click_coords(wid, 110, 590, delay=0.5)
        exec_in_x11(f"xdotool windowfocus --sync {wid} key Return space")
        time.sleep(1.0)
        try:
            for hb_p in heartbeat_paths:
                if os.path.exists(hb_p):
                    with open(hb_p) as f:
                        content = f.read()
                        if "IN_BATTLE=true" in content or "STATE=5" in content:
                            print(f"[+] Battle conflict on {CONTAINER} successfully started!")
                            return True
        except Exception:
            pass
    return True

if __name__ == "__main__":
    launch_quick_battle()
