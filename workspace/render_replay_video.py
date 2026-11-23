#!/usr/bin/env python3
"""
Renders a 2D Tactical Battle Map video (.mp4) from a recorded replay JSON.
Pipes rendered frames directly to ffmpeg inside the medieval2-ubuntu container.
"""
import os
import sys
import json
import glob
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 720
FPS = 5

ROLE_NAMES = {
    "I": "INF", "S": "SPEAR", "A": "ARCHER", "C": "CAV", "G": "GEN", "T": "ARTY"
}

STATUS_NAMES = {
    3: ("HOLD", (100, 180, 255)),
    4: ("ATK", (255, 120, 50)),
    5: ("FIRE", (255, 220, 50)),
    6: ("ROUT", (255, 50, 50)),
    0: ("MOVE", (180, 255, 180))
}

def render_frame(frame, prev_frame, bounds, battle_id, total_t, terrain=None):
    img = Image.new("RGB", (WIDTH, HEIGHT), (18, 24, 22))
    draw = ImageDraw.Draw(img)
    terrain = terrain or {}
    btype = terrain.get("battle_type", "open_field")
    
    # Map area: x from 40 to 980, y from 60 to 660 (940x600)
    MAP_X0, MAP_Y0, MAP_W, MAP_H = 40, 60, 940, 620
    
    # Draw tactical grid
    draw.rectangle([MAP_X0, MAP_Y0, MAP_X0 + MAP_W, MAP_Y0 + MAP_H], outline=(40, 60, 50), fill=(24, 32, 28), width=2)
    for gx in range(MAP_X0 + 50, MAP_X0 + MAP_W, 60):
        draw.line([(gx, MAP_Y0), (gx, MAP_Y0 + MAP_H)], fill=(32, 44, 38), width=1)
    for gy in range(MAP_Y0 + 50, MAP_Y0 + MAP_H, 60):
        draw.line([(MAP_X0, gy), (MAP_X0 + MAP_W, gy)], fill=(32, 44, 38), width=1)
        
    min_x, max_x, min_y, max_y = bounds
    span_x = max(100.0, max_x - min_x)
    span_y = max(100.0, max_y - min_y)
    
    def to_screen(wx, wy):
        sx = MAP_X0 + int(((wx - min_x) / span_x) * (MAP_W - 80)) + 40
        sy = MAP_Y0 + int(((wy - min_y) / span_y) * (MAP_H - 80)) + 40
        return max(MAP_X0 + 10, min(MAP_X0 + MAP_W - 10, sx)), max(MAP_Y0 + 10, min(MAP_Y0 + MAP_H - 10, sy))

    # Draw River Crossing or Siege Fortress overlays
    if btype == "river_crossing":
        ry = float(terrain.get("river_y", 0.0))
        rhw = float(terrain.get("river_half_w", 22.0))
        _, sy1 = to_screen(0, ry - rhw)
        _, sy2 = to_screen(0, ry + rhw)
        r_top, r_bot = min(sy1, sy2), max(sy1, sy2)
        draw.rectangle([MAP_X0 + 2, r_top, MAP_X0 + MAP_W - 2, max(r_top + 14, r_bot)], fill=(22, 65, 115), outline=(40, 110, 180))
        for b in terrain.get("bridges", []):
            bx = float(b.get("x", 0.0))
            bw = float(b.get("width", 28.0))
            bx1, _ = to_screen(bx - bw * 0.5, ry)
            bx2, _ = to_screen(bx + bw * 0.5, ry)
            is_ford = b.get("kind") == "ford"
            b_col = (70, 125, 155) if is_ford else (130, 105, 75)
            draw.rectangle([min(bx1, bx2) - 6, r_top - 3, max(bx1, bx2) + 6, max(r_top + 14, r_bot) + 3], fill=b_col, outline=(210, 190, 140), width=2)
            draw.text((min(bx1, bx2) - 4, r_top - 16), "FORD" if is_ford else "BRIDGE", fill=(220, 210, 170))
    elif btype in ("siege_assault", "siege_defense"):
        wy = float(terrain.get("wall_y", 75.0))
        gx = float(terrain.get("gate_x", 0.0))
        px = float(terrain.get("plaza_x", 0.0))
        py = float(terrain.get("plaza_y", 215.0))
        pr = float(terrain.get("plaza_r", 45.0))
        gate_hp = float(frame.get("gate_hp", 100.0))
        ladders = frame.get("ladders", [False, False])
        p_timer = float(frame.get("plaza_timer", 0.0))

        # Town Plaza circle
        psx, psy = to_screen(px, py)
        pr_px = max(22, int((pr / span_x) * (MAP_W - 80)))
        draw.ellipse([psx - pr_px, psy - pr_px, psx + pr_px, psy + pr_px], outline=(240, 200, 60), width=2)
        draw.text((psx - 32, psy - 6), f"PLAZA {p_timer:.0f}/30s", fill=(240, 210, 80))

        # Stone Fortress Wall line
        _, w_sy = to_screen(0, wy)
        draw.rectangle([MAP_X0 + 4, w_sy - 5, MAP_X0 + MAP_W - 4, w_sy + 5], fill=(110, 110, 118), outline=(170, 170, 180), width=2)

        # Gatehouse
        gsx, _ = to_screen(gx, wy)
        g_col = (40, 160, 70) if gate_hp <= 0 else (165, 85, 45)
        draw.rectangle([gsx - 24, w_sy - 10, gsx + 24, w_sy + 10], fill=g_col, outline=(240, 210, 120), width=2)
        g_txt = "GATE OPEN!" if gate_hp <= 0 else f"GATE {gate_hp:.0f}%"
        draw.text((gsx - 28, w_sy - 24), g_txt, fill=(100, 255, 130) if gate_hp <= 0 else (255, 200, 110))

        # Left / Right Ladder & Siege Tower sections
        for idx_l, lx in enumerate((gx - 95.0, gx + 95.0)):
            lsx, _ = to_screen(lx, wy)
            docked = ladders[idx_l] if idx_l < len(ladders) else False
            l_col = (60, 190, 110) if docked else (85, 85, 95)
            draw.rectangle([lsx - 14, w_sy - 8, lsx + 14, w_sy + 8], fill=l_col, outline=(200, 200, 210), width=1)
            if docked:
                draw.text((lsx - 22, w_sy + 11), "LADDERS UP", fill=(120, 255, 160))

    # Parse units: [side, id, x, y, hp_percent, morale, status, lastAction, symbol, (optional) fleeing]
    units = frame.get("units", [])
    blue_count, red_count = 0, 0
    blue_hp, red_hp = 0, 0
    blue_flee, red_flee = 0, 0
    
    for u in units:
        side, uid, wx, wy, hp, morale, st, action, sym = u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]
        fleeing = bool(u[9]) if len(u) > 9 else (st == 6 or morale == 6)
        sx, sy = to_screen(wx, wy)
        is_blue = (side == 1)
        
        if is_blue:
            blue_count += (1 if hp > 0 else 0)
            blue_hp += hp
            blue_flee += (1 if hp > 0 and fleeing else 0)
            color = (45, 95, 145) if fleeing else (60, 160, 255)
            border_color = (255, 200, 50) if fleeing else (120, 210, 255)
        else:
            red_count += (1 if hp > 0 else 0)
            red_hp += hp
            red_flee += (1 if hp > 0 and fleeing else 0)
            color = (145, 50, 50) if fleeing else (235, 65, 65)
            border_color = (255, 200, 50) if fleeing else (255, 130, 130)
            
        if hp <= 0:
            continue
            
        # Draw unit badge
        R = 14
        draw.ellipse([sx - R, sy - R, sx + R, sy + R], fill=color, outline=border_color, width=3 if fleeing else 2)
        if fleeing:
            draw.line([(sx - R + 3, sy - R + 3), (sx + R - 3, sy + R - 3)], fill=(255, 200, 50), width=2)
        
        # Symbol & Siege Equipment tag ([RAM] / [LAD])
        role_txt = sym[0].upper() if sym else "U"
        draw.text((sx - 4, sy - 7), role_txt, fill=(255, 255, 255))
        equip = u[10] if len(u) > 10 else ""
        if equip:
            draw.rectangle([sx - 15, sy - R - 19, sx + 15, sy - R - 8], fill=(180, 110, 30), outline=(255, 220, 100))
            draw.text((sx - 11, sy - R - 19), equip, fill=(255, 255, 220))
        
        # Status text & HP bar
        if fleeing:
            st_name, st_col = "FLEE", (255, 200, 50)
        else:
            st_name, st_col = STATUS_NAMES.get(st, ("IDLE", (200, 200, 200)))
        draw.text((sx - 16, sy + R + 2), f"{st_name}", fill=st_col)
        
        # HP bar
        bar_w = 26
        hp_w = int((hp / 100.0) * bar_w)
        draw.rectangle([sx - bar_w//2, sy - R - 7, sx + bar_w//2, sy - R - 3], fill=(40, 40, 40))
        bar_color = (50, 220, 80) if hp > 50 else ((240, 200, 30) if hp > 25 else (240, 40, 40))
        draw.rectangle([sx - bar_w//2, sy - R - 7, sx - bar_w//2 + hp_w, sy - R - 3], fill=bar_color)

    # HUD Banner (Top & Right)
    t = frame.get("t", 0)
    draw.rectangle([0, 0, WIDTH, 50], fill=(12, 16, 14))
    draw.line([(0, 50), (WIDTH, 50)], fill=(50, 75, 65), width=2)
    
    draw.text((40, 14), f"BATTLE #{battle_id} [{btype.upper().replace('_', ' ')}]", fill=(230, 230, 230))
    draw.text((320, 14), f"TIME: {t:.1f}s / {total_t:.1f}s", fill=(200, 255, 120))
    
    # Side indicators
    draw.rectangle([575, 10, 805, 40], fill=(20, 35, 55), outline=(60, 120, 200))
    draw.text((585, 18), f"RL: {blue_count}u ({blue_hp}HP, {blue_flee} flee)", fill=(100, 200, 255))
    
    draw.rectangle([820, 10, 1060, 40], fill=(50, 20, 20), outline=(200, 70, 70))
    draw.text((830, 18), f"EN: {red_count}u ({red_hp}HP, {red_flee} flee)", fill=(255, 120, 120))
    
    # Right Sidebar (Legend & Info)
    SIDE_X0 = 1000
    draw.rectangle([SIDE_X0, 60, WIDTH - 20, MAP_Y0 + MAP_H], fill=(24, 30, 28), outline=(40, 55, 48), width=2)
    draw.text((SIDE_X0 + 20, 80), "TACTICAL LEGEND", fill=(220, 220, 180))
    draw.line([(SIDE_X0 + 20, 105), (WIDTH - 40, 105)], fill=(50, 65, 58))
    
    legend = [
        ("[I] Line Infantry", (200, 200, 200)),
        ("[S] Spear Infantry", (200, 200, 200)),
        ("[A] Missile / Bow", (200, 200, 200)),
        ("[C] Cavalry", (200, 200, 200)),
        ("[G] General Unit", (240, 210, 60)),
        ("", (0,0,0)),
        ("STATUS CODES:", (220, 220, 180)),
        ("HOLD: Stationary", (100, 180, 255)),
        ("ATK : In Melee", (255, 120, 50)),
        ("FIRE: Shooting", (255, 220, 50)),
        ("ROUT: Fleeing", (255, 50, 50)),
        ("MOVE: Advancing", (180, 255, 180))
    ]
    
    ly = 120
    for txt, col in legend:
        if txt:
            draw.text((SIDE_X0 + 20, ly), txt, fill=col)
        ly += 22
        
    return img

def expand_frames_for_duration(frames, target_secs=30.0, fps=10):
    """Interpolate unit (x, y, hp) between keyframes so the output video lasts target_secs at fps."""
    if len(frames) < 2:
        return frames
    total_out = int(round(target_secs * fps))
    out = []
    n = len(frames)
    for k in range(total_out):
        pos = k * (n - 1) / max(1, total_out - 1)
        i = min(n - 2, int(pos))
        alpha = pos - i
        fa, fb = frames[i], frames[i + 1]
        b_map = {(u[0], u[1]): u for u in fb.get("units", [])}
        interp_units = []
        for ua in fa.get("units", []):
            ub = b_map.get((ua[0], ua[1]))
            if ub:
                nx = ua[2] + alpha * (ub[2] - ua[2])
                ny = ua[3] + alpha * (ub[3] - ua[3])
                nhp = int(round(ua[4] + alpha * (ub[4] - ua[4])))
                src = ua if alpha < 0.5 else ub
                row = [ua[0], ua[1], nx, ny, nhp, src[5], src[6], src[7], ua[8]]
                if len(src) > 9:
                    row.append(src[9])
                if len(src) > 10:
                    row.append(src[10])
                interp_units.append(row)
            else:
                interp_units.append(ua)
        nt = fa.get("t", 0) + alpha * (fb.get("t", 0) - fa.get("t", 0))
        out.append({
            "t": round(nt, 1),
            "units": interp_units,
            "gate_hp": fa.get("gate_hp", 100.0),
            "ladders": fa.get("ladders", [False, False]),
            "plaza_timer": fa.get("plaza_timer", 0.0),
        })
    return out


def render_replay(json_path, output_mp4=None):
    if not os.path.exists(json_path):
        print(f"[-] Replay file {json_path} not found.")
        return False
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    frames = data.get("frames", [])
    if not frames:
        print("[-] Replay has no frames!")
        return False
        
    battle_id = data.get("battle_id", 1)
    is_long = bool(data.get("long_battle")) or (isinstance(battle_id, int) and battle_id % 10 == 0)
    fps = 10 if is_long else FPS
    if is_long:
        frames = expand_frames_for_duration(frames, target_secs=30.0, fps=fps)
    if not output_mp4:
        output_mp4 = json_path.replace(".json", ".mp4")
        
    print(f"[*] Rendering {len(frames)} frames ({len(frames)/fps:.1f}s @ {fps}fps) for Battle #{battle_id} to {output_mp4}...")
    
    # Calculate global bounding box
    min_x, max_x, min_y, max_y = 1e9, -1e9, 1e9, -1e9
    for fr in frames:
        for u in fr.get("units", []):
            wx, wy = u[2], u[3]
            min_x = min(min_x, wx)
            max_x = max(max_x, wx)
            min_y = min(min_y, wy)
            max_y = max(max_y, wy)
    bounds = (min_x, max_x, min_y, max_y)
    total_t = frames[-1].get("t", 0) if frames else 0
    terrain = data.get("terrain") or {"battle_type": data.get("battle_type", "open_field")}
    
    # Pipe raw RGB24 frames into local ffmpeg (fallback to imageio_ffmpeg)
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", output_mp4,
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    prev_fr = None
    for idx, fr in enumerate(frames):
        img = render_frame(fr, prev_fr, bounds, battle_id, total_t, terrain=terrain)
        prev_fr = fr
        proc.stdin.write(img.tobytes())
        
    proc.stdin.close()
    proc.wait()
    
    if proc.returncode == 0:
        print(f"[+] Successfully rendered replay video: {output_mp4}")
        return True
    else:
        err = proc.stderr.read().decode('utf-8')
        print(f"[-] FFmpeg error: {err}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        replays = sorted(glob.glob("/home/tomek/medieval2-docker/workspace/replays/replay_*.json"))
        target = replays[-1] if replays else "/home/tomek/medieval2-docker/workspace/replays/replay_1.json"
    render_replay(target)
