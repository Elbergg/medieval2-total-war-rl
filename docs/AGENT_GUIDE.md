# Agent & Developer Guide — Medieval II: Total War RL Battle AI

This guide is designed for **autonomous AI agents** and **human developers** continuing work on this repository. Read this before making changes to the battle environment, training pipelines, or Lua plugins.

---

## 1. Quick Orientation

| Component | Host Location | Container Location | Purpose |
|:---|:---|:---|:---|
| **Game Container** | `docker-compose.yml` (`medieval2-ubuntu`) | `/home/tomek/` | Runs Wine32, Xvfb, M2TWEOP v3.3.1, `medieval2.exe` |
| **Trainer Container** | `docker-compose.yml` (`medieval2-trainer`) | `/trainer/` | Watches `/workspace/episodes`, updates SARSA weights, triggers Gemini judge |
| **Shared IPC Volume** | `./workspace/` | `/home/tomek/workspace/` | Inter-container communication (episodes, replays, weights, state) |
| **Game Installation** | `./game/` | `/home/tomek/game/` | Hardlinked copy of Steam game files + M2TWEOP (Host Steam install untouched) |
| **In-Game Lua AI** | `workspace/rl_battle_ai.lua` & `game/youneuoy_Data/plugins/lua/rl_battle_ai.lua` | Hooked into `luaPluginScript.lua` | In-battle feature extraction, action selection, and episode dumping |
| **Battle Supervisor** | `workspace/run_continuous_battles.py` | Host process or container | Autonomous loop: boots game, starts Quick Battle, collects episode, loops |
| **Replay Viewer CLI** | `./view_latest.sh` / `workspace/view_latest_battle.py` | Host CLI | Terminal tactical summary + auto-renders 2D MP4 video |

---

## 2. Daily Operations & CLI Commands

### Checking Live Status
```bash
# Check current curriculum phase and battle count
cat workspace/state.json

# Check trainer log
docker logs --tail 25 medieval2-trainer

# Check container resource utilization
docker stats --no-stream

# View the latest battle in terminal and open the rendered video
./view_latest.sh --open
```

### Starting the Autonomous Battle Loop
To run continuous training in the background:
```bash
cd /home/tomek/medieval2-docker
python3 -u workspace/run_continuous_battles.py
```
*Note: In agent environments, launch this as a background task / daemon.*

### Gracefully Stopping Training
```bash
# 1. Kill the continuous supervisor process (e.g. pkill -f run_continuous_battles.py)
# 2. Cleanly tear down any running Wine session:
docker exec medieval2-ubuntu pkill -9 -f medieval2.exe
docker exec medieval2-ubuntu wineserver -k
docker exec medieval2-ubuntu pkill -9 -f Xvfb
```
*Weights and state are flushed atomically after each battle, so stopping at any time preserves progress.*

---

## 3. Critical Engine & Lua Gotchas (DO NOT BREAK)

1. **`battle.sides` is 1-Indexed in Lua**:
   - `battle.sides[1]` is Side 0 (the player army).
   - `battle.sides[2]` is Side 1 (the enemy army).
   - `battle.sides[0]` evaluates to `nil` in Lua! Always use `for si = 1, 2`.

2. **Unit Methods in M2TWEOP**:
   - `unit:isEngaged()` $\rightarrow$ **Valid** (boolean).
   - `unit:isUnderFire()` $\rightarrow$ **Valid** (boolean).
   - `unit:isFiring()` $\rightarrow$ **DOES NOT EXIST**. Calling it will crash the Lua tick. Firing is inferred: unit has missile capability, is stationary ($\Delta x^2 + \Delta y^2 < 4$), and is not engaged in melee.

3. **`eduEntry` Structure**:
   - `eduEntry` does **not** have a nested `primaryStats` sub-table.
   - Properties are top-level fields: `e.range`, `e.ammunition`, `e.aiUnitValuePerSoldier`, `e.category`, `e.mass`.

4. **Wine & Address Space (4GB LAA Patch)**:
   - `medieval2.exe` MUST retain the `IMAGE_FILE_LARGE_ADDRESS_AWARE` bit (`0x0020`) in its PE header.
   - Without it, Wine limits virtual address space to 2GB, causing `"A fatal graphics error has occurred"` crashes after consecutive battles.

5. **Graphics Configuration (`medieval2.preference.cfg`)**:
   - Do **NOT** set `unit_detail = low` (causes CPU software rasterizer downsampling thrashing). Keep `unit_detail = high`.
   - Keep `vegetation = 0`, `grass_distance = 0`, and `depth_shadows = 0` to preserve 100% of CPU cycles for simulation ticks.

6. **Host Steam Safety Guarantee**:
   - Host Steam installation at `/home/tomek/.steam/` must remain completely untouched.
   - Game files are isolated inside `./game/` and `./wineprefix/`.

---

## 4. Video Rendering Pipeline

Replay videos are rendered top-down at 1280x720 H.264 without saving temporary image frames to disk:
1. `workspace/render_replay_video.py` parses `workspace/replays/replay_{N}.json`.
2. Generates tactical frames with Pillow:
   - Cyan/Blue tokens = Our AI (`[I]` Line, `[S]` Spear, `[A]` Archer, `[C]` Cav, `[G]` General)
   - Crimson/Red tokens = Default Total War AI
   - Real-time HP bars and state labels (`HOLD`, `ATK`, `FIRE`, `ROUT`, `MOVE`).
3. Streams raw RGBA bytes directly into container `ffmpeg` stdin (`-f rawvideo -pix_fmt rgba`).
4. Output file: `workspace/replays/replay_{N}.mp4`.
