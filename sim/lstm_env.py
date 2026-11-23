"""Simulator battle environment for the Full-Army Monolithic LSTM.

Builds the exact 1,068-float observation vector (12 global + 24 own x 22 + 24 enemy x 22)
with exact Sim-to-Real parity against rl_battle_ai.lua, and issues commands to ALL
active friendly units simultaneously every 4.5s game tick.
"""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from combat import BattleSim, SimUnit, clamp
from lstm_model import BATTLE_TYPES, GLOBAL_FEAT, MAX_UNITS, OBS_DIM, UNIT_FEAT, FullArmyLSTM
from physics_world_model import UnitSelfAttentionNet, obs_to_unit_nodes
from rl_env import _army_spread, _cohesion_reward, _inf_centroid, _issue_action, _to_rel
from spec import NUM_ACTIONS, ROLES, imitation_reward
from vanilla_ai import step_vanilla_ai

DECISION_DT = 4.0
SUB_DT = 1.0


def _active_centroid(units: List[SimUnit]) -> Tuple[float, float]:
    """Unweighted centroid of active units — exact parity with rl_battle_ai.lua."""
    sx = sy = 0.0
    n = 0
    for u in units:
        if u.active:
            sx += u.x
            sy += u.y
            n += 1
    if n == 0:
        return 0.0, 0.0
    return sx / n, sy / n


def _army_frame(own: List[SimUnit], en: List[SimUnit]) -> dict:
    cx, cy = _active_centroid(own)
    ex, ey = _active_centroid(en)
    fx, fy = ex - cx, ey - cy
    n = math.hypot(fx, fy)
    if n < 1e-3:
        fx, fy = 0.0, 1.0
    else:
        fx, fy = fx / n, fy / n
    return {"cx": cx, "cy": cy, "fx": fx, "fy": fy, "rx": fy, "ry": -fx}


def _bind_slots(units: List[SimUnit], fr: dict) -> List[Optional[SimUnit]]:
    """Sort deterministically by (role_index, initial_rx) and bind to fixed slots 0..MAX_UNITS-1."""
    keyed = []
    for u in units:
        rx, _ = _to_rel(fr, u.x, u.y)
        keyed.append((u.ri, rx, u.uid, u))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    slots: List[Optional[SimUnit]] = [None] * MAX_UNITS
    for i, item in enumerate(keyed[:MAX_UNITS]):
        slots[i] = item[3]
    return slots


def _encode_unit(u: Optional[SimUnit], fr: dict, out: np.ndarray, off: int, sim: Optional[BattleSim] = None, mean_z: float = 0.0):
    if u is None or u.alive <= 0:
        return
    rx, ry = _to_rel(fr, u.x, u.y)
    out[off + 0] = 1.0
    out[off + 1] = clamp(rx / 300.0, -1.5, 1.5)
    out[off + 2] = clamp(ry / 300.0, -1.5, 1.5)
    out[off + 3] = clamp(u.alive / max(1, u.start), 0.0, 1.0)
    out[off + 4] = clamp(u.morale_level / 6.0, 0.0, 1.0)
    out[off + 5] = clamp(u.fatigue / 6.0, 0.0, 1.0)
    if u.role == "artillery":
        out[off + 6] = 1.0 if (u.max_ammo <= 0 or u.ammo > 0) else 0.0
    else:
        out[off + 6] = clamp(u.ammo / u.max_ammo, 0.0, 1.0) if u.max_ammo > 0 else 0.0
    out[off + 7] = 1.0 if u.engaged else 0.0
    out[off + 8] = 1.0 if u.is_under_fire else 0.0
    out[off + 9] = 1.0 if u.routing else 0.0
    ri0 = u.ri - 1
    if 0 <= ri0 < 7:
        out[off + 10 + ri0] = 1.0
    # 7 stats matching rl_battle_ai.lua parity (including base morale & unit size)
    out[off + 17] = clamp(u.tpl.ai_value_per_soldier / 25.0, 0.0, 1.0)
    out[off + 18] = clamp((u.tpl.armour + u.tpl.def_skill) / 30.0, 0.0, 1.0)
    rng = u.tpl.pri_range if u.tpl.is_missile else 0.0
    out[off + 19] = clamp(rng / 250.0, 0.0, 1.0)
    out[off + 20] = clamp(u.tpl.move_speed_mod / 2.5, 0.0, 1.0)
    out[off + 21] = clamp(u.tpl.mass / 5.0, 0.0, 1.0)
    out[off + 22] = clamp(u.tpl.morale / 15.0, 0.0, 1.0)
    out[off + 23] = clamp(u.start / 150.0, 0.0, 1.5)
    # 2 terrain/position features: relative hill elevation + on_walls flag
    z = sim.elevation(u.x, u.y) if sim is not None else 0.0
    out[off + 24] = clamp((z - mean_z) / 25.0, -1.5, 1.5)
    on_wall = (
        sim is not None
        and sim.battle_type in ("siege_assault", "siege_defense")
        and abs(u.y - sim.wall_y) <= 10.0
        and abs(u.x - sim.gate_x) > 15.0
        and not u.tpl.is_cav
    )
    out[off + 25] = 1.0 if on_wall else 0.0


def build_obs(
    sim: BattleSim,
    own_slots: List[Optional[SimUnit]],
    en_slots: List[Optional[SimUnit]],
    fr: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.zeros(MAX_UNITS, dtype=np.float32)

    own = sim.sides[0]
    en = sim.sides[1]

    # 12 global features
    bt_idx = BATTLE_TYPES.index(sim.battle_type) if sim.battle_type in BATTLE_TYPES else 0
    obs[bt_idx] = 1.0
    own_alive = sum(u.alive for u in own)
    en_alive = sum(u.alive for u in en)
    obs[4] = clamp(own_alive / max(1, sim.own_start), 0.0, 1.0)
    obs[5] = clamp(en_alive / max(1, sim.en_start), 0.0, 1.0)
    own_m = [u.morale_level / 6.0 for u in own if u.alive > 0]
    en_m = [u.morale_level / 6.0 for u in en if u.alive > 0]
    obs[6] = float(sum(own_m) / len(own_m)) if own_m else 0.0
    obs[7] = float(sum(en_m) / len(en_m)) if en_m else 0.0
    obs[8] = clamp(sim.t / 900.0, 0.0, 1.0)
    obs[9] = 1.0 if any(u.role == "general" and u.active for u in own) else 0.0
    obs[10] = 1.0 if any(u.role == "general" and u.active for u in en) else 0.0
    if sim.battle_type in ("siege_assault", "siege_defense"):
        breach = 1.0 if (sim.gate_hp <= 0 or sim.wall_left_hp <= 0 or sim.wall_right_hp <= 0) else 0.0
        obs[11] = max(breach, clamp(sim.plaza_timer / 30.0, 0.0, 1.0))

    zs = [sim.elevation(u.x, u.y) for u in own + en if u.alive > 0]
    mean_z = float(sum(zs) / len(zs)) if zs else 0.0

    # 24 own slots
    base_own = GLOBAL_FEAT
    for i, u in enumerate(own_slots):
        if u is not None and u.alive > 0:
            _encode_unit(u, fr, obs, base_own + i * UNIT_FEAT, sim=sim, mean_z=mean_z)
            if u.active:
                mask[i] = 1.0

    # 24 enemy slots
    base_en = GLOBAL_FEAT + MAX_UNITS * UNIT_FEAT
    for i, u in enumerate(en_slots):
        if u is not None and u.alive > 0:
            _encode_unit(u, fr, obs, base_en + i * UNIT_FEAT, sim=sim, mean_z=mean_z)

    return obs, mask


def _vanilla_target_action(u: SimUnit, prev_x: float, prev_y: float, en: List[SimUnit], fr: dict) -> int:
    """Infer the discrete action 0..12 that best matches Vanilla AI's physical move/attack over this tick."""
    vq = None
    if u.order_kind == "attack" and u.target_uid is not None:
        tgt = next((e for e in en if e.uid == u.target_uid and e.alive > 0), None)
        if tgt is not None:
            dx, dy = tgt.x - u.x, tgt.y - u.y
            rx = dx * fr["rx"] + dy * fr["ry"]
            fy = dx * fr["fx"] + dy * fr["fy"]
            ang = math.degrees(math.atan2(rx, fy))
            if abs(ang) <= 45.0:
                vq = 0
            elif -135.0 <= ang < -45.0:
                vq = 1
            elif 45.0 < ang <= 135.0:
                vq = 2
            else:
                vq = 3
    dx, dy = u.x - prev_x, u.y - prev_y
    disp = (dx * fr["rx"] + dy * fr["ry"], dx * fr["fx"] + dy * fr["fy"])
    best_a, best_r = 8, -1e9
    for a in range(NUM_ACTIONS):
        r = imitation_reward(a, disp, vq)
        if r > best_r:
            best_r = r
            best_a = a
    return best_a


@torch.no_grad()
def _apply_neural_physics(
    sim: BattleSim,
    obs: np.ndarray,
    own_slots: List[Optional[SimUnit]],
    en_slots: List[Optional[SimUnit]],
    dmg_net: UnitSelfAttentionNet,
    mor_net: UnitSelfAttentionNet,
):
    """Apply Phase-0 Self-Attention DamageNet (subtract HP) & MoraleNet (assign morale) to every alive unit."""
    nodes, alive = obs_to_unit_nodes(obs)
    if alive.sum() < 1.0:
        return
    n_t = torch.from_numpy(nodes[None])
    m_t = torch.from_numpy(alive[None])
    pred_dhp = dmg_net(n_t, m_t)[0].numpy()
    pred_mor = mor_net(n_t, m_t)[0].numpy()

    all_slots = list(own_slots) + list(en_slots)
    for idx, u in enumerate(all_slots):
        if u is None or u.alive <= 0:
            continue
        # Subtract predicted damage fraction from unit HP
        dhp = float(pred_dhp[idx])
        if dhp > 0.002:
            hp_loss = dhp * float(u.start * u.tpl.hp)
            u.hp_pool = max(0.0, u.hp_pool - hp_loss)
            new_alive = max(0, min(u.alive, int(math.ceil(u.hp_pool / max(1.0, float(u.tpl.hp))))))
            lost = u.alive - new_alive
            if lost > 0:
                u.alive = new_alive
                u.recent_loss += lost / max(1, u.start)
                if u.alive <= 0 and u.role == "general" and sim.general_dead_time[u.side] is None:
                    sim.general_dead_time[u.side] = sim.t
        # Assign predicted morale directly to unit
        if u.alive > 0:
            m_val = float(clamp(float(pred_mor[idx]), 0.0, 1.0))
            u.morale = m_val * 12.0
            u.morale_level = int(round(m_val * 6.0))
            if u.morale_level <= 0:
                u.routing = True
                u.rout_count += 1
            elif u.routing and u.morale_level >= 2 and not u.shattered:
                u.routing = False
                u.order_kind = "hold"


def run_lstm_battle(
    sim: BattleSim,
    battle_id: int,
    model: FullArmyLSTM,
    inv_temp: float = 1.0,
    greedy: bool = False,
    record_replay: bool = False,
    phase: int = 2,
    dmg_net: Optional[UnitSelfAttentionNet] = None,
    mor_net: Optional[UnitSelfAttentionNet] = None,
) -> Tuple[dict, Optional[dict]]:
    use_neural_phys = (dmg_net is not None and mor_net is not None)
    sim.neural_physics = use_neural_phys
    own = sim.sides[0]
    en = sim.sides[1]
    fr0 = _army_frame(own, en)
    own_slots = _bind_slots(own, fr0)
    en_slots = _bind_slots(en, fr0)

    hx = None
    steps = []
    replay_frames = []

    prev_own_alive = sum(u.alive for u in own)
    prev_en_alive = sum(u.alive for u in en)
    prev_u_alive = [u.alive if u is not None else 0 for u in own_slots]
    prev_u_killed = [u.killed if u is not None else 0 for u in own_slots]

    max_ticks = int(900.0 / DECISION_DT)
    for tick in range(max_ticks):
        if sim.is_done():
            break

        fr = _army_frame(own, en)
        obs, mask = build_obs(sim, own_slots, en_slots, fr)
        prev_pos = [(u.x, u.y) if u is not None else (0.0, 0.0) for u in own_slots]

        if phase == 1:
            acts = [8] * MAX_UNITS
        else:
            # Single forward pass gives commands to ALL friendly units at once
            acts, hx = model.act_step(obs, mask, hx=hx, inv_temp=inv_temp, greedy=greedy)
            for i, u in enumerate(own_slots):
                if u is not None and u.active:
                    _issue_action(acts[i], u, en, fr, sim=sim)
                    u.last_a = acts[i]

        # Advance physics for DECISION_DT (4.0s) in 1.0s substeps
        n_sub = int(round(DECISION_DT / SUB_DT))
        for _ in range(n_sub):
            if sim.is_done():
                break
            if phase == 1:
                step_vanilla_ai(sim, 0)
            step_vanilla_ai(sim, 1)
            sim.step(SUB_DT)

        # Apply Phase-0 learned DamageNet & MoraleNet once per tick when active
        if use_neural_phys and not sim.is_done():
            _apply_neural_physics(sim, obs, own_slots, en_slots, dmg_net, mor_net)

        if phase == 1:
            for i, u in enumerate(own_slots):
                if u is not None and mask[i] > 0.5:
                    acts[i] = _vanilla_target_action(u, prev_pos[i][0], prev_pos[i][1], en, fr)
                    u.last_a = acts[i]

        # Compute step deltas
        cur_own_alive = sum(u.alive for u in own)
        cur_en_alive = sum(u.alive for u in en)
        dk = max(0.0, (prev_en_alive - cur_en_alive) / max(1, sim.en_start))
        dl = max(0.0, (prev_own_alive - cur_own_alive) / max(1, sim.own_start))
        prev_own_alive, prev_en_alive = cur_own_alive, cur_en_alive

        u_dk = [0.0] * MAX_UNITS
        u_dl = [0.0] * MAX_UNITS
        inf_cx, inf_cy = _inf_centroid(own, fr)
        spread = _army_spread(own)
        cohs = []
        for i, u in enumerate(own_slots):
            if u is not None:
                u_dk[i] = max(0.0, (u.killed - prev_u_killed[i]) / max(1, u.start))
                u_dl[i] = max(0.0, (prev_u_alive[i] - u.alive) / max(1, u.start))
                prev_u_killed[i] = u.killed
                prev_u_alive[i] = u.alive
                if u.active:
                    cohs.append(_cohesion_reward(u, inf_cx, inf_cy, spread))
        mean_coh = float(sum(cohs) / len(cohs)) if cohs else 0.0

        moved_dists = []
        for i, u in enumerate(own_slots):
            if u is not None and u.active and mask[i] > 0.5:
                moved_dists.append(math.hypot(u.x - prev_pos[i][0], u.y - prev_pos[i][1]))
        mean_move_m = float(sum(moved_dists) / len(moved_dists)) if moved_dists else 0.0

        steps.append({
            "obs": obs.tolist(),
            "mask": mask.tolist(),
            "acts": acts,
            "move_m": round(mean_move_m, 3),
            "dk": round(dk, 5),
            "dl": round(dl, 5),
            "coh": round(mean_coh, 5),
            "u_dk": [round(x, 5) for x in u_dk],
            "u_dl": [round(x, 5) for x in u_dl],
        })

        if record_replay and (tick % 2 == 0 or sim.is_done()):
            sym_map = {"line_inf": "I", "spear_inf": "S", "shock_cav": "C", "missile_foot": "M", "missile_cav": "H", "artillery": "A", "general": "G"}
            frame_units = []
            for s_idx, side_num in ((0, 1), (1, 2)):
                for u in sim.sides[s_idx]:
                    if u.alive <= 0:
                        continue
                    hp_pct = int(round(100.0 * u.alive / max(1, u.start)))
                    if u.routing:
                        st_code = 6
                    elif u.engaged:
                        st_code = 4
                    elif u.tpl.is_missile and u.ammo > 0 and u.fire_cooldown > 0:
                        st_code = 5
                    elif u.order_kind in ("move", "attack"):
                        st_code = 0
                    else:
                        st_code = 3
                    frame_units.append([
                        side_num, u.uid, round(u.x, 1), round(u.y, 1),
                        hp_pct, u.morale_level, st_code, u.last_a if s_idx == 0 else -1,
                        sym_map.get(u.role, "I"), u.routing,
                        "RAM" if u.has_ram else ("LAD" if u.has_ladder else ""),
                    ])
            replay_frames.append({
                "t": round(sim.t, 1),
                "units": frame_units,
                "gate_hp": round(sim.gate_hp, 1),
                "ladders": [sim.ladder_left_docked, sim.ladder_right_docked],
                "plaza_timer": round(sim.plaza_timer, 1),
            })

    res_i = int(sim.result())
    own_surv = sum(u.alive for u in own) / max(1, sim.own_start)
    en_surv = sum(u.alive for u in en) / max(1, sim.en_start)
    en_dmg = clamp(1.0 - en_surv, 0.0, 1.0)
    own_loss = clamp(1.0 - own_surv, 0.0, 1.0)
    all_en_flee = all(not u.active for u in en)
    ep = {
        "battle_id": battle_id,
        "arch": "lstm",
        "battle_type": sim.battle_type,
        "duration_sec": round(sim.t, 1),
        "result": res_i,
        "enemy_damage_frac": round(en_dmg, 4),
        "own_loss_frac": round(own_loss, 4),
        "decisive_win": (res_i == 1 and (en_surv < 0.35 or all_en_flee)),
        "decisive_loss": (res_i == -1 and own_surv < 0.35),
        "steps": steps,
    }
    rep = None
    if record_replay:
        rep = {
            "battle_id": battle_id,
            "battle_type": sim.battle_type,
            "terrain": sim.terrain,
            "result": ep["result"],
            "enemy_damage_frac": ep["enemy_damage_frac"],
            "frames": replay_frames,
        }
    return ep, rep
