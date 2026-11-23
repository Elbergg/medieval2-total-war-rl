"""RL environment & feature/action/episode bridge matching rl_battle_ai.lua and spec.py."""
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

TRAINER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace", "trainer")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

from spec import (  # noqa: E402
    ATTACK_ACTIONS, CMD_RNN_HIDDEN, CMD_RNN_OUT, GRID, GRID_CH,
    MOVE_OFFSETS, NUM_ACTIONS, NUM_GLOBAL, NUM_LOCAL, ROLES, UNIT_MLP_HIDDEN,
)
from combat import BattleSim, SimUnit, clamp  # noqa: E402
from vanilla_ai import step_vanilla_ai  # noqa: E402

CMD_INTERVAL = 12.0
UNIT_COMMIT_SEC = 6.0
SIM_DT = 2.0  # 3 analytical simulation steps per 6s unit decision window


def _softmax_sample(logits: np.ndarray, rng: np.random.Generator, inv_temp: float = 2.2) -> int:
    z = (logits - np.max(logits)) * inv_temp
    e = np.exp(np.clip(z, -30.0, 30.0))
    p = e / np.sum(e)
    return int(rng.choice(len(p), p=p))


def _unit_rnn_step(rw: dict, feats: List[Tuple[int, float]], h_prev: np.ndarray, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    if feats:
        idx = np.fromiter((p[0] for p in feats), dtype=np.int64, count=len(feats))
        val = np.fromiter((p[1] for p in feats), dtype=np.float64, count=len(feats))
        h_kb = np.tanh(rw["W_kb"][:, idx] @ val + rw["b_kb"])
        ctx_mask = idx < (NUM_GLOBAL + NUM_LOCAL)
        has_ctx = bool(np.any(ctx_mask))
        ctx_idx = idx[ctx_mask] if has_ctx else None
        ctx_val = val[ctx_mask] if has_ctx else None
        ctx_term = rw["W_top_x"][:, ctx_idx] @ ctx_val if has_ctx else 0.0
    else:
        h_kb = np.tanh(rw["b_kb"])
        has_ctx = False
        ctx_idx = ctx_val = None
        ctx_term = 0.0
    h_top = np.tanh(rw["W_top_kb"] @ h_kb + ctx_term + rw["Wh"] @ h_prev + rw["b_top"])
    logits = (rw["W_kb_out"] @ h_kb + rw["b_kb_out"]) + (rw["W2"] @ h_top)
    if "W_adapt_out" in rw and "W_adapt_h" in rw:
        ad_x = rw["W_adapt_x"][:, ctx_idx] @ ctx_val if (has_ctx and "W_adapt_x" in rw) else 0.0
        h_ad = np.tanh(rw["W_adapt_h"] @ h_top + ad_x + rw.get("b_adapt", 0.0))
        logits = logits + np.clip(rw["W_adapt_out"] @ h_ad, -1.0, 1.0)
    return _softmax_sample(logits, rng), h_top


def _cmd_rnn_step(rnn: dict, feats: List[Tuple[int, float]], h_prev: np.ndarray, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    if feats:
        idx = np.fromiter((p[0] for p in feats), dtype=np.int64, count=len(feats))
        val = np.fromiter((p[1] for p in feats), dtype=np.float64, count=len(feats))
        h_kb = np.tanh(rnn["W_kb"][:, idx] @ val + rnn["b_kb"])
        x_term = rnn["Wx"][:, idx] @ val
    else:
        idx = val = None
        h_kb = np.tanh(rnn["b_kb"])
        x_term = 0.0
    h_top = np.tanh(rnn["W_top_kb"] @ h_kb + x_term + rnn["Wh"] @ h_prev + rnn["bh"])
    logits = (rnn["W_kb_out"] @ h_kb + rnn["b_kb_out"]) + (rnn["Wo"] @ h_top)
    if "W_adapt_out" in rnn and "W_adapt_h" in rnn:
        ad_x = rnn["W_adapt_x"][:, idx] @ val if (idx is not None and "W_adapt_x" in rnn) else 0.0
        h_ad = np.tanh(rnn["W_adapt_h"] @ h_top + ad_x + rnn.get("b_adapt", 0.0))
        logits = logits + np.clip(rnn["W_adapt_out"] @ h_ad, -1.0, 1.0)
    return _softmax_sample(logits, rng), h_top


def _centroid(units: List[SimUnit]) -> Tuple[float, float]:
    sx = sy = w = 0.0
    for u in units:
        if u.alive > 0:
            wt = 1 if u.routing else max(1, u.alive)
            sx += u.x * wt
            sy += u.y * wt
            w += wt
    if w <= 0:
        return 0.0, 0.0
    return sx / w, sy / w


def _inf_centroid(units: List[SimUnit], fr: dict) -> Tuple[float, float]:
    sx = sy = w = 0.0
    for u in units:
        if u.active and u.role in ("line_inf", "spear_inf"):
            sx += u.x * u.alive
            sy += u.y * u.alive
            w += u.alive
    if w <= 0:
        return fr["cx"], fr["cy"]
    return sx / w, sy / w


def _frame(own: List[SimUnit], en: List[SimUnit]) -> dict:
    cx, cy = _centroid(own)
    ex, ey = _centroid(en)
    fx, fy = ex - cx, ey - cy
    n = math.hypot(fx, fy)
    if n < 1e-3:
        fx, fy = 0.0, 1.0
    else:
        fx, fy = fx / n, fy / n
    return {"cx": cx, "cy": cy, "fx": fx, "fy": fy, "rx": fy, "ry": -fx}


def _to_rel(fr: dict, x: float, y: float) -> Tuple[float, float]:
    dx, dy = x - fr["cx"], y - fr["cy"]
    return dx * fr["rx"] + dy * fr["ry"], dx * fr["fx"] + dy * fr["fy"]


def _strength(units: List[SimUnit], pred=None) -> float:
    s = 0.0
    for u in units:
        if u.active and (pred is None or pred(u)):
            s += u.alive * (1.0 + u.tpl.ai_value_per_soldier / 10.0)
    return s


def _global_features(sim: BattleSim, side: int, fr: dict) -> List[float]:
    own = sim.sides[side]
    en = sim.sides[1 - side]
    o_start = sim.own_start if side == 0 else sim.en_start
    e_start = sim.en_start if side == 0 else sim.own_start
    o_alive = sum(u.alive for u in own)
    e_alive = sum(u.alive for u in en)
    oS, eS = _strength(own), _strength(en)
    tot = max(1.0, oS + eS)

    def zone(lst, pred):
        return _strength(lst, lambda u: pred(*_to_rel(fr, u.x, u.y)))

    g = [0.0] * NUM_GLOBAL
    g[0] = 1.0
    g[1] = min(1.0, sim.t / 1800.0)
    g[2] = o_alive / max(1, o_start)
    g[3] = e_alive / max(1, e_start)
    g[4] = clamp((1.0 - g[3]) - (1.0 - g[2]), -1.0, 1.0)
    g[5] = math.tanh((zone(own, lambda x, y: abs(x) <= 80 and abs(y) <= 60) - zone(en, lambda x, y: abs(x) <= 80)) / tot)
    g[6] = math.tanh((zone(own, lambda x, y: x < -80) - zone(en, lambda x, y: x < -80)) / tot)
    g[7] = math.tanh((zone(own, lambda x, y: x > 80) - zone(en, lambda x, y: x > 80)) / tot)
    g[8] = zone(own, lambda x, y: y < -50) / max(1.0, oS)
    om = _strength(own, lambda u: u.tpl.is_missile)
    em = _strength(en, lambda u: u.tpl.is_missile)
    g[9] = (om - em) / (om + em + 1.0)
    oc = _strength(own, lambda u: u.tpl.is_cav)
    ec = _strength(en, lambda u: u.tpl.is_cav)
    g[10] = (oc - ec) / (oc + ec + 1.0)
    act = [u for u in own if u.active]
    g[11] = sum(1 for u in act if u.engaged) / len(act) if act else 0.0
    g[12] = 1.0 if any(u.role == "general" and u.active for u in own) else 0.0
    g[13] = 1.0 if any(u.role == "general" and u.active for u in en) else 0.0
    mo = sum(0 if u.routing else 1 for u in own if u.alive > 0)
    no = sum(1 for u in own if u.alive > 0)
    me = sum(0 if u.routing else 1 for u in en if u.alive > 0)
    ne = sum(1 for u in en if u.alive > 0)
    g[14] = mo / no if no > 0 else 0.0
    g[15] = me / ne if ne > 0 else 0.0
    g[16] = (oS - eS) / tot * 0.5 + 0.5
    if sim.battle_type in ("siege_assault", "siege_defense"):
        g[17] = 1.0 if side != sim.defender_side else -1.0
        breach = 1.0 if (sim.gate_hp <= 0 or sim.wall_left_hp <= 0 or sim.wall_right_hp <= 0) else (
            0.5 if (sim.ladder_left_docked or sim.ladder_right_docked) else (1.0 - sim.gate_hp / 100.0)
        )
        g[18] = clamp(breach, 0.0, 1.0)
        p_prog = clamp(sim.plaza_timer / 30.0, 0.0, 1.0)
        g[19] = p_prog if side != sim.defender_side else -p_prog
    elif sim.battle_type == "river_crossing":
        g[17] = 0.5
        g[18] = clamp(1.0 - abs(fr["cy"] - sim.river_y) / 250.0, 0.0, 1.0)
    return g


def _unit_status(u: SimUnit) -> int:
    if u.routing:
        return 6
    if u.engaged:
        return 4
    if u.tpl.is_missile and u.ammo > 0 and u.fire_cooldown > 0:
        return 5
    if u.order_kind in ("move", "attack", "withdraw"):
        return 0
    return 3


def _unit_sparse(gl: List[float], u: SimUnit, own: List[SimUnit], en: List[SimUnit], fr: dict, inf_cx: float, inf_cy: float) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for i in range(NUM_GLOBAL):
        v = gl[i] * (3.0 if 17 <= i < 50 else 1.0)
        if v != 0.0:
            out.append((i, v))

    t = u.tpl
    atk = t.ai_value_per_soldier
    l = [0.0] * NUM_LOCAL
    l[0] = 0.0 if t.is_cav else 1.0
    l[1] = 1.0 if t.is_cav else 0.0
    l[2] = 1.0 if t.uclass == "heavy" else 0.0
    l[3] = 1.0 if (t.uclass != "heavy" and not t.is_missile) else 0.0
    l[4] = 1.0 if t.is_missile else 0.0
    l[5] = 1.0 if (t.pri_spear or t.pri_long_pike) else 0.0
    l[7] = 1.0 if t.can_phalanx else 0.0
    l[9] = clamp(atk / 25.0, 0.0, 1.0)
    l[10] = clamp(t.melee_chg / 15.0, 0.0, 1.0)
    l[11] = clamp((t.pri_atk if t.is_missile else 0.0) / 20.0, 0.0, 1.0)
    l[12] = clamp(t.pri_range / 250.0, 0.0, 1.0)
    l[13] = clamp((t.armour + t.def_skill + t.shield) / 30.0, 0.0, 1.0)
    l[14] = clamp(t.move_speed_mod / 2.5, 0.0, 1.0)
    l[15] = clamp(t.mass / 5.0, 0.0, 1.0)
    rx, ry = _to_rel(fr, u.x, u.y)
    l[16] = clamp(rx / 300.0, -1.0, 1.0)
    l[17] = clamp(ry / 300.0, -1.0, 1.0)
    l[18] = u.alive / max(1, u.start)
    l[19] = clamp(u.morale_level / 6.0, 0.0, 1.0)
    l[20] = clamp(u.fatigue / 6.0, 0.0, 1.0)
    l[21] = (u.ammo / u.max_ammo) if u.max_ammo > 0 else 0.0
    l[22] = 1.0 if u.engaged else 0.0
    l[23] = 1.0 if u.is_under_fire else 0.0
    nf = sum(o.alive for o in own if o is not u and o.active and abs(o.x - u.x) < 80 and abs(o.y - u.y) < 80)
    ne = sum(o.alive for o in en if o.active and abs(o.x - u.x) < 80 and abs(o.y - u.y) < 80)
    l[24] = clamp(nf / 500.0, 0.0, 1.0)
    l[25] = clamp(ne / 500.0, 0.0, 1.0)
    idx_x = (u.x - inf_cx) * fr["rx"] + (u.y - inf_cy) * fr["ry"]
    idx_y = (u.x - inf_cx) * fr["fx"] + (u.y - inf_cy) * fr["fy"]
    l[26] = clamp(idx_x / 200.0, -1.0, 1.0)
    l[27] = clamp(idx_y / 200.0, -1.0, 1.0)
    for i, v in enumerate(l):
        if v != 0.0:
            out.append((NUM_GLOBAL + i, v))

    seen = set()
    for lst, is_en in ((own, False), (en, True)):
        for o in lst:
            if (not is_en and o is u) or o.alive <= 0:
                continue
            dx, dy = o.x - u.x, o.y - u.y
            gx = dx * fr["rx"] + dy * fr["ry"]
            gy = dx * fr["fx"] + dy * fr["fy"]
            if gx < -100 or gx >= 100 or gy < -100 or gy >= 100:
                continue
            c = int((gx + 100) // 25)
            rr = int((gy + 100) // 25)
            base = NUM_GLOBAL + NUM_LOCAL + (rr * GRID + c) * GRID_CH
            st = _unit_status(o)
            ch_list = [base + (1 if is_en else 0)]
            if st >= 3:
                ch_list.append(base + st - 1)
            for ch_idx in ch_list:
                if ch_idx not in seen:
                    seen.add(ch_idx)
                    out.append((ch_idx, 1.0))
    return out


def _build_cmd_base_phi(own: List[SimUnit], en: List[SimUnit], fr: dict, g: List[float]) -> List[Tuple[int, float]]:
    base: List[Tuple[int, float]] = []
    for i in range(20):
        v = g[i] * (3.0 if 17 <= i < 20 else 1.0)
        if abs(v) > 1e-6:
            base.append((i, round(v, 4)))
    tot = max(1.0, _strength(own) + _strength(en))
    cells: Dict[int, float] = {}
    for lst, ch in ((own, 0), (en, 1)):
        for u in lst:
            if u.active:
                rx, ry = _to_rel(fr, u.x, u.y)
                mc = int(clamp(math.floor((rx + 240.0) / 80.0), 0, 5))
                mr = int(clamp(math.floor((ry + 240.0) / 80.0), 0, 5))
                idx0 = 20 + (mr * 6 + mc) * 2 + ch
                cells[idx0] = cells.get(idx0, 0.0) + u.alive * (1.0 + u.tpl.ai_value_per_soldier / 10.0) / tot
    for idx0, val in cells.items():
        v = clamp(val * 3.0, 0.0, 1.0)
        if v > 1e-6:
            base.append((idx0, round(v, 4)))

    for lst, base_idx in ((own, 92), (en, 172)):
        items = []
        for u in lst:
            if u.active:
                rx, ry = _to_rel(fr, u.x, u.y)
                items.append((u.ri, rx, ry, u.alive / max(1, u.start)))
        items.sort(key=lambda x: (x[0], x[1]))
        for k, (ri, rx, ry, hp) in enumerate(items[:20]):
            ub = base_idx + k * 4
            for off, val in enumerate((clamp(rx / 300.0, -1.0, 1.0), clamp(ry / 300.0, -1.0, 1.0), clamp(hp, 0.0, 1.0), ri / 7.0)):
                if abs(val) > 1e-6:
                    base.append((ub + off, round(val, 4)))
    return base


def _nearest_enemy_quad(u: SimUnit, en: List[SimUnit], fr: dict, quad: int) -> Optional[SimUnit]:
    best = None
    best_d = 1e9
    for e in en:
        if not e.active:
            continue
        dx, dy = e.x - u.x, e.y - u.y
        rx = dx * fr["rx"] + dy * fr["ry"]
        fy = dx * fr["fx"] + dy * fr["fy"]
        ang = math.degrees(math.atan2(rx, fy))
        ok = (
            quad < 0
            or (quad == 0 and abs(ang) <= 45.0)
            or (quad == 1 and -135.0 <= ang < -45.0)
            or (quad == 2 and 45.0 < ang <= 135.0)
            or (quad == 3 and abs(ang) > 135.0)
        )
        if ok:
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best = e
    return best


def _issue_action(a: int, u: SimUnit, en: List[SimUnit], fr: dict, sim: Optional[BattleSim] = None):
    if sim is not None and sim.battle_type in ("siege_assault", "siege_defense"):
        att_sign = 1.0 if sim.defender_side == 1 else -1.0
        if u.side != sim.defender_side:
            if (u.has_ram and sim.gate_hp > 0) or u.has_ladder:
                if a not in (5, 6, 7):  # Unless explicitly ordered to fall back, push ram/ladder to wall
                    u.order_kind = "move"
                    u.target_x = sim.gate_x if u.has_ram else u.ladder_x
                    u.target_y = sim.wall_y
                    u.run = False
                    return
        else:
            # Defender inside fortress: don't sally outside the wall while gate/walls are holding
            enemy_inside = [e for e in en if e.active and (e.y - sim.wall_y) * att_sign > -8.0]
            if not enemy_inside and a in ATTACK_ACTIONS:
                u.order_kind = "hold"
                u.fire_at_will = True
                tgt = _nearest_enemy_quad(u, en, fr, a - 9) or _nearest_enemy_quad(u, en, fr, -1)
                if tgt is not None:
                    u.target_uid = tgt.uid
                return
            elif enemy_inside and a in ATTACK_ACTIONS:
                en = enemy_inside
    open_field = sim is not None and sim.battle_type == "open_field"
    # ---- Layer 3 reflexes (same low-level micro the vanilla bot and the real engine have; open field only) ----
    if open_field and not u.engaged:
        dists = [(math.hypot(e.x - u.x, e.y - u.y), e) for e in en if e.active]
        if dists:
            d0, e0 = min(dists, key=lambda t: t[0])
            # Missile units skirmish back when the enemy closes in
            if u.role in ("missile_foot", "missile_cav") and u.ammo > 0 and d0 < (35.0 if u.tpl.is_cav else 40.0):
                dist = max(1.0, d0)
                u.order_kind = "move"
                u.target_x = u.x - (e0.x - u.x) / dist * 60.0
                u.target_y = u.y - (e0.y - u.y) / dist * 60.0
                u.run = True
                return
            # Spearmen brace (hold) when enemy cavalry is near and not yet in melee range
            if u.role == "spear_inf" and a not in (5, 6, 7, 8) and d0 > 25.0 and any(
                e.tpl.is_cav and math.hypot(e.x - u.x, e.y - u.y) <= 70.0 for _, e in dists
            ):
                u.order_kind = "hold"
                return
    if a == 8:
        u.order_kind = "hold"
        u.fire_at_will = True
        return
    if a in ATTACK_ACTIONS:
        if u.engaged:
            return
        tgt = _nearest_enemy_quad(u, en, fr, a - 9) or _nearest_enemy_quad(u, en, fr, -1)
        if tgt is None:
            return
        # Layer 3 target filter: shock cavalry avoids charging into spearmen when other targets are within 135m
        if open_field and u.role == "shock_cav":
            ns = [e for e in en if e.active and e.role != "spear_inf" and math.hypot(e.x - u.x, e.y - u.y) <= 135.0]
            if ns:
                tgt = min(ns, key=lambda e: math.hypot(e.x - u.x, e.y - u.y))
        d = math.hypot(tgt.x - u.x, tgt.y - u.y)
        if u.tpl.is_missile and u.ammo > 0:
            r = max(40.0, u.tpl.pri_range) * 0.90
            if d <= r:
                u.order_kind = "hold"
                u.fire_at_will = True
                u.target_uid = tgt.uid
            else:
                u.order_kind = "move"
                u.target_x = tgt.x
                u.target_y = tgt.y
                u.run = False
            return
        max_eng = 120.0 if u.tpl.is_cav else 60.0
        if d <= max_eng:
            u.order_kind = "attack"
            u.target_uid = tgt.uid
            u.run = True
        else:
            u.order_kind = "move"
            u.target_x = tgt.x
            u.target_y = tgt.y
            u.run = d < 110.0
        return

    ox, oy = MOVE_OFFSETS[a]
    u.order_kind = "move"
    if (
        sim is not None
        and sim.battle_type in ("siege_assault", "siege_defense")
        and u.side != sim.defender_side
        and a in (0, 1, 2)
        and (sim.gate_hp <= 0 or sim.ladder_left_docked or sim.ladder_right_docked)
    ):
        # In siege assault, advancing orders steer toward the Town Plaza
        u.target_x = sim.plaza_x + ox * 0.4
        u.target_y = sim.plaza_y
        u.run = True
        return
    u.target_x = u.x + fr["rx"] * ox + fr["fx"] * oy
    u.target_y = u.y + fr["ry"] * ox + fr["fy"] * oy
    near_e = _nearest_enemy_quad(u, en, fr, -1)
    near_d = math.hypot(near_e.x - u.x, near_e.y - u.y) if near_e is not None else 999.0
    u.run = (near_d < 150.0 or u.is_under_fire)


def _army_spread(own: List[SimUnit]) -> float:
    xs = [u.x for u in own if u.active]
    ys = [u.y for u in own if u.active]
    if len(xs) < 2:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _cohesion_reward(u: SimUnit, inf_cx: float, inf_cy: float, spread: float) -> float:
    d = math.hypot(u.x - inf_cx, u.y - inf_cy)
    is_cav = u.role in ("shock_cav", "missile_cav", "general")
    hard = 250.0 if is_cav else 160.0
    coh = 0.04 if d <= hard else -min(2.5, (d - hard) / 50.0)
    if spread > 650.0:
        coh -= min(0.6, (spread - 650.0) / 400.0)
    return coh


def _assign_squads(own: List[SimUnit], inf_cx: float, inf_cy: float, fr: dict) -> Tuple[Dict[int, int], Optional[SimUnit]]:
    # 1 = left flank, 2 = center, 3 = right flank in army coordinate frame
    sq_map: Dict[int, int] = {}
    lead = None
    best_x = 1e9
    for u in own:
        if not u.active:
            continue
        rx = (u.x - inf_cx) * fr["rx"] + (u.y - inf_cy) * fr["ry"]
        if rx < -45.0:
            sq = 1
        elif rx > 45.0:
            sq = 3
        else:
            sq = 2
        sq_map[u.uid] = sq
        u.squad = sq
        if u.role == "line_inf" and abs(rx) < best_x:
            best_x = abs(rx)
            lead = u
    return sq_map, lead


def run_battle(
    sim: BattleSim,
    battle_id: int,
    mode: str,
    w_rl: Optional[List[dict]],
    rnn_rl: Optional[dict],
    w_opp: Optional[List[dict]] = None,
    rnn_opp: Optional[dict] = None,
    seed: int = 1,
    record_replay: bool = False,
) -> Tuple[dict, Optional[dict]]:
    rng = np.random.default_rng(seed)

    # Per-side RL state
    side_state = {}
    for s in (0, 1):
        side_state[s] = {
            "cmd_h": np.zeros(CMD_RNN_HIDDEN, dtype=np.float64),
            "last_cmd_t": -999.0,
            "global_cmd": 0,
            "sub_cmds": {1: 0, 2: 0, 3: 0},
            "lead_cmd": 0,
            "cmd_steps": [],
            "pending_cmd": [],
            "prev_tot_killed": 0,
            "prev_tot_alive": sum(u.alive for u in sim.sides[s]),
        }
        for u in sim.sides[s]:
            u.last_h = np.zeros(UNIT_MLP_HIDDEN, dtype=np.float64)
            u.prev_killed = 0
            u.prev_alive = u.alive
            u.was_engaged = False
            u.pending = None

    replay_frames = []

    def step_rl_side(side: int, w: List[dict], rnn: dict, is_learner: bool, imitate: bool = False):
        own = sim.sides[side]
        en = sim.sides[1 - side]
        if not any(u.active for u in own) or not any(u.active for u in en):
            return
        fr = _frame(own, en)
        inf_cx, inf_cy = _inf_centroid(own, fr)
        g = _global_features(sim, side, fr)
        st = side_state[side]
        sq_map, lead = _assign_squads(own, inf_cx, inf_cy, fr)
        spread = _army_spread(own)

        # Commander RNN decision every CMD_INTERVAL (aligned with rl_battle_ai.lua features 252..292)
        if not imitate and sim.t - st["last_cmd_t"] >= CMD_INTERVAL:
            if is_learner and st["pending_cmd"]:
                tot_k = sum(u.killed for u in own)
                tot_a = sum(u.alive for u in own)
                n_act = max(1, sum(1 for u in own if u.active))
                c_dk = (tot_k - st["prev_tot_killed"]) / (20.0 * n_act)
                c_dl = max(0, st["prev_tot_alive"] - tot_a) / (20.0 * n_act)
                c_coh = sum(_cohesion_reward(u, inf_cx, inf_cy, spread) for u in own if u.active) / n_act
                for pc in st["pending_cmd"]:
                    pc["dk"], pc["dl"], pc["coh"] = c_dk, c_dl, c_coh
                    st["cmd_steps"].append(pc)
                st["pending_cmd"] = []
                st["prev_tot_killed"] = tot_k
                st["prev_tot_alive"] = tot_a

            st["last_cmd_t"] = sim.t
            base_phi = _build_cmd_base_phi(own, en, fr, g)
            prev_g = st["global_cmd"]
            # Step 0: Global order (252=1, 263+prev_g=1, 273+prev_g=1)
            phi0 = list(base_phi) + [(252, 1.0), (263 + prev_g, 1.0)]
            if 273 + prev_g != 263 + prev_g:
                phi0.append((273 + prev_g, 1.0))
            h0 = st["cmd_h"].copy()
            g_cmd, st["cmd_h"] = _cmd_rnn_step(rnn, phi0, st["cmd_h"], rng)
            st["global_cmd"] = g_cmd
            if is_learner:
                st["pending_cmd"].append({"phi": phi0, "h": h0.tolist(), "a": g_cmd})
            prev_step_cmd = g_cmd
            # Steps 1..3: Detachment orders (matching Lua indices 253..292)
            for sq in (1, 2, 3):
                sq_units = [u for u in own if u.active and sq_map.get(u.uid, 2) == sq]
                if sq_units:
                    sq_rx = sum(((u.x - fr["cx"]) * fr["rx"] + (u.y - fr["cy"]) * fr["ry"]) for u in sq_units) / len(sq_units)
                    sq_ry = sum(((u.x - fr["cx"]) * fr["fx"] + (u.y - fr["cy"]) * fr["fy"]) for u in sq_units) / len(sq_units)
                    sq_hp = sum(u.alive for u in sq_units) / max(1, sum(u.start for u in sq_units))
                    sq_ri = sq_units[0].ri
                else:
                    sq_rx, sq_ry, sq_hp, sq_ri = (sq - 2) * 90.0, 0.0, 0.0, 1
                own_prev = st["sub_cmds"].get(sq, 0)
                phis = list(base_phi) + [
                    (253 + g_cmd, 1.0),
                    (263 + prev_step_cmd, 1.0),
                    (273 + own_prev, 1.0),
                    (282 + sq_ri, 1.0),
                    (290, round(clamp(sq_rx / 250.0, -1.0, 1.0), 4)),
                    (291, round(clamp(sq_ry / 250.0, -1.0, 1.0), 4)),
                    (292, round(clamp(sq_hp, 0.0, 1.0), 4)),
                ]
                hs = st["cmd_h"].copy()
                sc, st["cmd_h"] = _cmd_rnn_step(rnn, phis, st["cmd_h"], rng)
                st["sub_cmds"][sq] = sc
                prev_step_cmd = sc
                if is_learner:
                    st["pending_cmd"].append({"phi": phis, "h": hs.tolist(), "a": sc})
            # Step 4: Lead line_inf order
            if lead is not None:
                st["lead_cmd"] = st["sub_cmds"].get(lead.squad, 0)

        for i in range(20, 30):
            g[i] = 0.0
        g[20 + st["global_cmd"]] = 1.0
        for i in range(40, 50):
            g[i] = 0.0
        g[40 + st["lead_cmd"]] = 1.0

        for u in own:
            if not u.active:
                continue
            engaged_changed = (u.engaged != getattr(u, "was_engaged", False))
            u.was_engaged = u.engaged
            if not imitate and sim.t - u.last_cmd_t < UNIT_COMMIT_SEC and not engaged_changed:
                continue
            if imitate and sim.t - u.last_cmd_t < UNIT_COMMIT_SEC:
                continue
            u.last_cmd_t = sim.t

            if is_learner and u.pending is not None:
                p = u.pending
                dk = (u.killed - u.prev_killed) / 20.0
                dl = max(0, u.prev_alive - u.alive) / 20.0
                coh = _cohesion_reward(u, inf_cx, inf_cy, spread)
                step_d = {"phi": p["phi"], "h": p["h"], "a": p["a"], "dk": dk, "dl": dl, "coh": coh}
                if imitate and "frm" in p:
                    dx, dy = u.x - p["sx"], u.y - p["sy"]
                    pf = p["frm"]
                    step_d["disp"] = [dx * pf["rx"] + dy * pf["ry"], dx * pf["fx"] + dy * pf["fy"]]
                    if u.engaged:
                        tgt = _nearest_enemy_quad(u, en, pf, -1)
                        if tgt is not None:
                            ex, ey = tgt.x - u.x, tgt.y - u.y
                            ang = math.degrees(math.atan2(ex * pf["rx"] + ey * pf["ry"], ex * pf["fx"] + ey * pf["fy"]))
                            step_d["vq"] = 0 if abs(ang) <= 45 else (1 if -135 <= ang < -45 else (2 if 45 < ang <= 135 else 3))
                u.traj.append(step_d)
                u.prev_killed = u.killed
                u.prev_alive = u.alive
                u.pending = None

            for i in range(30, 40):
                g[i] = 0.0
            sq_cmd = st["sub_cmds"].get(sq_map.get(u.uid, 2), 0)
            g[30 + sq_cmd] = 1.0

            feats = _unit_sparse(g, u, own, en, fr, inf_cx, inf_cy)
            h_before = u.last_h.copy()
            if imitate:
                _, u.last_h = _unit_rnn_step(w[u.ri - 1], feats, u.last_h, rng)
                if is_learner:
                    u.pending = {"phi": feats, "h": h_before.tolist(), "a": -1, "sx": u.x, "sy": u.y, "frm": fr}
            else:
                a, u.last_h = _unit_rnn_step(w[u.ri - 1], feats, u.last_h, rng)
                _issue_action(a, u, en, fr, sim=sim)
                u.last_a = a
                if is_learner:
                    u.pending = {"phi": feats, "h": h_before.tolist(), "a": a}

    # Main simulation loop
    while not sim.is_done():
        # Side 0
        if mode == "vanilla_vs_vanilla":
            step_vanilla_ai(sim, 0)
            if w_rl is not None and rnn_rl is not None:
                step_rl_side(0, w_rl, rnn_rl, is_learner=True, imitate=True)
        else:
            step_rl_side(0, w_rl, rnn_rl, is_learner=True, imitate=False)

        # Side 1
        if mode == "rl_vs_frozen" and w_opp is not None and rnn_opp is not None:
            step_rl_side(1, w_opp, rnn_opp, is_learner=False)
        else:
            step_vanilla_ai(sim, 1)

        sim.step(SIM_DT)

        if record_replay:
            sym_map = {
                "line_inf": "I", "spear_inf": "S", "missile_foot": "A",
                "shock_cav": "C", "missile_cav": "C", "general": "G", "artillery": "T",
            }
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
                        hp_pct, u.morale_level, st_code, u.last_a,
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

    # Flush final pending steps
    fr0 = _frame(sim.sides[0], sim.sides[1])
    inf_cx0, inf_cy0 = _inf_centroid(sim.sides[0], fr0)
    spread0 = _army_spread(sim.sides[0])
    if side_state[0]["pending_cmd"]:
        own0 = sim.sides[0]
        tot_k = sum(u.killed for u in own0)
        tot_a = sum(u.alive for u in own0)
        n_act = max(1, sum(1 for u in own0 if u.active))
        c_dk = (tot_k - side_state[0]["prev_tot_killed"]) / (20.0 * n_act)
        c_dl = max(0, side_state[0]["prev_tot_alive"] - tot_a) / (20.0 * n_act)
        c_coh = sum(_cohesion_reward(u, inf_cx0, inf_cy0, spread0) for u in own0 if u.active) / n_act
        for pc in side_state[0]["pending_cmd"]:
            pc["dk"], pc["dl"], pc["coh"] = c_dk, c_dl, c_coh
            side_state[0]["cmd_steps"].append(pc)
        side_state[0]["pending_cmd"] = []
    for u in sim.sides[0]:
        if getattr(u, "pending", None) is not None:
            p = u.pending
            dk = (u.killed - u.prev_killed) / 20.0
            dl = max(0, u.prev_alive - u.alive) / 20.0
            coh = _cohesion_reward(u, inf_cx0, inf_cy0, spread0)
            u.traj.append({"phi": p["phi"], "h": p["h"], "a": p["a"], "dk": dk, "dl": dl, "coh": coh})
            u.pending = None

    res = sim.result()
    own_alive = sum(u.alive for u in sim.sides[0] if u.active)
    en_alive = sum(u.alive for u in sim.sides[1] if u.active)
    en_dmg = 1.0 - (en_alive / max(1, sim.en_start))
    decisive_win = (res > 0) and (en_alive / max(1, sim.en_start) < 0.35)
    decisive_loss = (res == -1) and (own_alive / max(1, sim.own_start) < 0.35)

    units_out = [{"role": u.role, "steps": u.traj} for u in sim.sides[0] if u.traj]
    if side_state[0]["cmd_steps"]:
        units_out.append({"role": "commander", "steps": side_state[0]["cmd_steps"]})

    ep = {
        "battle_id": battle_id,
        "mode": mode,
        "battle_type": sim.battle_type,
        "result": res,
        "decisive_win": decisive_win,
        "decisive_loss": decisive_loss,
        "enemy_damage_frac": round(en_dmg, 4),
        "own_start": sim.own_start,
        "own_alive": own_alive,
        "en_start": sim.en_start,
        "en_alive": en_alive,
        "duration_sec": round(sim.t, 1),
        "long_battle": (battle_id % 5 == 0),
        "units": units_out,
        "commander_steps": side_state[0]["cmd_steps"],
    }
    rep = None
    if record_replay:
        rep = {
            "battle_id": battle_id,
            "mode": mode,
            "battle_type": sim.battle_type,
            "terrain": sim.terrain,
            "result": res,
            "own_start": sim.own_start,
            "own_alive": own_alive,
            "en_start": sim.en_start,
            "en_alive": en_alive,
            "duration_sec": round(sim.t, 1),
            "frames": replay_frames,
        }
    return ep, rep
