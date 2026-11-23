"""Fast, M2TW-accurate Battle Physics, Combat, Missile, Fatigue, and Morale Simulator."""
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple

from edu_parser import ProjectileInfo, ROLES, UnitTemplate

MAP_HALF = 450.0
MELEE_HIT_RATE = 1.75  # battle_config.xml
ACC_INF = 0.79         # battle_config.xml
ACC_CAV = 0.56         # battle_config.xml


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)


@dataclass
class SimUnit:
    uid: int
    side: int                # 0 = RL/side0, 1 = Enemy/side1
    role: str
    ri: int                  # 1-based role index (1..7)
    tpl: UnitTemplate
    x: float
    y: float
    fx: float                # unit facing vector (normalized)
    fy: float
    alive: int
    start: int
    hp_pool: float
    ammo: int
    max_ammo: int
    # Dynamic combat state
    morale: float
    morale_level: int = 3
    routing: bool = False
    shattered: bool = False
    rout_count: int = 0
    rout_timer: float = 0.0
    fatigue_val: float = 0.0  # 0..100
    fatigue: int = 0          # 0..6
    engaged: bool = False
    under_fire_timer: float = 0.0
    charging: bool = False
    charge_timer: float = 0.0
    fire_cooldown: float = 0.0
    recent_loss: float = 0.0  # decaying casualty shock
    # Orders
    order_kind: str = "hold"  # hold | move | attack | withdraw
    target_x: float = 0.0
    target_y: float = 0.0
    target_uid: Optional[int] = None
    run: bool = False
    fire_at_will: bool = True
    # RL tracking fields (matching rl_battle_ai.lua)
    squad: int = 0
    has_ram: bool = False
    has_ladder: bool = False
    ladder_x: float = 0.0
    last_cmd_t: float = -999.0
    last_a: int = 8
    last_h: Optional[List[float]] = None
    last_feats: Optional[List[Tuple[int, float]]] = None
    prev_alive: int = 0
    killed: int = 0
    traj: List[dict] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.alive > 0 and not self.routing

    @property
    def is_under_fire(self) -> bool:
        return self.under_fire_timer > 0.0

    @property
    def width(self) -> float:
        cols = max(2, min(self.alive, math.ceil(self.alive / max(1, self.tpl.ranks))))
        return max(6.0, cols * self.tpl.spacing_x)

    @property
    def depth(self) -> float:
        cols = max(2, min(self.alive, math.ceil(self.alive / max(1, self.tpl.ranks))))
        rows = max(1, math.ceil(self.alive / cols))
        return max(3.0, rows * self.tpl.spacing_y)

    @property
    def radius(self) -> float:
        return clamp(0.5 * self.depth + 0.10 * self.width, 4.0, 10.0)


class BattleSim:
    def __init__(
        self,
        side0_templates: List[UnitTemplate],
        side1_templates: List[UnitTemplate],
        projectiles: Dict[str, ProjectileInfo],
        weather: str = "clear",
        seed: int = 1,
        deploy0: Optional[dict] = None,
        deploy1: Optional[dict] = None,
        terrain: Optional[dict] = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.projectiles = projectiles
        self.weather = weather
        self.terrain = terrain or {"battle_type": "open_field"}
        self.battle_type = self.terrain.get("battle_type", "open_field")
        # Smooth rolling terrain hills in open field (flat in sieges/river water)
        self.hill_amp = self.rng.uniform(4.0, 12.0) if self.battle_type == "open_field" else 0.0
        self.hill_phx = self.rng.uniform(0.0, 6.283)
        self.hill_phy = self.rng.uniform(0.0, 6.283)
        self.unit_size_scale = float(self.rng.choice([0.45, 0.65, 1.0]))
        self.t = 0.0
        self.units: List[SimUnit] = []
        self.sides: List[List[SimUnit]] = [[], []]
        self.general_dead_time: List[Optional[float]] = [None, None]
        self.own_start = 0
        self.en_start = 0

        # River crossing state
        self.river_y = float(self.terrain.get("river_y", 0.0))
        self.river_half_w = float(self.terrain.get("river_half_w", 22.0))
        self.bridges: List[dict] = self.terrain.get("bridges", [])

        # Siege state
        self.defender_side = int(self.terrain.get("defender_side", 1))
        self.wall_y = float(self.terrain.get("wall_y", 75.0))
        self.gate_x = float(self.terrain.get("gate_x", 0.0))
        self.gate_hp = 100.0
        self.wall_left_hp = 100.0
        self.wall_right_hp = 100.0
        self.ladder_left_docked = False
        self.ladder_right_docked = False
        self.tower_cooldown = 0.0
        self.plaza_x = float(self.terrain.get("plaza_x", 0.0))
        self.plaza_y = float(self.terrain.get("plaza_y", 215.0))
        self.plaza_r = float(self.terrain.get("plaza_r", 45.0))
        self.plaza_timer = 0.0
        self.plaza_captured = False

        uid = 0
        for side_idx, (tpls, dcfg) in enumerate(((side0_templates, deploy0), (side1_templates, deploy1))):
            placed = self._deploy_side(side_idx, tpls, uid, dcfg or {})
            uid += len(placed)
            self.sides[side_idx] = placed
            self.units.extend(placed)

        if self.battle_type in ("siege_assault", "siege_defense"):
            att_side = 1 - self.defender_side
            att_inf = [u for u in self.sides[att_side] if u.role in ("line_inf", "spear_inf")]
            att_inf.sort(key=lambda u: abs(u.x - self.gate_x))
            # Equip 2 center-most infantry units with Battering Rams
            for u in att_inf[:2]:
                u.has_ram = True
            # Equip up to 2 outer infantry units with Siege Ladders / Siege Towers
            flank_inf = [u for u in att_inf[2:] if u.role == "line_inf"] or att_inf[2:]
            if len(flank_inf) >= 1:
                flank_inf[0].has_ladder = True
                flank_inf[0].ladder_x = self.gate_x - 95.0
            if len(flank_inf) >= 2:
                flank_inf[1].has_ladder = True
                flank_inf[1].ladder_x = self.gate_x + 95.0

        self.own_start = sum(u.start for u in self.sides[0])
        self.en_start = sum(u.start for u in self.sides[1])
        self.unit_by_id: Dict[int, SimUnit] = {u.uid: u for u in self.units}

    def _deploy_side(self, side: int, tpls: List[UnitTemplate], start_uid: int, dcfg: dict) -> List[SimUnit]:
        sign = 1.0 if side == 0 else -1.0
        style = dcfg.get("style", "pitched_line")
        dx_shift = float(dcfg.get("dx", 0.0))
        y_dist = float(dcfg.get("y_dist", 125.0))
        if self.battle_type == "open_field":
            y_dist = min(y_dist, self.rng.uniform(100.0, 145.0))
        v_wing = float(dcfg.get("vanguard_wing", 1.0))
        y_base = -y_dist * sign
        fy = sign
        fx = 0.0

        buckets: Dict[str, List[UnitTemplate]] = {r: [] for r in ROLES}
        for t in tpls:
            buckets[t.role].append(t)

        placed: List[SimUnit] = []
        uid = start_uid

        def style_offset(x_loc: float, idx: int, total: int) -> Tuple[float, float]:
            frac = (idx - (max(1, total) - 1) / 2.0) / max(1.0, total / 2.0)
            if style == "oblique_left":
                return x_loc, -frac * 45.0 * sign
            if style == "oblique_right":
                return x_loc, frac * 45.0 * sign
            if style == "split_wings":
                gap = -95.0 if x_loc < 0 else 95.0
                return x_loc * 0.7 + gap, 15.0 * sign if abs(x_loc) > 60 else 0.0
            if style == "deep_column":
                col = (idx % 3) - 1
                row = idx // 3
                return col * 28.0, -row * 22.0 * sign
            return x_loc, 0.0

        def add_unit(t: UnitTemplate, x: float, y: float):
            nonlocal uid
            n = max(12, int(round(t.num_soldiers * self.unit_size_scale)))
            ri = ROLES.index(t.role) + 1
            px = clamp(x + dx_shift, -MAP_HALF + 30.0, MAP_HALF - 30.0)
            py = clamp(y, -MAP_HALF + 30.0, MAP_HALF - 30.0)
            u = SimUnit(
                uid=uid, side=side, role=t.role, ri=ri, tpl=t,
                x=px, y=py, fx=fx, fy=fy,
                alive=n, start=n, hp_pool=float(n * t.hp),
                ammo=t.pri_ammo, max_ammo=t.pri_ammo,
                morale=float(t.morale) + 2.0,
                prev_alive=n,
                target_x=px, target_y=py,
            )
            placed.append(u)
            uid += 1

        if style == "siege_garrison":
            # Defenders inside settlement: archers on wall battlements, infantry behind gate/walls, general/cav at plaza
            mf = buckets["missile_foot"]
            for i, t in enumerate(mf):
                off = (i - (len(mf) - 1) / 2.0) * 36.0
                add_unit(t, self.gate_x + off - dx_shift, self.wall_y - 8.0 * sign)
            inf = buckets["spear_inf"] + buckets["line_inf"]
            for i, t in enumerate(inf):
                if i < 3:
                    # Gatehouse defenders
                    off = (i - 1) * 22.0
                    add_unit(t, self.gate_x + off - dx_shift, self.wall_y - 26.0 * sign)
                else:
                    wing = -1.0 if (i % 2 == 1) else 1.0
                    slot = (i - 3) // 2
                    add_unit(t, self.gate_x + wing * (85.0 + slot * 24.0) - dx_shift, self.wall_y - 24.0 * sign)
            for i, t in enumerate(buckets["shock_cav"] + buckets["missile_cav"]):
                off = (i - 1) * 28.0
                add_unit(t, self.plaza_x + off - dx_shift, self.plaza_y + 20.0 * sign)
            for i, t in enumerate(buckets["general"]):
                add_unit(t, self.plaza_x - dx_shift, self.plaza_y)
            for i, t in enumerate(buckets["artillery"]):
                add_unit(t, self.gate_x + (i * 30.0) - dx_shift, self.wall_y - 45.0 * sign)
            return placed

        # Center infantry: spear_inf + line_inf
        inf = buckets["spear_inf"] + buckets["line_inf"]
        for i, t in enumerate(inf):
            off = (i - (len(inf) - 1) / 2.0) * 28.0
            sx, sy = style_offset(off, i, len(inf))
            add_unit(t, sx, y_base + sy)

        # Missile foot ahead of center
        mf = buckets["missile_foot"]
        for i, t in enumerate(mf):
            off = (i - (len(mf) - 1) / 2.0) * 30.0
            sx, sy = style_offset(off, i, len(mf))
            add_unit(t, sx, y_base + 28.0 * sign + sy)

        # Cavalry split on left/right wings (or detached flanking vanguard)
        cav = buckets["shock_cav"] + buckets["missile_cav"]
        half_w = max(80.0, len(inf) * 15.0 + 25.0)
        for i, t in enumerate(cav):
            if style == "flanking_vanguard":
                add_unit(t, v_wing * (half_w + 55.0 + i * 24.0), y_base + 70.0 * sign)
            else:
                wing = -1.0 if (i % 2 == 0) else 1.0
                slot = i // 2
                sx, sy = style_offset(wing * (half_w + slot * 28.0), i, len(cav))
                add_unit(t, sx, y_base - 5.0 * sign + sy)

        # General behind center
        for i, t in enumerate(buckets["general"]):
            off = (i - (len(buckets["general"]) - 1) / 2.0) * 30.0
            add_unit(t, off, y_base - 35.0 * sign)

        # Artillery behind general
        for i, t in enumerate(buckets["artillery"]):
            off = (i - (len(buckets["artillery"]) - 1) / 2.0) * 32.0
            add_unit(t, off, y_base - 55.0 * sign)

        return placed

    def is_done(self) -> bool:
        if self.t >= 1200.0 or self.plaza_captured:
            return True
        a0 = any(u.active for u in self.sides[0])
        a1 = any(u.active for u in self.sides[1])
        return not (a0 and a1)

    def result(self) -> int:
        if self.plaza_captured:
            att_side = 1 - self.defender_side
            return 1 if att_side == 0 else -1
        s0 = sum(u.alive for u in self.sides[0] if u.active)
        s1 = sum(u.alive for u in self.sides[1] if u.active)
        if s0 > 0 and s1 == 0:
            return 1
        if s1 > 0 and s0 == 0:
            return -1
        if self.battle_type in ("siege_assault", "siege_defense") and self.t >= 1200.0:
            # Defender wins on timeout if settlement plaza was not captured
            return 1 if self.defender_side == 0 else -1
        r0 = s0 / max(1, self.own_start)
        r1 = s1 / max(1, self.en_start)
        if r0 > r1 + 0.05:
            return 1
        if r1 > r0 + 0.05:
            return -1
        return 0

    def step(self, dt: float = 1.0):
        self.t += dt
        for u in self.units:
            u.engaged = False
            if u.under_fire_timer > 0:
                u.under_fire_timer = max(0.0, u.under_fire_timer - dt)
            if u.fire_cooldown > 0:
                u.fire_cooldown = max(0.0, u.fire_cooldown - dt)
            if u.charge_timer > 0:
                u.charge_timer = max(0.0, u.charge_timer - dt)
                if u.charge_timer <= 0:
                    u.charging = False
            u.recent_loss *= math.exp(-dt / 8.0)

        # 1. Movement & target resolution
        for u in self.units:
            if u.alive <= 0:
                continue
            self._move_unit(u, dt)

        # 2. Melee combat (simultaneous mutual melee with frontage split across multiple opponents)
        engaged_pairs = []
        fought_pairs = set()
        opp_count: Dict[int, int] = {}
        was_engaged = {u.uid: u.engaged for u in self.units}
        for u in self.units:
            if u.alive <= 0 or u.routing:
                continue
            tgt = self.unit_by_id.get(u.target_uid) if (u.order_kind == "attack" and u.target_uid is not None) else None
            # Also auto-engage any enemy physically colliding with a holding/moving unit
            if tgt is None or tgt.alive <= 0:
                best_e = None
                best_d = 1e9
                for e in self.sides[1 - u.side]:
                    if e.alive <= 0:
                        continue
                    d = math.hypot(e.x - u.x, e.y - u.y)
                    if d <= u.radius + e.radius + 2.0 and d < best_d:
                        best_d = d
                        best_e = e
                tgt = best_e
            if tgt is not None and tgt.alive > 0:
                dx = tgt.x - u.x
                dy = tgt.y - u.y
                dist = math.hypot(dx, dy)
                engage_dist = u.radius + tgt.radius + 2.0
                if dist <= engage_dist and not self._blocked_by_barrier(u, tgt):
                    if not was_engaged.get(u.uid) and u.run and u.charge_timer <= 0:
                        u.charging = True
                        u.charge_timer = 4.0
                    u.engaged = True
                    tgt.engaged = True
                    pair_key = (min(u.uid, tgt.uid), max(u.uid, tgt.uid))
                    if pair_key not in fought_pairs:
                        fought_pairs.add(pair_key)
                        engaged_pairs.append((u, tgt))
                        opp_count[u.uid] = opp_count.get(u.uid, 0) + 1
                        opp_count[tgt.uid] = opp_count.get(tgt.uid, 0) + 1

        # Snapshot pre-step fighter counts so Side 0 and Side 1 strike simultaneously
        alive_snap = {u.uid: u.alive for u in self.units}
        for u, tgt in engaged_pairs:
            self._apply_melee(u, tgt, dt, att_alive=alive_snap[u.uid], att_div=opp_count.get(u.uid, 1))
            if not tgt.routing and alive_snap[tgt.uid] > 0:
                self._apply_melee(tgt, u, dt, att_alive=alive_snap[tgt.uid], att_div=opp_count.get(tgt.uid, 1))

        # 3. Missile & artillery fire
        for u in self.units:
            if u.alive <= 0 or u.routing or u.engaged:
                continue
            if u.tpl.is_missile and u.ammo > 0 and u.fire_at_will and u.fire_cooldown <= 0.0:
                self._try_fire_missile(u)

        # 4. Siege towers/arrows & plaza capture
        if self.battle_type in ("siege_assault", "siege_defense"):
            self._update_siege(dt)

        # 5. Morale & routing
        self._update_morale(dt)

    def _blocked_by_barrier(self, a: SimUnit, b: SimUnit) -> bool:
        if self.battle_type == "river_crossing" and self.bridges:
            if (a.y - self.river_y) * (b.y - self.river_y) < 0:
                mx = 0.5 * (a.x + b.x)
                if any(abs(mx - br["x"]) <= br["width"] * 0.6 for br in self.bridges):
                    return False
                return True
        if self.battle_type in ("siege_assault", "siege_defense"):
            if (a.y - self.wall_y) * (b.y - self.wall_y) < 0:
                mx = 0.5 * (a.x + b.x)
                if self.gate_hp <= 0 and abs(mx - self.gate_x) <= 22.0:
                    return False
                if (self.wall_left_hp <= 0 or self.ladder_left_docked) and abs(mx - (self.gate_x - 95.0)) <= 20.0:
                    return False
                if (self.wall_right_hp <= 0 or self.ladder_right_docked) and abs(mx - (self.gate_x + 95.0)) <= 20.0:
                    return False
                return True
        return False

    def _update_siege(self, dt: float):
        att_side = 1 - self.defender_side
        def_side = self.defender_side
        # Reassign dropped battering ram or ladders if carriers routed or died before breaching
        if self.gate_hp > 0 and not any(u.active and u.has_ram for u in self.sides[att_side]):
            cand = [u for u in self.sides[att_side] if u.active and u.role in ("line_inf", "spear_inf") and not u.has_ladder]
            if cand:
                nearest_c = min(cand, key=lambda u: math.hypot(u.x - self.gate_x, u.y - self.wall_y))
                nearest_c.has_ram = True
        if not self.ladder_left_docked and not any(u.active and u.has_ladder and u.ladder_x < self.gate_x for u in self.sides[att_side]):
            cand = [u for u in self.sides[att_side] if u.active and u.role in ("line_inf", "spear_inf") and not u.has_ram and not u.has_ladder]
            if cand:
                c = min(cand, key=lambda u: abs(u.x - (self.gate_x - 95.0)))
                c.has_ladder = True
                c.ladder_x = self.gate_x - 95.0
        if not self.ladder_right_docked and not any(u.active and u.has_ladder and u.ladder_x > self.gate_x for u in self.sides[att_side]):
            cand = [u for u in self.sides[att_side] if u.active and u.role in ("line_inf", "spear_inf") and not u.has_ram and not u.has_ladder]
            if cand:
                c = min(cand, key=lambda u: abs(u.x - (self.gate_x + 95.0)))
                c.has_ladder = True
                c.ladder_x = self.gate_x + 95.0
        # Gatehouse arrow towers shoot approaching attackers while gate is intact
        if self.gate_hp > 0:
            self.tower_cooldown = max(0.0, self.tower_cooldown - dt)
            if self.tower_cooldown <= 0.0:
                near_att = [
                    u for u in self.sides[att_side]
                    if u.active and math.hypot(u.x - self.gate_x, u.y - self.wall_y) <= 135.0
                ]
                if near_att:
                    tgt = min(near_att, key=lambda u: math.hypot(u.x - self.gate_x, u.y - self.wall_y))
                    self._deal_damage(tgt, 1.4 if not tgt.has_ram else 0.4)
                    tgt.under_fire_timer = 3.5
                    self.tower_cooldown = 3.0

        # Town Plaza capture (battle_config.xml <plaza-capture>: soldier-ratio 0.8)
        att_in = sum(u.alive for u in self.sides[att_side] if u.active and math.hypot(u.x - self.plaza_x, u.y - self.plaza_y) <= self.plaza_r)
        def_in = sum(u.alive for u in self.sides[def_side] if u.active and math.hypot(u.x - self.plaza_x, u.y - self.plaza_y) <= self.plaza_r)
        if att_in >= 15 and att_in > def_in * 1.25:
            self.plaza_timer += dt
            if self.plaza_timer >= 30.0:
                self.plaza_captured = True
        elif def_in >= att_in:
            self.plaza_timer = max(0.0, self.plaza_timer - dt)

    def _base_speed(self, u: SimUnit, running: bool, charging: bool) -> float:
        if u.tpl.category == "siege":
            base = 1.6
        elif u.tpl.is_cav:
            base = 12.5 if charging else (8.5 if running else 4.5)
        else:
            base = 6.5 if charging else (4.8 if running else 2.6)
        fat_mult = 1.0 - 0.04 * u.fatigue
        return base * u.tpl.move_speed_mod * max(0.65, fat_mult)

    def _turn_toward(self, u: SimUnit, tx: float, ty: float, dt: float):
        n = math.hypot(tx, ty)
        if n < 1e-4:
            return
        tx, ty = tx / n, ty / n
        cur_ang = math.atan2(u.fy, u.fx)
        tgt_ang = math.atan2(ty, tx)
        diff = (tgt_ang - cur_ang + math.pi) % (2 * math.pi) - math.pi
        max_turn = (2.2 if u.tpl.is_cav else 1.6) * dt
        if abs(diff) <= max_turn:
            u.fx, u.fy = tx, ty
        else:
            new_ang = cur_ang + max_turn * (1.0 if diff > 0 else -1.0)
            u.fx, u.fy = math.cos(new_ang), math.sin(new_ang)

    def _move_unit(self, u: SimUnit, dt: float):
        gx, gy = u.x, u.y
        is_charging = False
        is_running = u.run

        if u.routing:
            # Flee toward own map edge (while still respecting river bridges and fortress walls below)
            gx = u.x
            gy = -MAP_HALF - 20.0 if u.side == 0 else MAP_HALF + 20.0
            is_running = True
            if abs(u.y) >= MAP_HALF - 8.0 or abs(u.x) >= MAP_HALF - 8.0:
                # Shatter at map edge without zeroing u.alive (avoids artificial dl spike vs real M2TW)
                u.shattered = True
                return
        elif (u.has_ram and self.gate_hp > 0) or u.has_ladder:
            gx = self.gate_x if u.has_ram else u.ladder_x
            gy = self.wall_y
        elif u.order_kind == "attack" and u.target_uid is not None:
            tgt = self.unit_by_id.get(u.target_uid)
            if tgt is None or tgt.alive <= 0:
                u.order_kind = "hold"
            else:
                gx, gy = tgt.x, tgt.y
                dx, dy = gx - u.x, gy - u.y
                dist = math.hypot(dx, dy)
                engage_dist = u.radius + tgt.radius + 2.0
                if dist <= engage_dist and not self._blocked_by_barrier(u, tgt):
                    self._turn_toward(u, dx, dy, dt)
                    if u.charge_timer <= 0 and not u.engaged:
                        u.charging = True
                        u.charge_timer = 4.0
                    u.fatigue_val = min(100.0, u.fatigue_val + 1.2 * dt)
                    return
                if dist <= u.tpl.charge_dist + engage_dist:
                    is_charging = True
                    is_running = True
                    if not u.charging:
                        u.charging = True
                        u.charge_timer = 4.5
        elif u.order_kind in ("move", "withdraw"):
            gx, gy = u.target_x, u.target_y
        else:
            # Hold: recover fatigue slowly, face nearest active enemy so braced spears stay oriented
            u.fatigue_val = max(0.0, u.fatigue_val - 1.0 * dt)
            u.fatigue = min(6, int(u.fatigue_val / 16.0))
            if not u.engaged:
                en_act = [e for e in self.sides[1 - u.side] if e.active]
                if en_act:
                    ne = min(en_act, key=lambda e: math.hypot(e.x - u.x, e.y - u.y))
                    self._turn_toward(u, ne.x - u.x, ne.y - u.y, dt)
            return

        speed_mult = 1.0

        # River crossing terrain constraint: funnel through bridges/fords
        if self.battle_type == "river_crossing" and self.bridges:
            in_river = abs(u.y - self.river_y) <= self.river_half_w
            wants_cross = (u.y - self.river_y) * (gy - self.river_y) <= 0
            on_bridge = None
            for b in self.bridges:
                if abs(u.x - b["x"]) <= b["width"] * 0.55:
                    on_bridge = b
                    break
            if on_bridge is not None and in_river and on_bridge.get("kind") == "ford":
                speed_mult *= 0.58
            if wants_cross or in_river:
                if on_bridge is None:
                    best_b = min(self.bridges, key=lambda b: abs(u.x - b["x"]) + 0.35 * abs(gx - b["x"]))
                    gx = best_b["x"]
                    bank_sign = -1.0 if u.y < self.river_y else 1.0
                    gy = self.river_y + bank_sign * (self.river_half_w + 4.0)
                else:
                    # Stay aligned inside the bridge/ford corridor until safely across the river
                    gx = clamp(gx, on_bridge["x"] - on_bridge["width"] * 0.35, on_bridge["x"] + on_bridge["width"] * 0.35)
                    if (u.y - self.river_y) * (gy - self.river_y) <= 0:
                        exit_sign = 1.0 if u.y < self.river_y else -1.0
                        gy = self.river_y + exit_sign * (self.river_half_w + 14.0)

        # Siege terrain constraint: Battering Ram, Siege Ladders/Towers, and Wall Breaches
        elif self.battle_type in ("siege_assault", "siege_defense"):
            att_sign = 1.0 if self.defender_side == 1 else -1.0
            if u.side != self.defender_side:
                if u.has_ram and self.gate_hp > 0:
                    gx = self.gate_x
                    gy = self.wall_y - 8.0 * att_sign
                    speed_mult = min(speed_mult, 0.45)  # Pushing heavy battering ram
                    if math.hypot(u.x - gx, u.y - gy) <= 14.0:
                        self.gate_hp = max(0.0, self.gate_hp - 16.0 * dt)
                        if self.gate_hp <= 0:
                            for o in self.sides[u.side]:
                                o.has_ram = False
                        return
                elif u.has_ladder:
                    gx = u.ladder_x
                    gy = self.wall_y - 6.0 * att_sign
                    speed_mult = min(speed_mult, 0.55)  # Carrying ladders / pushing siege tower
                    if math.hypot(u.x - gx, u.y - gy) <= 14.0:
                        if u.ladder_x < self.gate_x:
                            self.ladder_left_docked = True
                        else:
                            self.ladder_right_docked = True
                        u.has_ladder = False
                        return

            wants_cross_wall = (u.y - self.wall_y) * (gy - self.wall_y) <= 0
            if wants_cross_wall:
                open_passages = []
                if self.gate_hp <= 0:
                    open_passages.append((self.gate_x, 20.0))
                if self.wall_left_hp <= 0 or (not u.tpl.is_cav and self.ladder_left_docked):
                    open_passages.append((self.gate_x - 95.0, 18.0))
                if self.wall_right_hp <= 0 or (not u.tpl.is_cav and self.ladder_right_docked):
                    open_passages.append((self.gate_x + 95.0, 18.0))

                if not open_passages:
                    # Wall still intact: stage outside wall (or hold inside if defender)
                    side_sign = -att_sign if u.side != self.defender_side else att_sign
                    gy = self.wall_y + side_sign * 26.0
                else:
                    aligned_p = None
                    for px, pw in open_passages:
                        if abs(u.x - px) <= pw:
                            aligned_p = (px, pw)
                            break
                    if aligned_p is None:
                        best_px, _ = min(open_passages, key=lambda p: abs(u.x - p[0]) + 0.3 * abs(gx - p[0]))
                        gx = best_px
                        side_sign = -att_sign if (u.y - self.wall_y) * att_sign < 0 else att_sign
                        gy = self.wall_y + side_sign * 10.0
                    elif abs(u.y - self.wall_y) <= 18.0:
                        # Walk straight through the gate/breach corridor before turning laterally
                        gx = aligned_p[0]
                        exit_sign = att_sign if (u.y - self.wall_y) * att_sign < 0 else -att_sign
                        gy = self.wall_y + exit_sign * 22.0
                        # Climbing a docked ladder (wall still intact) takes time
                        using_ladder = (
                            (abs(aligned_p[0] - (self.gate_x - 95.0)) < 1.0 and self.wall_left_hp > 0)
                            or (abs(aligned_p[0] - (self.gate_x + 95.0)) < 1.0 and self.wall_right_hp > 0)
                        )
                        if using_ladder:
                            speed_mult = min(speed_mult, 0.35)

            # Climbing onto or descending from fortress battlements via wall stairs takes time
            if not u.tpl.is_cav and u.role != "artillery":
                in_cur = (u.y - self.wall_y) * att_sign
                in_goal = (gy - self.wall_y) * att_sign
                if (14.0 < in_cur <= 30.0 and in_goal <= 14.0) or (0.0 <= in_cur <= 14.0 and in_goal > 14.0):
                    speed_mult = min(speed_mult, 0.30)

        dx, dy = gx - u.x, gy - u.y
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            u.fatigue_val = max(0.0, u.fatigue_val - 0.6 * dt)
            u.fatigue = min(6, int(u.fatigue_val / 16.0))
            return

        # Wheeling/reforming lag: sharp turns (>60 deg) slow regiment movement while ranks pivot
        align = u.fx * (dx / dist) + u.fy * (dy / dist)
        if align < 0.5:
            speed_mult *= (0.45 if align < 0.0 else 0.70)

        # Friendly regiment collision drag (prevents blobbing/stacking multiple units on one spot)
        crowded = 0
        for o in self.sides[u.side]:
            if o is u or not o.active:
                continue
            od = math.hypot(u.x - o.x, u.y - o.y)
            if od < 10.0:
                crowded += 1
                if od > 0.5:
                    u.x += ((u.x - o.x) / od) * 0.6 * dt
                    u.y += ((u.y - o.y) / od) * 0.6 * dt
        if crowded > 0:
            speed_mult *= max(0.55, 1.0 - 0.18 * crowded)

        # Uphill slope drag & extra fatigue
        uphill_fat = 0.0
        if self.hill_amp > 0.0:
            z_cur = self.elevation(u.x, u.y)
            z_ahead = self.elevation(u.x + (dx / dist) * 10.0, u.y + (dy / dist) * 10.0)
            slope = (z_ahead - z_cur) / 10.0
            if slope > 0.03:
                speed_mult *= clamp(1.0 - slope * 2.0, 0.72, 1.0)
                uphill_fat = 0.35

        self._turn_toward(u, dx, dy, dt)
        spd = self._base_speed(u, is_running, is_charging) * speed_mult
        max_step = dist
        if u.order_kind == "attack" and u.target_uid is not None:
            tgt = self.unit_by_id.get(u.target_uid)
            if tgt is not None and tgt.alive > 0:
                stop_d = (u.radius + tgt.radius) * 0.85
                if dist > stop_d:
                    max_step = dist - stop_d
        move_d = min(max_step, spd * dt)
        u.x = clamp(u.x + (dx / dist) * move_d, -MAP_HALF + 5.0, MAP_HALF - 5.0)
        u.y = clamp(u.y + (dy / dist) * move_d, -MAP_HALF + 5.0, MAP_HALF - 5.0)

        fat_rate = (1.4 if is_charging else (0.9 if is_running else 0.15)) + uphill_fat
        u.fatigue_val = clamp(u.fatigue_val + fat_rate * dt, 0.0, 100.0)
        u.fatigue = min(6, int(u.fatigue_val / 16.0))

    def elevation(self, x: float, y: float) -> float:
        if self.hill_amp <= 0.0:
            return 0.0
        return self.hill_amp * (
            0.65 * math.sin(x / 170.0 + self.hill_phx) * math.cos(y / 170.0 + self.hill_phy)
            + 0.35 * math.sin((x + y) / 240.0 + self.hill_phx)
        )

    def _hit_quadrant(self, att: SimUnit, defu: SimUnit) -> str:
        # Vector from defender to attacker
        dx, dy = att.x - defu.x, att.y - defu.y
        d = max(1e-4, math.hypot(dx, dy))
        vx, vy = dx / d, dy / d
        dot = defu.fx * vx + defu.fy * vy
        if dot > 0.35:
            return "front"
        if dot < -0.35:
            return "rear"
        cross = defu.fx * vy - defu.fy * vx
        return "left" if cross > 0 else "right"

    def _apply_melee(self, att: SimUnit, defu: SimUnit, dt: float, att_alive: Optional[int] = None, att_div: int = 1):
        quad = self._hit_quadrant(att, defu)
        shield_ok = quad in ("front", "left")

        eff_atk = float(att.tpl.melee_atk)
        chg = float(att.tpl.melee_chg) if att.charging else 0.0

        # Braced spear/pike vs frontal cavalry charge
        braced_spear = (
            (defu.tpl.pri_spear or defu.tpl.pri_long_pike)
            and not defu.routing
            and defu.order_kind == "hold"
            and quad == "front"
            and att.tpl.is_cav
        )
        if braced_spear:
            chg = 0.0
            # Spear reflection damage to charging cav
            if att.charging:
                refl_p = clamp(0.05 + 0.02 * defu.tpl.melee_spear_bonus, 0.05, 0.35)
                self._deal_damage(att, min(att.alive, 15) * refl_p * dt, attacker=defu)

        if defu.tpl.is_cav and att.tpl.melee_spear_bonus > 0:
            eff_atk += att.tpl.melee_spear_bonus

        if quad == "rear":
            eff_atk += 4.0
        elif quad == "right":
            eff_atk += 2.0

        # Elevation advantage (downhill bonus / uphill penalty in open field)
        if self.hill_amp > 0.0:
            dz = self.elevation(att.x, att.y) - self.elevation(defu.x, defu.y)
            eff_atk += clamp(dz * 0.35, -2.5, 2.5)
            if dz < -1.0:
                chg *= 0.65
            elif dz > 1.0:
                chg *= 1.15

        eff_atk += chg - 0.5 * att.fatigue

        armour = defu.tpl.armour * (0.5 if att.tpl.melee_ap else 1.0)
        shield = defu.tpl.shield if shield_ok else 0.0
        braced_bonus = 2.0 if (defu.order_kind == "hold" and quad == "front") else 0.0
        if self.battle_type in ("siege_assault", "siege_defense") and defu.side == self.defender_side and abs(defu.y - self.wall_y) <= 22.0:
            braced_bonus += 3.0  # Defender holding gatehouse/wall breach chokepoint
        eff_def = armour + defu.tpl.def_skill + shield + braced_bonus - 0.4 * defu.fatigue

        # Frontage: overlapping front ranks split across att_div simultaneous opponents
        eff_alive = float(att_alive if att_alive is not None else att.alive) / max(1, att_div)
        frontage = max(6.0, min(att.width, defu.width) / max(0.8, att.tpl.spacing_x) * 1.5)
        n_fighters = min(eff_alive, frontage)

        hit_prob = clamp(0.018 + 0.006 * (eff_atk - eff_def), 0.005, 0.16) * (MELEE_HIT_RATE / 1.75)
        if defu.routing:
            hit_prob *= 2.0

        dmg = n_fighters * hit_prob * dt
        self._deal_damage(defu, dmg, attacker=att)

        # Morale shock for flanked/rear attacks
        if quad == "rear":
            defu.morale -= 1.2 * dt
        elif quad in ("left", "right"):
            defu.morale -= 0.4 * dt

    def _try_fire_missile(self, u: SimUnit):
        enemies = self.sides[1 - u.side]
        best = None
        best_d = 1e9
        if u.target_uid is not None:
            t = self.unit_by_id.get(u.target_uid)
            if t is not None and t.alive > 0 and t.side != u.side:
                d = math.hypot(t.x - u.x, t.y - u.y)
                if d <= u.tpl.pri_range:
                    best, best_d = t, d
        if best is None:
            for e in enemies:
                if e.alive <= 0:
                    continue
                d = math.hypot(e.x - u.x, e.y - u.y)
                if d <= u.tpl.pri_range and d < best_d:
                    best, best_d = e, d
        if best is None:
            # Allow siege artillery to bombard intact walls/gates even if no defender unit is within range
            if (
                u.tpl.pri_area
                and self.battle_type in ("siege_assault", "siege_defense")
                and u.side != self.defender_side
                and (self.gate_hp > 0 or self.wall_left_hp > 0 or self.wall_right_hp > 0)
                and math.hypot(self.gate_x - u.x, self.wall_y - u.y) <= u.tpl.pri_range
            ):
                u.ammo = max(0, u.ammo - 1)
                u.fire_cooldown = max(2.5, u.tpl.pri_delay + u.tpl.fire_delay + 2.0)
                if self.gate_hp > 0:
                    self.gate_hp = max(0.0, self.gate_hp - 18.0)
                elif self.wall_left_hp > 0:
                    self.wall_left_hp = max(0.0, self.wall_left_hp - 22.0)
                elif self.wall_right_hp > 0:
                    self.wall_right_hp = max(0.0, self.wall_right_hp - 22.0)
            return

        # Firing volley
        u.ammo = max(0, u.ammo - 1)
        u.fire_cooldown = max(2.5, u.tpl.pri_delay + u.tpl.fire_delay + 2.0)

        quad = self._hit_quadrant(u, best)
        shield_ok = quad in ("front", "left")
        base_acc = ACC_CAV if best.tpl.is_cav else ACC_INF
        proj = self.projectiles.get(u.tpl.pri_proj)
        if proj and proj.affected_by_rain and self.weather in ("rain", "storm"):
            base_acc *= 0.72
        if self.weather == "fog":
            base_acc *= 0.82
        range_factor = clamp(1.10 - 0.45 * (best_d / max(1.0, u.tpl.pri_range)), 0.35, 1.0)

        armour = best.tpl.armour * (0.5 if u.tpl.pri_ap else 1.0)
        shield = best.tpl.shield if shield_ok else 0.0
        if best.has_ram:
            shield += 6.0  # Battering ram roof cover against missiles
        if self.battle_type in ("siege_assault", "siege_defense") and best.side == self.defender_side and abs(best.y - self.wall_y) <= 18.0:
            shield += 4.0  # Wall crenellation cover
        eff_def = armour + shield * 0.85

        kill_per_shot = clamp(0.018 + 0.010 * (u.tpl.pri_atk - eff_def), 0.004, 0.25)
        if u.tpl.category == "siege":
            pieces = max(2.0, float(u.tpl.extras or 2)) * (u.alive / max(1, u.start))
            shots = pieces * 4.5
        else:
            shots = float(u.alive)
        expected_dmg = shots * base_acc * range_factor * kill_per_shot * 0.25
        self._deal_damage(best, expected_dmg, attacker=u)
        best.under_fire_timer = 4.5
        if u.tpl.pri_area:
            best.morale -= 1.2
            # Siege artillery breaches walls and gatehouse
            if self.battle_type in ("siege_assault", "siege_defense") and u.side != self.defender_side:
                if self.gate_hp > 0:
                    self.gate_hp = max(0.0, self.gate_hp - 18.0)
                elif self.wall_left_hp > 0:
                    self.wall_left_hp = max(0.0, self.wall_left_hp - 22.0)
                elif self.wall_right_hp > 0:
                    self.wall_right_hp = max(0.0, self.wall_right_hp - 22.0)

    def _deal_damage(self, target: SimUnit, dmg: float, attacker: Optional[SimUnit] = None):
        if target.alive <= 0 or dmg <= 0:
            return
        # Stochastic rounding for fractional damage so small skirmishes still kill men
        whole = int(dmg)
        frac = dmg - whole
        actual = whole + (1 if self.rng.random() < frac else 0)
        if actual <= 0:
            return
        old_alive = target.alive
        target.hp_pool = max(0.0, target.hp_pool - float(actual))
        new_alive = math.ceil(target.hp_pool / max(1, target.tpl.hp))
        if new_alive < old_alive:
            lost = old_alive - new_alive
            target.alive = new_alive
            target.recent_loss += lost / max(1, target.start)
            if attacker is not None:
                attacker.killed += lost
            if new_alive <= 0 and target.role == "general" and self.general_dead_time[target.side] is None:
                self.general_dead_time[target.side] = self.t

    def _update_morale(self, dt: float):
        for side_idx in (0, 1):
            own = self.sides[side_idx]
            en = self.sides[1 - side_idx]
            gens = [g for g in own if g.role == "general" and g.active]
            gen_dead_shock = 0.0
            if self.general_dead_time[side_idx] is not None:
                since = self.t - self.general_dead_time[side_idx]
                if since < 60.0:
                    gen_dead_shock = -4.0 * (1.0 - since / 60.0)

            for u in own:
                if u.alive <= 0:
                    continue
                if u.tpl.lock_morale:
                    u.morale = 10.0
                    u.morale_level = 5
                    continue

                # Base morale baseline (+3.0 so units hold lines like real M2TW until flanked/heavy losses)
                target_m = float(u.tpl.morale) + 3.0
                if u.role == "general":
                    target_m += 4.0
                if u.tpl.discipline == "disciplined":
                    shock_mult = 0.75
                    target_m += 1.0
                elif u.tpl.discipline == "impetuous":
                    shock_mult = 0.90
                    target_m += 0.5
                elif u.tpl.discipline == "low":
                    shock_mult = 1.30
                else:
                    shock_mult = 1.0

                # General proximity
                if any(math.hypot(g.x - u.x, g.y - u.y) <= 85.0 for g in gens):
                    target_m += 2.5
                target_m += gen_dead_shock * shock_mult

                # Cumulative & recent casualties
                loss_frac = 1.0 - (u.alive / max(1, u.start))
                target_m -= 6.5 * loss_frac
                target_m -= 20.0 * u.recent_loss * shock_mult

                # Local balance & nearby routing friendlies (80m)
                f_str = 0.0
                e_str = 0.0
                near_routs = 0
                for o in own:
                    if o.alive > 0 and math.hypot(o.x - u.x, o.y - u.y) <= 80.0:
                        if o.routing:
                            near_routs += 1
                        else:
                            f_str += o.alive
                for e in en:
                    if e.active and math.hypot(e.x - u.x, e.y - u.y) <= 80.0:
                        e_str += e.alive
                if e_str > f_str * 1.4 and e_str > 20:
                    target_m -= 2.2 * shock_mult
                elif f_str > e_str * 1.5 and u.engaged:
                    target_m += 1.5
                target_m -= min(3.5, near_routs * 1.2 * shock_mult)

                if u.is_under_fire:
                    target_m -= 1.2 * shock_mult
                target_m -= 0.30 * u.fatigue

                # Smoothly move morale toward target_m
                u.morale += (target_m - u.morale) * min(1.0, 0.35 * dt)

                # Update routing / rally state
                if not u.routing:
                    if u.morale <= 0.0:
                        u.routing = True
                        u.rout_timer = 0.0
                        u.rout_count += 1
                        if u.rout_count >= 2 or (u.alive / max(1, u.start)) < 0.20:
                            u.shattered = True
                else:
                    u.rout_timer += dt
                    if (
                        not u.shattered
                        and u.rout_timer > 14.0
                        and (u.alive / max(1, u.start)) >= 0.25
                        and u.morale > 2.5
                        and not any(e.active and math.hypot(e.x - u.x, e.y - u.y) < 60.0 for e in en)
                    ):
                        u.routing = False
                        u.order_kind = "hold"

                # Map continuous morale to M2TWEOP 0..6
                if u.routing:
                    u.morale_level = 0
                elif u.morale < 2.0:
                    u.morale_level = 1
                elif u.morale < 4.0:
                    u.morale_level = 2
                elif u.morale < 6.5:
                    u.morale_level = 3
                elif u.morale < 9.0:
                    u.morale_level = 4
                elif u.morale < 11.5:
                    u.morale_level = 5
                else:
                    u.morale_level = 6
