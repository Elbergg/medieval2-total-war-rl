"""Medieval II: Total War Grand Tactical AI (<gta>) reconstructed from sim_data/config_ai_battle.xml.

Directly parses config_ai_battle.xml at import time and implements M2TW's real managers:
  1. <battle-analyser>   : Attack vs Defend-Line decision from melee/ranged strength ratios
  2. <gta> Open Field    : 4-phase approach (500m -> 230m -> 120m -> 20m), missiles-in-front,
                           <outflanking> (single flank vs double envelopment when cav >= 4)
  3. <melee-manager>     : Open (40m inf / 120m cav) & Settlement (100m / 200m) engage ranges,
                           <attack-brace> (150m), <outflank-analyser> (phalanx & router weights),
                           <retreat-analyser> (40m / 60m cav pullback)
  4. <missile-analyser>  : Counter-battery priority for enemy missile/artillery units
  5. <defend-settlement> : 4-stage siege defense (repel -> perimeter -> streets -> pull-back/plaza)
                           + sally-out-ratio (2.0); <attack-settlement> breach & plaza assault
  6. <attack-crossing> & <defend-crossing> : 2-phase bridge assault, cav sneak-attack on weak
                           secondary fords (<0.25 ratio), and missile-fire pullback defense
"""
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from combat import BattleSim, SimUnit

_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim_data", "config_ai_battle.xml")


def _load_xml_cfg(path: str) -> Dict[str, float]:
    """Parse key numeric parameters from M2TW's config_ai_battle.xml (medium/hard defaults)."""
    cfg = {
        # melee-manager
        "open_inf_engage": 40.0,
        "open_mis_engage": 40.0,
        "open_cav_engage": 120.0,
        "settle_inf_engage": 100.0,
        "settle_mis_engage": 100.0,
        "settle_cav_engage": 200.0,
        "brace_dist": 150.0,
        "retreat_dist": 40.0,
        "retreat_cav_mult": 1.5,
        "cav_vs_phalanx": 16.0,
        "vs_routers": 0.25,
        # missile-analyser
        "missile_prio_mod": 0.003,
        # gta distances
        "assault_dist": 20.0,
        "contact_dist": 120.0,
        "close_approach_dist": 230.0,
        "distant_approach_dist": 500.0,
        # outflanking
        "outflank_str_ratio": 0.66,
        "double_env_min_cav": 4.0,
        # attack-battlegroup
        "fight_missile_ratio": 2.0,
        # battle-analyser
        "ba_str_ratio": 0.8,
        "ba_en_melee_mult": 3.0,
        "ba_fr_ranged_mult": 3.0,
        "ba_fr_ranged_div": 5.0,
        "ba_line_buffer": 25.0,
        # attack-settlement
        "att_settle_commit_ratio": 1.6,
        "att_settle_plaza_reform": 120.0,
        # defend-settlement
        "sally_out_ratio": 2.0,
        "def_penetration_streets": 0.25,
        "def_streets_force_ratio": 1.6,
        "def_streets_loss_ratio": 0.7,
        "def_breach_offset": 15.0,
        "def_pullback_stand_dist": 20.0,  # sqrt(400)
        # crossing
        "cross_p1_melee_off": -20.0,
        "cross_p1_art_off": 15.0,
        "cross_p2_melee_off": 70.0,
        "cross_p2_mis_off": 40.0,
        "cross_outshoot_ratio": 1.4,
        "sneak_str_ratio": 0.25,
        "sneak_standoff": 150.0,
        "sneak_cross_off": 50.0,
        "def_cross_standoff": 30.0,
        "def_cross_max_retreat": 190.0,
        "def_cross_melee_threat": 0.2,
    }
    if not os.path.exists(path):
        return cfg
    try:
        root = ET.parse(path).getroot()

        def _f(xpath: str, default: float) -> float:
            el = root.find(xpath)
            if el is not None and el.text:
                try:
                    return float(el.text.strip())
                except ValueError:
                    pass
            return default

        cfg["open_inf_engage"] = _f("melee-manager/open/infantry/max-engage-dist/hard", 40.0)
        cfg["open_mis_engage"] = _f("melee-manager/open/missile/max-engage-dist/hard", 40.0)
        cfg["open_cav_engage"] = _f("melee-manager/open/cavalry/max-engage-dist/hard", 120.0)
        cfg["settle_inf_engage"] = _f("melee-manager/settlement/infantry/max-engage-dist/hard", 100.0)
        cfg["settle_mis_engage"] = _f("melee-manager/settlement/missile/max-engage-dist/hard", 100.0)
        cfg["settle_cav_engage"] = _f("melee-manager/settlement/cavalry/max-engage-dist/hard", 200.0)
        cfg["brace_dist"] = _f("melee-manager/attack-brace/distance", 150.0)
        cfg["retreat_dist"] = _f("melee-manager/retreat-analyser/retreat-point/default/distance", 40.0)
        cfg["retreat_cav_mult"] = _f("melee-manager/retreat-analyser/retreat-point/cavalry/multiplier", 1.5)
        cfg["cav_vs_phalanx"] = _f("melee-manager/outflank-analyser/unit-priority/cavalry-vs-phalanx", 16.0)
        cfg["vs_routers"] = _f("melee-manager/outflank-analyser/unit-priority/vs-routers", 0.25)
        cfg["missile_prio_mod"] = _f("missile-analyser/missile-strength-base-priority-modifier", 0.003)

        cfg["assault_dist"] = _f("gta/assault-dist", 20.0)
        cfg["contact_dist"] = _f("gta/contact-dist", 120.0)
        cfg["close_approach_dist"] = _f("gta/close-approach-dist", 230.0)
        cfg["distant_approach_dist"] = _f("gta/distant-approach-dist", 500.0)
        cfg["outflank_str_ratio"] = _f("gta/outflanking/strength-ratio", 0.66)
        cfg["double_env_min_cav"] = _f("gta/outflanking/double-envelopment/min-units", 4.0)
        cfg["fight_missile_ratio"] = _f("gta/attack-battlegroup/fight/missile-ratio", 2.0)

        cfg["ba_str_ratio"] = _f("gta/battle-analyser/friendly-to-enemy-strength-ratio", 0.8)
        cfg["ba_en_melee_mult"] = _f("gta/battle-analyser/enemy-melee-strength-multiplier", 3.0)
        cfg["ba_fr_ranged_mult"] = _f("gta/battle-analyser/friendly-ranged-strength-multiplier", 3.0)
        cfg["ba_fr_ranged_div"] = _f("gta/battle-analyser/friendly-ranged-strength-divisor", 5.0)
        cfg["ba_line_buffer"] = _f("gta/battle-analyser/enemy-position-buffer-distance", 25.0)

        cfg["att_settle_commit_ratio"] = _f("gta/attack-settlement/detachment/commit/strength-ratio", 1.6)
        cfg["att_settle_plaza_reform"] = _f("gta/attack-settlement/tactics/capture-plaza/reform-dist", 120.0)
        cfg["sally_out_ratio"] = _f("gta/defend-settlement/threat-assessment/sally-out-ratio", 2.0)
        cfg["def_penetration_streets"] = _f("gta/defend-settlement/stage-configuration/defend-perimeter/progression-criteria/enemy-penetration", 0.25)
        cfg["def_streets_force_ratio"] = _f("gta/defend-settlement/stage-configuration/defend-streets/progression-criteria/force-ratio", 1.6)
        cfg["def_streets_loss_ratio"] = _f("gta/defend-settlement/stage-configuration/defend-streets/progression-criteria/loss-ratio", 0.7)
        cfg["def_breach_offset"] = _f("gta/defend-settlement/tactics/defend-breaches/breach-offset-pos", 15.0)
        cfg["def_pullback_stand_dist"] = math.sqrt(max(1.0, _f("gta/defend-settlement/tactics/pull-back/state-criteria/stand-ground/nearest-enemy", 400.0)))

        cfg["cross_p1_melee_off"] = _f("gta/attack-crossing/tactics/assault/phases/phase1/melee-offset", -20.0)
        cfg["cross_p1_art_off"] = _f("gta/attack-crossing/tactics/assault/phases/phase1/artillery-offset", 15.0)
        cfg["cross_p2_melee_off"] = _f("gta/attack-crossing/tactics/assault/phases/phase2/melee", 70.0)
        cfg["cross_p2_mis_off"] = _f("gta/attack-crossing/tactics/assault/phases/phase2/missile", 40.0)
        cfg["cross_outshoot_ratio"] = _f("gta/attack-crossing/tactics/assault/phases/phase2/entry-conditions/outshoot-ratio", 1.4)
        cfg["sneak_str_ratio"] = _f("gta/attack-crossing/tactics/sneak-attack/crossing-suitability/strength-ratio", 0.25)
        cfg["sneak_standoff"] = _f("gta/attack-crossing/tactics/sneak-attack/standoff-offset", 150.0)
        cfg["sneak_cross_off"] = _f("gta/attack-crossing/tactics/sneak-attack/crossing-offset", 50.0)
        cfg["def_cross_standoff"] = _f("gta/defend-crossing/tactics/defend-distance/standoff-distance", 30.0)
        cfg["def_cross_max_retreat"] = _f("gta/defend-crossing/tactics/defend-distance/max-retreat-distance", 190.0)
        cfg["def_cross_melee_threat"] = _f("gta/defend-crossing/detachment/melee-threat-threshold", 0.2)
    except Exception:
        pass
    return cfg


CFG = _load_xml_cfg(_XML_PATH)


def _unit_power(u: SimUnit) -> float:
    return float(u.alive) * (1.0 + 0.12 * u.tpl.ai_value_per_soldier)


def _sum_power(units: List[SimUnit], pred=None) -> float:
    return sum(_unit_power(u) for u in units if u.active and (pred is None or pred(u)))


def _centroid(units: List[SimUnit]) -> Tuple[float, float]:
    sx = sy = w = 0.0
    for u in units:
        if u.active:
            sx += u.x * u.alive
            sy += u.y * u.alive
            w += u.alive
    if w <= 0:
        return 0.0, 0.0
    return sx / w, sy / w


def _best_melee_target(u: SimUnit, candidates: List[SimUnit]) -> Optional[Tuple[SimUnit, float]]:
    """<melee-manager> + <outflank-analyser> target selector (penalizes cav-vs-phalanx & routers)."""
    best = None
    best_score = 1e18
    best_d = 1e9
    for e in candidates:
        if e.alive <= 0:
            continue
        d = math.hypot(e.x - u.x, e.y - u.y)
        score = d
        # <vs-routers>0.25 -> 4x distance penalty against routing enemies so AI fights active threats first
        if e.routing:
            score /= max(0.05, CFG["vs_routers"])
        # <cavalry-vs-phalanx>16 -> heavy penalty for cavalry charging spearmen/phalanx head-on
        if u.tpl.is_cav and (e.role == "spear_inf" or e.tpl.can_phalanx):
            score += CFG["cav_vs_phalanx"] * 8.0
        if score < best_score:
            best_score = score
            best = e
            best_d = d
    return (best, best_d) if best is not None else None


def _best_missile_target(u: SimUnit, candidates: List[SimUnit]) -> Optional[Tuple[SimUnit, float]]:
    """<missile-analyser> target selector: prioritizes enemy missile threats in range."""
    best = None
    best_score = -1e18
    best_d = 1e9
    max_r = max(80.0, u.tpl.pri_range)
    for e in candidates:
        if not e.active:
            continue
        d = math.hypot(e.x - u.x, e.y - u.y)
        in_r = 1.0 if d <= max_r else 0.0
        mis_bonus = (CFG["missile_prio_mod"] * _unit_power(e) * 120.0) if e.tpl.is_missile else 0.0
        # Closer & higher-value missile targets score highest
        score = in_r * 100.0 + mis_bonus - 0.35 * d
        if score > best_score:
            best_score = score
            best = e
            best_d = d
    return (best, best_d) if best is not None else None


def _step_siege_gta(sim: BattleSim, side: int, own: List[SimUnit], en: List[SimUnit]):
    att_sign = 1.0 if sim.defender_side == 1 else -1.0
    breached = (
        sim.gate_hp <= 0
        or sim.wall_left_hp <= 0
        or sim.wall_right_hp <= 0
        or sim.ladder_left_docked
        or sim.ladder_right_docked
    )

    if side == sim.defender_side:
        own_pow = _sum_power(own)
        en_pow = _sum_power(en)
        # <sally-out-ratio>2.0: if defender is >= 2x stronger than attacker, sally out!
        if own_pow >= CFG["sally_out_ratio"] * max(1.0, en_pow) and breached:
            _step_open_field_gta(sim, side, own, en, force_attack=True)
            return

        ram_or_lad = [e for e in en if e.has_ram or e.has_ladder]
        enemy_inside = [e for e in en if (e.y - sim.wall_y) * att_sign >= -4.0]
        pen_frac = len(enemy_inside) / max(1, len(en))
        own_start = sim.own_start if side == 0 else sim.en_start
        loss_frac = 1.0 - (sum(u.alive for u in own) / max(1, own_start))
        plaza_threat = sum(1 for e in enemy_inside if math.hypot(e.x - sim.plaza_x, e.y - sim.plaza_y) < 90.0)

        # 4-stage <defend-settlement> state machine from config_ai_battle.xml:
        # 1=repel-attackers -> 2=defend-perimeter -> 3=defend-streets -> 4=pull-back/defend-plaza
        in_pow = _sum_power(enemy_inside)
        if plaza_threat > 0 or loss_frac >= CFG["def_streets_loss_ratio"] or in_pow >= CFG["def_streets_force_ratio"] * max(1.0, own_pow):
            stage = 4  # pull-back -> defend-plaza
        elif pen_frac >= CFG["def_penetration_streets"]:
            stage = 3  # defend-streets
        elif breached or enemy_inside:
            stage = 2  # defend-perimeter
        else:
            stage = 1  # repel-attackers

        ladders_coming = any(e.has_ladder for e in en) or sim.ladder_left_docked or sim.ladder_right_docked
        breach_off = CFG["def_breach_offset"]
        stand_d = CFG["def_pullback_stand_dist"]

        for u in own:
            if u.engaged:
                continue
            near = _best_melee_target(u, en)
            if near is None:
                continue
            tgt, d = near

            if u.role in ("missile_foot", "artillery"):
                m_tgt = _best_missile_target(u, ram_or_lad if ram_or_lad else en)
                u.order_kind = "hold"
                u.fire_at_will = True
                u.target_uid = m_tgt[0].uid if m_tgt else tgt.uid
                continue

            if stage == 4:
                # <pull-back> -> <defend-plaza> (<stand-ground> if enemy within 20m)
                if d <= stand_d or math.hypot(u.x - sim.plaza_x, u.y - sim.plaza_y) <= 45.0:
                    near_in = _best_melee_target(u, enemy_inside) if enemy_inside else near
                    if near_in and near_in[1] <= CFG["settle_inf_engage"]:
                        u.order_kind = "attack"
                        u.target_uid = near_in[0].uid
                        u.run = True
                    else:
                        u.order_kind = "hold"
                else:
                    u.order_kind = "move"
                    u.target_x, u.target_y = sim.plaza_x, sim.plaza_y
                    u.run = True
                continue

            if u.role in ("shock_cav", "missile_cav", "general"):
                # Mobile plaza/street reserve; counter-charges inside threats within settle_cav_engage (200m)
                if enemy_inside:
                    near_in = _best_melee_target(u, enemy_inside)
                    if near_in and near_in[1] <= CFG["settle_cav_engage"]:
                        u.order_kind = "attack"
                        u.target_uid = near_in[0].uid
                        u.run = True
                        continue
                u.order_kind = "move" if math.hypot(u.x - sim.plaza_x, u.y - sim.plaza_y) > 30.0 else "hold"
                u.target_x, u.target_y = sim.plaza_x, sim.plaza_y
                u.run = False
                continue

            # Keep defending infantry inside fortress
            if (u.y - sim.wall_y) * att_sign < 2.0:
                u.order_kind = "move"
                u.target_x, u.target_y = u.x, sim.wall_y + breach_off * att_sign
                u.run = True
                continue

            if stage in (2, 3) and enemy_inside:
                near_in = _best_melee_target(u, enemy_inside)
                if near_in and near_in[1] <= CFG["settle_inf_engage"]:
                    u.order_kind = "attack"
                    u.target_uid = near_in[0].uid
                    u.run = True
                    continue

            # Stage 1/2: man walls against ladders or hold breach offset behind gate
            if ladders_coming and abs(u.x - sim.gate_x) > 45.0:
                wall_x = (sim.gate_x - 95.0) if u.x < sim.gate_x else (sim.gate_x + 95.0)
                wall_y = sim.wall_y + 8.0 * att_sign
                if abs(u.y - wall_y) > 4.0 or abs(u.x - wall_x) > 28.0:
                    u.order_kind = "move"
                    u.target_x = u.x if abs(u.x - wall_x) <= 28.0 else wall_x
                    u.target_y = wall_y
                    u.run = False
                else:
                    u.order_kind = "hold"
            else:
                gate_y = sim.wall_y + breach_off * att_sign
                if abs(u.y - gate_y) > 6.0 and abs(u.x - sim.gate_x) <= 55.0:
                    u.order_kind = "move"
                    u.target_x, u.target_y = sim.gate_x + (u.x - sim.gate_x) * 0.5, gate_y
                    u.run = False
                else:
                    u.order_kind = "hold"
        return

    # <attack-settlement> tactics
    for i, u in enumerate(own):
        if u.engaged:
            continue
        near = _best_melee_target(u, en)
        if near is None:
            continue
        tgt, d = near

        if u.has_ram or u.has_ladder:
            u.order_kind = "move"
            u.target_x = sim.gate_x if u.has_ram else u.ladder_x
            u.target_y = sim.wall_y
            u.run = False
            continue

        if u.role == "artillery":
            u.order_kind = "hold"
            u.fire_at_will = True
            u.target_uid = tgt.uid
            continue

        if u.role in ("missile_foot", "missile_cav") and u.ammo > 0:
            m_tgt = _best_missile_target(u, en)
            stage_y = sim.wall_y - 85.0 * att_sign
            if abs(u.y - stage_y) > 20.0 and (u.y - sim.wall_y) * att_sign < 0:
                u.order_kind = "move"
                u.target_x, u.target_y = u.x, stage_y
                u.run = False
            else:
                u.order_kind = "hold"
                u.fire_at_will = True
                u.target_uid = m_tgt[0].uid if m_tgt else tgt.uid
            continue

        if not breached:
            # Stage outside arrow tower range while ram/artillery breaches gate
            stage_y = sim.wall_y - 80.0 * att_sign
            off_x = ((i % 7) - 3) * 24.0
            u.order_kind = "move" if math.hypot(u.x - (sim.gate_x + off_x), u.y - stage_y) > 15.0 else "hold"
            u.target_x, u.target_y = sim.gate_x + off_x, stage_y
            u.run = False
            continue

        # Breached! Storm through gate & push to Town Plaza (reform-dist = 120m)
        inside_self = (u.y - sim.wall_y) * att_sign > 5.0
        eng_d = CFG["settle_cav_engage"] if u.tpl.is_cav else CFG["settle_inf_engage"]
        if inside_self and d > eng_d * 0.7:
            u.order_kind = "move"
            u.target_x, u.target_y = sim.plaza_x, sim.plaza_y
            u.run = True
        else:
            u.order_kind = "attack"
            u.target_uid = tgt.uid
            u.run = True


def _step_crossing_gta(sim: BattleSim, side: int, own: List[SimUnit], en: List[SimUnit]):
    """<attack-crossing> & <defend-crossing> from config_ai_battle.xml."""
    bank_sign = -1.0 if side == 0 else 1.0
    main_b = sim.bridges[0] if sim.bridges else {"x": 0.0, "width": 28.0}
    flank_b = sim.bridges[1] if len(sim.bridges) > 1 else main_b

    own_ranged = _sum_power(own, lambda u: u.tpl.is_missile and u.ammo > 0)
    en_ranged = _sum_power(en, lambda u: u.tpl.is_missile and u.ammo > 0)
    own_melee = _sum_power(own, lambda u: not u.tpl.is_missile)
    en_melee = _sum_power(en, lambda u: not u.tpl.is_missile)

    # Decide attack-crossing vs defend-crossing: stronger ranged holds bank unless enemy crosses or outshot
    is_defender = (own_ranged > en_ranged * 1.15 and en_melee < own_melee * 1.3 and side == 1 and sim.t < 240.0)
    en_crossing = _sum_power(en, lambda e: (e.y - sim.river_y) * bank_sign > -sim.river_half_w)

    # Check secondary ford suitability for <sneak-attack> cavalry (<strength-ratio>0.25)
    cavs = [u for u in own if u.role == "shock_cav"]
    cav_pow = _sum_power(cavs)
    en_at_flank = _sum_power(en, lambda e: math.hypot(e.x - flank_b["x"], e.y - sim.river_y) < 110.0)
    sneak_ok = (len(sim.bridges) > 1) and (en_at_flank < CFG["sneak_str_ratio"] * max(1.0, cav_pow))

    # Phase 2 entry for attack-crossing: push across once formed (t > 20s) or outshot (>1.4x)
    push_phase2 = (not is_defender) and (sim.t >= 20.0 or en_ranged > own_ranged * CFG["cross_outshoot_ratio"])

    for u in own:
        if u.engaged:
            continue
        near = _best_melee_target(u, en)
        if near is None:
            continue
        tgt, d = near

        if u.role == "artillery":
            m_tgt = _best_missile_target(u, en)
            u.order_kind = "hold"
            u.fire_at_will = True
            u.target_uid = m_tgt[0].uid if m_tgt else tgt.uid
            continue

        if u.role in ("missile_foot", "missile_cav") and u.ammo > 0:
            m_tgt = _best_missile_target(u, en)
            bank_y = sim.river_y + bank_sign * (sim.river_half_w + CFG["def_cross_standoff"])
            if m_tgt and m_tgt[1] <= u.tpl.pri_range * 0.90:
                u.order_kind = "hold"
                u.fire_at_will = True
                u.target_uid = m_tgt[0].uid
            else:
                u.order_kind = "move"
                u.target_x = main_b["x"] + (u.x - main_b["x"]) * 0.5
                u.target_y = bank_y
                u.run = False
            continue

        if u.role == "shock_cav":
            on_own_bank = (u.y - sim.river_y) * bank_sign > 0
            if on_own_bank and d > CFG["open_cav_engage"]:
                if sneak_ok:
                    u.order_kind = "move"
                    u.target_x = flank_b["x"]
                    u.target_y = sim.river_y - bank_sign * (sim.river_half_w + CFG["sneak_cross_off"])
                    u.run = True
                else:
                    # Standoff at 150m if ford is heavily guarded
                    u.order_kind = "move"
                    u.target_x = main_b["x"] + (60.0 if u.x >= main_b["x"] else -60.0)
                    u.target_y = sim.river_y + bank_sign * CFG["sneak_standoff"]
                    u.run = False
            else:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            continue

        if u.role == "general":
            if d < 35.0:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            else:
                u.order_kind = "move"
                u.target_x = main_b["x"]
                u.target_y = sim.river_y + bank_sign * (sim.river_half_w + 55.0)
                u.run = False
            continue

        # Infantry: <defend-crossing> vs <attack-crossing> Phase 1/2
        if is_defender:
            # Pull back if taking heavy missile fire and no enemy melee crossing (<melee-threat-threshold>0.2)
            under_fire_pullback = u.is_under_fire and (en_crossing < CFG["def_cross_melee_threat"] * max(1.0, own_melee))
            if d <= CFG["open_inf_engage"]:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            elif under_fire_pullback:
                u.order_kind = "move"
                u.target_x = main_b["x"]
                u.target_y = sim.river_y + bank_sign * min(CFG["def_cross_max_retreat"], sim.river_half_w + 110.0)
                u.run = True
            else:
                block_y = sim.river_y + bank_sign * (sim.river_half_w + CFG["def_cross_standoff"])
                u.order_kind = "move" if math.hypot(u.x - main_b["x"], u.y - block_y) > 20.0 else "hold"
                u.target_x, u.target_y = main_b["x"], block_y
                u.run = False
        else:
            if d <= CFG["open_inf_engage"]:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            elif not push_phase2:
                # Phase 1: form up on near bank (-20m offset)
                stage_y = sim.river_y + bank_sign * (sim.river_half_w - CFG["cross_p1_melee_off"])
                u.order_kind = "move"
                u.target_x, u.target_y = main_b["x"], stage_y
                u.run = False
            else:
                # Phase 2: push across bridge to far bank (+70m) or enemy
                on_own_bank = (u.y - sim.river_y) * bank_sign > 0
                u.order_kind = "move"
                if on_own_bank:
                    u.target_x = main_b["x"]
                    u.target_y = sim.river_y - bank_sign * (sim.river_half_w + CFG["cross_p2_melee_off"])
                else:
                    u.target_x, u.target_y = tgt.x, tgt.y
                u.run = d < CFG["contact_dist"]


def _step_open_field_gta(sim: BattleSim, side: int, own: List[SimUnit], en: List[SimUnit], force_attack: bool = False):
    """<gta> Open Field: <battle-analyser>, 4-phase approach, missiles-in-front, <outflanking>, <melee-manager>."""
    inf_own = [u for u in own if u.role in ("line_inf", "spear_inf")]
    icx, icy = _centroid(inf_own if inf_own else own)
    ecx, ecy = _centroid(en)
    dx_line, dy_line = ecx - icx, ecy - icy
    dist_line = max(1e-3, math.hypot(dx_line, dy_line))
    fx, fy = dx_line / dist_line, dy_line / dist_line
    rx, ry = fy, -fx

    # 1. <battle-analyser>: decide attack-battlegroup vs defend-line from real strength ratios
    own_tot = _sum_power(own)
    en_tot = _sum_power(en)
    own_rng = _sum_power(own, lambda u: u.tpl.is_missile and u.ammo > 0)
    en_rng = _sum_power(en, lambda u: u.tpl.is_missile and u.ammo > 0)
    own_mel = _sum_power(own, lambda u: not u.tpl.is_missile)
    en_mel = _sum_power(en, lambda u: not u.tpl.is_missile)

    # Must attack if: forced, stronger overall (>= 0.8x), outgunned in ranged (en_rng > 2x own_rng),
    # or own total > 5x enemy ranged. Hold defensive line only when we have superior ranged & weaker melee.
    must_attack = (
        force_attack
        or own_tot >= CFG["ba_str_ratio"] * max(1.0, en_tot)
        or en_rng > CFG["fight_missile_ratio"] * max(1.0, own_rng)
        or own_tot > CFG["ba_fr_ranged_div"] * max(1.0, en_rng)
        or sim.t >= 240.0
    )
    prefer_defend = (not must_attack) and (own_rng > en_rng * 1.2) and (en_mel > own_mel)
    any_engaged = any(u.engaged for u in own)
    min_en_d = min(math.hypot(e.x - icx, e.y - icy) for e in en)
    # <enemy-position-buffer-distance>25 -> break defensive hold if enemy closes within contact/buffer
    hold_defensive_line = prefer_defend and (not any_engaged) and (min_en_d > CFG["contact_dist"] - CFG["ba_line_buffer"])

    # 2. <outflanking>: double envelopment if cav >= 4, else single-flank concentration if cav_ratio >= 0.66
    cav_own = [u for u in own if u.role == "shock_cav"]
    cav_en_pow = _sum_power(en, lambda e: e.tpl.is_cav)
    cav_own_pow = _sum_power(cav_own)
    double_envelop = len(cav_own) >= int(CFG["double_env_min_cav"])
    can_outflank = cav_own_pow >= CFG["outflank_str_ratio"] * max(1.0, cav_en_pow)
    # Single-flank side: pick the flank where our cavalry already has more mass
    flank_bias = 1.0 if sum((c.x - icx) * rx + (c.y - icy) * ry for c in cav_own) >= 0 else -1.0

    # Approach phase from <gta> distances (500m -> 230m -> 120m -> 20m)
    in_contact = dist_line <= CFG["contact_dist"] or any_engaged
    in_close_approach = (not in_contact) and (dist_line <= CFG["close_approach_dist"])

    # Second-line infantry reserve (~25% of melee infantry)
    n_res = max(1, len(inf_own) // 4) if len(inf_own) >= 4 else 0
    reserve_ids = {u.uid for u in inf_own[-n_res:]} if n_res else set()
    engaged_enemies = [e for e in en if e.engaged]
    retreat_d = CFG["retreat_dist"]
    cav_retreat_d = retreat_d * CFG["retreat_cav_mult"]

    for u in own:
        if u.engaged:
            continue
        near = _best_melee_target(u, en)
        if near is None:
            continue
        tgt, d = near

        if u.role == "artillery":
            m_tgt = _best_missile_target(u, en)
            u.order_kind = "hold"
            u.fire_at_will = True
            u.target_uid = m_tgt[0].uid if m_tgt else tgt.uid
            continue

        if u.role == "missile_foot":
            if u.ammo > 0:
                m_tgt = _best_missile_target(u, en)
                # <retreat-analyser>: pull back 40m behind infantry line when enemy closes within open_mis_engage (40m)
                if d < CFG["open_mis_engage"]:
                    u.order_kind = "move"
                    u.target_x = icx - fx * retreat_d
                    u.target_y = icy - fy * retreat_d
                    u.run = True
                elif m_tgt and (m_tgt[1] <= u.tpl.pri_range * 0.90 or hold_defensive_line):
                    u.order_kind = "hold"
                    u.fire_at_will = True
                    u.target_uid = m_tgt[0].uid
                elif not in_contact:
                    # ai_triple_missiles_in_front: screen 25m ahead of infantry during approach
                    u.order_kind = "move"
                    u.target_x = icx + fx * 25.0 + ((u.x - icx) * rx + (u.y - icy) * ry) * rx * 0.7
                    u.target_y = icy + fy * 25.0 + ((u.x - icx) * rx + (u.y - icy) * ry) * ry * 0.7
                    u.run = False
                else:
                    u.order_kind = "hold"
                    u.fire_at_will = True
                    u.target_uid = m_tgt[0].uid if m_tgt else tgt.uid
            elif d <= CFG["assault_dist"] + 10.0:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            else:
                u.order_kind = "move" if math.hypot(u.x - (icx - fx * retreat_d), u.y - (icy - fy * retreat_d)) > 15.0 else "hold"
                u.target_x, u.target_y = icx - fx * retreat_d, icy - fy * retreat_d
                u.run = False
            continue

        if u.role == "missile_cav":
            if u.ammo > 0:
                m_tgt = _best_missile_target(u, en)
                if d < CFG["open_mis_engage"]:
                    u.order_kind = "move"
                    u.target_x = u.x - (tgt.x - u.x) / max(1.0, d) * cav_retreat_d
                    u.target_y = u.y - (tgt.y - u.y) / max(1.0, d) * cav_retreat_d
                    u.run = True
                elif m_tgt and (m_tgt[1] <= u.tpl.pri_range * 0.85 or hold_defensive_line):
                    u.order_kind = "hold"
                    u.fire_at_will = True
                    u.target_uid = m_tgt[0].uid
                else:
                    u.order_kind = "move"
                    u.target_x = tgt.x - fx * (u.tpl.pri_range * 0.70)
                    u.target_y = tgt.y - fy * (u.tpl.pri_range * 0.70)
                    u.run = True
            elif d <= CFG["open_cav_engage"] * 0.5:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            else:
                u.order_kind = "move"
                u.target_x, u.target_y = icx - fx * retreat_d, icy - fy * retreat_d
                u.run = True
            continue

        if u.role == "general":
            if d < 30.0:
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            else:
                gx, gy = icx - fx * 60.0, icy - fy * 60.0
                u.order_kind = "hold" if (hold_defensive_line and math.hypot(u.x - gx, u.y - gy) <= 25.0) else "move"
                u.target_x, u.target_y = gx, gy
                u.run = dist_line > CFG["close_approach_dist"]
            continue

        if u.role == "shock_cav":
            if hold_defensive_line and d > CFG["open_cav_engage"]:
                side_sign = (1.0 if ((u.x - icx) * rx + (u.y - icy) * ry) >= 0 else -1.0) if double_envelop else flank_bias
                gx, gy = icx + rx * side_sign * 90.0 - fx * 15.0, icy + ry * side_sign * 90.0 - fy * 15.0
                u.order_kind = "move" if math.hypot(u.x - gx, u.y - gy) > 18.0 else "hold"
                u.target_x, u.target_y = gx, gy
                u.run = False
                continue
            if d > CFG["open_cav_engage"] and can_outflank:
                side_sign = (1.0 if ((u.x - ecx) * rx + (u.y - ecy) * ry) >= 0 else -1.0) if double_envelop else flank_bias
                u.order_kind = "move"
                u.target_x = ecx + rx * side_sign * 95.0 - fx * 15.0
                u.target_y = ecy + ry * side_sign * 95.0 - fy * 15.0
                u.run = True
            else:
                # <melee-manager> open/cavalry/max-engage-dist = 120m + <outflank-analyser> avoids phalanx/routers
                u.order_kind = "attack"
                u.target_uid = tgt.uid
                u.run = True
            continue

        if hold_defensive_line and d > CFG["open_inf_engage"] + 25.0:
            u.order_kind = "hold"
            continue

        # Second-line tactical reserve
        if u.uid in reserve_ids:
            if not any_engaged and d > CFG["open_inf_engage"] + 25.0:
                rx_pos, ry_pos = icx - fx * retreat_d, icy - fy * retreat_d
                u.order_kind = "move" if math.hypot(u.x - rx_pos, u.y - ry_pos) > 22.0 else "hold"
                u.target_x, u.target_y = rx_pos, ry_pos
                u.run = False
                continue
            elif engaged_enemies:
                near_eng = _best_melee_target(u, engaged_enemies)
                if near_eng:
                    u.order_kind = "attack"
                    u.target_uid = near_eng[0].uid
                    u.run = True
                    continue

        # <attack-brace> distance = 150m: spearmen brace if enemy cavalry is charging within 150m (and > assault-dist 20m)
        if u.role == "spear_inf" and d > CFG["assault_dist"]:
            cav_threat = any(
                e.tpl.is_cav and math.hypot(e.x - u.x, e.y - u.y) <= CFG["brace_dist"]
                and (e.order_kind == "attack" or math.hypot(e.x - u.x, e.y - u.y) <= 85.0)
                for e in en
            )
            if cav_threat:
                u.order_kind = "hold"
                continue

        # <melee-manager> open/infantry/max-engage-dist = 40m
        if d <= CFG["open_inf_engage"]:
            u.order_kind = "attack"
            u.target_uid = tgt.uid
            u.run = True
        else:
            u.order_kind = "move"
            u.target_x, u.target_y = tgt.x, tgt.y
            u.run = in_contact or (in_close_approach and en_rng > CFG["fight_missile_ratio"] * max(1.0, own_rng))


def step_vanilla_ai(sim: BattleSim, side: int):
    own = [u for u in sim.sides[side] if u.active]
    en = [u for u in sim.sides[1 - side] if u.active]
    if not own or not en:
        return

    if sim.battle_type in ("siege_assault", "siege_defense"):
        _step_siege_gta(sim, side, own, en)
        return

    if sim.battle_type == "river_crossing" and sim.bridges:
        _step_crossing_gta(sim, side, own, en)
        return

    _step_open_field_gta(sim, side, own, en)
