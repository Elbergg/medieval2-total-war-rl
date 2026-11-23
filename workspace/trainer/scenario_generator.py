"""Randomized battle scenarios: open field, river crossings, and sieges (with battering rams, ladders & artillery)."""
import json
import random

FACTIONS = [
    "england", "france", "hre", "spain", "portugal", "scotland", "milan", "venice", "sicily",
    "denmark", "russia", "poland", "hungary", "byzantium", "turks", "egypt", "moors",
]
WEATHER = ["clear", "cloudy", "rain", "storm", "fog", "snow"]
TIMES = [6, 9, 12, 15, 18]

ARCHETYPES = {
    "balanced": {"line_inf": 5, "spear_inf": 3, "missile_foot": 4, "shock_cav": 3, "missile_cav": 1, "general": 1, "artillery": 0},
    "heavy_vanguard": {"line_inf": 7, "spear_inf": 2, "missile_foot": 2, "shock_cav": 4, "missile_cav": 0, "general": 1, "artillery": 0},
    "cavalry_host": {"line_inf": 2, "spear_inf": 2, "missile_foot": 2, "shock_cav": 5, "missile_cav": 4, "general": 1, "artillery": 0},
    "archer_pike_wall": {"line_inf": 2, "spear_inf": 5, "missile_foot": 6, "shock_cav": 1, "missile_cav": 0, "general": 1, "artillery": 2},
    "skirmish_raiders": {"line_inf": 3, "spear_inf": 2, "missile_foot": 4, "shock_cav": 2, "missile_cav": 3, "general": 1, "artillery": 0},
    "grand_guard": {"line_inf": 6, "spear_inf": 4, "missile_foot": 4, "shock_cav": 3, "missile_cav": 1, "general": 1, "artillery": 1},
    "vanguard_clash": {"line_inf": 4, "spear_inf": 2, "missile_foot": 2, "shock_cav": 3, "missile_cav": 1, "general": 1, "artillery": 0},
    "siege_storm": {"line_inf": 7, "spear_inf": 3, "missile_foot": 4, "shock_cav": 1, "missile_cav": 0, "general": 1, "artillery": 2},
}

DEPLOYMENTS = [
    "pitched_line",
    "oblique_left",
    "oblique_right",
    "split_wings",
    "flanking_vanguard",
    "deep_column",
]


def army(rng, arch_name=None, is_siege_attacker=False):
    if is_siege_attacker and arch_name is None:
        arch_name = rng.choice(["siege_storm", "grand_guard", "heavy_vanguard", "archer_pike_wall"])
    arch_name = arch_name or rng.choice(list(ARCHETYPES.keys()))
    base = ARCHETYPES[arch_name]
    slots = {k: max(0, v + rng.choice([-1, 0, 0, 1])) for k, v in base.items()}
    slots["general"] = 1
    if is_siege_attacker:
        slots["line_inf"] = max(4, slots.get("line_inf", 0))
        slots["artillery"] = max(1, slots.get("artillery", 0))
    if slots.get("line_inf", 0) + slots.get("spear_inf", 0) == 0:
        slots["line_inf"] = 2
    return {k: v for k, v in slots.items() if v > 0}, arch_name


def make_deploy_cfg(rng, battle_type="open_field", is_defender=False):
    if battle_type in ("siege_assault", "siege_defense"):
        return {
            "style": "siege_garrison" if is_defender else "siege_lines",
            "dx": round(rng.uniform(-40.0, 40.0), 1),
            "y_dist": 140.0 if is_defender else 170.0,
            "vanguard_wing": rng.choice([-1, 1]),
        }
    if battle_type == "river_crossing":
        return {
            "style": rng.choice(["pitched_line", "split_wings", "deep_column"]),
            "dx": round(rng.uniform(-70.0, 70.0), 1),
            "y_dist": round(rng.uniform(115.0, 165.0), 1),
            "vanguard_wing": rng.choice([-1, 1]),
        }
    style = rng.choice(DEPLOYMENTS)
    return {
        "style": style,
        "dx": round(rng.uniform(-110.0, 110.0), 1),
        "y_dist": round(rng.uniform(135.0, 225.0), 1),
        "vanguard_wing": rng.choice([-1, 1]),
    }


def make_terrain_cfg(rng):
    r = rng.random()
    if r < 0.55:
        btype = "open_field"
    elif r < 0.70:
        btype = "river_crossing"
    elif r < 0.90:
        btype = "siege_assault"   # Side 0 (RL) attacks fortress with battering ram, siege towers/ladders & artillery
    else:
        btype = "siege_defense"   # Side 0 (RL) defends fortress against enemy battering ram, ladders & artillery

    cfg = {"battle_type": btype}
    if btype == "river_crossing":
        b1 = round(rng.uniform(-90.0, 40.0), 1)
        bridges = [{"x": b1, "width": 28.0, "kind": "bridge"}]
        if rng.random() < 0.65:
            b2 = round(b1 + rng.choice([-1, 1]) * rng.uniform(140.0, 210.0), 1)
            b2 = max(-280.0, min(280.0, b2))
            bridges.append({"x": b2, "width": 34.0, "kind": "ford"})
        cfg["river_y"] = 0.0
        cfg["river_half_w"] = 22.0
        cfg["bridges"] = bridges
    elif btype in ("siege_assault", "siege_defense"):
        def_side = 1 if btype == "siege_assault" else 0
        sign = 1.0 if def_side == 1 else -1.0
        cfg["defender_side"] = def_side
        cfg["wall_y"] = 75.0 * sign
        cfg["gate_x"] = round(rng.uniform(-35.0, 35.0), 1)
        cfg["plaza_x"] = 0.0
        cfg["plaza_y"] = 215.0 * sign
        cfg["plaza_r"] = 45.0
    return cfg


def make_scenario(battle_id, rng=None):
    rng = rng or random.Random(battle_id)
    f1 = rng.choice(FACTIONS)
    f2 = rng.choice([f for f in FACTIONS if f != f1])
    terrain = make_terrain_cfg(rng)
    btype = terrain["battle_type"]
    # 60% matched/symmetric archetypes (like M2TW Quick Battles) so cav/missile balance centers near 0
    if btype not in ("siege_assault", "siege_defense") and rng.random() < 0.60:
        shared_arch = rng.choice(["balanced", "grand_guard", "heavy_vanguard", "vanguard_clash", "skirmish_raiders"])
        a1, arch1 = army(rng, arch_name=shared_arch, is_siege_attacker=False)
        a2, arch2 = army(rng, arch_name=shared_arch, is_siege_attacker=False)
    else:
        a1, arch1 = army(rng, is_siege_attacker=(btype == "siege_assault"))
        a2, arch2 = army(rng, is_siege_attacker=(btype == "siege_defense"))
    return {
        "battle_id": battle_id,
        "map_seed": rng.randrange(1 << 30),
        "weather": rng.choice(WEATHER),
        "hour": rng.choice(TIMES),
        "terrain": terrain,
        "sides": [
            {"faction": f1, "army": a1, "archetype": arch1, "deploy": make_deploy_cfg(rng, btype, is_defender=(btype == "siege_defense"))},
            {"faction": f2, "army": a2, "archetype": arch2, "deploy": make_deploy_cfg(rng, btype, is_defender=(btype == "siege_assault"))},
        ],
    }


def write_scenario(path, battle_id, mode):
    sc = make_scenario(battle_id)
    sc["mode"] = mode
    json.dump(sc, open(path, "w"), indent=1)
    return sc
