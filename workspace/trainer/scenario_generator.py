"""Randomized battle scenarios: random factions per side, standardized army templates + variance."""
import json
import random

FACTIONS = ["england", "france", "hre", "spain", "portugal", "scotland", "milan", "venice", "sicily",
            "denmark", "russia", "poland", "hungary", "byzantium", "turks", "egypt", "moors"]
WEATHER = ["clear", "cloudy", "rain", "storm", "fog", "snow"]
TIMES = [6, 9, 12, 15, 18]
# role-slot template (slot -> count); filled from the faction's EDU pool by class at battle build time
STANDARD = {"line_inf": 5, "spear_inf": 3, "missile_foot": 4, "shock_cav": 3, "missile_cav": 1, "general": 1}
EXTREME_P = 0.05  # small tail probability for lopsided armies (e.g. all cavalry)


def army(rng, budget=20):
    if rng.random() < EXTREME_P:
        focus = rng.choice(["shock_cav", "missile_foot", "spear_inf", "line_inf"])
        slots = {"general": 1, focus: budget - 1}
    else:
        slots = {k: max(0, v + rng.choice([-1, 0, 0, 1])) for k, v in STANDARD.items()}
        slots["general"] = 1
        if slots["line_inf"] + slots["spear_inf"] == 0:
            slots["line_inf"] = 2
    return slots


def make_scenario(battle_id, rng=None):
    rng = rng or random.Random(battle_id)
    f1 = rng.choice(FACTIONS)
    f2 = rng.choice([f for f in FACTIONS if f != f1])
    return {
        "battle_id": battle_id,
        "map_seed": rng.randrange(1 << 30),
        "weather": rng.choice(WEATHER),
        "hour": rng.choice(TIMES),
        "sides": [{"faction": f1, "army": army(rng)}, {"faction": f2, "army": army(rng)}],
    }


def write_scenario(path, battle_id, mode):
    sc = make_scenario(battle_id)
    sc["mode"] = mode  # vanilla_vs_vanilla (phase 1) | rl_vs_vanilla (phase 2) | rl_vs_frozen (phase 3)
    json.dump(sc, open(path, "w"), indent=1)
    return sc
