"""Parser for Medieval II Total War export_descr_unit.txt and descr_projectile.txt."""
from dataclasses import dataclass, field
import os
import re
from typing import Dict, List, Optional

SIM_DATA = os.environ.get("SIM_DATA", "/home/tomek/medieval2-docker/sim_data")
UNIT_SCALE = 2.5  # UI_VIDEO_UNIT_SCALE_HUGE (descr_unit_sizes.txt)

ROLES = ["line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general", "artillery"]

FACTIONS = [
    "england", "france", "hre", "spain", "portugal", "scotland", "milan", "venice", "sicily",
    "denmark", "russia", "poland", "hungary", "byzantium", "turks", "egypt", "moors",
]

# Faction culture fallback if a specific role has no direct match
CULTURE_GROUPS = {
    "england": "northern_european", "france": "northern_european", "hre": "northern_european",
    "scotland": "northern_european", "denmark": "northern_european",
    "spain": "southern_european", "portugal": "southern_european", "milan": "southern_european",
    "venice": "southern_european", "sicily": "southern_european",
    "russia": "eastern_european", "poland": "eastern_european", "hungary": "eastern_european",
    "byzantium": "greek",
    "turks": "middle_eastern", "egypt": "middle_eastern", "moors": "middle_eastern",
}


@dataclass
class ProjectileInfo:
    name: str
    accuracy_vs_units: float = 0.05
    affected_by_rain: bool = False
    body_piercing: bool = False
    area_effect: bool = False
    radius: float = 0.1


@dataclass
class UnitTemplate:
    name: str
    category: str = "infantry"       # infantry | cavalry | siege | handler
    uclass: str = "light"            # heavy | light | missile | spearmen
    soldiers_base: int = 40
    extras: int = 0
    mass: float = 1.0
    spacing_x: float = 1.2
    spacing_y: float = 1.5
    ranks: int = 4
    can_phalanx: bool = False
    can_schiltrom: bool = False
    can_wedge: bool = False
    hp: int = 1
    mount_hp: int = 0
    # Primary weapon
    pri_atk: int = 4
    pri_chg: int = 2
    pri_proj: str = "no"
    pri_range: float = 0.0
    pri_ammo: int = 0
    pri_type: str = "melee"          # melee | missile | thrown | siege_missile
    pri_delay: float = 1.0           # seconds (from 1/10s in EDU)
    pri_ap: bool = False
    pri_bp: bool = False
    pri_spear: bool = False
    pri_long_pike: bool = False
    pri_spear_bonus: int = 0
    pri_area: bool = False
    # Secondary weapon (melee sidearm for missile units)
    sec_atk: int = 0
    sec_chg: int = 0
    sec_ap: bool = False
    sec_spear: bool = False
    sec_spear_bonus: int = 0
    # Armour & defence
    armour: int = 0
    def_skill: int = 0
    shield: int = 0
    # Mental & movement
    morale: int = 5
    discipline: str = "normal"       # low | normal | disciplined | impetuous
    training: str = "trained"        # untrained | trained | highly_trained
    lock_morale: bool = False
    charge_dist: float = 30.0
    fire_delay: float = 0.0
    move_speed_mod: float = 1.0
    general_unit: bool = False
    is_peasant: bool = False
    frighten_foot: bool = False
    frighten_mounted: bool = False
    can_skirmish: bool = False
    attributes: List[str] = field(default_factory=list)
    ownership: List[str] = field(default_factory=list)
    mount_effect: Dict[str, int] = field(default_factory=dict)

    @property
    def num_soldiers(self) -> int:
        if self.category == "siege":
            return max(2, int(round(self.soldiers_base * UNIT_SCALE)))
        return max(10, int(round(self.soldiers_base * UNIT_SCALE)))

    @property
    def is_cav(self) -> bool:
        return self.category == "cavalry"

    @property
    def is_missile(self) -> bool:
        return self.pri_type in ("missile", "thrown", "siege_missile") and self.pri_range > 0

    @property
    def melee_atk(self) -> int:
        return self.sec_atk if self.is_missile and self.sec_atk > 0 else self.pri_atk

    @property
    def melee_chg(self) -> int:
        return self.sec_chg if self.is_missile and self.sec_atk > 0 else self.pri_chg

    @property
    def melee_ap(self) -> bool:
        return self.sec_ap if self.is_missile and self.sec_atk > 0 else self.pri_ap

    @property
    def melee_spear_bonus(self) -> int:
        if self.is_missile and self.sec_atk > 0:
            return self.sec_spear_bonus
        return self.pri_spear_bonus

    @property
    def ai_value_per_soldier(self) -> float:
        atk = self.melee_atk + (self.pri_atk * 0.6 if self.is_missile else 0.0)
        tot_def = self.armour + self.def_skill + self.shield * 0.7
        return max(2.0, (atk * 0.7 + tot_def * 0.45 + self.morale * 0.35))

    @property
    def role(self) -> str:
        if self.category == "siege":
            return "artillery"
        if self.general_unit:
            return "general"
        if self.is_cav:
            return "missile_cav" if self.is_missile else "shock_cav"
        if self.is_missile:
            return "missile_foot"
        if self.uclass == "spearmen" or self.pri_spear or self.pri_long_pike:
            return "spear_inf"
        return "line_inf"


def parse_projectiles(path: Optional[str] = None) -> Dict[str, ProjectileInfo]:
    path = path or os.path.join(SIM_DATA, "descr_projectile.txt")
    projs: Dict[str, ProjectileInfo] = {}
    if not os.path.exists(path):
        return projs
    cur: Optional[ProjectileInfo] = None
    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0].lower()
            if key == "projectile" and len(parts) >= 2:
                cur = ProjectileInfo(name=parts[1].lower())
                projs[cur.name] = cur
            elif cur is not None:
                if key == "accuracy_vs_units" and len(parts) >= 2:
                    cur.accuracy_vs_units = float(parts[1])
                elif key == "affected_by_rain":
                    cur.affected_by_rain = True
                elif key == "body_piercing":
                    cur.body_piercing = True
                elif key == "area_effect":
                    cur.area_effect = True
                elif key == "radius" and len(parts) >= 2:
                    cur.radius = float(parts[1])
    return projs


def _parse_attrs(vals: List[str]):
    ap = "ap" in vals
    bp = "bp" in vals
    spear = "spear" in vals or "light_spear" in vals
    long_pike = "long_pike" in vals
    area = "area" in vals
    spear_bonus = 0
    for v in vals:
        if v.startswith("spear_bonus_"):
            try:
                spear_bonus = max(spear_bonus, int(v.split("_")[-1]))
            except ValueError:
                pass
    if (spear or long_pike) and spear_bonus == 0:
        spear_bonus = 4
    return ap, bp, spear, long_pike, area, spear_bonus


def parse_edu(path: Optional[str] = None) -> Dict[str, UnitTemplate]:
    path = path or os.path.join(SIM_DATA, "export_descr_unit.txt")
    units: Dict[str, UnitTemplate] = {}
    cur: Optional[UnitTemplate] = None
    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            m = re.match(r"^(\S+)\s+(.*)$", line)
            if not m:
                continue
            key, rest = m.group(1).lower(), m.group(2).strip()
            vals = [v.strip() for v in rest.split(",")]
            if key == "type":
                cur = UnitTemplate(name=rest)
                units[cur.name] = cur
            elif cur is None:
                continue
            elif key == "category":
                cur.category = vals[0].lower()
            elif key == "class":
                cur.uclass = vals[0].lower()
            elif key == "soldier" and len(vals) >= 2:
                try:
                    cur.soldiers_base = int(vals[1])
                    if len(vals) >= 3:
                        cur.extras = int(vals[2])
                    if len(vals) >= 4:
                        cur.mass = float(vals[3])
                except ValueError:
                    pass
            elif key == "formation" and len(vals) >= 5:
                try:
                    cur.spacing_x = float(vals[0])
                    cur.spacing_y = float(vals[1])
                    cur.ranks = max(1, int(vals[4]))
                    forms = " ".join(vals[5:]).lower()
                    cur.can_phalanx = "phalanx" in forms
                    cur.can_schiltrom = "schiltrom" in forms
                    cur.can_wedge = "wedge" in forms
                except ValueError:
                    pass
            elif key == "attributes":
                cur.attributes = [v.lower() for v in vals]
                cur.general_unit = "general_unit" in cur.attributes
                cur.is_peasant = "is_peasant" in cur.attributes
                cur.frighten_foot = "frighten_foot" in cur.attributes
                cur.frighten_mounted = "frighten_mounted" in cur.attributes
            elif key == "stat_health" and len(vals) >= 1:
                cur.hp = max(1, int(vals[0]))
                if len(vals) >= 2:
                    cur.mount_hp = int(vals[1])
            elif key == "stat_pri" and len(vals) >= 7:
                try:
                    cur.pri_atk = int(vals[0])
                    cur.pri_chg = int(vals[1])
                    cur.pri_proj = vals[2].lower()
                    cur.pri_range = float(vals[3])
                    cur.pri_ammo = int(vals[4])
                    cur.pri_type = vals[5].lower()
                    for tok in vals[9:]:
                        try:
                            cur.pri_delay = max(0.5, float(tok) * 0.1)
                            break
                        except ValueError:
                            pass
                except ValueError:
                    pass
            elif key == "stat_pri_attr":
                ap, bp, sp, lp, ar, sb = _parse_attrs([v.lower() for v in vals])
                cur.pri_ap, cur.pri_bp, cur.pri_spear, cur.pri_long_pike, cur.pri_area, cur.pri_spear_bonus = ap, bp, sp, lp, ar, sb
            elif key == "stat_sec" and len(vals) >= 2 and vals[0].lower() != "no":
                try:
                    cur.sec_atk = int(vals[0])
                    cur.sec_chg = int(vals[1])
                except ValueError:
                    pass
            elif key == "stat_sec_attr":
                ap, _, sp, _, _, sb = _parse_attrs([v.lower() for v in vals])
                cur.sec_ap, cur.sec_spear, cur.sec_spear_bonus = ap, sp, sb
            elif key == "stat_pri_armour" and len(vals) >= 3:
                try:
                    cur.armour = int(vals[0])
                    cur.def_skill = int(vals[1])
                    cur.shield = int(vals[2])
                except ValueError:
                    pass
            elif key == "stat_mental" and len(vals) >= 3:
                try:
                    cur.morale = int(vals[0])
                    cur.discipline = vals[1].lower()
                    cur.training = vals[2].lower()
                    cur.lock_morale = "lock_morale" in [v.lower() for v in vals]
                except ValueError:
                    pass
            elif key == "stat_charge_dist":
                try:
                    cur.charge_dist = float(vals[0])
                except ValueError:
                    pass
            elif key == "stat_fire_delay":
                try:
                    cur.fire_delay = max(0.0, float(vals[0]) * 0.1)
                except ValueError:
                    pass
            elif key == "move_speed_mod":
                try:
                    cur.move_speed_mod = float(vals[0])
                except ValueError:
                    pass
            elif key == "ownership":
                cur.ownership = [v.lower() for v in vals]
    return units


def build_faction_pools(units: Dict[str, UnitTemplate]) -> Dict[str, Dict[str, List[UnitTemplate]]]:
    import copy
    pools: Dict[str, Dict[str, List[UnitTemplate]]] = {f: {r: [] for r in ROLES} for f in FACTIONS}
    for u in units.values():
        if u.category not in ("infantry", "cavalry", "siege"):
            continue
        if "no_custom" in u.attributes and not u.general_unit:
            continue
        if "mercenary_unit" in u.attributes:
            continue
        r = u.role
        # Regular recruitable heavy cavalry that also have general_unit should populate shock_cav/missile_cav too
        non_gen_copy = None
        if u.general_unit and "no_custom" not in u.attributes and u.is_cav:
            non_gen_copy = copy.copy(u)
            non_gen_copy.general_unit = False
        # Ownership can contain faction names OR culture names
        for f in FACTIONS:
            cg = CULTURE_GROUPS[f]
            if f in u.ownership or cg in u.ownership or "all" in u.ownership:
                pools[f][r].append(u)
                if non_gen_copy is not None:
                    pools[f][non_gen_copy.role].append(non_gen_copy)
    # Fallback for any empty role slot using culture peers or global standard
    for f in FACTIONS:
        for r in ROLES:
            if not pools[f][r]:
                cg = CULTURE_GROUPS[f]
                for f2 in FACTIONS:
                    if CULTURE_GROUPS[f2] == cg and pools[f2][r]:
                        pools[f][r].extend(pools[f2][r])
                if not pools[f][r]:
                    for f2 in FACTIONS:
                        if pools[f2][r]:
                            pools[f][r].extend(pools[f2][r])
                            break
    return pools


if __name__ == "__main__":
    projs = parse_projectiles()
    units = parse_edu()
    pools = build_faction_pools(units)
    print(f"Loaded {len(units)} units, {len(projs)} projectiles")
    for f in ["england", "moors", "byzantium", "turks"]:
        summary = {r: len(pools[f][r]) for r in ROLES}
        print(f"  {f}: {summary}")
