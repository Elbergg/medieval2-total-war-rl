"""Shared spec between the Lua plugin and the Python trainer. Keep in sync with rl_battle_ai.lua."""
import math

NUM_ACTIONS = 13
NUM_CMD_ACTIONS = 10
NUM_GLOBAL = 50  # 20 army/map + 10 global order + 10 detachment order + 10 lead line_inf order (unnamed orders 0..9)
NUM_LOCAL = 28   # 26 unit/local + 2 (dx, dy) offset from friendly infantry centroid
GRID = 8
GRID_CH = 6  # friendly, enemy, hold, attacking, firing, fleeing
NUM_GRID = GRID * GRID * GRID_CH  # 384
NUM_FEATURES = NUM_GLOBAL + NUM_LOCAL + NUM_GRID  # 462
UNIT_ROLES = ["line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general", "artillery"]
ROLES = UNIT_ROLES
NUM_ROLES = len(ROLES)
UNIT_MLP_HIDDEN = 32

# Commander RNN: 20 global + 72 (6x6x2 map) + 160 (40 units x [x,y,hp,role]) + 41 (step/squad + 3x10 recursive orders) = 293
CMD_RNN_IN = 293
CMD_RNN_HIDDEN = 48
CMD_RNN_OUT = NUM_CMD_ACTIONS  # 10

# Movement offsets (x_right, y_fwd) in metres, army frame. a8 = hold, a9..a12 = attack front/left/right/rear
_D = 45.0
MOVE_OFFSETS = {
    0: (0.0, 60.0), 1: (-_D, _D), 2: (_D, _D), 3: (-60.0, 0.0), 4: (60.0, 0.0),
    5: (-_D, -_D), 6: (_D, -_D), 7: (0.0, -60.0), 8: (0.0, 0.0),
}
ATTACK_ACTIONS = {9: "front", 10: "left", 11: "right", 12: "rear"}


def imitation_reward(action: int, disp, engaged_quadrant):
    """Imitation reward from Vanilla's physical displacement and melee engagement.

    disp: (dx, dy) actual displacement in army frame (metres) or None.
    engaged_quadrant: 0..3 (front,left,right,rear) if vanilla was attacking, else None.
    """
    if engaged_quadrant is not None:
        if action == 9 + engaged_quadrant:
            return 0.5
        if action in ATTACK_ACTIONS:
            return -0.2
        return -1.0
    if disp is None:
        return -1.0
    dx, dy = disp[0], disp[1]
    dist = math.hypot(dx, dy)
    if dist < 4.0:
        return 0.0 if action == 8 else -0.8
    if action == 8:
        return -1.0
    if action in ATTACK_ACTIONS:
        # Charging forward toward enemy before contact aligns with attack_front (9)
        if action == 9 and dy > 8.0 and abs(dx) < dy:
            return -0.1
        return -1.0
    ox, oy = MOVE_OFFSETS[action]
    od = math.hypot(ox, oy)
    cos_sim = (ox * dx + oy * dy) / max(1e-6, od * dist)
    return cos_sim - 1.0
