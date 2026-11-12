"""Shared spec between the Lua plugin and the Python trainer. Keep in sync with rl_battle_ai.lua."""
import math

NUM_ACTIONS = 13
NUM_GLOBAL = 20
NUM_LOCAL = 26
GRID = 8
GRID_CH = 6  # friendly, enemy, hold, attacking, firing, fleeing
NUM_GRID = GRID * GRID * GRID_CH  # 384
NUM_FEATURES = NUM_GLOBAL + NUM_LOCAL + NUM_GRID  # 430
ROLES = ["line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general"]
NUM_ROLES = len(ROLES)

# Movement offsets (x_right, y_fwd) in metres, army frame. a8 = hold, a9..a12 = attack front/left/right/rear
_D = 45.0
MOVE_OFFSETS = {
    0: (0.0, 60.0), 1: (-_D, _D), 2: (_D, _D), 3: (-60.0, 0.0), 4: (60.0, 0.0),
    5: (-_D, -_D), 6: (_D, -_D), 7: (0.0, -60.0), 8: (0.0, 0.0),
}
ATTACK_ACTIONS = {9: "front", 10: "left", 11: "right", 12: "rear"}


def imitation_reward(action: int, disp, engaged_quadrant):
    """Way-2 reward: penalise divergence from what Vanilla physically did in the 3s window.

    disp: (dx, dy) actual displacement in army frame (metres) or None.
    engaged_quadrant: 0..3 (front,left,right,rear) if vanilla was attacking, else None.
    """
    if engaged_quadrant is not None:
        return 0.0 if action == 9 + engaged_quadrant else -1.0
    if action in ATTACK_ACTIONS or disp is None:
        return -1.0
    ox, oy = MOVE_OFFSETS[action]
    d = math.hypot(ox - disp[0], oy - disp[1]) / 60.0
    return -min(d, 2.0) ** 2
