"""Towell & Shavlik (1994) KBANN Compiler + Stacked Zero-Initialized Top Layer.

Architecture:
  1. Per-role Unit Network (2-Layer Stacked RNN):
     - Layer 1 (KBANN Rule Layer, 16 neurons):
         h_kb = tanh(W_kb @ x + b_kb)
       Compiled deterministically from symbolic AND / OR / NOT / Threshold rules
       using Towell & Shavlik's (1994) exact omega=4.0 formulas.
     - Layer 2 (Learned Recurrent Layer ON TOP of Layer 1, 32 neurons):
         h_top = tanh(W_top_kb @ h_kb + W_top_x @ x[:78] + Wh @ h_top_prev + b_top)
     - Output Logits (13 actions):
         logits = (W_kb_out @ h_kb + b_kb_out) + W2 @ h_top
       where W2 is zero-initialized (0.0) so at Battle #1 the policy is 100% pure
       Towell-Shavlik KBANN, and Layer 2 learns higher-order combinations on top of h_kb.
       Layer 1 (W_kb, b_kb, W_kb_out) remains fully differentiable and can be unfrozen
       at any point via kbann_lr_scale > 0.

  2. Commander Network (2-Layer Stacked RNN):
     - Layer 1 (KBANN Grand-Tactics Layer, 10 neurons):
         h_kb = tanh(W_kb @ x + b_kb)
     - Layer 2 (Learned Recurrent Layer ON TOP of Layer 1, 48 neurons):
         h_top = tanh(W_top_kb @ h_kb + Wx @ x + Wh @ h_top_prev + bh)
     - Output Logits (10 orders):
         logits = (W_kb_out @ h_kb + b_kb_out) + Wo @ h_top   (Wo initialized to 0.0)
"""
import numpy as np
from spec import (
    CMD_RNN_HIDDEN, CMD_RNN_IN, CMD_RNN_OUT,
    NUM_ACTIONS, NUM_FEATURES, NUM_GLOBAL, NUM_LOCAL, NUM_ROLES, UNIT_MLP_HIDDEN,
)

UNIT_KB_HIDDEN = 24
UNIT_CTX_IN = NUM_GLOBAL + NUM_LOCAL  # 78 (50 global + 28 local context features)
CMD_KB_HIDDEN = 19
OMEGA = 4.0  # Towell & Shavlik (1994) standard sharpness constant


def _grid_idx(r, c, ch):
    """Return 0-based feature index for 8x8x6 egocentric grid cell (row r, col c, channel ch)."""
    return 78 + (r * 8 + c) * 6 + ch


def _emit_outputs(W_out, b_out, neuron, outputs):
    """Attach rule neuron to output logits with exact tanh(-1) centering:
    when h_kb[neuron] == -1 (inactive), W_out[a, neuron]*(-1) + b_out[a] == 0;
    when h_kb[neuron] == +1 (active), contribution is +2 * wt.
    """
    for act, wt in outputs:
        W_out[act, neuron] += wt
        b_out[act] += wt


def compile_and(W, b, W_out, b_out, neuron, pos_indices, neg_indices, outputs, omega=OMEGA):
    """Towell & Shavlik (1994) exact AND rule compiler:
      - Positive antecedents: +omega
      - Negated (NOT) antecedents: -omega
      - Bias: -(P - 0.5) * omega, where P = len(pos_indices)
    """
    P = len(pos_indices)
    for idx in pos_indices:
        W[neuron, idx] += omega
    for idx in neg_indices:
        W[neuron, idx] -= omega
    b[neuron] = -(max(1, P) - 0.5) * omega
    _emit_outputs(W_out, b_out, neuron, outputs)


def compile_or(W, b, W_out, b_out, neuron, or_indices, neg_indices, outputs, omega=OMEGA):
    """Towell & Shavlik (1994) exact OR rule compiler with hard NOT blockers:
      - Disjunctive antecedents: +omega (any >=1 active gives >= +omega)
      - Negated (NOT) blockers: -len(or_indices) * omega (overrides even if all OR inputs are active)
      - Bias: -0.5 * omega
    """
    K = max(1, len(or_indices))
    for idx in or_indices:
        W[neuron, idx] += omega
    for idx in neg_indices:
        W[neuron, idx] -= K * omega
    b[neuron] = -0.5 * omega
    _emit_outputs(W_out, b_out, neuron, outputs)


def compile_threshold(W, b, W_out, b_out, neuron, feat_idx, threshold, direction, neg_indices, outputs, width=0.25, omega=OMEGA):
    """Towell & Shavlik continuous threshold compiler for (x[feat_idx] > threshold) or (x[feat_idx] < threshold):
      - direction = +1 for x > threshold, -1 for x < threshold
      - Crosses 0 at x[feat_idx] == threshold with slope = direction * (omega / width)
      - Negated blockers receive -2 * omega
    """
    slope = direction * (omega / max(1e-3, width))
    W[neuron, feat_idx] += slope
    b[neuron] = -slope * threshold
    for idx in neg_indices:
        W[neuron, idx] -= 2.0 * omega
    _emit_outputs(W_out, b_out, neuron, outputs)


def compile_linear(W, b, W_out, b_out, neuron, terms, bias, outputs):
    """Weighted-sum rule: active when sum(w_i * x_i) + bias > 0 (used for conjunctions of continuous features)."""
    for idx, w in terms:
        W[neuron, idx] += w
    b[neuron] = bias
    _emit_outputs(W_out, b_out, neuron, outputs)


def build_kbann_unit_rnns(seed=123):
    """Build 7 per-role 2-layer stacked Unit RNNs (Layer 1: 16 Towell-Shavlik KBANN rules -> Layer 2: 32 learned RNN on top)."""
    rng = np.random.default_rng(seed)
    roles = []
    for ri in range(NUM_ROLES):
        # Layer 1: KBANN symbolic rule layer (16 neurons over 462 features)
        # Small N(0, 0.005^2) background per Towell & Shavlik so rules can adapt when unfrozen
        W_kb = (rng.standard_normal((UNIT_KB_HIDDEN, NUM_FEATURES)) * 0.005).astype(np.float64)
        b_kb = np.zeros(UNIT_KB_HIDDEN, dtype=np.float64)
        W_kb_out = np.zeros((NUM_ACTIONS, UNIT_KB_HIDDEN), dtype=np.float64)
        b_kb_out = np.zeros(NUM_ACTIONS, dtype=np.float64)

        is_cav = ri in (3, 4, 5)

        # Rule 0 (AND): Default role doctrine when alive/active (x[0]=1), NOT engaged (x[72]=0), and NO rear enemy (r=0, c=0..7)
        rear_row0 = [_grid_idx(0, c, 1) for c in range(8)]
        if ri in (2, 6):  # missile_foot, artillery
            r0_out = [(9, 1.05), (8, 0.45), (5, -0.50), (6, -0.50), (7, -0.55), (12, -0.50)]
        elif ri == 4:     # missile_cav
            r0_out = [(9, 0.90), (1, 0.35), (2, 0.35), (7, -0.45), (12, -0.45)]
        elif ri == 3:     # shock_cav
            r0_out = [(9, 0.95), (1, 0.30), (2, 0.30), (5, -0.55), (6, -0.55), (7, -0.65), (12, -0.50)]
        else:             # line_inf, spear_inf, general
            r0_out = [(9, 1.05), (0, 0.30), (5, -0.55), (6, -0.55), (7, -0.65), (12, -0.50)]
        compile_and(W_kb, b_kb, W_kb_out, b_kb_out, 0, pos_indices=[0], neg_indices=[72] + rear_row0, outputs=r0_out)

        # Rule 1 (OR): Enemy in front grid (r=4..7, c=2..5) & NOT engaged (x[72]=0) -> attack_front (9)
        front_cells = [_grid_idx(r, c, 1) for r in range(4, 8) for c in range(2, 6)]
        compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 1, or_indices=front_cells, neg_indices=[72], outputs=[(9, 0.90)])

        # Rule 2 (OR): Enemy on left flank (r=2..7, c=0..2) & NOT engaged -> attack_left (10)
        left_cells = [_grid_idx(r, c, 1) for r in range(2, 8) for c in range(0, 3)]
        compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 2, or_indices=left_cells, neg_indices=[72], outputs=[(10, 0.75)])

        # Rule 3 (OR): Enemy on right flank (r=2..7, c=5..7) & NOT engaged -> attack_right (11)
        right_cells = [_grid_idx(r, c, 1) for r in range(2, 8) for c in range(5, 8)]
        compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 3, or_indices=right_cells, neg_indices=[72], outputs=[(11, 0.75)])

        # Rule 4 (OR): Enemy in rear (r=0..2, c=0..7) & NOT engaged -> attack_rear (12)
        rear_cells = [_grid_idx(r, c, 1) for r in range(0, 3) for c in range(0, 8)]
        compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 4, or_indices=rear_cells, neg_indices=[72], outputs=[(12, 0.75)])

        # Rule 5 (Threshold): Far right of infantry centroid (x[76] > +0.75) & NOT engaged -> angle forward-left (1, 3)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 5, feat_idx=76, threshold=0.75, direction=+1.0, neg_indices=[72], outputs=[(1, 0.65), (3, 0.45)])

        # Rule 6 (Threshold): Far left of infantry centroid (x[76] < -0.75) & NOT engaged -> angle forward-right (2, 4)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 6, feat_idx=76, threshold=-0.75, direction=-1.0, neg_indices=[72], outputs=[(2, 0.65), (4, 0.45)])

        # Rule 7 (Threshold): Lagging behind infantry centroid (x[77] < -0.45) & NOT engaged -> catch up forward (0)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 7, feat_idx=77, threshold=-0.45, direction=-1.0, neg_indices=[72], outputs=[(0, 1.25), (9, -0.50)])

        # Rule 8: Role-specific skirmish retreat / melee lock
        if ri == 2:  # missile_foot: enemy close in front (r=3..4, c=2..5) & NOT engaged -> retreat (7)
            close_front = [_grid_idx(r, c, 1) for r in (3, 4) for c in range(2, 6)]
            compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 8, or_indices=close_front, neg_indices=[72], outputs=[(7, 0.75), (9, 0.25)])
        elif ri == 4:  # missile_cav: enemy in r=3..5, c=1..6 & NOT engaged -> kite (5, 6)
            kite_cells = [_grid_idx(r, c, 1) for r in range(3, 6) for c in range(1, 7)]
            compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 8, or_indices=kite_cells, neg_indices=[72], outputs=[(5, 0.60), (6, 0.60)])
        elif ri == 6:  # artillery: hold & fire when not engaged
            compile_and(W_kb, b_kb, W_kb_out, b_kb_out, 8, pos_indices=[0], neg_indices=[72], outputs=[(8, 0.45), (9, 0.65)])
        else:          # melee units: stay locked in melee when engaged (x[72]=1)
            compile_and(W_kb, b_kb, W_kb_out, b_kb_out, 8, pos_indices=[72], neg_indices=[], outputs=[(9, 0.80)])

        # Rule 9: Over-extended ahead of infantry line (x[77] > +0.65) or cavalry wing positioning
        if is_cav and ri == 3:
            compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 9, feat_idx=76, threshold=0.35, direction=+1.0, neg_indices=[72], outputs=[(2, 0.50), (11, 0.40)])
        elif is_cav and ri == 5:
            compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 9, feat_idx=68, threshold=0.50, direction=+1.0, neg_indices=[72], outputs=[(0, 0.50), (9, 0.40)])
        else:
            compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 9, feat_idx=77, threshold=0.65, direction=+1.0, neg_indices=[72], outputs=[(7, 1.25), (9, -0.75)])

        # Rules 10..15 (Threshold): Coupling to Commander detachment order x[30..35] (scaled to 3.0 when active)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 10, feat_idx=30, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(8, 0.10), (9, 0.15)], width=0.8)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 11, feat_idx=31, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(0, 0.15), (9, 0.25)], width=0.8)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 12, feat_idx=32, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(9, 0.30), (10, 0.15), (11, 0.15)], width=0.8)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 13, feat_idx=33, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(1, 0.20), (10, 0.25)], width=0.8)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 14, feat_idx=34, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(2, 0.20), (11, 0.25)], width=0.8)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 15, feat_idx=35, threshold=1.5, direction=+1.0, neg_indices=[], outputs=[(5, 0.15), (6, 0.15), (7, 0.20)], width=0.8)

        # ---- Rules 16..23: additional vanilla Layer-2 tactics (march, standoff, flank approach, siege staging/storm/defense) ----
        enemy_all = [_grid_idx(r, c, 1) for r in range(8) for c in range(8)]
        # Rule 16 (AND): no enemy within grid & NOT engaged -> march to contact (artillery holds)
        compile_and(W_kb, b_kb, W_kb_out, b_kb_out, 16, pos_indices=[0], neg_indices=[72] + enemy_all,
                    outputs=[(8, 0.5)] if ri == 6 else [(0, 0.8), (8, -0.3)])
        # Rule 17: melee/cav: enemy within ~50m ahead -> charge; ranged/artillery: enemy 25-100m ahead -> hold & fire
        if ri in (2, 4, 6):
            standoff = [_grid_idx(r, c, 1) for r in (5, 6, 7) for c in range(1, 7)]
            compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 17, or_indices=standoff, neg_indices=[72], outputs=[(8, 0.5), (9, 0.3)])
        else:
            close_ahead = [_grid_idx(r, c, 1) for r in (4, 5) for c in range(1, 7)]
            compile_or(W_kb, b_kb, W_kb_out, b_kb_out, 17, or_indices=close_ahead, neg_indices=[72], outputs=[(9, 0.6)])
        # Rules 18/19: cavalry outflank approach (no enemy in grid): drift further toward own side; general stays behind line
        if ri in (3, 4):
            compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 18, [(66, 16.0), (72, -16.0)] + [(i, -16.0) for i in enemy_all], -1.6, [(2, 0.7), (0, 0.3)])
            compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 19, [(66, -16.0), (72, -16.0)] + [(i, -16.0) for i in enemy_all], -1.6, [(1, 0.7), (0, 0.3)])
        elif ri == 5:
            compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 18, feat_idx=77, threshold=-0.2, direction=+1.0, neg_indices=[72], outputs=[(7, 0.8), (0, -0.4)])
        # Rule 20: siege assault & gate intact & NOT engaged -> stage outside tower range (hold) until breach (g[17..19] scaled 3.0x)
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 20, feat_idx=17, threshold=2.25, direction=+1.0, neg_indices=[72, 18], outputs=[(8, 0.55), (9, -0.35), (0, -0.3)], width=0.6)
        # Rule 21: siege assault & breached -> storm the breach
        compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 21, [(17, 8.0), (18, 5.33)], -32.0,
                       [(9, 0.5), (8, 0.2)] if ri in (2, 6) else [(9, 0.9), (0, 0.4)])
        # Rule 22: siege defense & walls holding & NOT engaged -> hold position / fire from battlements
        compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 22, feat_idx=17, threshold=-2.25, direction=-1.0, neg_indices=[72, 18], outputs=[(8, 0.6), (9, -0.3)], width=0.6)
        # Rule 23: siege defense & breached -> counter-charge into the breach
        compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 23, [(17, -8.0), (18, 5.33)], -32.0,
                       [(8, 0.3)] if ri in (2, 6) else [(9, 0.9), (0, 0.4)])

        # Layer 2 (Purely Additive on top of Layer 1): 32 homogeneous recurrent neurons with W2 initialized to 0.0
        # logits = (W_kb_out @ h_kb + b_kb_out) + (W2 @ h_top)
        W_top_kb = (
            rng.standard_normal((UNIT_MLP_HIDDEN, UNIT_KB_HIDDEN)) * (0.55 / np.sqrt(UNIT_KB_HIDDEN))
        ).astype(np.float64)
        W_top_x = (
            rng.standard_normal((UNIT_MLP_HIDDEN, UNIT_CTX_IN)) * (0.25 / np.sqrt(UNIT_CTX_IN))
        ).astype(np.float64)
        Wh = (
            rng.standard_normal((UNIT_MLP_HIDDEN, UNIT_MLP_HIDDEN)) * (0.25 / np.sqrt(UNIT_MLP_HIDDEN))
        ).astype(np.float64)
        for j in range(UNIT_MLP_HIDDEN):
            Wh[j, j] += 0.40
        b_top = np.zeros(UNIT_MLP_HIDDEN, dtype=np.float64)
        W2 = np.zeros((NUM_ACTIONS, UNIT_MLP_HIDDEN), dtype=np.float64)

        roles.append({
            "W_kb": W_kb,
            "b_kb": b_kb,
            "W_kb_out": W_kb_out,
            "b_kb_out": b_kb_out,
            "W_top_kb": W_top_kb,
            "W_top_x": W_top_x,
            "Wh": Wh,
            "b_top": b_top,
            "W2": W2,
        })
    return roles


def build_kbann_commander_rnn(seed=42):
    """Build 2-layer stacked Commander RNN (Layer 1: 10 Towell-Shavlik grand-tactics rules -> Layer 2: 48 learned RNN on top)."""
    rng = np.random.default_rng(seed)
    W_kb = (rng.standard_normal((CMD_KB_HIDDEN, CMD_RNN_IN)) * 0.005).astype(np.float64)
    b_kb = np.zeros(CMD_KB_HIDDEN, dtype=np.float64)
    W_kb_out = np.zeros((CMD_RNN_OUT, CMD_KB_HIDDEN), dtype=np.float64)
    b_kb_out = np.zeros(CMD_RNN_OUT, dtype=np.float64)

    # Rule 0 (Threshold): Ranged advantage (g[9] > +0.25) & early battle -> Order 0 (form up / shootout)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 0, feat_idx=9, threshold=0.25, direction=+1.0, neg_indices=[11], outputs=[(0, 0.45)])
    # Rule 1 (Threshold): Strength ratio >= 0.8 (g[4] > -0.11) & not engaged -> Order 1 (advance battlegroup)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 1, feat_idx=4, threshold=-0.11, direction=+1.0, neg_indices=[11], outputs=[(1, 0.50)])
    # Rule 2 (Threshold): Army engaged in melee (g[11] > 0.25) -> Order 2 (press attack)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 2, feat_idx=11, threshold=0.25, direction=+1.0, neg_indices=[], outputs=[(2, 0.50)])
    # Rule 3 (Threshold): Left flank squad (x[290] < -0.20) -> Order 3 (left flank maneuver)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 3, feat_idx=290, threshold=-0.20, direction=-1.0, neg_indices=[], outputs=[(3, 0.45)])
    # Rule 4 (Threshold): Right flank squad (x[290] > +0.20) -> Order 4 (right flank maneuver)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 4, feat_idx=290, threshold=0.20, direction=+1.0, neg_indices=[], outputs=[(4, 0.45)])
    # Rule 5 (Threshold): Outnumbered (g[4] < -0.20) -> Order 5 (defensive regroup)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 5, feat_idx=4, threshold=-0.20, direction=-1.0, neg_indices=[], outputs=[(5, 0.45)])
    # Rule 6 (Threshold): Siege assault (g[17]*3 > +2.25) & gate intact -> Order 1 (push siege engines)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 6, feat_idx=17, threshold=2.25, direction=+1.0, neg_indices=[], outputs=[(1, 0.45), (2, 0.35)], width=0.6)
    # Rule 7 (Threshold): Siege defense (g[17]*3 < -2.25) -> Order 0 (hold fortress walls & gatehouse)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 7, feat_idx=17, threshold=-2.25, direction=-1.0, neg_indices=[], outputs=[(0, 0.55), (5, 0.30)], width=0.6)
    # Rule 8 (Threshold): River crossing (g[17]*3 > 0.75) -> Order 1/2 (control bridge & ford crossings)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 8, feat_idx=17, threshold=0.75, direction=+1.0, neg_indices=[], outputs=[(1, 0.35), (3, 0.25), (4, 0.25)], width=0.6)
    # Rule 9 (Threshold): Enemy morale wavering (g[15] < 0.65) -> Order 2 (general charge / pursuit)
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 9, feat_idx=15, threshold=0.65, direction=-1.0, neg_indices=[], outputs=[(2, 0.50)])

    # ---- Rules 10..18: additional vanilla grand-tactics ----
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 10, feat_idx=10, threshold=0.25, direction=+1.0, neg_indices=[11], outputs=[(3, 0.30), (4, 0.30)])   # cavalry superiority -> outflank
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 11, feat_idx=4, threshold=0.25, direction=+1.0, neg_indices=[], outputs=[(2, 0.40)])                # clearly ahead -> press
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 12, feat_idx=14, threshold=0.60, direction=-1.0, neg_indices=[], outputs=[(5, 0.50)])               # own morale low -> regroup
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 13, feat_idx=1, threshold=0.10, direction=-1.0, neg_indices=[], outputs=[(0, 0.30), (1, 0.30)])     # early battle -> form up & advance
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 14, feat_idx=6, threshold=-0.30, direction=-1.0, neg_indices=[], outputs=[(3, 0.30)])               # left wing weak
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 15, feat_idx=7, threshold=-0.30, direction=-1.0, neg_indices=[], outputs=[(4, 0.30)])               # right wing weak
    compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 16, [(17, 8.0), (18, 5.33)], -32.0, [(2, 0.60)])                                                       # siege assault breached -> press
    compile_linear(W_kb, b_kb, W_kb_out, b_kb_out, 17, [(17, -8.0), (18, 5.33)], -32.0, [(2, 0.60), (5, -0.30)])                                          # siege defense breached -> counter-charge
    compile_threshold(W_kb, b_kb, W_kb_out, b_kb_out, 18, feat_idx=13, threshold=0.50, direction=-1.0, neg_indices=[], outputs=[(2, 0.50)])              # enemy general dead -> press

    # Layer 2 (Purely Additive on top of Layer 1): 48 homogeneous recurrent neurons with Wo initialized to 0.0
    # logits = (W_kb_out @ h_kb + b_kb_out) + (Wo @ h_top)
    W_top_kb = (
        rng.standard_normal((CMD_RNN_HIDDEN, CMD_KB_HIDDEN)) * (0.55 / np.sqrt(CMD_KB_HIDDEN))
    ).astype(np.float64)
    Wx = (
        rng.standard_normal((CMD_RNN_HIDDEN, CMD_RNN_IN)) * (0.25 / np.sqrt(CMD_RNN_IN))
    ).astype(np.float64)
    Wh = (
        rng.standard_normal((CMD_RNN_HIDDEN, CMD_RNN_HIDDEN)) * (0.25 / np.sqrt(CMD_RNN_HIDDEN))
    ).astype(np.float64)
    for j in range(CMD_RNN_HIDDEN):
        Wh[j, j] += 0.40
    bh = np.zeros(CMD_RNN_HIDDEN, dtype=np.float64)
    Wo = np.zeros((CMD_RNN_OUT, CMD_RNN_HIDDEN), dtype=np.float64)

    return {
        "W_kb": W_kb,
        "b_kb": b_kb,
        "W_kb_out": W_kb_out,
        "b_kb_out": b_kb_out,
        "W_top_kb": W_top_kb,
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "Wo": Wo,
    }
