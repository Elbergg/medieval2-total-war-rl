"""Post-battle Linear SARSA(lambda) learner (NumPy). Features arrive sparse: [[idx, val], ...]."""
import json
import os
import numpy as np
from spec import NUM_ACTIONS, NUM_FEATURES, NUM_ROLES, ROLES, imitation_reward

GAMMA, LAMBDA, ALPHA = 0.995, 0.95, 0.01
W_LOC, W_GLOB, W_MAP, TERMINAL = 0.1, 2.0, 1.0, 150.0


def new_weights():
    return np.zeros((NUM_ROLES, NUM_ACTIONS, NUM_FEATURES), dtype=np.float64)


def load_weights(path):
    if os.path.exists(path):
        w = np.array(json.load(open(path)), dtype=np.float64)
        if w.shape[0] < NUM_ROLES:
            pad = np.zeros((NUM_ROLES - w.shape[0], NUM_ACTIONS, NUM_FEATURES), dtype=np.float64)
            w = np.concatenate([w, pad], axis=0)
        return w
    return new_weights()


def save_weights(w, path):
    tmp = path + ".tmp"
    json.dump(np.round(w, 6).tolist(), open(tmp, "w"))
    os.replace(tmp, path)


def step_reward(step, phase, terminal=0.0, llm_penalty=0.0):
    """Phase 1: imitation only. Phase 2/3: combat+map (+terminal) only; imitation is gone entirely."""
    if phase == 1:
        return imitation_reward(step["a"], step.get("disp"), step.get("vq"))
    r = (W_LOC * (step.get("dk", 0.0) - 0.25 * step.get("dl", 0.0))
         + W_GLOB * step.get("gc", 0.0) + W_MAP * step.get("gm", 0.0))
    return r + terminal + llm_penalty


def _phi(sparse):
    idx = np.fromiter((p[0] for p in sparse), dtype=np.int64, count=len(sparse))
    val = np.fromiter((p[1] for p in sparse), dtype=np.float64, count=len(sparse))
    return idx, val


def update_phase1(w, ep):
    """Way-2 warm start: for every logged state, move Q(s,a) for ALL 13 actions toward its imitation reward
    (-dist^2 to what Vanilla physically did in the 3s window), so argmax_a Q reproduces Vanilla."""
    n = 0
    for u in ep["units"]:
        ri = ROLES.index(u["role"])
        for s in u["steps"]:
            idx, val = _phi(s["phi"])
            norm = max(1.0, val.sum())
            for a in range(NUM_ACTIONS):
                target = imitation_reward(a, s.get("disp"), s.get("vq"))
                err = target - w[ri, a][idx] @ val
                w[ri, a][idx] += (ALPHA / norm) * err * val
            n += 1
    return n


def update_from_episode(w, ep, phase, penalties=None, llm_terminal_penalty=0.0, weight_mult=1.0):
    if phase == 1:
        return update_phase1(w, ep)
    """penalties: list of dicts {role, action, t_start, t_end, penalty} from the LLM judge;
    llm_terminal_penalty: summed negative squared score in [-300, 0] applied at battle end;
    weight_mult: multiplier on learning rate ALPHA (e.g. 3.0 on 30-min LLM-judged battles)."""
    result = ep.get("result", 0)
    tick = ep.get("tick_seconds", 3.0)
    alpha_eff = ALPHA * weight_mult
    n_updates = 0
    for u in ep["units"]:
        role, steps = u["role"], u["steps"]
        ri = ROLES.index(role)
        if not steps:
            continue
        E = {}  # action -> (dense trace)
        for t, s in enumerate(steps):
            last = t == len(steps) - 1
            terminal = (TERMINAL * result + llm_terminal_penalty) if (last and phase != 1) else 0.0
            pen = 0.0
            for p in penalties or []:
                if p["role"] in (role, "any") and p["action"] == s["a"] and p["t_start"] <= t * tick <= p["t_end"]:
                    pen += p["penalty"]
            r = step_reward(s, phase, terminal, pen)
            idx, val = _phi(s["phi"])
            q = w[ri, s["a"]][idx] @ val
            if last:
                q_next = 0.0
            else:
                ni, nv = _phi(steps[t + 1]["phi"])
                q_next = w[ri, steps[t + 1]["a"]][ni] @ nv
            delta = r + GAMMA * q_next - q
            for a in list(E):
                E[a] *= GAMMA * LAMBDA
            tr = E.setdefault(s["a"], np.zeros(NUM_FEATURES))
            tr[idx] += val
            norm = max(1.0, val.sum())
            for a, tr_a in E.items():
                w[ri, a] += (alpha_eff / norm) * delta * tr_a
            n_updates += 1
    return n_updates
