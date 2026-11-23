"""Full-Army Monolithic LSTM Policy-Value Network + Large-Batch Policy Gradient (A2C, no PPO).

Architecture:
  obs_t (1068 floats):
    - 12 global floats: battle_type (4 one-hot), own_hp, en_hp, own_morale, en_morale,
                        time_frac, own_gen_alive, en_gen_alive, siege_progress
    - 24 friendly unit slots x 22 floats = 528
    - 24 enemy unit slots    x 22 floats = 528
  Network:
    Linear(1068 -> 256) -> LayerNorm -> SiLU -> LSTM(256 -> 256)
      -> Actor Head: Linear(256 -> 24 * 13)  (commands for ALL friendly units at once)
      -> Value Head: Linear(256 -> 1)        (learned baseline for Policy Gradient)
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from spec import NUM_ACTIONS, ROLES

MAX_UNITS = 24
UNIT_FEAT = 26
GLOBAL_FEAT = 12
OBS_DIM = GLOBAL_FEAT + 2 * MAX_UNITS * UNIT_FEAT  # 1260
HIDDEN_DIM = 512

BATTLE_TYPES = ("open_field", "river_crossing", "siege_assault", "siege_defense")

# Reward weights: ONLY win/loss + decisive + LLM judge (zero dense dk/dl/coh shaping)
TERMINAL = 100.0
DECISIVE_BONUS = 340.0
LLM_MULT = 85.0

DEFAULT_LR = 1.0e-3
WEIGHT_DECAY = 1.0e-3
L2_COEF = 1.0e-4


class FullArmyLSTM(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.actor = nn.Linear(hidden_dim, MAX_UNITS * NUM_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

        # Clean orthogonal init for stable large-batch RL
        nn.init.orthogonal_(self.fc_in.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(self.fc_in.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.05)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor, hx=None):
        """obs: (B, T, OBS_DIM) -> logits (B, T, MAX_UNITS, NUM_ACTIONS), values (B, T), hx_new."""
        x = F.silu(self.ln_in(self.fc_in(obs)))
        out, hx_new = self.lstm(x, hx)
        B, T, _ = out.shape
        logits = self.actor(out).view(B, T, MAX_UNITS, NUM_ACTIONS)
        values = self.critic(out).squeeze(-1)
        return logits, values, hx_new

    @torch.no_grad()
    def act_step(self, obs_np: np.ndarray, mask_np: np.ndarray, hx=None, inv_temp: float = 1.0, greedy: bool = False):
        """Single-step inference for Sim or M2TW. Returns (actions_list[24], hx_new)."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).view(1, 1, self.obs_dim)
        logits, _, hx_new = self.forward(obs_t, hx)
        logits = logits[0, 0]  # (24, 13)
        if greedy:
            acts = torch.argmax(logits, dim=-1).tolist()
        else:
            probs = F.softmax(logits * float(inv_temp), dim=-1)
            acts = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()
        # Inactive slots default to Hold (8)
        for i in range(MAX_UNITS):
            if mask_np[i] <= 0.5:
                acts[i] = 8
        return acts, hx_new


def save_lstm(model: FullArmyLSTM, path: str, opt: optim.Optimizer = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    payload = {
        "obs_dim": model.obs_dim,
        "hidden_dim": model.hidden_dim,
        "state_dict": model.state_dict(),
    }
    if opt is not None:
        payload["opt_state"] = opt.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _make_optimizer(model: FullArmyLSTM, lr: float):
    """AdamW with L2 weight decay applied to weights only (excluding biases & LayerNorm)."""
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "ln_" not in name:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)


def _l2_penalty(model: FullArmyLSTM) -> torch.Tensor:
    reg = torch.tensor(0.0)
    n = 0
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "ln_" not in name:
            reg = reg + p.pow(2).mean()
            n += 1
    return reg / max(1, n)


def load_lstm(path: str, lr: float = DEFAULT_LR):
    model = FullArmyLSTM()
    opt = _make_optimizer(model, lr=lr)
    if path and os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if ckpt.get("obs_dim") == OBS_DIM and ckpt.get("hidden_dim") == HIDDEN_DIM:
                model.load_state_dict(ckpt["state_dict"])
                if "opt_state" in ckpt:
                    try:
                        opt.load_state_dict(ckpt["opt_state"])
                        for pg in opt.param_groups:
                            pg["lr"] = lr
                    except Exception:
                        pass
        except Exception as e:
            print(f"[lstm_model] Warning loading {path}: {e}", flush=True)
    return model, opt


MOVE_PEN_COEF = -0.02   # small per-meter movement penalty per tick (max -0.6/tick -> ~-18 over 30 ticks)
LINE_REW_COEF = 0.40    # straight-line formation bonus per tick (max +0.4/tick -> ~+12 over 30 ticks)


def compute_step_shaping(steps: list) -> tuple:
    """Compute per-tick (move_pen[T], line_rew[T]) with strict anti-exploit guards.

    1. move_pen: small penalty for actual world-space movement (uses exact world meters `st["move_m"]`
       when available, immune to army-frame rotation; falls back to fraction of active units issued
       non-Hold/non-Brace orders * 10m).
    2. line_rew: reward for keeping >= 3 active non-routing front-line infantry (line_inf + spear_inf,
       or all active units if < 3 infantry) in a genuine enemy-facing straight battle line:
         - Requires k >= 3 units (2 points are trivially collinear).
         - Stable 2D covariance PCA: disc = (c00-c11)^2 + 4*c01^2 (zero cancellation).
         - Anti-column guard: multiplied by |vx| (cosine of angle to enemy-facing horizontal axis),
           so a single-file conga line pointing at the enemy scores ~0.
         - Anti-stacking guard: requires rms_along >= 15m and median adjacent gap >= 10m.
         - Anti-split-wings guard: decays smoothly if max adjacent gap along the line > 55m.
    """
    T = len(steps)
    move_pen = np.zeros(T, dtype=np.float32)
    line_rew = np.zeros(T, dtype=np.float32)

    for t, st in enumerate(steps):
        # 1. True movement penalty (world-space meters, never fooled by army-frame rotation)
        if "move_m" in st:
            mean_d = min(30.0, max(0.0, float(st["move_m"])))
        else:
            acts = st.get("acts") or []
            mask = st.get("mask") or []
            n_act = 0
            n_mov = 0
            for a, m in zip(acts, mask):
                if m > 0.5:
                    n_act += 1
                    if int(a) not in (8, 9):  # 8=Hold, 9=Defend/Brace
                        n_mov += 1
            mean_d = 10.0 * (n_mov / max(1, n_act)) if n_act > 0 else 0.0
        move_pen[t] = MOVE_PEN_COEF * mean_d

        # 2. Straight battle-line formation reward
        obs = np.asarray(st["obs"], dtype=np.float32)
        uf = (obs.shape[0] - GLOBAL_FEAT) // (2 * MAX_UNITS)
        inf_pts = []
        all_pts = []
        for i in range(MAX_UNITS):
            off = GLOBAL_FEAT + i * uf
            if obs[off + 0] > 0.5 and obs[off + 9] < 0.5:  # alive & not routing
                x = float(obs[off + 1]) * 300.0
                y = float(obs[off + 2]) * 300.0
                all_pts.append((x, y))
                if obs[off + 10] > 0.5 or obs[off + 11] > 0.5:  # line_inf or spear_inf
                    inf_pts.append((x, y))

        pts_list = inf_pts if len(inf_pts) >= 3 else all_pts
        k = len(pts_list)
        if k >= 3:
            pts = np.asarray(pts_list, dtype=np.float32)
            c = pts - pts.mean(axis=0, keepdims=True)
            c00 = float((c[:, 0] * c[:, 0]).mean())
            c11 = float((c[:, 1] * c[:, 1]).mean())
            c01 = float((c[:, 0] * c[:, 1]).mean())
            tr = c00 + c11
            disc = math.sqrt((c00 - c11) * (c00 - c11) + 4.0 * c01 * c01)
            lam_max = max(0.0, 0.5 * (tr + disc))
            lam_min = max(0.0, 0.5 * (tr - disc))
            rms_along = math.sqrt(lam_max)
            rms_perp = math.sqrt(lam_min)

            if rms_along >= 15.0:
                # Principal axis unit vector (vx, vy); |vx|=1 is broadside facing enemy, |vx|=0 is single-file column
                if abs(c01) > 1e-6:
                    vx, vy = lam_max - c11, c01
                    vn = math.hypot(vx, vy)
                    vx, vy = (vx / vn, vy / vn) if vn > 1e-6 else (1.0, 0.0)
                else:
                    vx, vy = (1.0, 0.0) if c00 >= c11 else (0.0, 1.0)
                broadside = abs(vx)

                # Project onto line axis and check adjacent spacing (anti-stacking & anti-split-wings)
                proj = np.sort(c[:, 0] * vx + c[:, 1] * vy)
                gaps = np.diff(proj)
                med_gap = float(np.median(gaps))
                max_gap = float(gaps.max())
                if med_gap >= 10.0:
                    gap_ok = math.exp(-max(0.0, max_gap - 55.0) / 35.0)
                    straight = math.exp(-rms_perp / 18.0) * broadside * gap_ok * min(1.0, (k - 2) / 3.0)
                    line_rew[t] = LINE_REW_COEF * straight

    return move_pen, line_rew


def _terminal_reward(ep: dict) -> float:
    result = float(ep.get("result", 0))
    dmg_frac = float(ep.get("enemy_damage_frac", 0.0))
    is_dec = bool(ep.get("decisive_win", False))
    is_dec_loss = bool(ep.get("decisive_loss", False))
    llm_pen = float(ep.get("llm_terminal_penalty", 0.0))
    r = TERMINAL * result
    if result > 0 and is_dec:
        r += DECISIVE_BONUS * max(0.5, dmg_frac)
    elif result <= 0 and is_dec_loss:
        r -= 0.5 * DECISIVE_BONUS
    r += LLM_MULT * llm_pen
    return r


def update_batch(
    model: FullArmyLSTM,
    opt: optim.Optimizer,
    episodes: list,
    gamma: float = 0.98,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    target_kl: float = 0.03,
    ent_coef: float = 0.015,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
) -> dict:
    """Large-batch Per-Unit Clipped PPO with learned Value Critic & L2 regularization.

    Pure outcome reward: ONLY win/loss + decisive win/loss + LLM judge score.
    Computes per-unit ratio r_{t,i} = pi_new / pi_old clipped to [1-eps, 1+eps],
    recomputing LSTM hidden states cleanly on every PPO epoch, with KL early-stopping.
    """
    valid_eps = [ep for ep in episodes if ep.get("steps")]
    if not valid_eps:
        return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "l2": 0.0, "steps": 0}

    # Pre-build episode tensors and snapshot old log-probs & baseline advantages (no grad)
    cached = []
    all_advs = []
    total_steps = 0

    model.eval()
    with torch.no_grad():
        for ep in valid_eps:
            steps = ep["steps"]
            T = len(steps)
            total_steps += T
            term = _terminal_reward(ep)
            w_mult = float(ep.get("weight_mult", 1.0))

            obs_t = torch.from_numpy(np.asarray([s["obs"] for s in steps], dtype=np.float32)).unsqueeze(0)
            mask_t = torch.from_numpy(np.asarray([s["mask"] for s in steps], dtype=np.float32))
            acts_t = torch.from_numpy(np.asarray([s["acts"] for s in steps], dtype=np.int64))

            move_p, line_r = compute_step_shaping(steps)
            step_r = move_p + line_r
            rets = np.zeros(T, dtype=np.float32)
            g = term
            for t in range(T - 1, -1, -1):
                g_t = g + float(step_r[t])
                rets[t] = g_t / 100.0
                g = g_t * gamma
            rets_t = torch.from_numpy(rets)

            logits_old, vals_old, _ = model(obs_t)
            lprobs_old = F.log_softmax(logits_old[0], dim=-1)
            old_lp = lprobs_old.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1)  # (T, 24)
            adv_t = rets_t - vals_old[0]                                      # (T,)
            all_advs.append(adv_t)
            cached.append((obs_t, mask_t, acts_t, rets_t, old_lp, adv_t, w_mult))

    cat_adv = torch.cat(all_advs, dim=0)
    adv_mean = cat_adv.mean()
    adv_std = cat_adv.std().clamp_min(1e-4)
    w_total = sum(x[6] for x in cached)

    model.train()
    last_pg = last_vf = last_ent = last_kl = last_cf = last_l2 = 0.0

    for _ep_idx in range(max(1, ppo_epochs)):
        opt.zero_grad()
        pg_loss_sum = torch.tensor(0.0)
        vf_loss_sum = torch.tensor(0.0)
        ent_sum = torch.tensor(0.0)
        kl_sum = 0.0
        cf_sum = 0.0

        for obs_t, mask_t, acts_t, rets_t, old_lp, adv_t, w_mult in cached:
            norm_adv = ((adv_t - adv_mean) / adv_std).clamp(-3.0, 3.0).unsqueeze(-1)  # (T, 1)

            logits, vals, _ = model(obs_t)
            logits = logits[0]                                                        # (T, 24, 13)
            vals = vals[0]                                                            # (T,)
            lprobs = F.log_softmax(logits, dim=-1)
            probs = lprobs.exp()
            new_lp = lprobs.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1)              # (T, 24)
            ent = -(probs * lprobs).sum(dim=-1)                                       # (T, 24)

            log_ratio = (new_lp - old_lp).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * norm_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv

            active_cnt = mask_t.sum(dim=-1).clamp_min(1.0)                            # (T,)
            step_pg = -(torch.min(surr1, surr2) * mask_t).sum(dim=-1) / active_cnt
            step_ent = (ent * mask_t).sum(dim=-1) / active_cnt

            wt = w_mult / max(1.0, w_total)
            pg_loss_sum = pg_loss_sum + wt * step_pg.mean()
            ent_sum = ent_sum + wt * step_ent.mean()
            vf_loss_sum = vf_loss_sum + wt * F.mse_loss(vals, rets_t)

            with torch.no_grad():
                m_sum = mask_t.sum().clamp_min(1.0)
                kl_sum += wt * float((((ratio - 1.0) - log_ratio) * mask_t).sum().item() / m_sum.item())
                cf_sum += wt * float((((ratio - 1.0).abs() > clip_eps).float() * mask_t).sum().item() / m_sum.item())

        l2_reg = _l2_penalty(model)
        loss = pg_loss_sum + vf_coef * vf_loss_sum - ent_coef * ent_sum + L2_COEF * l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        opt.step()
        opt.zero_grad()

        last_pg = float(pg_loss_sum.item())
        last_vf = float(vf_loss_sum.item())
        last_ent = float(ent_sum.item())
        last_kl = kl_sum
        last_cf = cf_sum
        last_l2 = float(l2_reg.item())

        if target_kl > 0.0 and kl_sum > target_kl:
            break

    return {
        "pg_loss": round(last_pg, 4),
        "vf_loss": round(last_vf, 4),
        "entropy": round(last_ent, 4),
        "kl": round(last_kl, 4),
        "clip_frac": round(last_cf, 3),
        "l2": round(last_l2, 5),
        "steps": total_steps,
    }


def update_imitation_batch(
    model: FullArmyLSTM,
    opt: optim.Optimizer,
    episodes: list,
    max_grad_norm: float = 1.0,
) -> dict:
    """Phase-1 Behavioral Cloning / Imitation on large batches (cross-entropy over all 24 friendly units)."""
    valid_eps = [ep for ep in episodes if ep.get("steps")]
    if not valid_eps:
        return {"ce_loss": 0.0, "acc": 0.0, "steps": 0}

    model.train()
    opt.zero_grad()
    ce_sum = torch.tensor(0.0)
    correct = 0.0
    total_active = 0.0
    total_steps = 0
    n_eps = float(len(valid_eps))

    for ep in valid_eps:
        steps = ep["steps"]
        T = len(steps)
        total_steps += T
        obs_t = torch.from_numpy(np.asarray([s["obs"] for s in steps], dtype=np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(np.asarray([s["mask"] for s in steps], dtype=np.float32))
        tgts_t = torch.from_numpy(np.asarray([s["acts"] for s in steps], dtype=np.int64))

        logits, _, _ = model(obs_t)
        logits = logits[0]  # (T, 24, 13)
        ce = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), tgts_t.reshape(-1), reduction="none").view(T, MAX_UNITS)
        active_cnt = mask_t.sum(dim=-1).clamp_min(1.0)
        step_ce = (ce * mask_t).sum(dim=-1) / active_cnt
        ce_sum = ce_sum + step_ce.mean() / n_eps

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += float(((preds == tgts_t).float() * mask_t).sum().item())
            total_active += float(mask_t.sum().item())

    l2_reg = _l2_penalty(model)
    (ce_sum + L2_COEF * l2_reg).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    opt.step()
    opt.zero_grad()

    return {
        "ce_loss": round(float(ce_sum.item()), 4),
        "acc": round(100.0 * correct / max(1.0, total_active), 1),
        "l2": round(float(l2_reg.item()), 5),
        "steps": total_steps,
    }
