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
HIDDEN_DIM = 256

BATTLE_TYPES = ("open_field", "river_crossing", "siege_assault", "siege_defense")

# Reward weights: ONLY win/loss + decisive + LLM judge (zero dense dk/dl/coh shaping)
TERMINAL = 100.0
DECISIVE_BONUS = 340.0
LLM_MULT = 85.0

DEFAULT_LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4


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


def load_lstm(path: str, lr: float = DEFAULT_LR):
    model = FullArmyLSTM()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
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
    ent_coef: float = 0.015,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
) -> dict:
    """Large-batch Policy Gradient with learned Value baseline (A2C, no PPO).

    Pure outcome reward: ONLY win/loss + decisive win/loss + LLM judge score.
    """
    valid_eps = [ep for ep in episodes if ep.get("steps")]
    if not valid_eps:
        return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "steps": 0}

    model.train()
    opt.zero_grad()

    ep_tensors = []
    all_advs = []
    total_steps = 0

    for ep in valid_eps:
        steps = ep["steps"]
        T = len(steps)
        total_steps += T
        term = _terminal_reward(ep)
        w_mult = float(ep.get("weight_mult", 1.0))

        obs_arr = np.asarray([s["obs"] for s in steps], dtype=np.float32)       # (T, 1068)
        mask_arr = np.asarray([s["mask"] for s in steps], dtype=np.float32)     # (T, 24)
        acts_arr = np.asarray([s["acts"] for s in steps], dtype=np.int64)       # (T, 24)

        # Pure terminal return (win + decisive + LLM judge), discounted backward across the battle
        rets = np.zeros(T, dtype=np.float32)
        g = term
        for t in range(T - 1, -1, -1):
            rets[t] = g / 100.0
            g *= gamma

        obs_t = torch.from_numpy(obs_arr).unsqueeze(0)  # (1, T, 1068)
        logits, vals, _ = model(obs_t)
        logits = logits[0]                              # (T, 24, 13)
        vals = vals[0]                                  # (T,)
        rets_t = torch.from_numpy(rets)
        adv_t = (rets_t - vals.detach())                # (T,)
        all_advs.append(adv_t)

        ep_tensors.append((
            logits, vals, rets_t, adv_t,
            torch.from_numpy(mask_arr),
            torch.from_numpy(acts_arr),
            w_mult,
        ))

    # Standardize advantages across the entire large batch for low variance
    cat_adv = torch.cat(all_advs, dim=0)
    adv_mean = cat_adv.mean()
    adv_std = cat_adv.std().clamp_min(1e-4)

    pg_loss_sum = torch.tensor(0.0)
    vf_loss_sum = torch.tensor(0.0)
    ent_sum = torch.tensor(0.0)
    w_total = sum(x[6] for x in ep_tensors)

    for logits, vals, rets_t, adv_t, mask_t, acts_t, w_mult in ep_tensors:
        norm_adv = ((adv_t - adv_mean) / adv_std).clamp(-3.0, 3.0).unsqueeze(-1)  # (T, 1)

        lprobs = F.log_softmax(logits, dim=-1)                                  # (T, 24, 13)
        probs = lprobs.exp()
        chosen_lp = lprobs.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1)         # (T, 24)
        ent = -(probs * lprobs).sum(dim=-1)                                     # (T, 24)

        active_cnt = mask_t.sum(dim=-1).clamp_min(1.0)                          # (T,)
        step_pg = -((chosen_lp * norm_adv) * mask_t).sum(dim=-1) / active_cnt   # (T,)
        step_ent = (ent * mask_t).sum(dim=-1) / active_cnt                      # (T,)

        wt = w_mult / max(1.0, w_total)
        pg_loss_sum = pg_loss_sum + wt * step_pg.mean()
        ent_sum = ent_sum + wt * step_ent.mean()
        vf_loss_sum = vf_loss_sum + wt * F.mse_loss(vals, rets_t)

    loss = pg_loss_sum + vf_coef * vf_loss_sum - ent_coef * ent_sum
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    opt.step()
    opt.zero_grad()

    return {
        "pg_loss": round(float(pg_loss_sum.item()), 4),
        "vf_loss": round(float(vf_loss_sum.item()), 4),
        "entropy": round(float(ent_sum.item()), 4),
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

    ce_sum.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    opt.step()
    opt.zero_grad()

    return {
        "ce_loss": round(float(ce_sum.item()), 4),
        "acc": round(100.0 * correct / max(1.0, total_active), 1),
        "steps": total_steps,
    }
