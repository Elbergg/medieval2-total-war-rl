"""Learned Self-Attention World Models for M2TW Battle Physics: DamageNet & MoraleNet.

Two separate lightweight Transformer/Self-Attention networks:
  - DamageNet: predicts per-unit HP fraction lost over the next 4.5s tick (d_hp in [0, 1])
  - MoraleNet: predicts per-unit next morale level (morale_{t+1} in [0, 1], 0=routing, 1=eager)

Both take ALL units on the battlefield (up to 48 = 24 friendly + 24 enemy, each 23 floats:
22 unit features + side flag +1/-1) with self-attention so every unit attends to nearby
enemies, flankers, generals, archers, and routing allies.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_model import GLOBAL_FEAT, MAX_UNITS, UNIT_FEAT

TOTAL_SLOTS = 2 * MAX_UNITS  # 48 (24 own + 24 enemy)
NODE_FEAT = UNIT_FEAT + 1    # 23 (22 unit feats + side flag +1.0/-1.0)


def obs_to_unit_nodes(obs: np.ndarray) -> tuple:
    """Convert flat obs (supports both 24-feat [1164] and legacy 22-feat [1068]) into (nodes[48, 25], alive_mask[48])."""
    nodes = np.zeros((TOTAL_SLOTS, NODE_FEAT), dtype=np.float32)
    alive = np.zeros((TOTAL_SLOTS,), dtype=np.float32)
    uf = (obs.shape[0] - GLOBAL_FEAT) // TOTAL_SLOTS
    uf_copy = min(uf, UNIT_FEAT)
    off_own = GLOBAL_FEAT
    off_en = GLOBAL_FEAT + MAX_UNITS * uf
    for i in range(MAX_UNITS):
        u = obs[off_own + i * uf : off_own + (i + 1) * uf]
        if u[0] > 0.5:
            nodes[i, :uf_copy] = u[:uf_copy]
            if uf < UNIT_FEAT:
                nodes[i, 22] = 5.0 / 15.0   # fallback base_morale for legacy 22-feat eps
                nodes[i, 23] = 60.0 / 150.0  # fallback start size
            nodes[i, UNIT_FEAT] = 1.0
            alive[i] = 1.0
    for i in range(MAX_UNITS):
        u = obs[off_en + i * uf : off_en + (i + 1) * uf]
        if u[0] > 0.5:
            nodes[MAX_UNITS + i, :uf_copy] = u[:uf_copy]
            if uf < UNIT_FEAT:
                nodes[MAX_UNITS + i, 22] = 5.0 / 15.0
                nodes[MAX_UNITS + i, 23] = 60.0 / 150.0
            nodes[MAX_UNITS + i, UNIT_FEAT] = -1.0
            alive[MAX_UNITS + i] = 1.0
    return nodes, alive


def extract_transitions(episodes: list) -> tuple:
    """Build supervised (X[M, 48, 25], mask[M, 48], y_dmg[M, 48], y_mor[M, 48]) from M2TW episodes."""
    xs, ms, yd, ym = [], [], [], []
    for ep in episodes:
        steps = ep.get("steps") or []
        if len(steps) < 2:
            continue
        for t in range(len(steps) - 1):
            o0 = np.asarray(steps[t]["obs"], dtype=np.float32)
            o1 = np.asarray(steps[t + 1]["obs"], dtype=np.float32)
            if o0.shape[0] < GLOBAL_FEAT + TOTAL_SLOTS * 22:
                continue
            n0, m0 = obs_to_unit_nodes(o0)
            n1, m1 = obs_to_unit_nodes(o1)
            if m0.sum() < 1.0:
                continue
            # hp_frac is index 3, morale is index 4
            hp0 = n0[:, 3]
            hp1 = n1[:, 3] * m1  # if unit died (m1=0), hp1=0
            dhp = np.clip(hp0 - hp1, 0.0, 1.0)
            mor1 = n1[:, 4] * m1  # if dead, morale=0
            xs.append(n0)
            ms.append(m0)
            yd.append(dhp)
            ym.append(mor1)
    if not xs:
        return None, None, None, None
    return (
        np.stack(xs, axis=0),
        np.stack(ms, axis=0),
        np.stack(yd, axis=0),
        np.stack(ym, axis=0),
    )


class UnitSelfAttentionNet(nn.Module):
    """Small Self-Attention network predicting a scalar per unit across all 48 army slots."""

    def __init__(self, in_dim: int = NODE_FEAT, d_model: int = 64, n_heads: int = 4, target_kind: str = "damage"):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.target_kind = target_kind

        self.in_proj = nn.Linear(in_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=0.05)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, nodes: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        """nodes: (B, 48, 23), alive_mask: (B, 48) -> pred: (B, 48) in [0, 1]."""
        h = F.silu(self.ln1(self.in_proj(nodes)))
        key_pad = (alive_mask <= 0.5)  # True = ignore
        # Guard against all-masked rows
        all_pad = key_pad.all(dim=-1, keepdim=True)
        safe_pad = key_pad & (~all_pad)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=safe_pad, need_weights=False)
        h = self.ln2(h + attn_out)
        h = self.ln3(h + self.ffn(h))
        raw = self.head(h).squeeze(-1)
        if self.target_kind == "damage":
            # Small non-negative per-tick damage fraction in [0, 1]
            return torch.sigmoid(raw - 3.0)
        # Residual update on current morale (index 4) clamped to [0, 1]
        cur_mor = nodes[:, :, 4]
        return torch.clamp(cur_mor + 0.5 * torch.tanh(raw), 0.0, 1.0)


def save_physics_net(model: UnitSelfAttentionNet, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "in_dim": model.in_dim,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "target_kind": model.target_kind,
        "state_dict": model.state_dict(),
    }, tmp)
    os.replace(tmp, path)


def load_physics_net(path: str, target_kind: str = "damage") -> UnitSelfAttentionNet:
    model = UnitSelfAttentionNet(target_kind=target_kind)
    if path and os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = UnitSelfAttentionNet(
                in_dim=ckpt.get("in_dim", NODE_FEAT),
                d_model=ckpt.get("d_model", 64),
                n_heads=ckpt.get("n_heads", 4),
                target_kind=ckpt.get("target_kind", target_kind),
            )
            model.load_state_dict(ckpt["state_dict"])
        except Exception as e:
            print(f"[physics_world_model] Warning loading {path}: {e}", flush=True)
    return model


@torch.no_grad()
def evaluate_physics_net(model: UnitSelfAttentionNet, val_episodes: list) -> dict:
    """Evaluate per-tick MAE (%) AND full-battle cumulative HP/morale error (%) on held-out M2TW battles."""
    model.eval()
    X, M, Yd, Ym = extract_transitions(val_episodes)
    if X is None:
        return {"mae_pct": 100.0, "cum_err_pct": 100.0, "n_ticks": 0}
    Y = Yd if model.target_kind == "damage" else Ym
    x_t = torch.from_numpy(X)
    m_t = torch.from_numpy(M)
    y_t = torch.from_numpy(Y)
    preds = []
    for i in range(0, len(X), 128):
        preds.append(model(x_t[i : i + 128], m_t[i : i + 128]))
    p_t = torch.cat(preds, dim=0)
    abs_err = (p_t - y_t).abs() * m_t
    mae = float(abs_err.sum().item() / max(1.0, m_t.sum().item())) * 100.0

    # True autoregressive full-battle rollout error (%) for BOTH DamageNet & MoraleNet:
    #   - DamageNet: start from true hp_0, subtract pred_dhp every tick (feeding predicted hp back in)
    #   - MoraleNet: start from true morale_0, assign pred_morale every tick (feeding predicted morale back in)
    rollout_errs = []
    for ep in val_episodes:
        ex, em, eyd, eym = extract_transitions([ep])
        if ex is None or len(ex) < 2:
            continue
        T_ep = len(ex)
        cur_hp = ex[0, :, 3].copy()
        cur_mor = ex[0, :, 4].copy()
        step_errs = []
        for t in range(T_ep):
            x_step = ex[t].copy()
            m_step = em[t]
            if model.target_kind == "damage":
                x_step[:, 3] = cur_hp
                pred_dhp = model(torch.from_numpy(x_step[None]), torch.from_numpy(m_step[None]))[0].numpy()
                cur_hp = np.clip(cur_hp - pred_dhp * m_step, 0.0, 1.0)
                true_hp_next = np.clip(ex[t, :, 3] - eyd[t], 0.0, 1.0)
                valid = m_step > 0.5
                if valid.any():
                    step_errs.append(float(np.abs(cur_hp[valid] - true_hp_next[valid]).mean()) * 100.0)
            else:
                x_step[:, 4] = cur_mor
                pred_mor = model(torch.from_numpy(x_step[None]), torch.from_numpy(m_step[None]))[0].numpy()
                cur_mor = np.where(m_step > 0.5, pred_mor, 0.0)
                valid = m_step > 0.5
                if valid.any():
                    step_errs.append(float(np.abs(cur_mor[valid] - eym[t, valid]).mean()) * 100.0)
        if step_errs:
            rollout_errs.append(float(np.mean(step_errs)))
    cum_err = float(np.mean(rollout_errs)) if rollout_errs else mae
    return {"mae_pct": round(mae, 2), "cum_err_pct": round(cum_err, 2), "n_ticks": int(len(X))}


def train_physics_epoch(
    model: UnitSelfAttentionNet,
    opt: torch.optim.Optimizer,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    epochs: int = 8,
) -> float:
    model.train()
    n = len(X)
    rng = np.random.default_rng()
    last_loss = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(X[idx])
            mb = torch.from_numpy(M[idx])
            yb = torch.from_numpy(Y[idx])
            pred = model(xb, mb)
            if model.target_kind == "damage":
                # Upweight active-combat damage ticks (y > 0) + penalize batch mean bias so cumulative battle error stays <5%
                wt = mb * torch.where(yb > 1e-4, 3.0, 1.0)
                tick_loss = (F.l1_loss(pred, yb, reduction="none") * wt).sum() / wt.sum().clamp_min(1.0)
                bias_loss = ((pred * mb).sum() - (yb * mb).sum()).abs() / mb.sum().clamp_min(1.0)
                loss = tick_loss + 2.0 * bias_loss
            else:
                loss_mat = F.smooth_l1_loss(pred, yb, beta=0.02, reduction="none") * mb
                loss = loss_mat.sum() / mb.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            batches += 1
        last_loss = ep_loss / max(1, batches)
    return round(last_loss, 5)
