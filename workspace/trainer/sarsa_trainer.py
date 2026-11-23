"""2-Layer Stacked Towell-Shavlik KBANN + Zero-Initialized Top RNN Trainer (PyTorch Autograd + AdamW)."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from spec import (
    CMD_RNN_HIDDEN, CMD_RNN_IN, CMD_RNN_OUT,
    NUM_ACTIONS, NUM_FEATURES, NUM_ROLES, ROLES, UNIT_MLP_HIDDEN, imitation_reward,
)
from kbann_init import (
    CMD_KB_HIDDEN, UNIT_CTX_IN, UNIT_KB_HIDDEN,
    build_kbann_commander_rnn, build_kbann_unit_rnns,
)

W_LOC, W_LOSS, W_COH = 22.5, 8.5, 4.25
TERMINAL, DMG_BONUS, DECISIVE_BONUS, LLM_MULT = 100.0, 0.0, 340.0, 85.0
UNIT_KEYS = ("W_kb", "b_kb", "W_kb_out", "b_kb_out", "W_top_kb", "W_top_x", "Wh", "b_top", "W2")
CMD_KEYS = ("W_kb", "b_kb", "W_kb_out", "b_kb_out", "W_top_kb", "Wx", "Wh", "bh", "Wo")
ADAPT_KEYS = ("W_adapt_h", "W_adapt_x", "b_adapt", "W_adapt_out")
UNIT_ALL_KEYS = UNIT_KEYS + ADAPT_KEYS
CMD_ALL_KEYS = CMD_KEYS + ADAPT_KEYS
UNIT_LORA_KEYS = ("W_kb", "W_kb_out", "W_top_kb", "W_top_x", "Wh", "W2", "W_adapt_h", "W_adapt_x", "W_adapt_out")
CMD_LORA_KEYS = ("W_kb", "W_kb_out", "W_top_kb", "Wx", "Wh", "Wo", "W_adapt_h", "W_adapt_x", "W_adapt_out")
ADAPT_HIDDEN = 24
LORA_RANK = 4
LORA_ALPHA = 4.0
LORA_SCALE = LORA_ALPHA / LORA_RANK

LR_BASE = 2.0e-3
ADAPT_LR = 2.0e-3
LORA_LR = 1.5e-3
WEIGHT_DECAY = 1.0e-4
ADAPT_WEIGHT_DECAY = 1.0e-3
LORA_WEIGHT_DECAY = 1.0e-3


def _ensure_unit_adapter(rw, role_idx=0):
    rng = np.random.default_rng(1000 + role_idx)
    if "W_adapt_h" not in rw or np.asarray(rw["W_adapt_h"]).shape != (ADAPT_HIDDEN, UNIT_MLP_HIDDEN):
        rw["W_adapt_h"] = rng.normal(0.0, 0.08, size=(ADAPT_HIDDEN, UNIT_MLP_HIDDEN)).astype(np.float64)
    if "W_adapt_x" not in rw or np.asarray(rw["W_adapt_x"]).shape != (ADAPT_HIDDEN, UNIT_CTX_IN):
        rw["W_adapt_x"] = rng.normal(0.0, 0.05, size=(ADAPT_HIDDEN, UNIT_CTX_IN)).astype(np.float64)
    if "b_adapt" not in rw or np.asarray(rw["b_adapt"]).shape != (ADAPT_HIDDEN,):
        rw["b_adapt"] = np.zeros((ADAPT_HIDDEN,), dtype=np.float64)
    if "W_adapt_out" not in rw or np.asarray(rw["W_adapt_out"]).shape != (NUM_ACTIONS, ADAPT_HIDDEN):
        rw["W_adapt_out"] = np.zeros((NUM_ACTIONS, ADAPT_HIDDEN), dtype=np.float64)
    return rw


def _ensure_cmd_adapter(rnn):
    rng = np.random.default_rng(2000)
    if "W_adapt_h" not in rnn or np.asarray(rnn["W_adapt_h"]).shape != (ADAPT_HIDDEN, CMD_RNN_HIDDEN):
        rnn["W_adapt_h"] = rng.normal(0.0, 0.08, size=(ADAPT_HIDDEN, CMD_RNN_HIDDEN)).astype(np.float64)
    if "W_adapt_x" not in rnn or np.asarray(rnn["W_adapt_x"]).shape != (ADAPT_HIDDEN, CMD_RNN_IN):
        rnn["W_adapt_x"] = rng.normal(0.0, 0.05, size=(ADAPT_HIDDEN, CMD_RNN_IN)).astype(np.float64)
    if "b_adapt" not in rnn or np.asarray(rnn["b_adapt"]).shape != (ADAPT_HIDDEN,):
        rnn["b_adapt"] = np.zeros((ADAPT_HIDDEN,), dtype=np.float64)
    if "W_adapt_out" not in rnn or np.asarray(rnn["W_adapt_out"]).shape != (CMD_RNN_OUT, ADAPT_HIDDEN):
        rnn["W_adapt_out"] = np.zeros((CMD_RNN_OUT, ADAPT_HIDDEN), dtype=np.float64)
    return rnn


class UnitRNNModule(nn.Module):
    def __init__(self, rw, lora_state=None):
        super().__init__()
        self.use_lora = False
        base_d = (lora_state or {}).get("base", {})
        a_d = (lora_state or {}).get("A", {})
        b_d = (lora_state or {}).get("B", {})
        for k in UNIT_ALL_KEYS:
            arr = base_d[k] if k in base_d else rw[k]
            self.register_parameter(k, nn.Parameter(torch.tensor(arr, dtype=torch.float32)))
        for k in UNIT_LORA_KEYS:
            out_d, in_d = rw[k].shape
            if k in a_d and k in b_d:
                A = torch.tensor(a_d[k], dtype=torch.float32)
                B = torch.tensor(b_d[k], dtype=torch.float32)
            else:
                A = torch.randn(LORA_RANK, in_d, dtype=torch.float32) * 0.02
                B = torch.zeros(out_d, LORA_RANK, dtype=torch.float32)
            self.register_parameter(f"lora_A_{k}", nn.Parameter(A))
            self.register_parameter(f"lora_B_{k}", nn.Parameter(B))

    def _eff(self, k):
        w = getattr(self, k)
        if not self.use_lora:
            return w
        return w + LORA_SCALE * (getattr(self, f"lora_B_{k}") @ getattr(self, f"lora_A_{k}"))

    def lora_params(self):
        out = []
        for k in UNIT_LORA_KEYS:
            out.append(getattr(self, f"lora_A_{k}"))
            out.append(getattr(self, f"lora_B_{k}"))
        return out

    def forward_traj(self, X, acts, advs):
        T = X.shape[0]
        W_kb, W_kb_out = self._eff("W_kb"), self._eff("W_kb_out")
        W_top_kb, W_top_x, Wh, W2 = self._eff("W_top_kb"), self._eff("W_top_x"), self._eff("Wh"), self._eff("W2")
        W_ah, W_ax, W_ao = self._eff("W_adapt_h"), self._eff("W_adapt_x"), self._eff("W_adapt_out")
        H_kb = torch.tanh(X @ W_kb.T + self.b_kb)
        Ctx = X[:, :UNIT_CTX_IN] @ W_top_x.T
        Pre = H_kb @ W_top_kb.T + Ctx + self.b_top
        h_list = []
        hp = torch.zeros(UNIT_MLP_HIDDEN, dtype=X.dtype, device=X.device)
        for t in range(T):
            hp = torch.tanh(Pre[t] + Wh @ hp)
            h_list.append(hp)
        H_top = torch.stack(h_list, dim=0)
        H_adapt = torch.tanh(H_top @ W_ah.T + X[:, :UNIT_CTX_IN] @ W_ax.T + self.b_adapt)
        adapt_delta = torch.clamp(H_adapt @ W_ao.T, -1.0, 1.0)
        logits = (H_kb @ W_kb_out.T + self.b_kb_out) + (H_top @ W2.T) + adapt_delta
        lprobs = torch.log_softmax(logits * 2.2, dim=-1)
        chosen_lprobs = lprobs[torch.arange(T, device=X.device), acts]
        return -torch.sum(advs * chosen_lprobs)

    def forward_imitation(self, X, tgts):
        T = X.shape[0]
        W_kb, W_kb_out = self._eff("W_kb"), self._eff("W_kb_out")
        W_top_kb, W_top_x, Wh, W2 = self._eff("W_top_kb"), self._eff("W_top_x"), self._eff("Wh"), self._eff("W2")
        W_ah, W_ax, W_ao = self._eff("W_adapt_h"), self._eff("W_adapt_x"), self._eff("W_adapt_out")
        H_kb = torch.tanh(X @ W_kb.T + self.b_kb)
        Ctx = X[:, :UNIT_CTX_IN] @ W_top_x.T
        Pre = H_kb @ W_top_kb.T + Ctx + self.b_top
        h_list = []
        hp = torch.zeros(UNIT_MLP_HIDDEN, dtype=X.dtype, device=X.device)
        for t in range(T):
            hp = torch.tanh(Pre[t] + Wh @ hp)
            h_list.append(hp)
        H_top = torch.stack(h_list, dim=0)
        H_adapt = torch.tanh(H_top @ W_ah.T + X[:, :UNIT_CTX_IN] @ W_ax.T + self.b_adapt)
        adapt_delta = torch.clamp(H_adapt @ W_ao.T, -1.0, 1.0)
        logits = (H_kb @ W_kb_out.T + self.b_kb_out) + (H_top @ W2.T) + adapt_delta
        lprobs = torch.log_softmax(logits * 2.2, dim=-1)
        return -torch.sum(tgts * lprobs)


class CommanderRNNModule(nn.Module):
    def __init__(self, rnn, lora_state=None):
        super().__init__()
        self.use_lora = False
        base_d = (lora_state or {}).get("base", {})
        a_d = (lora_state or {}).get("A", {})
        b_d = (lora_state or {}).get("B", {})
        for k in CMD_ALL_KEYS:
            arr = base_d[k] if k in base_d else rnn[k]
            self.register_parameter(k, nn.Parameter(torch.tensor(arr, dtype=torch.float32)))
        for k in CMD_LORA_KEYS:
            out_d, in_d = rnn[k].shape
            if k in a_d and k in b_d:
                A = torch.tensor(a_d[k], dtype=torch.float32)
                B = torch.tensor(b_d[k], dtype=torch.float32)
            else:
                A = torch.randn(LORA_RANK, in_d, dtype=torch.float32) * 0.02
                B = torch.zeros(out_d, LORA_RANK, dtype=torch.float32)
            self.register_parameter(f"lora_A_{k}", nn.Parameter(A))
            self.register_parameter(f"lora_B_{k}", nn.Parameter(B))

    def _eff(self, k):
        w = getattr(self, k)
        if not self.use_lora:
            return w
        return w + LORA_SCALE * (getattr(self, f"lora_B_{k}") @ getattr(self, f"lora_A_{k}"))

    def lora_params(self):
        out = []
        for k in CMD_LORA_KEYS:
            out.append(getattr(self, f"lora_A_{k}"))
            out.append(getattr(self, f"lora_B_{k}"))
        return out

    def forward_traj(self, X, acts, advs):
        T = X.shape[0]
        W_kb, W_kb_out = self._eff("W_kb"), self._eff("W_kb_out")
        W_top_kb, Wx, Wh, Wo = self._eff("W_top_kb"), self._eff("Wx"), self._eff("Wh"), self._eff("Wo")
        W_ah, W_ax, W_ao = self._eff("W_adapt_h"), self._eff("W_adapt_x"), self._eff("W_adapt_out")
        H_kb = torch.tanh(X @ W_kb.T + self.b_kb)
        Pre = H_kb @ W_top_kb.T + X @ Wx.T + self.bh
        h_list = []
        hp = torch.zeros(CMD_RNN_HIDDEN, dtype=X.dtype, device=X.device)
        for t in range(T):
            hp = torch.tanh(Pre[t] + Wh @ hp)
            h_list.append(hp)
        H_top = torch.stack(h_list, dim=0)
        H_adapt = torch.tanh(H_top @ W_ah.T + X @ W_ax.T + self.b_adapt)
        adapt_delta = torch.clamp(H_adapt @ W_ao.T, -1.0, 1.0)
        logits = (H_kb @ W_kb_out.T + self.b_kb_out) + (H_top @ Wo.T) + adapt_delta
        lprobs = torch.log_softmax(logits * 2.2, dim=-1)
        chosen_lprobs = lprobs[torch.arange(T, device=X.device), acts]
        loss = -torch.sum(advs * chosen_lprobs)
        probs = torch.softmax(logits * 2.2, dim=-1)
        ent = -torch.sum(probs * lprobs)
        return loss - 0.01 * ent


_GLOBAL_TORCH = {
    "role_models": [None] * NUM_ROLES,
    "role_opts": [None] * NUM_ROLES,
    "cmd_model": None,
    "cmd_opt": None,
}


def _sync_train_state(model, opt, kbann_lr_scale, adapter_only=False, train_all=False, use_lora=False):
    top_params = opt.param_groups[0]["params"]
    kb_params = opt.param_groups[1]["params"]
    adapt_params = opt.param_groups[2]["params"]
    lora_params = opt.param_groups[3]["params"]
    if use_lora:
        model.use_lora = True
        opt.param_groups[0]["lr"] = 0.0
        opt.param_groups[1]["lr"] = 0.0
        opt.param_groups[2]["lr"] = 0.0
        opt.param_groups[3]["lr"] = LORA_LR
        for p in top_params + kb_params + adapt_params:
            p.requires_grad = False
        for p in lora_params:
            p.requires_grad = True
    elif train_all:
        model.use_lora = False
        unfrozen_kb = (kbann_lr_scale > 0.0)
        opt.param_groups[0]["lr"] = LR_BASE
        opt.param_groups[1]["lr"] = LR_BASE * kbann_lr_scale if unfrozen_kb else 0.0
        opt.param_groups[2]["lr"] = ADAPT_LR
        opt.param_groups[3]["lr"] = 0.0
        for p in top_params:
            p.requires_grad = True
        for p in kb_params:
            p.requires_grad = unfrozen_kb
        for p in adapt_params:
            p.requires_grad = True
        for p in lora_params:
            p.requires_grad = False
    elif adapter_only:
        model.use_lora = False
        opt.param_groups[0]["lr"] = 0.0
        opt.param_groups[1]["lr"] = 0.0
        opt.param_groups[2]["lr"] = ADAPT_LR
        opt.param_groups[3]["lr"] = 0.0
        for p in top_params + kb_params + lora_params:
            p.requires_grad = False
        for p in adapt_params:
            p.requires_grad = True
    else:
        model.use_lora = False
        unfrozen_kb = (kbann_lr_scale > 0.0)
        opt.param_groups[0]["lr"] = LR_BASE
        opt.param_groups[1]["lr"] = LR_BASE * kbann_lr_scale if unfrozen_kb else 0.0
        opt.param_groups[2]["lr"] = 0.0
        opt.param_groups[3]["lr"] = 0.0
        for p in top_params:
            p.requires_grad = True
        for p in kb_params:
            p.requires_grad = unfrozen_kb
        for p in adapt_params + lora_params:
            p.requires_grad = False


def _get_role_engine(ri, rw, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False):
    _ensure_unit_adapter(rw, ri)
    model = _GLOBAL_TORCH["role_models"][ri]
    opt = _GLOBAL_TORCH["role_opts"][ri]
    if model is None:
        model = UnitRNNModule(rw, lora_state=rw.get("_lora"))
        top_params = [model.W_top_kb, model.W_top_x, model.Wh, model.b_top, model.W2]
        kb_params = [model.W_kb, model.b_kb, model.W_kb_out, model.b_kb_out]
        adapt_params = [model.W_adapt_h, model.W_adapt_x, model.b_adapt, model.W_adapt_out]
        lora_params = model.lora_params()
        opt = optim.AdamW([
            {"params": top_params, "lr": LR_BASE, "weight_decay": WEIGHT_DECAY},
            {"params": kb_params, "lr": 0.0, "weight_decay": WEIGHT_DECAY},
            {"params": adapt_params, "lr": 0.0, "weight_decay": ADAPT_WEIGHT_DECAY},
            {"params": lora_params, "lr": 0.0, "weight_decay": LORA_WEIGHT_DECAY},
        ], betas=(0.9, 0.999), eps=1e-8)
        _GLOBAL_TORCH["role_models"][ri] = model
        _GLOBAL_TORCH["role_opts"][ri] = opt
    _sync_train_state(model, opt, kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    return model, opt


def _get_cmd_engine(rnn, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False):
    _ensure_cmd_adapter(rnn)
    model = _GLOBAL_TORCH["cmd_model"]
    opt = _GLOBAL_TORCH["cmd_opt"]
    if model is None:
        model = CommanderRNNModule(rnn, lora_state=rnn.get("_lora"))
        top_params = [model.W_top_kb, model.Wx, model.Wh, model.bh, model.Wo]
        kb_params = [model.W_kb, model.b_kb, model.W_kb_out, model.b_kb_out]
        adapt_params = [model.W_adapt_h, model.W_adapt_x, model.b_adapt, model.W_adapt_out]
        lora_params = model.lora_params()
        opt = optim.AdamW([
            {"params": top_params, "lr": LR_BASE, "weight_decay": WEIGHT_DECAY},
            {"params": kb_params, "lr": 0.0, "weight_decay": WEIGHT_DECAY},
            {"params": adapt_params, "lr": 0.0, "weight_decay": ADAPT_WEIGHT_DECAY},
            {"params": lora_params, "lr": 0.0, "weight_decay": LORA_WEIGHT_DECAY},
        ], betas=(0.9, 0.999), eps=1e-8)
        _GLOBAL_TORCH["cmd_model"] = model
        _GLOBAL_TORCH["cmd_opt"] = opt
    _sync_train_state(model, opt, kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    return model, opt


def new_weights():
    roles = build_kbann_unit_rnns()
    for ri, rw in enumerate(roles):
        _ensure_unit_adapter(rw, ri)
    return roles


def _lora_sidecar(path):
    return path + ".lora.npz"


def _load_lora_roles(path, roles):
    sc = _lora_sidecar(path)
    if not os.path.exists(sc):
        return
    try:
        z = np.load(sc)
        for ri, rw in enumerate(roles):
            base_d, a_d, b_d = {}, {}, {}
            for k in UNIT_ALL_KEYS:
                bk = f"r{ri}_base_{k}"
                if bk in z:
                    base_d[k] = z[bk].astype(np.float64)
            for k in UNIT_LORA_KEYS:
                ak, bk = f"r{ri}_A_{k}", f"r{ri}_B_{k}"
                if ak in z and bk in z:
                    a_d[k] = z[ak].astype(np.float64)
                    b_d[k] = z[bk].astype(np.float64)
            if base_d or a_d:
                rw["_lora"] = {"base": base_d, "A": a_d, "B": b_d}
    except Exception:
        pass


def _load_lora_cmd(path, rnn):
    sc = _lora_sidecar(path)
    if not os.path.exists(sc):
        return
    try:
        z = np.load(sc)
        base_d, a_d, b_d = {}, {}, {}
        for k in CMD_ALL_KEYS:
            bk = f"base_{k}"
            if bk in z:
                base_d[k] = z[bk].astype(np.float64)
        for k in CMD_LORA_KEYS:
            ak, bk = f"A_{k}", f"B_{k}"
            if ak in z and bk in z:
                a_d[k] = z[ak].astype(np.float64)
                b_d[k] = z[bk].astype(np.float64)
        if base_d or a_d:
            rnn["_lora"] = {"base": base_d, "A": a_d, "B": b_d}
    except Exception:
        pass


def load_weights(path):
    if os.path.exists(path):
        try:
            data = json.load(open(path))
            if isinstance(data, list) and len(data) == NUM_ROLES and isinstance(data[0], dict):
                roles = []
                ok = True
                for ri, d in enumerate(data):
                    if not all(k in d for k in UNIT_KEYS):
                        ok = False
                        break
                    rw = {k: np.array(d[k], dtype=np.float64) for k in UNIT_KEYS}
                    for k in ADAPT_KEYS:
                        if k in d:
                            rw[k] = np.array(d[k], dtype=np.float64)
                    if (
                        rw["W_kb"].shape != (UNIT_KB_HIDDEN, NUM_FEATURES)
                        or rw["b_kb"].shape != (UNIT_KB_HIDDEN,)
                        or rw["W_kb_out"].shape != (NUM_ACTIONS, UNIT_KB_HIDDEN)
                        or rw["b_kb_out"].shape != (NUM_ACTIONS,)
                        or rw["W_top_kb"].shape != (UNIT_MLP_HIDDEN, UNIT_KB_HIDDEN)
                        or rw["W_top_x"].shape != (UNIT_MLP_HIDDEN, UNIT_CTX_IN)
                        or rw["Wh"].shape != (UNIT_MLP_HIDDEN, UNIT_MLP_HIDDEN)
                        or rw["b_top"].shape != (UNIT_MLP_HIDDEN,)
                        or rw["W2"].shape != (NUM_ACTIONS, UNIT_MLP_HIDDEN)
                    ):
                        ok = False
                        break
                    _ensure_unit_adapter(rw, ri)
                    roles.append(rw)
                if ok:
                    _load_lora_roles(path, roles)
                    # Invalidate cached models to re-sync with loaded weights
                    for i in range(NUM_ROLES):
                        _GLOBAL_TORCH["role_models"][i] = None
                        _GLOBAL_TORCH["role_opts"][i] = None
                    return roles
        except Exception:
            pass
    return new_weights()


def new_commander_rnn():
    return _ensure_cmd_adapter(build_kbann_commander_rnn())


def load_commander_rnn(path):
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            if isinstance(d, dict) and all(k in d for k in CMD_KEYS):
                rnn = {k: np.array(d[k], dtype=np.float64) for k in CMD_KEYS}
                for k in ADAPT_KEYS:
                    if k in d:
                        rnn[k] = np.array(d[k], dtype=np.float64)
                if (
                    rnn["W_kb"].shape == (CMD_KB_HIDDEN, CMD_RNN_IN)
                    and rnn["b_kb"].shape == (CMD_KB_HIDDEN,)
                    and rnn["W_kb_out"].shape == (CMD_RNN_OUT, CMD_KB_HIDDEN)
                    and rnn["b_kb_out"].shape == (CMD_RNN_OUT,)
                    and rnn["W_top_kb"].shape == (CMD_RNN_HIDDEN, CMD_KB_HIDDEN)
                    and rnn["Wx"].shape == (CMD_RNN_HIDDEN, CMD_RNN_IN)
                    and rnn["Wh"].shape == (CMD_RNN_HIDDEN, CMD_RNN_HIDDEN)
                    and rnn["bh"].shape == (CMD_RNN_HIDDEN,)
                    and rnn["Wo"].shape == (CMD_RNN_OUT, CMD_RNN_HIDDEN)
                ):
                    _ensure_cmd_adapter(rnn)
                    _load_lora_cmd(path, rnn)
                    _GLOBAL_TORCH["cmd_model"] = None
                    _GLOBAL_TORCH["cmd_opt"] = None
                    return rnn
        except Exception:
            pass
    return new_commander_rnn()


def save_commander_rnn(rnn, path):
    tmp = path + ".tmp"
    out = {k: np.round(v, 6).tolist() for k, v in rnn.items() if not k.startswith("_")}
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)
    if "_lora" in rnn:
        arrs = {}
        ld = rnn["_lora"]
        for k, v in ld.get("base", {}).items():
            arrs[f"base_{k}"] = np.asarray(v, dtype=np.float32)
        for k, v in ld.get("A", {}).items():
            arrs[f"A_{k}"] = np.asarray(v, dtype=np.float32)
        for k, v in ld.get("B", {}).items():
            arrs[f"B_{k}"] = np.asarray(v, dtype=np.float32)
        sc = _lora_sidecar(path)
        tmp_sc = sc + ".tmp.npz"
        np.savez(tmp_sc, **arrs)
        os.replace(tmp_sc, sc)


def save_weights(w, path):
    tmp = path + ".tmp"
    out = [{k: np.round(v, 6).tolist() for k, v in rw.items() if not k.startswith("_")} for rw in w]
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)
    if any("_lora" in rw for rw in w):
        arrs = {}
        for ri, rw in enumerate(w):
            ld = rw.get("_lora")
            if not ld:
                continue
            for k, v in ld.get("base", {}).items():
                arrs[f"r{ri}_base_{k}"] = np.asarray(v, dtype=np.float32)
            for k, v in ld.get("A", {}).items():
                arrs[f"r{ri}_A_{k}"] = np.asarray(v, dtype=np.float32)
            for k, v in ld.get("B", {}).items():
                arrs[f"r{ri}_B_{k}"] = np.asarray(v, dtype=np.float32)
        sc = _lora_sidecar(path)
        tmp_sc = sc + ".tmp.npz"
        np.savez(tmp_sc, **arrs)
        os.replace(tmp_sc, sc)


def step_reward(step, phase, terminal=0.0, llm_penalty=0.0):
    r = (W_LOC * step.get("dk", 0.0) - W_LOSS * step.get("dl", 0.0) + W_COH * step.get("coh", 0.0))
    return r + terminal + llm_penalty


def _softmax(z, inv_temp=2.2):
    z_shift = (z - np.max(z)) * inv_temp
    e = np.exp(np.clip(z_shift, -30.0, 30.0))
    return e / np.sum(e)


def init_role_grads(w):
    for ri in range(NUM_ROLES):
        _, opt = _get_role_engine(ri, w[ri])
        opt.zero_grad()
    return [{"cnt": 0} for _ in range(NUM_ROLES)]


def init_cmd_grads(rnn):
    _, opt = _get_cmd_engine(rnn)
    opt.zero_grad()
    return {"cnt": 0}


def _terminal_components(ep):
    result = ep.get("result", 0)
    dmg_frac = float(ep.get("enemy_damage_frac", 0.0))
    dmg_bonus = DMG_BONUS * dmg_frac
    is_dec = bool(ep.get("decisive_win"))
    is_dec_loss = bool(ep.get("decisive_loss"))
    if result > 0 and is_dec:
        decisive_bonus = DECISIVE_BONUS * max(0.5, dmg_frac)
    elif result <= 0 and is_dec_loss:
        decisive_bonus = -0.5 * DECISIVE_BONUS
    else:
        decisive_bonus = 0.0
    term_base = TERMINAL * result
    return result, term_base, dmg_bonus, decisive_bonus


P1_STATS = {"loss": 0.0, "steps": 0}


def update_phase1(w, ep, kbann_lr_scale=0.0, apply_in_place=True, role_grads=None, adapter_only=False, train_all=False, use_lora=False):
    """Phase 1 warm start: train 3-Layer Unit RNN via cross-entropy imitation."""
    n = 0
    if role_grads is None:
        role_grads = init_role_grads(w)
    for u in ep["units"]:
        if u["role"] not in ROLES or u["role"] == "commander" or not u.get("steps"):
            continue
        ri = ROLES.index(u["role"])
        model, _ = _get_role_engine(ri, w[ri], kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
        steps = u["steps"]
        T = len(steps)
        X = torch.zeros(T, NUM_FEATURES, dtype=torch.float32)
        tgts = torch.zeros(T, NUM_ACTIONS, dtype=torch.float32)
        for t, s in enumerate(steps):
            for idx, val in s["phi"]:
                if 0 <= idx < NUM_FEATURES:
                    X[t, idx] = float(val)
            raw_r = np.array([imitation_reward(a, s.get("disp"), s.get("vq")) for a in range(NUM_ACTIONS)], dtype=np.float32)
            tgts[t] = torch.tensor(_softmax(raw_r * 2.0), dtype=torch.float32)

        loss = model.forward_imitation(X, tgts)
        P1_STATS["loss"] += float(loss.item())
        P1_STATS["steps"] += T
        scale = 1.0 / max(1.0, float(T) ** 0.5)
        (scale * loss).backward()
        role_grads[ri]["cnt"] += 1
        n += 1

    if apply_in_place:
        apply_role_grads(w, role_grads, kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    return n


def update_from_episode(
    w, ep, phase=2, penalties=None, llm_terminal_penalty=0.0,
    weight_mult=1.0, apply_in_place=True, role_grads=None, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False,
):
    """Train each role's 3-Layer Unit RNN (or LoRA adapters when use_lora=True) with PyTorch Autograd BPTT + AdamW."""
    if phase == 1:
        return update_phase1(
            w, ep, kbann_lr_scale=kbann_lr_scale,
            apply_in_place=apply_in_place, role_grads=role_grads, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora,
        )
    result, term_base, dmg_bonus, decisive_bonus = _terminal_components(ep)
    tick = ep.get("tick_seconds", 3.0)
    n_updates = 0

    if role_grads is None:
        role_grads = init_role_grads(w)

    for u in ep["units"]:
        role, steps = u["role"], u["steps"]
        if role == "commander" or role not in ROLES or not steps:
            continue
        ri = ROLES.index(role)
        model, _ = _get_role_engine(ri, w[ri], kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
        T = len(steps)
        rewards = np.zeros(T, dtype=np.float64)

        for t, s in enumerate(steps):
            last = (t == T - 1)
            terminal = (term_base + dmg_bonus + decisive_bonus + LLM_MULT * llm_terminal_penalty) if last else 0.0
            pen = 0.0
            for p in penalties or []:
                if p["role"] in (role, "any") and p["action"] == s["a"] and p["t_start"] <= t * tick <= p["t_end"]:
                    pen += p["penalty"]
            rewards[t] = step_reward(s, phase, terminal, pen)

        term_total = term_base + dmg_bonus + decisive_bonus + LLM_MULT * llm_terminal_penalty
        local_ret = np.zeros(T, dtype=np.float64)
        g_loc = 0.0
        for t in range(T - 1, -1, -1):
            r_loc = rewards[t] - (term_total if t == T - 1 else 0.0)
            g_loc = r_loc + 0.85 * g_loc
            local_ret[t] = g_loc
        loc_adv = np.clip(local_ret / 4.0, -2.0, 2.0)
        ep_adv = np.clip(term_total / 220.0, -1.5, 1.8)
        adv = np.clip(0.55 * loc_adv + ep_adv, -3.0, 3.0)

        # Build trajectory tensors
        X = torch.zeros(T, NUM_FEATURES, dtype=torch.float32)
        acts = torch.tensor([int(s["a"]) for s in steps], dtype=torch.long)
        advs = torch.tensor(adv, dtype=torch.float32)
        for t, s in enumerate(steps):
            for idx, val in s["phi"]:
                if 0 <= idx < NUM_FEATURES:
                    X[t, idx] = float(val)

        loss = model.forward_traj(X, acts, advs)
        scale = float(weight_mult) / max(1.0, float(T) ** 0.5)
        (scale * loss).backward()
        role_grads[ri]["cnt"] += 1
        n_updates += 1

    if apply_in_place:
        apply_role_grads(w, role_grads, kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    return n_updates


def apply_role_grads(w, role_grads, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False):
    for ri in range(NUM_ROLES):
        rg = role_grads[ri]
        if rg["cnt"] <= 0:
            continue
        model, opt = _get_role_engine(ri, w[ri], kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
        if use_lora:
            for p in opt.param_groups[0]["params"] + opt.param_groups[1]["params"] + opt.param_groups[2]["params"]:
                p.grad = None
        elif train_all:
            for p in opt.param_groups[3]["params"]:
                p.grad = None
            if kbann_lr_scale <= 0.0:
                for p in opt.param_groups[1]["params"]:
                    p.grad = None
        elif adapter_only:
            for p in opt.param_groups[0]["params"] + opt.param_groups[1]["params"] + opt.param_groups[3]["params"]:
                p.grad = None
        else:
            for p in opt.param_groups[2]["params"] + opt.param_groups[3]["params"]:
                p.grad = None
            if kbann_lr_scale <= 0.0:
                for p in opt.param_groups[1]["params"]:
                    p.grad = None

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        opt.step()
        opt.zero_grad()
        # Sync updated parameters back to numpy dictionary (merged effective weights when use_lora=True)
        if use_lora:
            base_d, a_d, b_d = {}, {}, {}
            for k in UNIT_ALL_KEYS:
                base_np = getattr(model, k).detach().cpu().numpy().astype(np.float64)
                base_d[k] = base_np
                if k in UNIT_LORA_KEYS:
                    eff_np = model._eff(k).detach().cpu().numpy().astype(np.float64)
                    w[ri][k][:] = eff_np
                    a_d[k] = getattr(model, f"lora_A_{k}").detach().cpu().numpy().astype(np.float64)
                    b_d[k] = getattr(model, f"lora_B_{k}").detach().cpu().numpy().astype(np.float64)
                else:
                    w[ri][k][:] = base_np
            w[ri]["_lora"] = {"base": base_d, "A": a_d, "B": b_d}
        else:
            sync_keys = ADAPT_KEYS if (adapter_only and not train_all) else UNIT_ALL_KEYS
            for k in sync_keys:
                w[ri][k][:] = getattr(model, k).detach().cpu().numpy().astype(np.float64)


def _dense_cmd(sparse):
    x = np.zeros(CMD_RNN_IN, dtype=np.float64)
    for idx, val in sparse:
        if 0 <= idx < CMD_RNN_IN:
            x[idx] = val
    return x


def update_commander_rnn(
    rnn, ep, phase=2, llm_terminal_penalty=0.0,
    weight_mult=1.0, apply_in_place=True, cmd_grads=None, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False,
):
    """Train the 3-Layer Commander RNN (or LoRA adapters when use_lora=True) with PyTorch Autograd BPTT + AdamW."""
    result, term_base, dmg_bonus, decisive_bonus = _terminal_components(ep)
    cmd_units = [u for u in ep["units"] if u["role"] == "commander" and u.get("steps")]
    if not cmd_units:
        return 0
    n_updates = 0

    if cmd_grads is None:
        cmd_grads = init_cmd_grads(rnn)

    model, _ = _get_cmd_engine(rnn, kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)

    for u in cmd_units:
        steps = u["steps"]
        T = len(steps)
        rewards = np.zeros(T, dtype=np.float64)

        for t, s in enumerate(steps):
            last = (t == T - 1)
            terminal = (term_base + dmg_bonus + decisive_bonus + LLM_MULT * llm_terminal_penalty) if last else 0.0
            rewards[t] = step_reward(s, phase, terminal, 0.0)

        term_total = term_base + dmg_bonus + decisive_bonus + LLM_MULT * llm_terminal_penalty
        local_ret = np.zeros(T, dtype=np.float64)
        g_loc = 0.0
        for t in range(T - 1, -1, -1):
            r_loc = rewards[t] - (term_total if t == T - 1 else 0.0)
            g_loc = r_loc + 0.85 * g_loc
            local_ret[t] = g_loc
        loc_adv = np.clip(local_ret / 4.0, -2.0, 2.0)
        ep_adv = np.clip(term_total / 220.0, -1.5, 1.8)
        adv = np.clip(0.55 * loc_adv + ep_adv, -3.0, 3.0)

        X = torch.zeros(T, CMD_RNN_IN, dtype=torch.float32)
        acts = torch.tensor([int(s["a"]) for s in steps], dtype=torch.long)
        advs = torch.tensor(adv, dtype=torch.float32)
        for t, s in enumerate(steps):
            for idx, val in s["phi"]:
                if 0 <= idx < CMD_RNN_IN:
                    X[t, idx] = float(val)

        loss = model.forward_traj(X, acts, advs)
        scale = float(weight_mult) / (max(1, len(cmd_units)) ** 0.5 * max(1.0, float(T) ** 0.5))
        (scale * loss).backward()
        cmd_grads["cnt"] += 1
        n_updates += 1

    if apply_in_place:
        apply_cmd_grads(rnn, cmd_grads, kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    return n_updates


def apply_cmd_grads(rnn, cmd_grads, kbann_lr_scale=0.0, adapter_only=False, train_all=False, use_lora=False):
    if cmd_grads["cnt"] <= 0:
        return
    model, opt = _get_cmd_engine(rnn, kbann_lr_scale=kbann_lr_scale, adapter_only=adapter_only, train_all=train_all, use_lora=use_lora)
    if use_lora:
        for p in opt.param_groups[0]["params"] + opt.param_groups[1]["params"] + opt.param_groups[2]["params"]:
            p.grad = None
    elif train_all:
        for p in opt.param_groups[3]["params"]:
            p.grad = None
        if kbann_lr_scale <= 0.0:
            for p in opt.param_groups[1]["params"]:
                p.grad = None
    elif adapter_only:
        for p in opt.param_groups[0]["params"] + opt.param_groups[1]["params"] + opt.param_groups[3]["params"]:
            p.grad = None
    else:
        for p in opt.param_groups[2]["params"] + opt.param_groups[3]["params"]:
            p.grad = None
        if kbann_lr_scale <= 0.0:
            for p in opt.param_groups[1]["params"]:
                p.grad = None

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
    opt.step()
    opt.zero_grad()
    if use_lora:
        base_d, a_d, b_d = {}, {}, {}
        for k in CMD_ALL_KEYS:
            base_np = getattr(model, k).detach().cpu().numpy().astype(np.float64)
            base_d[k] = base_np
            if k in CMD_LORA_KEYS:
                eff_np = model._eff(k).detach().cpu().numpy().astype(np.float64)
                rnn[k][:] = eff_np
                a_d[k] = getattr(model, f"lora_A_{k}").detach().cpu().numpy().astype(np.float64)
                b_d[k] = getattr(model, f"lora_B_{k}").detach().cpu().numpy().astype(np.float64)
            else:
                rnn[k][:] = base_np
        rnn["_lora"] = {"base": base_d, "A": a_d, "B": b_d}
    else:
        sync_keys = ADAPT_KEYS if (adapter_only and not train_all) else CMD_ALL_KEYS
        for k in sync_keys:
            rnn[k][:] = getattr(model, k).detach().cpu().numpy().astype(np.float64)
