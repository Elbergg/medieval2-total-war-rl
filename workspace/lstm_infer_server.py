#!/usr/bin/env python3
"""Persistent lightweight PyTorch inference server for Full-Army LSTM inside M2TW Docker containers.

Watches workspace/infer_req_{1,2,3}.json, runs FullArmyLSTM.act_step() (<0.5ms on CPU),
maintains per-worker LSTM hidden state (h_t, c_t) keyed by battle_id, and atomically
writes workspace/infer_res_{wid}.json. Hot-reloads weights_lstm.pt whenever mtime changes.
"""
import json
import os
import sys
import time
import numpy as np
import torch

WORKSPACE = "/home/tomek/medieval2-docker/workspace"
sys.path.insert(0, os.path.join(WORKSPACE, "trainer"))
from lstm_model import MAX_UNITS, OBS_DIM, load_lstm  # noqa: E402


def main():
    torch.set_num_threads(2)
    weights_path = os.path.join(WORKSPACE, "weights_lstm.pt")
    model, _ = load_lstm(weights_path)
    model.eval()
    last_mtime = os.path.getmtime(weights_path) if os.path.exists(weights_path) else 0.0

    # Per-worker LSTM hidden state: wid -> (battle_id, hx)
    worker_states = {}
    wids = ("1", "2", "3")
    print(f"[lstm_infer_server] Watching {WORKSPACE}/infer_req_{{1,2,3}}.json (weights={weights_path})", flush=True)

    last_check = 0.0
    while True:
        now = time.time()
        if now - last_check >= 1.0:
            last_check = now
            if os.path.exists(weights_path):
                mt = os.path.getmtime(weights_path)
                if mt > last_mtime + 1e-3:
                    try:
                        model, _ = load_lstm(weights_path)
                        model.eval()
                        last_mtime = mt
                        print(f"[lstm_infer_server] Reloaded {weights_path}", flush=True)
                    except Exception as e:
                        print(f"[lstm_infer_server] Reload failed: {e}", flush=True)

        handled = False
        for wid in wids:
            req_p = os.path.join(WORKSPACE, f"infer_req_{wid}.json")
            if not os.path.exists(req_p):
                continue
            try:
                raw = open(req_p, "r").read()
                if not raw:
                    continue
                req = json.loads(raw)
            except Exception:
                continue

            try:
                os.remove(req_p)
            except OSError:
                pass

            bid = req.get("battle_id", 0)
            tick = req.get("tick", 0)
            obs = np.asarray(req["obs"], dtype=np.float32)
            mask = np.asarray(req["mask"], dtype=np.float32)
            inv_temp = float(req.get("inv_temp", 1.0))
            greedy = bool(req.get("greedy", False))

            prev_bid, hx = worker_states.get(wid, (None, None))
            if prev_bid != bid or tick == 0:
                hx = None

            if obs.shape[0] == OBS_DIM and mask.shape[0] == MAX_UNITS:
                acts, hx_new = model.act_step(obs, mask, hx=hx, inv_temp=inv_temp, greedy=greedy)
                worker_states[wid] = (bid, hx_new)
            else:
                acts = [8] * MAX_UNITS

            res_p = os.path.join(WORKSPACE, f"infer_res_{wid}.json")
            tmp_p = res_p + ".tmp"
            with open(tmp_p, "w") as f:
                json.dump({"battle_id": bid, "tick": tick, "actions": acts}, f)
            os.replace(tmp_p, res_p)
            handled = True

        if not handled:
            time.sleep(0.003)


if __name__ == "__main__":
    main()
