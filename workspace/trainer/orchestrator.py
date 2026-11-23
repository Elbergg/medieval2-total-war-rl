"""Training orchestrator (runs in medieval2-trainer). Watches episodes/, updates weights, runs curriculum + judge.

IPC layout under /workspace:
  scenario.json            next battle (written here, consumed by the game container)
  weights_latest.json      learner weights [roles][actions][features]
  weights_frozen.json      phase-3 opponent
  episodes/episode_N.json  written by Lua after each battle
  replays/replay_N.json    compact top-down replay
  state.json               phase / battle counter / rolling results
"""
import glob
import json
import os
import shutil
import time
from collections import deque

import sarsa_trainer as st
import scenario_generator as sg

ROOT = os.environ.get("WORKSPACE", "/workspace")
P = lambda *a: os.path.join(ROOT, *a)
P2_TO_P3_WINRATE, P3_TO_P4_WINRATE, P4_PROMOTE_WINRATE, JUDGE_EVERY = 0.75, 0.60, 0.55, 5
MODES = {
    2: "rl_vs_vanilla",
    3: "rl_vs_vanilla_disadvantaged",
    4: "rl_vs_frozen",
}


def load_state():
    if os.path.exists(P("state.json")):
        s = json.load(open(P("state.json")))
        s["phase"] = max(2, int(s.get("phase", 2)))
        s["recent"] = deque(s["recent"], maxlen=50)
        s["recent_frozen"] = deque(s.get("recent_frozen", []), maxlen=100)
        return s
    return {"phase": 2, "battle": 0, "recent": deque(maxlen=50), "recent_frozen": deque(maxlen=100)}


def save_state(s):
    d = dict(s, recent=list(s["recent"]), recent_frozen=list(s["recent_frozen"]))
    tmp = P("state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, P("state.json"))


def winrate(q):
    return sum(1 for r in q if r > 0) / len(q) if q else 0.0


def main():
    for d in ("episodes", "replays", "processed"):
        os.makedirs(P(d), exist_ok=True)
    s = load_state()
    w = st.load_weights(P("weights_latest.json"))
    if not os.path.exists(P("weights_latest.json")):
        st.save_weights(w, P("weights_latest.json"))
    rnn = st.load_commander_rnn(P("commander_rnn.json"))
    if not os.path.exists(P("commander_rnn.json")):
        st.save_commander_rnn(rnn, P("commander_rnn.json"))
    if not os.path.exists(P("scenario.json")):
        sg.write_scenario(P("scenario.json"), s["battle"] + 1, MODES[s["phase"]])
    BATCH_SIZE = 10
    batch_cnt = 0
    role_grads = st.init_role_grads(w)
    cmd_grads = st.init_cmd_grads(rnn)
    while True:
        ep_files = sorted(
            glob.glob(P("episodes", "episode_*.json")),
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]) if os.path.basename(p).split("_")[-1].split(".")[0].isdigit() else 0,
        )
        for path in ep_files:
            try:
                ep = json.load(open(path))
            except json.JSONDecodeError:
                continue  # still being written
            n = int(ep.get("battle_id") or (s["battle"] + 1))
            s["battle"] = max(s["battle"], n)
            penalties = None
            llm_terminal = 0.0
            is_judge_battle = bool(ep.get("long_battle")) or (n % JUDGE_EVERY == 0)
            weight_mult = 3.0 if is_judge_battle else 1.0
            rep_name = ep.get("replay_file") or f"replay_{ep.get('battle_id', n)}.json"
            replay = P("replays", rep_name)
            if is_judge_battle and os.path.exists(replay):
                try:
                    import llm_judge
                    verdict = llm_judge.judge(replay, battle_id=n)
                    if verdict is not None:
                        json.dump(verdict, open(P("processed", f"verdict_{n}.json"), "w"), indent=1)
                        llm_terminal = verdict["penalties"]["total"]
                        print(
                            f"llm_judge battle {n} (long={ep.get('long_battle')}, weight_mult={weight_mult}x): "
                            f"fair={verdict['fairness']}/10 hist={verdict['historical_accuracy']}/10 "
                            f"reas={verdict['reasonability']}/10 penalty={llm_terminal}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"llm_judge battle {n} failed: {e}", flush=True)
            updates = st.update_from_episode(
                w, ep, s["phase"], penalties, llm_terminal, weight_mult,
                apply_in_place=False, role_grads=role_grads, kbann_lr_scale=0.0, adapter_only=True,
            )
            rnn_updates = st.update_commander_rnn(
                rnn, ep, s["phase"], llm_terminal, weight_mult,
                apply_in_place=False, cmd_grads=cmd_grads, kbann_lr_scale=0.0, adapter_only=True,
            )
            batch_cnt += 1
            if batch_cnt >= BATCH_SIZE:
                st.apply_role_grads(w, role_grads, kbann_lr_scale=0.0, adapter_only=True)
                st.apply_cmd_grads(rnn, cmd_grads, kbann_lr_scale=0.0, adapter_only=True)
                st.save_weights(w, P("weights_latest.json"))
                st.save_commander_rnn(rnn, P("commander_rnn.json"))
                u_norm = max(float(abs(rw["W_adapt_out"]).max()) for rw in w)
                c_norm = float(abs(rnn["W_adapt_out"]).max())
                print(
                    f"[Batch Update @ battle {n}] Applied {batch_cnt}-battle accumulated adapter grads "
                    f"(max|W_adapt_out| unit={u_norm:.4f}, cmd={c_norm:.4f})",
                    flush=True,
                )
                role_grads = st.init_role_grads(w)
                cmd_grads = st.init_cmd_grads(rnn)
                batch_cnt = 0
            # Milestone checkpointing every 10 battles + dedicated 50-iteration snapshots
            if n % 10 == 0 or n == 1:
                os.makedirs(P("checkpoints"), exist_ok=True)
                shutil.copy(P("weights_latest.json"), P("checkpoints", f"weights_battle_{n}.json"))
                shutil.copy(P("commander_rnn.json"), P("checkpoints", f"commander_rnn_battle_{n}.json"))
            if n % 50 == 0:
                os.makedirs(P("checkpoints"), exist_ok=True)
                shutil.copy(P("weights_latest.json"), P("checkpoints", f"weights_iter_{n}.json"))
                shutil.copy(P("commander_rnn.json"), P("checkpoints", f"commander_rnn_iter_{n}.json"))
            (s["recent_frozen"] if s["phase"] == 4 else s["recent"]).append(ep.get("result", 0))
            # curriculum (starts directly in phase 2: rl_vs_vanilla)
            if s["phase"] == 2 and len(s["recent"]) >= 50 and winrate(s["recent"]) > P2_TO_P3_WINRATE:
                s["phase"] = 3
                s["recent"].clear()
            elif s["phase"] == 3 and len(s["recent"]) >= 50 and winrate(s["recent"]) > P3_TO_P4_WINRATE:
                s["phase"] = 4
                shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
                shutil.copy(P("commander_rnn.json"), P("commander_rnn_frozen.json"))
            elif s["phase"] == 4 and n % 100 == 0 and winrate(s["recent_frozen"]) > P4_PROMOTE_WINRATE:
                shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
                shutil.copy(P("commander_rnn.json"), P("commander_rnn_frozen.json"))
            shutil.move(path, P("processed", os.path.basename(path)))
            sg.write_scenario(P("scenario.json"), n + 1, MODES.get(s["phase"], "rl_vs_vanilla"))
            save_state(s)
            print(f"battle {n} phase {s['phase']} updates {updates} rnn_updates {rnn_updates} result {ep.get('result')}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
