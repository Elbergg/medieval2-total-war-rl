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
PHASE1_BATTLES = 50
P2_TO_P3_WINRATE, P3_TO_P4_WINRATE, P4_PROMOTE_WINRATE, JUDGE_EVERY = 0.75, 0.60, 0.55, 10
MODES = {
    1: "vanilla_vs_vanilla",
    2: "rl_vs_vanilla",
    3: "rl_vs_vanilla_disadvantaged",
    4: "rl_vs_frozen",
}


def load_state():
    if os.path.exists(P("state.json")):
        s = json.load(open(P("state.json")))
        s["recent"] = deque(s["recent"], maxlen=50)
        return s
    return {"phase": 1, "battle": 0, "recent": deque(maxlen=50), "recent_frozen": deque(maxlen=100)}


def save_state(s):
    d = dict(s, recent=list(s["recent"]), recent_frozen=list(s["recent_frozen"]))
    json.dump(d, open(P("state.json"), "w"))


def winrate(q):
    return sum(1 for r in q if r > 0) / len(q) if q else 0.0


def main():
    for d in ("episodes", "replays", "processed"):
        os.makedirs(P(d), exist_ok=True)
    s = load_state()
    w = st.load_weights(P("weights_latest.json"))
    if not os.path.exists(P("weights_latest.json")):
        st.save_weights(w, P("weights_latest.json"))
    if not os.path.exists(P("scenario.json")):
        sg.write_scenario(P("scenario.json"), s["battle"] + 1, MODES[s["phase"]])
    while True:
        for path in sorted(glob.glob(P("episodes", "episode_*.json"))):
            try:
                ep = json.load(open(path))
            except json.JSONDecodeError:
                continue  # still being written
            s["battle"] += 1
            n = s["battle"]
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
            updates = st.update_from_episode(w, ep, s["phase"], penalties, llm_terminal, weight_mult)
            st.save_weights(w, P("weights_latest.json"))
            # Milestone checkpointing every 10 battles
            if n % 10 == 0 or n == 1:
                os.makedirs(P("checkpoints"), exist_ok=True)
                shutil.copy(P("weights_latest.json"), P("checkpoints", f"weights_battle_{n}.json"))
            (s["recent_frozen"] if s["phase"] == 4 else s["recent"]).append(ep.get("result", 0))
            # curriculum
            if s["phase"] == 1 and n >= PHASE1_BATTLES:
                s["phase"] = 2  # imitation disappears entirely
            elif s["phase"] == 2 and len(s["recent"]) >= 50 and winrate(s["recent"]) > P2_TO_P3_WINRATE:
                s["phase"] = 3
                s["recent"].clear()
            elif s["phase"] == 3 and len(s["recent"]) >= 50 and winrate(s["recent"]) > P3_TO_P4_WINRATE:
                s["phase"] = 4
                shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
            elif s["phase"] == 4 and n % 100 == 0 and winrate(s["recent_frozen"]) > P4_PROMOTE_WINRATE:
                shutil.copy(P("weights_latest.json"), P("weights_frozen.json"))
            shutil.move(path, P("processed", os.path.basename(path)))
            sg.write_scenario(P("scenario.json"), n + 1, MODES[s["phase"]])
            save_state(s)
            print(f"battle {n} phase {s['phase']} updates {updates} result {ep.get('result')}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
