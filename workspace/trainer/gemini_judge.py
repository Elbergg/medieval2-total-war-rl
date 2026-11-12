"""Gemini judge: every 100th battle, send the compact replay and get bounded corrective penalties."""
import json
import os

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
PENALTY_SCALE = 25.0  # strong but bounded: half a battle defeat; nudges, never wipes weights

PROMPT = """You review a top-down replay of a Medieval II battle played by a reinforcement-learning AI (UPPERCASE symbols = learner).
Symbols: I=heavy inf, S=spear, A=missile, C=cavalry, G=general. Status suffix: H=hold, A=attacking, F=firing, R=fleeing.
Score two axes: (1) anti-cheese/fairness: corner/red-line camping, endless kiting, single-coordinate stacking, 1-rank spaghetti lines;
(2) human-like historical cohesion: coherent battle line, reserves, flank protection, coordinated wings.
Return JSON: {"cohesion_score": -1..1, "fairness_anti_cheese_score": -1..1, "exploits_detected": [str],
"flagged": [{"role": str|"any", "action": int, "t_start": sec, "t_end": sec, "penalty_weight": 0..1}], "reasoning": str}
Replay:
"""


def judge(replay_path):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    text = open(replay_path).read()
    resp = client.models.generate_content(
        model=MODEL, contents=PROMPT + text,
        config={"response_mime_type": "application/json"})
    return json.loads(resp.text)


def to_penalties(verdict):
    return [{"role": f.get("role", "any"), "action": int(f["action"]),
             "t_start": f["t_start"], "t_end": f["t_end"],
             "penalty": -PENALTY_SCALE * abs(float(f.get("penalty_weight", 1.0)))}
            for f in verdict.get("flagged", [])]
