"""Client for the medieval2-llm Gemini Judge container (called every 10th battle)."""
import json
import os
import urllib.request

LLM_URL = os.environ.get("LLM_URL", "http://llm:8000").rstrip("/")


def judge(replay_path, battle_id=None, timeout=60):
    text = open(replay_path).read()
    payload = {"replay": text, "battle_id": battle_id}
    req = urllib.request.Request(
        f"{LLM_URL}/judge",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
