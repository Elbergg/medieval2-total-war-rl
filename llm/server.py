"""Dedicated LLM Judge microservice container calling Gemini API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

PROMPT = """You are a tactical judge reviewing a Medieval II: Total War battle replay.
Each frame shows time `t` (seconds) and units for side 0 (our RL agent) and side 1 (enemy): role, position (x,y), men alive, state.

Score our RL agent (side 0) on THREE axes from 0 to 10 (10 = best, 0 = worst):
1. fairness (0-10): 10 = honorable, fair play; 5 = somewhat unfair/cheesy; 0 = blatant exploits (corner/red-line camping, endless kiting, stacking all units on one spot).
2. historical_accuracy (0-10): 10 = authentic medieval tactics (coherent battle line, reserves, flank protection, proper infantry/missile/cavalry coordination); 0 = completely ahistorical chaos.
3. reasonability (0-10): 10 = sensible, purposeful tactical maneuvers; 0 = absurd, suicidal, or broken behavior.

Return ONLY valid JSON with these exact keys:
{
  "fairness": <number 0..10>,
  "historical_accuracy": <number 0..10>,
  "reasonability": <number 0..10>,
  "reasoning": "<short 1-2 sentence explanation>"
}

Replay:
"""


def axis_penalty(score):
    """Convert 0..10 score into negative squared penalty in [-100, 0]: -(10 - s)^2."""
    s = max(0.0, min(10.0, float(score)))
    neg = s - 10.0  # -10..0
    return -(neg * neg)


def score_verdict(raw):
    fair = max(0.0, min(10.0, float(raw.get("fairness", 5.0))))
    hist = max(0.0, min(10.0, float(raw.get("historical_accuracy", 5.0))))
    reas = max(0.0, min(10.0, float(raw.get("reasonability", 5.0))))
    p_fair = axis_penalty(fair)
    p_hist = axis_penalty(hist)
    p_reas = axis_penalty(reas)
    total = p_fair + p_hist + p_reas
    return {
        "fairness": fair,
        "historical_accuracy": hist,
        "reasonability": reas,
        "penalties": {
            "fairness": round(p_fair, 2),
            "historical_accuracy": round(p_hist, 2),
            "reasonability": round(p_reas, 2),
            "total": round(total, 2),
        },
        "reasoning": str(raw.get("reasoning", "")),
    }


def evaluate_replay(replay_text):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in medieval2-llm container")
    from google import genai
    import time
    client = genai.Client(api_key=api_key)
    candidates = [GEMINI_MODEL, "gemini-3.7-flash", "gemini-3.5-flash"]
    last_err = None
    for idx, model in enumerate(candidates):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=PROMPT + replay_text,
                config={"response_mime_type": "application/json"},
            )
            raw = json.loads(resp.text)
            return score_verdict(raw)
        except Exception as e:
            last_err = e
            if idx < len(candidates) - 1:
                time.sleep(2)
    raise last_err


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "ok": True,
                "model": GEMINI_MODEL,
                "has_key": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/judge":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            verdict = evaluate_replay(req["replay"])
            print(
                f"[LLM_JUDGE] battle={req.get('battle_id')} fair={verdict['fairness']}/10 "
                f"hist={verdict['historical_accuracy']}/10 reas={verdict['reasonability']}/10 "
                f"total_penalty={verdict['penalties']['total']}",
                flush=True,
            )
            self._send(200, verdict)
        except Exception as e:
            print(f"[LLM_JUDGE] ERROR: {e}", flush=True)
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"[LLM_JUDGE] Listening on :8000 (model={GEMINI_MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
