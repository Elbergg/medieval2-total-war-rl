"""Dedicated LLM Judge microservice container calling Gemini API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

PROMPT = """You are a strict tactical judge reviewing a Medieval II: Total War battle replay.
Below are the battle frames rendered as 2D ASCII tactical maps for Side 0 (our RL agent) and Side 1 (enemy).
Legend:
- Side 0 (our RL AI): UPPERCASE role + state (e.g. IH, CH, AA, TF)
- Side 1 (Enemy): lowercase role + state (e.g. iH, cR, tA)
- Roles: I=line_inf, S=spear_inf, A=missile_foot, C=cavalry, G=general, T=artillery
- States: M=Moving, H=Holding, A=Attacking, F=Firing, R=Routing/Fleeing

Score our RL agent (Side 0) on THREE axes from 0 to 10 (10 = best, 0 = worst):
1. fairness (0-10): 10 = honorable, fair play; 5 = somewhat unfair/cheesy; 0 = blatant exploits (corner/red-line camping, endless kiting, stacking all units on one spot).
2. historical_accuracy (0-10): 10 = authentic medieval tactics (coherent battle line, reserves, flank protection, proper infantry/missile/cavalry coordination); 0 = completely ahistorical chaos.
3. reasonability (0-10): 10 = sensible, purposeful tactical maneuvers; 0 = absurd, suicidal, or broken behavior.

STRICT CALIBRATION RULES (follow these strictly):
- Do NOT award 6+ on historical_accuracy or reasonability merely for avoiding corner-camping.
- If Side 0 units scatter/drift apart to opposite corners or edges (e.g. mean spread > 150m or max spread > 250m without a coherent flanking maneuver), score historical_accuracy <= 3 and reasonability <= 3.
- If Side 0 retreats/wanders away from the enemy without engaging, or lets cavalry/general suicide alone while infantry walks away, score historical_accuracy <= 3 and reasonability <= 3.
- Only score 7+ if Side 0 maintains a cohesive battle line, moves purposefully toward/around the enemy, and coordinates infantry, ranged, and cavalry together.

Return ONLY valid JSON with these exact keys:
{
  "fairness": <number 0..10>,
  "historical_accuracy": <number 0..10>,
  "reasonability": <number 0..10>,
  "reasoning": "<short 1-2 sentence explanation>"
}

"""


STATE_CHAR = {0: "M", 3: "H", 4: "A", 5: "F", 6: "R"}


def format_replay_text(replay_text):
    """Render evenly spaced replay frames as 2D ASCII tactical maps for LLM evaluation."""
    data = json.loads(replay_text)
    frames = data.get("frames") or []
    if not frames:
        return "Battle Replay: no frames recorded.\n"

    # Global bounding box across all frames so coordinates stay fixed across time
    xs, ys = [], []
    for fr in frames:
        for u in fr.get("units", []):
            if u[4] > 0:
                xs.append(u[2])
                ys.append(u[3])
    if not xs:
        return "Battle Replay: no active units.\n"
    min_x, max_x = min(xs) - 40, max(xs) + 40
    min_y, max_y = min(ys) - 40, max(ys) + 40
    span_x = max(200.0, max_x - min_x)
    span_y = max(200.0, max_y - min_y)

    W, H = 24, 12
    n = len(frames)
    idxs = list(range(n)) if n <= 10 else sorted(set(round(i * (n - 1) / 9) for i in range(10)))

    terrain = data.get("terrain") or {}
    btype = data.get("battle_type") or terrain.get("battle_type", "open_field")
    t_info = f"terrain={btype}"
    if btype in ("siege_assault", "siege_defense"):
        role_s = "ATTACKER" if btype == "siege_assault" else "DEFENDER"
        t_info += f" (Side0={role_s}, wall_y={terrain.get('wall_y', '?')}, gate_x={terrain.get('gate_x', '?')}, plaza_y={terrain.get('plaza_y', '?')})"
    elif btype == "river_crossing":
        b_xs = [b.get("x") for b in terrain.get("bridges", [])]
        t_info += f" (river_y={terrain.get('river_y', 0)}, bridges_x={b_xs})"

    lines = [
        f"Battle #{data.get('battle_id', '?')} | mode={data.get('mode', '?')} | {t_info} | result={data.get('result', '?')} "
        f"(1=Side0 win, -1=Side0 loss) | map=[{min_x:.0f}..{max_x:.0f}]x[{min_y:.0f}..{max_y:.0f}]m"
    ]
    border = "+" + "--" * W + "+"

    for idx in idxs:
        fr = frames[idx]
        s0 = [u for u in fr.get("units", []) if u[0] == 1 and u[4] > 0]
        s1 = [u for u in fr.get("units", []) if u[0] == 2 and u[4] > 0]
        if s0:
            cx0 = sum(u[2] for u in s0) / len(s0)
            cy0 = sum(u[3] for u in s0) / len(s0)
            dists = [((u[2] - cx0) ** 2 + (u[3] - cy0) ** 2) ** 0.5 for u in s0]
            mean_sp, max_sp = sum(dists) / len(dists), max(dists)
        else:
            cx0 = cy0 = mean_sp = max_sp = 0.0
        if s0 and s1:
            cx1 = sum(u[2] for u in s1) / len(s1)
            cy1 = sum(u[3] for u in s1) / len(s1)
            sep = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
        else:
            sep = 0.0

        grid = [[". " for _ in range(W)] for _ in range(H)]
        for u in s1 + s0:  # draw Side 0 on top if overlapping
            gx = max(0, min(W - 1, int((u[2] - min_x) / span_x * (W - 1))))
            gy = max(0, min(H - 1, int((max_y - u[3]) / span_y * (H - 1))))
            sym = str(u[8])[:1]
            tok = (sym.upper() if u[0] == 1 else sym.lower()) + STATE_CHAR.get(u[6], "?")
            grid[gy][gx] = tok

        lines.append(
            f"\n--- t={fr.get('t')}s | Side0={len(s0)} (mean_spread={mean_sp:.0f}m, "
            f"max_spread={max_sp:.0f}m, dist_to_enemy={sep:.0f}m) | Side1={len(s1)} ---"
        )
        lines.append(border)
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)

    return "\n".join(lines) + "\n"


def axis_penalty(score):
    """Convert 0..10 score into centered reward/penalty: steeper negative penalties below 5."""
    s = max(0.0, min(10.0, float(score)))
    diff = s - 5.0
    return diff * 1.6 if diff < 0 else diff


def score_verdict(raw):
    fair = max(0.0, min(10.0, float(raw.get("fairness", 5.0))))
    hist = max(0.0, min(10.0, float(raw.get("historical_accuracy", 5.0))))
    reas = max(0.0, min(10.0, float(raw.get("reasonability", 5.0))))
    p_fair = 1.6 * min(0.0, fair - 8.0)  # fairness is penalty-only (0 when >=8/10, steep negative on exploits)
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


_QUOTA_EXHAUSTED = False


def evaluate_replay(replay_text):
    global _QUOTA_EXHAUSTED
    if _QUOTA_EXHAUSTED:
        return None
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in medieval2-llm container")
    from google import genai
    import time
    client = genai.Client(api_key=api_key)
    candidates = [GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
    last_err = None
    formatted = format_replay_text(replay_text)
    for idx, model in enumerate(candidates):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=PROMPT + formatted,
                config={"response_mime_type": "application/json"},
            )
            raw = json.loads(resp.text)
            return score_verdict(raw)
        except Exception as e:
            last_err = e
            if "GenerateRequestsPerDay" in str(e):
                _QUOTA_EXHAUSTED = True
                return None
            if idx < len(candidates) - 1:
                time.sleep(0.5)
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
