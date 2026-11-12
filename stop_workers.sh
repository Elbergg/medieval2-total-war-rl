#!/usr/bin/env bash
# Stops the optional parallel workers (medieval2-worker2 & medieval2-worker3)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[*] Stopping parallel worker supervisors..."
if [ -f "$DIR/workspace/worker2.pid" ]; then
    kill -9 "$(cat "$DIR/workspace/worker2.pid")" 2>/dev/null || true
    rm -f "$DIR/workspace/worker2.pid"
fi
if [ -f "$DIR/workspace/worker3.pid" ]; then
    kill -9 "$(cat "$DIR/workspace/worker3.pid")" 2>/dev/null || true
    rm -f "$DIR/workspace/worker3.pid"
fi

echo "[*] Cleaning Wine inside worker containers..."
docker exec medieval2-worker2 wineserver -k 2>/dev/null || true
docker exec medieval2-worker3 wineserver -k 2>/dev/null || true

echo "[*] Stopping worker containers..."
docker compose stop medieval2-worker2 medieval2-worker3 2>/dev/null || true

echo "[+] Parallel workers 2 & 3 stopped cleanly. (Worker 1 / default continuous supervisor is unaffected)."
