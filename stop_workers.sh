#!/usr/bin/env bash
# Stops all parallel workers (medieval2-ubuntu, medieval2-worker2, medieval2-worker3)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[*] Stopping parallel worker supervisors..."
for wid in 1 2 3; do
    if [ -f "$DIR/workspace/worker${wid}.pid" ]; then
        kill -9 "$(cat "$DIR/workspace/worker${wid}.pid")" 2>/dev/null || true
        rm -f "$DIR/workspace/worker${wid}.pid"
    fi
done

echo "[*] Cleaning Wine inside all containers..."
docker exec medieval2-ubuntu wineserver -k 2>/dev/null || true
docker exec medieval2-worker2 wineserver -k 2>/dev/null || true
docker exec medieval2-worker3 wineserver -k 2>/dev/null || true

echo "[*] Stopping optional worker containers..."
docker compose stop medieval2-worker2 medieval2-worker3 2>/dev/null || true

echo "[+] All parallel workers stopped cleanly."
