#!/usr/bin/env bash
# Spins up the 2 optional parallel workers (medieval2-worker2 and medieval2-worker3)
# and starts their continuous battle supervisors.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[*] Spinning up container workers (medieval2-worker2 & medieval2-worker3)..."
docker compose --profile workers up -d medieval2-worker2 medieval2-worker3

echo "[*] Starting background supervisor for Worker 2..."
nohup python3 -u "$DIR/workspace/run_continuous_battles.py" --container medieval2-worker2 > "$DIR/workspace/worker2.log" 2>&1 &
echo $! > "$DIR/workspace/worker2.pid"

echo "[*] Starting background supervisor for Worker 3..."
nohup python3 -u "$DIR/workspace/run_continuous_battles.py" --container medieval2-worker3 > "$DIR/workspace/worker3.log" 2>&1 &
echo $! > "$DIR/workspace/worker3.pid"

echo "[+] Both parallel workers successfully started!"
echo "    Worker 2 Log: workspace/worker2.log (PID $(cat "$DIR/workspace/worker2.pid"))"
echo "    Worker 3 Log: workspace/worker3.log (PID $(cat "$DIR/workspace/worker3.pid"))"
echo "    Stop anytime with: ./stop_workers.sh"
