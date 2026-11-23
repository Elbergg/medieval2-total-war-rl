#!/usr/bin/env bash
# Open or start the live M2TW RL Training Monitor at http://localhost:8765
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! pgrep -f "training_monitor.py" >/dev/null 2>&1; then
    nohup python3 -u "$DIR/workspace/training_monitor.py" >/tmp/m2tw_monitor.log 2>&1 &
    sleep 1
fi
echo "[+] Training Monitor live at: http://localhost:8765"
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8765" >/dev/null 2>&1 || true
fi
