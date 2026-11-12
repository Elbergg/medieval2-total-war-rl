#!/usr/bin/env bash
# Convenience command to view the latest battle recording and render its video
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/workspace/view_latest_battle.py" "$@"
