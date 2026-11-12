#!/usr/bin/env bash
# Launch medieval2.exe headless under Wine + Xvfb with M2TWEOP injected and audio hotpatches.
set -euo pipefail

docker exec medieval2-ubuntu bash -c '
  # 1. Ensure PulseAudio dummy sink is running
  if ! pactl list sinks short 2>/dev/null | grep -q dummy; then
    pulseaudio -D --exit-idle-time=-1 --load="module-null-sink sink_name=dummy" 2>/dev/null || true
  fi

  cd /home/tomek/game
  export WINEDLLOVERRIDES="d3d9=n,b;mscoree,mshtml="

  # 2. Run m2tweop_starter under xvfb-run
  exec xvfb-run -a -s "-screen 0 1024x768x24" wine m2tweop_starter.exe . "$@"
' -- "$@"
