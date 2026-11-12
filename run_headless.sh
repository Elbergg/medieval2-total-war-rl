#!/usr/bin/env bash
# Launch medieval2.exe (Windows) headless under Wine + Xvfb with M2TWEOP (d3d9.dll) injected.
set -euo pipefail
docker exec medieval2-ubuntu bash -c '
  cd /home/tomek/game
  export WINEDLLOVERRIDES="d3d9=n,b;mscoree,mshtml="
  [ -d "$WINEPREFIX/drive_c" ] || wineboot -i
  exec xvfb-run -a -s "-screen 0 1280x720x24" wine medieval2.exe -ne -nm "$@"
' -- "$@"
