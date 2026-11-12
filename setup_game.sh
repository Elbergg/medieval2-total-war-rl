#!/usr/bin/env bash
# Install M2TWEOP 3.3.1 into ./game (hardlinked copy of the Steam install; the host install stays untouched).
set -euo pipefail
cd "$(dirname "$0")"
EOP=/home/tomek/m2tw-tools/M2TWEOP-3.3.1
for f in d3d9.dll M2TWEOPLibrary.dll lua5.1.dll libfbxsdk.dll discord_game_sdk.dll openal32.dll; do
  [ -f "$EOP/$f" ] && { rm -f "game/$f"; cp "$EOP/$f" "game/$f"; }
done
cp -r "$EOP/eopData" "$EOP/eopBattles" "$EOP/eopRedistributables" "$EOP/youneuoy_Data" game/ 2>/dev/null || true
echo "M2TWEOP installed into ./game"
