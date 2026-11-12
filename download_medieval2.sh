#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    read -rp "Enter your Steam username: " STEAM_USER
else
    STEAM_USER="$1"
fi

echo "==> Downloading Native Linux build of Total War: MEDIEVAL II (AppID 4700) into container..."
echo "    (You will be prompted for your Steam password and Steam Guard code if not cached)"

docker exec -it medieval2-ubuntu steamcmd \
    +@sSteamCmdForcePlatformType linux \
    +force_install_dir /home/tomek/medieval2 \
    +login "$STEAM_USER" \
    +app_update 4700 validate \
    +quit

# Create steam_appid.txt so the binary runs standalone without launching the Steam GUI
echo "4700" > /home/tomek/medieval2-docker/medieval2/steam_appid.txt
if [ -d "/home/tomek/medieval2-docker/medieval2/bin" ]; then
    echo "4700" > /home/tomek/medieval2-docker/medieval2/bin/steam_appid.txt
fi

echo "==> Download complete! Files are located at:"
echo "    Host:      /home/tomek/medieval2-docker/medieval2"
echo "    Container: /home/tomek/medieval2"
