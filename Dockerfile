FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Windows medieval2.exe (1.52) + M2TWEOP (32-bit DLL injector) -> Wine32 + Xvfb
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget sudo locales xvfb xauth x11-utils \
        wine wine32:i386 wine64 libwine libwine:i386 winbind cabextract \
        libgl1 libgl1:i386 libvulkan1 libvulkan1:i386 mesa-vulkan-drivers mesa-vulkan-drivers:i386 \
        procps tmux nano python3 xdotool imagemagick ffmpeg \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    WINEARCH=win32 \
    WINEPREFIX=/home/tomek/.wine \
    WINEDEBUG=-all \
    WINEDLLOVERRIDES="d3d9=n,b;mscoree,mshtml="

RUN usermod -l tomek -d /home/tomek -m ubuntu && groupmod -n tomek ubuntu && \
    echo "tomek ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/tomek && chmod 0440 /etc/sudoers.d/tomek

USER tomek
WORKDIR /home/tomek
RUN mkdir -p /home/tomek/game /home/tomek/workspace
CMD ["sleep", "infinity"]
