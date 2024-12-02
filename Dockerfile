
FROM ubuntu:focal AS build

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH"

RUN apt update && apt upgrade -y && apt install -q -y --no-install-recommends \
    python3-pip python3-venv curl gnupg2 lsb-release unzip ca-certificates cmake \
    ccache wget sudo git xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang \
    libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev python3-gdbm \
    libtbb-dev libosmesa6-dev libudev-dev autoconf libtool make cmake \
    zlib1g-dev libopenmpi-dev ffmpeg build-essential swig && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && \
    rm -rf /root/.cache

WORKDIR /
COPY scripts /scripts
COPY retro_examples /examples
COPY notebooks /notebooks
COPY roms /roms

RUN wget https://github.com/MatPoliquin/stable-retro-scripts/archive/refs/heads/main.zip && \
    unzip main.zip && mv stable-retro-scripts-main stable-retro-scripts
RUN wget https://github.com/Farama-Foundation/stable-retro/archive/refs/heads/master.zip && \
    unzip master.zip && mv stable-retro-master stable-retro
RUN cd stable-retro && pip3 install -e .

### Importing extra roms:
RUN cd /roms && \
    python3 -m retro.import .

ARG USERNAME=tinyagi
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -a -G audio,video $USERNAME
USER $USERNAME
    

#####################################################################
FROM scratch

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH"

COPY --from=build / /

WORKDIR /notebooks

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "9999", "--no-browser", \
    "--allow-root", "--ServerApp.token=docker_jupyter", \
    "--NotebookApp.allow_password_change=False"]
