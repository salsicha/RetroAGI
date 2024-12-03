
FROM ubuntu:focal AS build

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH="/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
    SHELL=/bin/bash

RUN apt update && apt upgrade -y && apt install -q -y --no-install-recommends \
    python3-pip python3-venv curl gnupg2 lsb-release unzip ca-certificates cmake \
    ccache wget sudo git xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang \
    libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev python3-gdbm \
    libtbb-dev libosmesa6-dev libudev-dev autoconf libtool make cmake \
    zlib1g-dev libopenmpi-dev ffmpeg build-essential swig && \
    rm -rf /var/lib/apt/lists/*

# CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
RUN apt update && apt upgrade -y && apt install -q -y --no-install-recommends \
    cuda-toolkit nvidia-gds && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY scripts /scripts
COPY retro_examples /examples
COPY notebooks /notebooks
COPY entrypoint.sh /entrypoint.sh
COPY requirements.txt /requirements.txt
RUN chmod +x /entrypoint.sh

RUN wget https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning/archive/69eace77a3437f98b1b437074adee5a578803581.zip && \
    unzip 69eace77a3437f98b1b437074adee5a578803581.zip && rm 69eace77a3437f98b1b437074adee5a578803581.zip && \
    mv Semantic-Segmentation-Boost-Reinforcement-Learning-69eace77a3437f98b1b437074adee5a578803581/ Semantic-Segmentation-Boost-Reinforcement-Learning
RUN wget https://github.com/MatPoliquin/stable-retro-scripts/archive/refs/heads/main.zip && \
    unzip main.zip && rm main.zip && mv stable-retro-scripts-main stable-retro-scripts
RUN wget https://github.com/Farama-Foundation/stable-retro/archive/refs/heads/master.zip && \
    unzip master.zip && rm master.zip && mv stable-retro-master stable-retro
RUN wget https://github.com/harikris001/Super-Mario-Reinforcement_Learning/archive/refs/heads/master.zip && \
    unzip master.zip && rm master.zip && mv Super-Mario-Reinforcement_Learning-master Super-Mario-Reinforcement_Learning

RUN python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install -r /requirements.txt && \
    cd stable-retro && pip3 install -e .

### Importing extra roms:
RUN mkdir /roms && cd /roms && \
    wget https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20\(Headered\)/Super%20Mario%20Bros.%20%28World%29.zip && \
    unzip Super\ Mario\ Bros.\ \(World\).zip && \
    wget https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20\(Headered\)/Super%20Mario%20Bros.%202%20%28USA%29.zip && \
    unzip Super\ Mario\ Bros.\ 2\ \(USA\).zip && \
    wget https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20\(Headered\)/Super%20Mario%20Bros.%203%20%28USA%29.zip && \
    unzip Super\ Mario\ Bros.\ 3\ \(USA\).zip && \
    . /venv/bin/activate && python3 -m retro.import .

# User
ARG USERNAME=retroagi
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

COPY --from=build / /

WORKDIR /notebooks

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "9999", "--no-browser", \
    "--allow-root", "--ServerApp.token=docker_jupyter", \
    "--NotebookApp.allow_password_change=False"]
