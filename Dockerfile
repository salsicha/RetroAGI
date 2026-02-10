FROM ubuntu:focal AS build

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH="/usr/local/cuda/bin:$PATH" \
    SHELL=/bin/bash

# Nvidia capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,display,video,utility

# Install System Dependencies
# Added libgl1-mesa-glx for opencv-python
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    python3-pip python3-venv curl gnupg2 lsb-release unzip ca-certificates cmake \
    ccache wget sudo git xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang \
    libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev python3-gdbm \
    libtbb-dev libosmesa6-dev libudev-dev autoconf libtool make zlib1g-dev \
    libopenmpi-dev ffmpeg build-essential swig vim libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA (Official Nvidia Repo)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    cuda-toolkit-12-8 nvidia-gds-12-8 nvidia-modprobe && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy Project Files
COPY files.zip /files.zip
COPY scripts /scripts
COPY src /src
# COPY notebooks /notebooks
COPY entrypoint.sh /entrypoint.sh
COPY requirements.txt /requirements.txt
RUN chmod +x /entrypoint.sh

# Install Python Dependencies
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip wheel
RUN pip3 install -r /requirements.txt

# Install Stable Retro (Source)
RUN wget https://github.com/Farama-Foundation/stable-retro/archive/refs/heads/master.zip && \
    unzip master.zip && rm master.zip && mv stable-retro-master stable-retro && \
    cd stable-retro && pip3 install -e .

# Import ROMs
# Unzip to a specific folder to avoid clutter and ensure correct import
RUN mkdir -p /roms && \
    unzip -o files.zip -d /roms && \
    python3 -m retro.import /roms

# Setup User
ARG USERNAME=retroagi
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=(root) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -a -G audio,video $USERNAME

# Final Stage (Flatten)
FROM scratch
COPY --from=build / /

WORKDIR /
USER retroagi
ENV PATH="/venv/bin:$PATH"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/scripts/run.py"]