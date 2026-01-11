FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    pkg-config libglvnd-dev dkms build-essential \
    libegl-dev libegl1 libgl-dev libgl1 libgles-dev libgles1 \
    libglvnd-core-dev libglx-dev libopengl-dev \
    gcc make screen nano isc-dhcp-client \
    python3-venv python3-pip wget git curl && \
    rm -rf /var/lib/apt/lists/*

# CUDA 키링 및 Toolkit 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*

# Python 가상환경 생성
RUN python3 -m venv /vllm-env
ENV PATH="/vllm-env/bin:$PATH"

# PyTorch + NCCL 설치
RUN /vllm-env/bin/pip install --upgrade pip setuptools wheel && \
    /vllm-env/bin/pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    /vllm-env/bin/pip install nvidia-nccl-cu12==2.27.3

# vLLM 설치 - 성공한 커밋으로 고정 (중요!)
WORKDIR /
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /vllm
RUN git checkout ba6f5f4 && \
    MAX_JOBS=32 /vllm-env/bin/pip install -e .

WORKDIR /vllm

CMD ["bash"]