FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. apt update && apt upgrade
RUN apt-get update && apt-get upgrade -y

# 2. 기본 패키지 설치
RUN apt-get install -y \
    pkg-config libglvnd-dev dkms build-essential \
    libegl-dev libegl1 libgl-dev libgl1 libgles-dev libgles1 \
    libglvnd-core-dev libglx-dev libopengl-dev \
    gcc make screen nano isc-dhcp-client python3-venv python3-pip \
    wget git curl

# 3. CUDA 키링 다운로드 및 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8

# 4. Python 가상환경 생성
RUN python3 -m venv /vllm-env

# 5. 가상환경 활성화 (ENV로 PATH 설정)
ENV PATH="/vllm-env/bin:$PATH"

# 6. pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 7. PyTorch + NCCL 설치
RUN pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install nvidia-nccl-cu12==2.27.3

# 8. vLLM 클론 및 설치
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /vllm
RUN MAX_JOBS=64 pip install -e .

# 작업 디렉토리
WORKDIR /vllm

CMD ["bash"]