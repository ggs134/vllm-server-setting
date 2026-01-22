# vLLM 서버 설정 가이드

이 프로젝트는 vLLM 기반의 LLM 서버를 Docker를 사용하여 설정하고 실행하는 방법을 안내합니다.

## 📦 프로젝트 구성

- `Dockerfile`: vLLM 서버를 위한 커스텀 Docker 이미지 빌드
- `docker-compose.yaml`: 컨테이너 실행 및 관리를 위한 Docker Compose 설정
- `install-nvidia-driver.sh`: 호스트 머신에 NVIDIA 드라이버 설치 스크립트
- `setup-docker-nvidia.sh`: 도커 및 NVIDIA 컨테이너 툴킷 설치 스크립트
- `config/`: 환경 변수 설정 파일 디렉토리
  - `.env.2gpu.qwen80b`: Qwen 80B 모델을 위한 2GPU 환경 설정 파일
  - `.env.4gpu.qwen235b`: Qwen 235B 모델을 위한 4GPU 환경 설정 파일
  - `.env.4gpu.qwen480b.coder`: Qwen 480B Coder 모델을 위한 4GPU 환경 설정 파일

## 🛠️ 사전 요구 사항

### 1. NVIDIA 드라이버 설치

호스트 머신에 NVIDIA 드라이버가 설치되어 있어야 합니다.

```bash
# NVIDIA 드라이버 설치
sudo ./install-nvidia-driver.sh

# 설치 후 시스템 재부팅
sudo reboot
```

### 2. 도커 및 NVIDIA 컨테이너 툴킷 설치

```bash
# 도커 및 NVIDIA 컨테이너 툴킷 설치
sudo ./setup-docker-nvidia.sh

# 설치 후 로그아웃 후 재로그인 (또는 재부팅)
# SSH 사용 중이면 SSH 재접속

# 그룹 확인
groups
# 출력에 'docker' 그룹이 포함되어야 함
```

## 🐳 Dockerfile 설명

`Dockerfile`은 vLLM 서버를 위한 커스텀 이미지를 빌드합니다.

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    pkg-config libglvnd-dev dkms build-essential \
    libegl-dev libegl1 libgl-dev libgl1 libgles-dev libgles1 \
    libglvnd-core-dev libglx-dev libopengl-dev \
    gcc make screen nano isc-dhcp-client \
    python3-venv python3-pip wget git curl \
    && rm -rf /var/lib/apt/lists/*

# Python 환경 설정
RUN python3 -m venv /vllm-env
ENV PATH="/vllm-env/bin:$PATH"

# PyTorch 및 NCCL 설치
RUN /vllm-env/bin/pip install --upgrade pip setuptools wheel && \
    /vllm-env/bin/pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    /vllm-env/bin/pip install nvidia-nccl-cu12==2.27.3

# vLLM 설치
WORKDIR /
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /vllm
RUN MAX_JOBS=32 /vllm-env/bin/pip install -e .

# 작업 디렉토리
WORKDIR /vllm

CMD ["bash"]
```

주요 특징:
- `nvidia/cuda:12.8.0-devel-ubuntu24.04` 베이스 이미지 사용
- Python 가상 환경 설정 (`/vllm-env`)
- PyTorch, torchvision, NCCL 설치
- vLLM 프로젝트 클론 및 설치

## 📋 docker-compose.yaml 설명

`docker-compose.yaml`은 컨테이너의 실행 환경을 정의합니다.

```yaml
services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile
    image: vllm-qwen:latest
    container_name: vllm-qwen-server
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    ipc: host
    shm_size: 20gb
    
    ports:
      - "${VLLM_PORT}:${VLLM_PORT}"
    
    volumes:
      - ~/workspace:/workspace
      - ~/models:/models
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.ssh:/root/.ssh:ro
    
    env_file:
      - .env
    
    environment:
      - OMP_NUM_THREADS=${OMP_NUM_THREADS}
      - MKL_NUM_THREADS=${MKL_NUM_THREADS}
      - OMP_PROC_BIND=${OMP_PROC_BIND}
      - OMP_PLACES=${OMP_PLACES}
      - VLLM_SLEEP_WHEN_IDLE=${VLLM_SLEEP_WHEN_IDLE}
      - VLLM_TUNE_FUSED_MOE=${VLLM_TUNE_FUSED_MOE}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      - NCCL_DEBUG=${NCCL_DEBUG}
      - NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}
      - NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
    
    command: >
      bash -c "
      source /vllm-env/bin/activate &&
      cd /vllm &&
      taskset -c ${TASKSET_CPUS} /vllm-env/bin/python3 /vllm-env/bin/vllm serve ${MODEL_NAME}
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
        --max-num-seqs ${MAX_NUM_SEQS}
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}
        --max-model-len ${MAX_MODEL_LEN}
        --enable-prefix-caching
        --enable-chunked-prefill
        --async-scheduling
        --enable-auto-tool-choice
        --tool-call-parser hermes
        --disable-log-stats
        --port ${VLLM_PORT}
      "
    
    restart: unless-stopped
```

주요 설정:
- **GPU 접근**: NVIDIA 드라이버를 통해 모든 GPU 사용
- **IPC 모드**: `host`로 설정하여 프로세스 간 통신 최적화
- **공유 메모리**: 20GB로 설정
- **포트 매핑**: 환경변수 `${VLLM_PORT}`를 사용하여 유연한 포트 설정
- **볼륨 마운트**: 로컬 디렉토리를 컨테이너에 마운트
- **환경변수**: `.env` 파일과 환경변수를 통해 다양한 설정 가능
- **명령어**: vLLM 서버를 자동으로 실행

## 🚀 사용 방법

### 1. 환경 변수 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.

```bash
# .env 파일 예시
VLLM_PORT=8000
MODEL_NAME=Qwen/Qwen2-72B-Instruct
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.95
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=4096
MAX_MODEL_LEN=32768
TASKSET_CPUS=0-31
OMP_NUM_THREADS=16
MKL_NUM_THREADS=16
OMP_PROC_BIND=true
OMP_PLACES=cores
VLLM_SLEEP_WHEN_IDLE=true
VLLM_TUNE_FUSED_MOE=true
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_DEBUG=INFO
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

### 2. 컨테이너 빌드 및 실행

```bash
# 빌드 및 실행 (백그라운드)
docker compose up -d

# 로그 확인
docker compose logs -f

# 컨테이너 내부에 접속
docker exec -it vllm-qwen-server bash

# 컨테이너 중지
docker compose down
```

### 3. 서버 테스트

```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 모델 정보 확인
curl http://localhost:8000/v1/models

# 추론 테스트
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-72B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

## 🔄 개발 모드

개발 중에는 컨테이너를 쉘 모드로 실행하여 디버깅할 수 있습니다.

```bash
# docker-compose.yaml의 command를 다음과 같이 수정
command: ["bash"]

# 또는 docker run 명령어 사용
docker run -it \
  --gpus all \
  --ipc=host \
  --shm-size=20g \
  -p 8000:8000 \
  -v ~/workspace:/workspace \
  -v ~/models:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.ssh:/root/.ssh:ro \
  --name gpu-dev \
  ubuntu:24.04 \
  sleep infinity \
  bash
```

## 🎯 사전 정의된 환경 설정

### Qwen 80B 모델 (2GPU)

`config/.env.2gpu.qwen80b` 파일은 Qwen 80B 모델을 2개의 GPU에서 실행하기 위한 최적의 설정을 포함합니다.

```bash
# Qwen 80B 모델을 위한 2GPU 환경 설정
# 사용 방법: docker compose --env-file config/.env.2gpu.qwen80b up -d

# CPU 스레드 설정
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
OMP_PROC_BIND=true
OMP_PLACES=cores

# vLLM 설정
VLLM_SLEEP_WHEN_IDLE=1
VLLM_TUNE_FUSED_MOE=1

# GPU 설정
CUDA_VISIBLE_DEVICES=0,1

# NCCL 설정
NCCL_DEBUG=WARN
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=1

# 모델 설정
MODEL_NAME=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=65535
MAX_MODEL_LEN=65535

# 포트 설정
VLLM_PORT=8000

# CPU 코어 할당 (taskset)
TASKSET_CPUS=0-63
```

**사용 방법:**
```bash
docker compose --env-file config/.env.2gpu.qwen80b up -d
```

### Qwen 235B 모델 (4GPU)

`config/.env.4gpu.qwen235b` 파일은 Qwen 235B 모델을 4개의 GPU에서 실행하기 위한 최적의 설정을 포함합니다.

```bash
# Qwen 235B 모델을 위한 4GPU 환경 설정
# 사용 방법: docker compose --env-file config/.env.4gpu.qwen235b up -d

# CPU 스레드 설정
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
OMP_PROC_BIND=true
OMP_PLACES=cores

# vLLM 설정
VLLM_SLEEP_WHEN_IDLE=1
VLLM_TUNE_FUSED_MOE=1

# GPU 설정
CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL 설정
NCCL_DEBUG=WARN
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=1

# 모델 설정
MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=98304
MAX_MODEL_LEN=131072

# 포트 설정
VLLM_PORT=8000

# CPU 코어 할당 (taskset)
TASKSET_CPUS=0-63
```

**사용 방법:**
```bash
docker compose --env-file config/.env.4gpu.qwen235b up -d
```

### Qwen 480B Coder 모델 (4GPU)

`config/.env.4gpu.qwen480b.coder` 파일은 Qwen 480B Coder 모델을 4개의 GPU에서 실행하기 위한 최적의 설정을 포함합니다.

```bash
# Qwen 480B Coder 모델을 위한 4GPU 환경 설정
# 사용 방법: docker compose --env-file config/.env.4gpu.qwen480b.coder up -d

# CPU 스레드 설정
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
OMP_PROC_BIND=true
OMP_PLACES=cores

# vLLM 설정
VLLM_SLEEP_WHEN_IDLE=1
VLLM_TUNE_FUSED_MOE=1

# GPU 설정
CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL 설정
NCCL_DEBUG=WARN
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=1

# 모델 설정
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=98304
MAX_MODEL_LEN=131072

# 포트 설정
VLLM_PORT=8000

# CPU 코어 할당 (taskset)
TASKSET_CPUS=0-63
```

**사용 방법:**
```bash
docker compose --env-file config/.env.4gpu.qwen480b.coder up -d
```

## 📚 참고 자료

- [vLLM 공식 문서](https://docs.vllm.ai/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [Docker Compose 문서](https://docs.docker.com/compose/)