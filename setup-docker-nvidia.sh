#!/bin/bash

# 도커 및 NVIDIA 컨테이너 툴킷 설치 스크립트
# 호스트 머신에서 실행 (Ubuntu/Debian)

set -e

echo "=== 도커 및 NVIDIA 컨테이너 툴킷 설치 시작 ==="

# root 권한 확인
if [ "$EUID" -ne 0 ]; then 
    echo "이 스크립트는 root 권한이 필요합니다."
    echo "사용법: sudo ./setup-docker-nvidia.sh"
    exit 1
fi

# 기존 도커 관련 패키지 제거
echo "기존 도커 관련 패키지 제거 중..."
dpkg --get-selections | grep -E "(docker.io|docker-compose|docker-compose-v2|docker-doc|podman-docker|containerd|runc)" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    sudo apt remove -y $(dpkg --get-selections | grep -E "(docker.io|docker-compose|docker-compose-v2|docker-doc|podman-docker|containerd|runc)" | cut -f1)
else
    echo "제거할 도커 관련 패키지가 없습니다."
fi

# 도커 그룹에 현재 사용자 추가
# 현재 스크립트는 root로 실행 중이므로, 사용자 이름을 명시적으로 지정해야 함
CURRENT_USER=$(logname 2>/dev/null || echo $SUDO_USER)
if [ -n "$CURRENT_USER" ]; then
    echo "도커 그룹에 사용자 $CURRENT_USER 추가 중..."
    usermod -aG docker $CURRENT_USER
else
    echo "현재 사용자 이름을 확인할 수 없습니다."
    exit 1
fi

echo ""
echo "=== NVIDIA 컨테이너 툴킷 설치 시작 ==="

echo "NVIDIA GPG 키 추가 중..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "저장소 추가 중..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "패키지 캐시 업데이트 중..."
apt-get update

echo "nvidia-container-toolkit 설치 중..."
apt-get install -y nvidia-container-toolkit

echo "Docker를 위한 NVIDIA 런타임 구성 중..."
nvidia-ctk runtime configure --runtime=docker

echo "Docker 서비스 재시작 중..."
systemctl restart docker

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "설정이 완료되었습니다."
echo ""
echo "다음 단계를 수행하세요:"
echo "1. 터미널을 완전히 종료하거나 로그아웃 후 재로그인"
echo "   - SSH 사용 중이면 SSH 연결을 끊고 재접속"
echo "2. 재접속 후 다음 명령어로 그룹 확인:"
echo "   groups"
echo "   - 출력에 'docker' 그룹이 포함되어야 함"
echo "3. 다음 명령어로 NVIDIA 런타임 확인:"
echo "   docker info | grep -i runtime"
echo "   - 출력에 'nvidia' 런타임이 포함되어야 함"
echo ""
echo "확인 후, 다음 명령어로 테스트 가능:"
echo "docker run --rm --gpus all nvidia/cuda:12.8.0-devel-ubuntu24.04 nvidia-smi"