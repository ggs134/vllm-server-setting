#!/bin/bash

# NVIDIA 드라이버 설치 스크립트
# 호스트 머신에서 실행 (Ubuntu/Debian)

set -e

echo "=== NVIDIA 드라이버 설치 시작 ==="

# root 권한 확인
if [ "$EUID" -ne 0 ]; then 
    echo "이 스크립트는 root 권한이 필요합니다."
    echo "사용법: sudo ./install-nvidia-driver.sh"
    exit 1
fi

# 패키지 목록 업데이트
echo "패키지 목록 업데이트 중..."
apt-get update

# NVIDIA 드라이버 패키지 설치
echo "NVIDIA 드라이버 패키지 설치 중..."
apt-get install -y \
    libnvidia-compute-580-server \
    nvidia-dkms-580-server-open \
    nvidia-utils-580-server

# DKMS 모듈 빌드 확인
echo "DKMS 모듈 빌드 확인 중..."
dkms status

echo ""
echo "=== 설치 완료 ==="
echo "시스템을 재부팅해야 드라이버가 활성화됩니다:"
echo "  sudo reboot"
