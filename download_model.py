#!/usr/bin/env python3
"""
Hugging Face 모델 다운로드 스크립트

사용 방법:
    python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
    python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --path ./models
    python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --method hub

참고:
    - 모델 크기가 매우 크므로 충분한 디스크 공간이 필요합니다
    - FP8 양자화 모델이므로 원본 bfloat16 모델보다 작습니다
    - 다운로드 시간이 오래 걸릴 수 있습니다
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, download_path=None):
    """모델과 토크나이저를 다운로드합니다."""
    print(f"다운로드 시작: {model_name}")
    print("=" * 60)
    
    try:
        # 토크나이저 다운로드
        print("1. 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=download_path
        )
        print("   ✓ 토크나이저 다운로드 완료")
        
        # 모델 다운로드 (가중치만, 메모리에 로드하지 않음)
        print("2. 모델 가중치 다운로드 중...")
        print("   (이 과정은 시간이 오래 걸릴 수 있습니다)")
        
        # trust_remote_code=True는 필요시 추가
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=download_path,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        print("   ✓ 모델 다운로드 완료")
        print("=" * 60)
        print(f"모델이 성공적으로 다운로드되었습니다!")
        
        if download_path:
            print(f"다운로드 경로: {download_path}")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"다운로드 경로 (기본 캐시): {cache_dir}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n문제 해결 방법:")
        print("1. Hugging Face 로그인이 필요할 수 있습니다: huggingface-cli login")
        print("2. 충분한 디스크 공간이 있는지 확인하세요")
        print("3. 네트워크 연결을 확인하세요")
        print("4. 모델 이름이 올바른지 확인하세요")
        raise


def download_with_huggingface_hub(model_name, download_path=None):
    """
    huggingface_hub를 사용한 대안 다운로드 방법
    메모리 사용량이 적고 더 세밀한 제어가 가능합니다.
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"huggingface_hub를 사용하여 다운로드: {model_name}")
        print("=" * 60)
        
        local_dir = download_path or f"./models/{model_name.split('/')[-1]}"
        os.makedirs(local_dir, exist_ok=True)
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"✓ 모델이 {local_dir}에 다운로드되었습니다!")
        return local_dir
        
    except ImportError:
        print("huggingface_hub가 설치되지 않았습니다.")
        print("설치: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hugging Face 모델 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용 (transformers 방법)
  python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
  
  # 특정 경로에 다운로드
  python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --path ./models
  
  # huggingface_hub 방법 사용
  python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --method hub
  
  # 환경변수로 모델 이름 지정
  MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 python download_model.py
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"),
        help="다운로드할 모델 이름 (예: Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) 또는 환경변수 MODEL_NAME 사용"
    )
    parser.add_argument(
        "--method",
        choices=["transformers", "hub"],
        default="transformers",
        help="다운로드 방법 선택 (기본값: transformers)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.getenv("MODEL_DOWNLOAD_PATH", None),
        help="다운로드 경로 지정 (기본값: Hugging Face 캐시) 또는 환경변수 MODEL_DOWNLOAD_PATH 사용"
    )
    
    args = parser.parse_args()
    
    # 모델 이름 검증
    if not args.model or "/" not in args.model:
        print("오류: 올바른 모델 이름을 지정해주세요. (형식: organization/model-name)")
        print(f"현재 지정된 모델: {args.model}")
        parser.print_help()
        exit(1)
    
    # 다운로드 경로 설정
    download_path = args.path
    if download_path:
        os.makedirs(download_path, exist_ok=True)
    
    print(f"모델: {args.model}")
    print(f"다운로드 방법: {args.method}")
    if download_path:
        print(f"다운로드 경로: {download_path}")
    print()
    
    # 다운로드 실행
    if args.method == "hub":
        download_with_huggingface_hub(args.model, download_path)
    else:
        download_model(args.model, download_path)