# Hugging Face 모델 다운로드 가이드

이 문서는 `download_model.py` 스크립트를 사용하여 Hugging Face 모델을 다운로드하는 방법을 설명합니다.

## 개요

`download_model.py`는 Hugging Face에서 모델을 다운로드하는 범용 스크립트입니다. 어떤 모델이든 동적으로 지정하여 다운로드할 수 있습니다.

## 사전 요구 사항

```bash
pip install transformers torch huggingface-hub
```

## 기본 사용법

### 1. 기본 다운로드

```bash
python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
```

### 2. 특정 경로에 다운로드

```bash
python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --path ./models
```

### 3. huggingface_hub 방법 사용

```bash
python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --method hub
```

### 4. 환경변수로 모델 이름 지정

```bash
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 python download_model.py
```

## 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 다운로드할 모델 이름 (형식: organization/model-name) | 환경변수 `MODEL_NAME` 또는 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| `--method` | 다운로드 방법 (`transformers` 또는 `hub`) | `transformers` |
| `--path` | 다운로드 경로 지정 | 환경변수 `MODEL_DOWNLOAD_PATH` 또는 Hugging Face 캐시 디렉토리 |

## 다운로드 방법 비교

### transformers 방법 (기본)

- **장점**: 모델과 토크나이저를 함께 다운로드하고 검증
- **단점**: 메모리를 더 많이 사용할 수 있음
- **사용 시나리오**: 모델을 바로 사용하려는 경우

### huggingface_hub 방법

- **장점**: 메모리 효율적, 더 세밀한 제어 가능
- **단점**: 추가 패키지 설치 필요 (`huggingface-hub`)
- **사용 시나리오**: 대용량 모델 다운로드, 특정 경로에 저장

## 사용 예제

### Qwen3-Coder 모델 다운로드

```bash
# FP8 양자화 버전
python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8

# 원본 bfloat16 버전
python download_model.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
```

### 다른 모델 다운로드

```bash
# Qwen3-Next 모델
python download_model.py --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8

# 다른 조직의 모델
python download_model.py --model meta-llama/Llama-3.1-8B-Instruct
```

### 배치 다운로드

```bash
# 여러 모델을 순차적으로 다운로드
for model in \
  "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"; do
  python download_model.py --model "$model" --path ./models
done
```

## 문제 해결

### 1. Hugging Face 로그인 필요

일부 모델은 로그인이 필요할 수 있습니다:

```bash
huggingface-cli login
```

### 2. 디스크 공간 부족

모델 크기를 확인하고 충분한 공간을 확보하세요:

```bash
# 모델 정보 확인
huggingface-cli repo info Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
```

### 3. 네트워크 연결 문제

다운로드가 중단되면 재시도하세요. `huggingface_hub` 방법은 자동으로 재개됩니다.

### 4. 메모리 부족

`--method hub` 옵션을 사용하면 메모리 사용량을 줄일 수 있습니다.

## 다운로드 경로

### 기본 캐시 디렉토리

- Linux/Mac: `~/.cache/huggingface/hub`
- Windows: `C:\Users\<username>\.cache\huggingface\hub`

### 사용자 지정 경로

```bash
python download_model.py --model <model-name> --path /path/to/models
```

## 참고사항

- **모델 크기**: 대형 모델(480B 등)은 수백 GB의 공간이 필요할 수 있습니다
- **다운로드 시간**: 네트워크 속도에 따라 수 시간이 걸릴 수 있습니다
- **FP8 양자화**: FP8 버전은 원본보다 작지만 여전히 큽니다
- **재개 다운로드**: `huggingface_hub` 방법은 중단된 다운로드를 자동으로 재개합니다

## 관련 링크

- [Hugging Face 모델 허브](https://huggingface.co/models)
- [Qwen3-Coder 모델 페이지](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8)
- [transformers 문서](https://huggingface.co/docs/transformers)
- [huggingface_hub 문서](https://huggingface.co/docs/huggingface_hub)