# Fine-tuning 가이드

반도체 ALD 공정 전문가 모델을 Fine-tuning하는 가이드입니다.

## 📁 디렉토리 구조

```
finetuning/
├── README.md              # 이 파일
├── data/                  # 학습 데이터
│   ├── train.jsonl       # 학습용 Q&A 쌍
│   └── eval.jsonl        # 검증용 Q&A 쌍
├── models/                # Fine-tuned 모델 저장소
│   └── qwen-ald-lora/    # LoRA 어댑터 모델
└── scripts/               # 실행 스크립트
    ├── prepare_finetuning_data.py  # 데이터 생성
    ├── finetune_llama.py           # Fine-tuning 실행
    └── run_finetuning.sh           # 전체 자동 실행
```

## 🚀 빠른 시작

### 자동 실행 (권장)

```bash
cd /home/keti_spark1/ald-rag-lab
./finetuning/scripts/run_finetuning.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
1. 학습 데이터 생성
2. Fine-tuning 실행
3. 모델 저장

### 수동 실행

#### 1단계: 학습 데이터 생성

```bash
cd /home/keti_spark1/ald-rag-lab
python finetuning/scripts/prepare_finetuning_data.py
```

**생성되는 데이터:**
- `docs/docs_ald.json`에서 질문-답변 쌍 생성 (개선된 버전)
- `feedback/feedback_data.json`에서 실제 사용자 질문 포함
- 더 다양하고 자연스러운 질문 패턴 생성
- `finetuning/data/train.jsonl`, `eval.jsonl` 생성

**데이터 특징:**
- 자연스러운 질문 패턴 포함 ("뭐야?", "어떻게?")
- 평균 답변 길이: ~90자
- 키워드 조합 질문 포함
- 피드백 데이터 반영

#### 2단계: Fine-tuning 실행

```bash
python finetuning/scripts/finetune_llama.py \
  --train_file finetuning/data/train.jsonl \
  --eval_file finetuning/data/eval.jsonl \
  --output_dir finetuning/models/qwen-ald-lora \
  --num_epochs 3 \
  --batch_size 4
```

**파라미터:**
- `--model_name`: 기본 모델 (기본값: Qwen/Qwen2.5-7B-Instruct)
- `--num_epochs`: 학습 에폭 수 (기본값: 3)
- `--batch_size`: 배치 크기 (기본값: 4)
- `--learning_rate`: 학습률 (기본값: 2e-4)

#### 3단계: Fine-tuned 모델 사용

`rag_core.py`에서 Fine-tuned 모델 경로 설정:

```python
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "qwen-ald-lora"
```

## 📦 필요 패키지

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## ⚙️ 기술 스택

- **기본 모델**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning 방법**: LoRA (Low-Rank Adaptation)
- **LoRA 설정**:
  - rank (r): 16
  - alpha: 32
  - target_modules: q_proj, v_proj, k_proj, o_proj
  - dropout: 0.1

## ⚠️ 주의사항

- **GPU 권장**: 16GB VRAM 이상
- **학습 시간**: 2-4시간 (데이터셋 크기 및 GPU 성능에 따라)
- **메모리**: LoRA 사용으로 메모리 효율적 (전체 모델 Fine-tuning 대비)
- **체크포인트**: 학습 중간 저장본은 `checkpoint-*/` 디렉토리에 저장됨

## 📊 데이터 통계

현재 학습 데이터:
- 학습 데이터: ~850개
- 검증 데이터: ~215개
- 평균 답변 길이: ~90자
- 질문 패턴: 다양 (자연스러운 질문 포함)

## 🔧 문제 해결

### 메모리 부족 오류
- `--batch_size`를 줄이기 (예: 2 또는 1)
- `gradient_accumulation_steps` 증가

### 학습이 너무 느림
- GPU 확인: `nvidia-smi`
- 배치 크기 조정
- LoRA rank 조정 (r 값 감소)

### 모델이 제대로 학습되지 않음
- 학습률 조정 (예: 1e-4)
- 에폭 수 증가
- 데이터 품질 확인

