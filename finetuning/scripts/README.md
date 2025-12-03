# Fine-tuning 가이드

## 빠른 시작

### 1단계: 학습 데이터 생성

```bash
cd /home/keti_spark1/ald-rag-lab
python finetuning/scripts/prepare_finetuning_data.py
```

이 스크립트는:
- `docs/docs_ald.json`에서 질문-답변 쌍 생성
- `finetuning/data/train.jsonl`, `eval.jsonl` 생성

### 2단계: Fine-tuning 실행

```bash
python finetuning/scripts/finetune_llama.py \
  --train_file finetuning/data/train.jsonl \
  --eval_file finetuning/data/eval.jsonl \
  --output_dir finetuning/models/ald-llama-lora \
  --num_epochs 3 \
  --batch_size 4
```

### 3단계: Fine-tuned 모델 사용

`rag_core.py`에서 Fine-tuned 모델 경로 설정:

```python
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "ald-llama-lora"
```

또는 환경 변수로 설정:

```bash
export FINETUNED_MODEL_PATH=/home/keti_spark1/ald-rag-lab/models/ald-llama-lora
```

## 필요 패키지

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## 주의사항

- GPU 권장 (16GB VRAM 이상)
- 학습 시간: 2-4시간 (데이터셋 크기에 따라)
- LoRA 사용으로 메모리 효율적

