# Qwen ALD LoRA 모델

반도체 ALD 공정 전문가 Fine-tuned 모델 (LoRA 어댑터)

## 모델 정보

- **기본 모델**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning 방법**: LoRA (Low-Rank Adaptation)
- **학습 데이터**: `docs/docs_ald.json` 기반 Q&A 쌍 (~1,000개)
- **학습 에폭**: 3
- **LoRA 설정**:
  - rank (r): 16
  - alpha: 32
  - target_modules: q_proj, v_proj, k_proj, o_proj

## 사용 방법

`rag_core.py`에서 모델 경로 설정:

```python
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "qwen-ald-lora"
```

## 파일 구조

- `adapter_model.safetensors`: LoRA 어댑터 가중치
- `adapter_config.json`: LoRA 설정
- `checkpoint-*/`: 학습 중간 체크포인트 (선택적)

## 참고

이 모델은 기본 Qwen 모델에 LoRA 어댑터를 추가한 형태입니다.
사용 시 기본 모델과 함께 로드해야 합니다.
