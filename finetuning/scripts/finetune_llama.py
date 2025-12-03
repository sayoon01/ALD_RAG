#!/usr/bin/env python3
# LLaMA Fine-tuning 스크립트 (LoRA 사용)

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# 프로젝트 루트 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"[!] 필요한 패키지가 설치되지 않았습니다: {e}")
    print("[!] 다음 명령어로 설치하세요:")
    print("    pip install transformers datasets peft accelerate bitsandbytes")
    DEPENDENCIES_AVAILABLE = False


def load_jsonl(filepath: Path) -> List[Dict[str, str]]:
    """JSONL 파일 로드"""
    data = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_instruction(item: Dict[str, str]) -> str:
    """Instruction 형식을 프롬프트로 변환"""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    
    return prompt


def prepare_dataset(train_file: Path, eval_file: Path, tokenizer):
    """데이터셋 준비"""
    
    def tokenize_function(examples):
        # batched=True일 때 examples는 딕셔너리 형태 (각 키의 값이 리스트)
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])
        
        # Instruction 형식을 프롬프트로 변환
        prompts = []
        for i in range(len(instructions)):
            item = {
                "instruction": instructions[i] if i < len(instructions) else "",
                "input": inputs[i] if i < len(inputs) else "",
                "output": outputs[i] if i < len(outputs) else ""
            }
            prompts.append(format_instruction(item))
        
        # 토크나이징
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels는 input_ids와 동일 (언어 모델링)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 데이터 로드
    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(eval_file)
    
    print(f"[+] 학습 데이터: {len(train_data)}개")
    print(f"[+] 검증 데이터: {len(eval_data)}개")
    
    # 데이터셋 생성
    from datasets import Dataset
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # 토크나이징
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="LLaMA Fine-tuning (LoRA)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-27b-it",
        help="기본 모델 이름"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(BASE_DIR / "finetuning" / "data" / "train.jsonl"),
        help="학습 데이터 파일"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=str(BASE_DIR / "finetuning" / "data" / "eval.jsonl"),
        help="검증 데이터 파일"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR / "finetuning" / "models" / "gemma-3-27b-ald-lora"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 크기"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="학습률"
    )
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        return
    
    print("[+] Fine-tuning 시작...")
    print(f"  - 모델: {args.model_name}")
    print(f"  - 학습 데이터: {args.train_file}")
    print(f"  - 출력: {args.output_dir}")
    
    # 모델 및 토크나이저 로드
    print("[+] 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # LoRA 설정
    print("[+] LoRA 설정 중...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 준비
    print("[+] 데이터셋 준비 중...")
    train_dataset, eval_dataset = prepare_dataset(
        Path(args.train_file),
        Path(args.eval_file),
        tokenizer
    )
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    print("[+] 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"[+] 모델 저장 중: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("[+] 완료!")


if __name__ == "__main__":
    main()

