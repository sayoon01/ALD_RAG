#!/usr/bin/env python3
# 학습 데이터 생성 스크립트: docs_ald.json → Q&A 쌍 생성

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import sys

# 프로젝트 루트 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"
OUTPUT_DIR = BASE_DIR / "finetuning" / "data"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval.jsonl"


def load_docs() -> List[Dict[str, Any]]:
    """docs_ald.json 로드"""
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {DOCS_PATH}")
    
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"잘못된 문서 형식: {DOCS_PATH}")


def generate_question_from_text(text: str, keywords: List[str]) -> List[str]:
    """텍스트와 키워드로부터 질문 생성"""
    questions = []
    
    # 기본 질문 패턴
    if keywords:
        for keyword in keywords:
            # 단순 질문
            questions.append(f"{keyword}는 무엇인가요?")
            questions.append(f"{keyword}의 역할은 무엇인가요?")
            questions.append(f"{keyword}에 대해 설명해주세요.")
    
    # 키워드가 2개 이상인 경우 비교 질문
    if len(keywords) >= 2:
        questions.append(f"{keywords[0]}와 {keywords[1]}의 차이는 무엇인가요?")
        questions.append(f"{keywords[0]}와 {keywords[1]}의 관계는?")
    
    # 키워드가 3개 이상인 경우 복합 질문
    if len(keywords) >= 3:
        questions.append(f"{keywords[0]}, {keywords[1]}, {keywords[2]}의 관계는 무엇인가요?")
    
    return questions[:5]  # 최대 5개 질문


def create_qa_pairs(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """문서 리스트에서 질문-답변 쌍 생성"""
    qa_pairs = []
    
    for doc in docs:
        text = str(doc.get("text", "")).strip()
        keywords = doc.get("keywords", [])
        
        if not text or not keywords:
            continue
        
        # 질문 생성
        questions = generate_question_from_text(text, keywords)
        
        for question in questions:
            qa_pairs.append({
                "instruction": "반도체 ALD 공정 전문가로서 질문에 답변하세요. 반드시 한국어로 답변해야 합니다.",
                "input": question,
                "output": text
            })
        
        # 키워드 기반 직접 질문
        for keyword in keywords:
            qa_pairs.append({
                "instruction": "반도체 ALD 공정 전문가로서 질문에 답변하세요. 반드시 한국어로 답변해야 합니다.",
                "input": f"{keyword}에 대해 설명해주세요.",
                "output": text
            })
    
    return qa_pairs


def create_keyword_combination_qa(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """키워드 조합 기반 질문-답변 쌍 생성"""
    qa_pairs = []
    
    # 키워드별 문서 그룹화
    keyword_to_docs = {}
    for doc in docs:
        keywords = doc.get("keywords", [])
        text = str(doc.get("text", "")).strip()
        
        if not text or not keywords:
            continue
        
        for keyword in keywords:
            if keyword not in keyword_to_docs:
                keyword_to_docs[keyword] = []
            keyword_to_docs[keyword].append(text)
    
    # 2개 키워드 조합
    keywords_list = list(keyword_to_docs.keys())
    for i, kw1 in enumerate(keywords_list):
        for kw2 in keywords_list[i+1:]:
            # 두 키워드가 모두 포함된 문서 찾기
            combined_texts = []
            for doc in docs:
                doc_keywords = doc.get("keywords", [])
                if kw1 in doc_keywords and kw2 in doc_keywords:
                    combined_texts.append(str(doc.get("text", "")).strip())
            
            if combined_texts:
                answer = " ".join(combined_texts[:3])  # 최대 3개 결합
                qa_pairs.append({
                    "instruction": "반도체 ALD 공정 전문가로서 질문에 답변하세요. 반드시 한국어로 답변해야 합니다.",
                    "input": f"{kw1}와 {kw2}의 관계는 무엇인가요?",
                    "output": answer
                })
    
    return qa_pairs


def save_jsonl(data: List[Dict[str, str]], filepath: Path):
    """JSONL 형식으로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with filepath.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[+] 저장 완료: {filepath} ({len(data)}개 항목)")


def main():
    print("[+] 학습 데이터 생성 시작...")
    
    # 문서 로드
    docs = load_docs()
    print(f"[+] 문서 로드 완료: {len(docs)}개")
    
    # Q&A 쌍 생성
    qa_pairs = []
    
    # 1. 기본 텍스트 기반 Q&A
    print("[+] 기본 Q&A 쌍 생성 중...")
    basic_qa = create_qa_pairs(docs)
    qa_pairs.extend(basic_qa)
    print(f"  - 생성된 기본 Q&A: {len(basic_qa)}개")
    
    # 2. 키워드 조합 Q&A
    print("[+] 키워드 조합 Q&A 생성 중...")
    combo_qa = create_keyword_combination_qa(docs)
    qa_pairs.extend(combo_qa)
    print(f"  - 생성된 조합 Q&A: {len(combo_qa)}개")
    
    # 중복 제거
    seen = set()
    unique_qa = []
    for qa in qa_pairs:
        key = (qa["input"], qa["output"])
        if key not in seen:
            seen.add(key)
            unique_qa.append(qa)
    
    print(f"[+] 총 Q&A 쌍: {len(unique_qa)}개 (중복 제거 후)")
    
    # 학습/검증 분할 (8:2)
    random.seed(42)
    random.shuffle(unique_qa)
    
    split_idx = int(len(unique_qa) * 0.8)
    train_data = unique_qa[:split_idx]
    eval_data = unique_qa[split_idx:]
    
    # 저장
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(eval_data, EVAL_FILE)
    
    print(f"[+] 완료!")
    print(f"  - 학습 데이터: {len(train_data)}개")
    print(f"  - 검증 데이터: {len(eval_data)}개")


if __name__ == "__main__":
    main()

