# rag_core.py

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

# ==============================
# 0) 경로 / 상수 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent           # ~/ald-rag-lab
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"

EMBED_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# 전역 상태
DOCS: List[str] = []
DOC_KEYWORDS: List[str] = []
DOC_ITEMS: List[Dict[str, Any]] = []
DOC_EMBEDS: Optional[np.ndarray] = None

DEVICE: Optional[torch.device] = None
SEMB_MODEL: Optional[SentenceTransformer] = None
TOKENIZER: Optional[AutoTokenizer] = None
LLM: Optional[AutoModelForCausalLM] = None

MODEL_INFO: Dict[str, Any] = {}


# ==============================
# 1) JSON 문서 로딩
# ==============================

def load_docs(path: Path = DOCS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"docs 파일이 없음: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs, keywords, pairs = [], [], []

    # 새 구조: {"documents": [{"id": 1, "keywords": ["ALD"], "text": "..."}, ...]}
    if isinstance(raw, dict) and "documents" in raw:
        documents = raw["documents"]
        
        if isinstance(documents, list):
            # 새 구조: keywords가 배열인 형태
            for i, item in enumerate(documents):
                if not isinstance(item, dict):
                    print(f"[!] 경고: 문서 #{i} 형식 오류. dict 아님 → 건너뜀")
                    continue

                text = str(item.get("text", "")).strip()
                item_keywords = item.get("keywords", [])
                
                # keywords가 배열인 경우
                if isinstance(item_keywords, list):
                    if not text or not item_keywords:
                        continue
                    
                    # 각 키워드마다 별도의 항목으로 추가 (검색 호환성)
                    for kw in item_keywords:
                        kw = str(kw).strip()
                        if not kw:
                            continue
                        docs.append(text)
                        keywords.append(kw)
                        pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
                
                # 기존 구조 호환: keyword가 단일 문자열인 경우
                elif isinstance(item_keywords, str):
                    kw = item_keywords.strip()
                    if not text or not kw:
                        continue
                    docs.append(text)
                    keywords.append(kw)
                    pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
                
                # 기존 구조 호환: keyword 필드가 있는 경우
                elif "keyword" in item:
                    kw = str(item.get("keyword", "unknown")).strip()
                    if not text or not kw:
                        continue
                    docs.append(text)
                    keywords.append(kw)
                    pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
        
        # 기존 구조 호환: documents가 dict인 경우 (키워드별 그룹화)
        elif isinstance(documents, dict):
            for keyword, text_list in documents.items():
                if not isinstance(text_list, list):
                    continue
                
                for item in text_list:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                    elif isinstance(item, str):
                        text = item.strip()
                    else:
                        continue
                    
                    if not text:
                        continue
                    
                    docs.append(text)
                    keywords.append(keyword)
                    pairs.append({"text": text, "keyword": keyword})
        else:
            raise ValueError("docs_ald.json 형식 오류 — documents는 list 또는 dict여야 함")
    else:
        raise ValueError("docs_ald.json 형식 오류 — 반드시 { 'documents': [...] } 형태여야 함")

    print(f"[+] 문서 로딩 완료 — 총 {len(docs)}개")
    return docs, keywords, pairs


# ==============================
# 2) 모델 초기화
# ==============================

def _init_models_if_needed():
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS
    global EMB_MODEL, TOKENIZER, LLM, DEVICE, DTYPE, MODEL_INFO

    if DOC_EMBEDS is not None:
        return  # 이미 초기화되었음

    # 문서 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()

    # 임베딩 모델
    print(f"[+] Embedding model 로딩 중: {EMBED_MODEL_NAME}")
    EMB_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

    DOC_EMBEDS = EMB_MODEL.encode(
        DOCS,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    print("[+] 문서 임베딩 shape:", DOC_EMBEDS.shape)

    # 디바이스 결정
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DTYPE = torch.float16
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DTYPE = torch.float16
    else:
        DEVICE = torch.device("cpu")
        DTYPE = torch.float32

    # LLaMA 로딩
    print(f"[+] LLaMA 로딩 중: {LLM_MODEL_NAME}")
    TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    LLM = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE.type != "cpu" else None
    )

    print(f"[+] LLaMA 로딩 완료 (device={DEVICE})")

    MODEL_INFO.update({
        "num_docs": len(DOCS),
        "keywords": sorted(list(set(DOC_KEYWORDS))),
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "device": str(DEVICE),
    })


def reload_documents():
    """
    docs_ald.json 파일을 다시 로드하고 임베딩을 재생성합니다.
    모델은 이미 로드되어 있어야 합니다 (EMB_MODEL이 있어야 함).
    """
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS, MODEL_INFO
    
    if EMB_MODEL is None:
        # 모델이 아직 초기화되지 않았으면 전체 초기화
        _init_models_if_needed()
        return
    
    print("[+] 문서 재로딩 중...")
    
    # 문서 다시 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()
    
    # 임베딩 재생성
    DOC_EMBEDS = EMB_MODEL.encode(
        DOCS,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    
    print(f"[+] 문서 재로딩 완료 — 총 {len(DOCS)}개")
    print(f"[+] 문서 임베딩 shape: {DOC_EMBEDS.shape}")
    
    # MODEL_INFO 업데이트
    MODEL_INFO.update({
        "num_docs": len(DOCS),
        "keywords": sorted(list(set(DOC_KEYWORDS))),
    })
    
    return len(DOCS)


# ==============================
# 3) 키워드 통계
# ==============================

def get_keyword_stats() -> Dict[str, int]:
    _init_models_if_needed()
    stats: Dict[str, int] = {}
    for kw in DOC_KEYWORDS:
        stats[kw] = stats.get(kw, 0) + 1
    return stats


# ==============================
# 4) 검색 (Retrieval)
# ==============================

def retrieve(
    query: str,
    top_k: int = 3,
    filter_keyword: Optional[str] = None,
):
    _init_models_if_needed()

    q_emb = EMB_MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")

    scores = np.dot(DOC_EMBEDS, q_emb)

    # 키워드 필터 적용
    idxs = range(len(DOCS))

    if filter_keyword:
        idxs = [i for i in idxs if DOC_KEYWORDS[i].lower() == filter_keyword.lower()]

    # 해당 keyword 문서가 없는 경우
    if filter_keyword and not idxs:
        return []

    # 상위 K 선택
    sorted_idx = sorted(idxs, key=lambda i: -scores[i])[:top_k]

    return [
        (DOCS[i], float(scores[i]), DOC_KEYWORDS[i])
        for i in sorted_idx
    ]


def debug_retrieval(query: str, retrieved):
    print("\n[🔍 검색 디버그]")
    print(f"- 질문: {query}")

    if not retrieved:
        print("  (검색 결과 없음)")
        return

    scores = [s for _, s, _ in retrieved]
    print(f"  * score range: min={min(scores):.3f}, max={max(scores):.3f}")

    for text, score, keyword in retrieved:
        print(f"    - [{keyword}] score={score:.3f} | {text}")


# ==============================
# 5) LLaMA 기반 RAG 생성
# ==============================

def generate_answer(
    query: str,
    top_k: int = 3,
    max_new_tokens: int = 256,
    filter_keyword: Optional[str] = None,
    context_only: bool = False,
    debug: bool = False,
):

    _init_models_if_needed()
    
    # 모델이 아직 로딩 중인지 확인
    if LLM is None or TOKENIZER is None:
        return (
            "모델이 아직 로딩 중입니다. 잠시 후 다시 시도해주세요.",
            []
        )

    # 검색 수행
    retrieved = retrieve(query, top_k=top_k, filter_keyword=filter_keyword)

    if debug:
        debug_retrieval(query, retrieved)

    if not retrieved:
        return (
            "해당 질문과 일치하는 문맥을 찾지 못했어.\n"
            "→ filter_keyword가 너무 좁거나\n"
            "→ docs_ald.json에 관련 문장이 부족할 수 있어.",
            []
        )

    scores = [s for _, s, _ in retrieved]
    max_score = max(scores)

    # 안전장치
    if max_score < 0.45:  
        return (
            "문맥과의 연관성이 너무 낮아서 답변을 생성하지 않았어.\n"
            "문서를 보강하거나 질문을 더 구체적으로 바꿔줘!",
            retrieved
        )

    # context만 반환 모드
    if context_only:
        return "컨텍스트만 반환했어.", retrieved

    # LLM 프롬프트 구성
    ctx = "\n".join([f"- ({kw}) {text}" for text, _, kw in retrieved])

    system_prompt = (
        "너는 반도체 ALD, 플라즈마, 유량, 압력, 챔버 등 ALD 개념을 설명하는 조수야.\n"
        "반드시 한국어로만 답해야 해.\n"
        "컨텍스트에 없는 내용을 추측하지 마.\n"
        "만약 정보가 없으면 '해당 노트에 없는 내용입니다.'라고만 말해."
    )

    user_prompt = f"""
아래는 관련 문맥이야:

{ctx}

[질문]
{query}

위 문맥만 근거로 한국어로 답변해줘.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    full_prompt = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = TOKENIZER(full_prompt, return_tensors="pt")
    
    # device_map="auto"를 사용하는 경우, 모델의 실제 디바이스를 확인하고
    # inputs를 그 디바이스로 이동시켜야 합니다.
    if hasattr(LLM, "device"):
        # 모델이 단일 디바이스에 있는 경우
        model_device = LLM.device
    elif hasattr(LLM, "hf_device_map"):
        # device_map="auto"로 여러 디바이스에 분산된 경우
        # 첫 번째 레이어의 디바이스를 확인
        first_param = next(LLM.parameters())
        model_device = first_param.device
    else:
        # 기본값으로 DEVICE 사용
        model_device = DEVICE if DEVICE is not None else torch.device("cpu")
    
    # inputs를 모델과 같은 디바이스로 이동
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = LLM.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = TOKENIZER.decode(gen_ids, skip_special_tokens=True)

    return answer.strip(), retrieved
