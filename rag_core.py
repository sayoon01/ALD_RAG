# rag_core.py

import json
import os
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
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
FEEDBACK_PATH = BASE_DIR / "feedback" / "feedback_data.json"

EMBED_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Fine-tuned 모델 경로 (None이면 기본 모델 사용)
FINETUNED_MODEL_PATH = None  # 예: BASE_DIR / "finetuning" / "models" / "ald-llama-lora"

# 전역 상태
DOCS: List[str] = []
DOC_KEYWORDS: List[str] = []
DOC_ITEMS: List[Dict[str, Any]] = []
DOC_EMBEDS: Optional[np.ndarray] = None

# 해시 테이블 인덱스 (키워드 -> 문서 인덱스 리스트)
KEYWORD_INDEX: Dict[str, List[int]] = {}  # keyword -> [doc_idx1, doc_idx2, ...]
KEYWORD_INDEX_NORMALIZED: Dict[str, List[int]] = {}  # normalized keyword -> [doc_idx1, doc_idx2, ...]

DEVICE: Optional[torch.device] = None
SEMB_MODEL: Optional[SentenceTransformer] = None
TOKENIZER: Optional[AutoTokenizer] = None
LLM: Optional[AutoModelForCausalLM] = None

MODEL_INFO: Dict[str, Any] = {}

# 동의어 매핑
SYNONYM_GROUPS: List[List[str]] = []
SYNONYM_MAP: Dict[str, str] = {}  # keyword -> normalized keyword

# 피드백 기반 점수 조정 (문서 인덱스 -> 점수 보정값)
FEEDBACK_SCORES: Dict[int, float] = {}  # 문서 인덱스 -> 점수 보정값 (양수: 보너스, 음수: 페널티)
FEEDBACK_LAST_MODIFIED: float = 0.0  # 피드백 파일 마지막 수정 시간 (캐싱용)
import os
FEEDBACK_LAST_MODIFIED: float = 0.0  # 피드백 파일 마지막 수정 시간 (캐싱용)


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

def _build_keyword_index():
    """키워드 기반 해시 테이블 인덱스 구축"""
    global KEYWORD_INDEX, KEYWORD_INDEX_NORMALIZED
    
    KEYWORD_INDEX = {}
    KEYWORD_INDEX_NORMALIZED = {}
    
    for idx, keyword in enumerate(DOC_KEYWORDS):
        # 정확한 키워드 매칭
        if keyword not in KEYWORD_INDEX:
            KEYWORD_INDEX[keyword] = []
        KEYWORD_INDEX[keyword].append(idx)
        
        # 정규화된 키워드 매칭 (동의어 지원)
        normalized = normalize_keyword(keyword)
        if normalized not in KEYWORD_INDEX_NORMALIZED:
            KEYWORD_INDEX_NORMALIZED[normalized] = []
        KEYWORD_INDEX_NORMALIZED[normalized].append(idx)
    
    print(f"[+] 키워드 인덱스 구축 완료: {len(KEYWORD_INDEX)}개 키워드")


def _init_models_if_needed():
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS
    global SEMB_MODEL, TOKENIZER, LLM, DEVICE, DTYPE, MODEL_INFO
    global KEYWORD_INDEX, KEYWORD_INDEX_NORMALIZED

    if DOC_EMBEDS is not None:
        # 문서는 이미 로드되었지만 피드백 점수는 다시 로드 (최신 상태 유지)
        load_feedback_scores()
        return  # 이미 초기화되었음

    # 동의어 로딩
    load_synonyms()
    
    # 문서 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()
    
    # 키워드 인덱스 구축 (해시 테이블)
    _build_keyword_index()

    # 임베딩 모델
    print(f"[+] Embedding model 로딩 중: {EMBED_MODEL_NAME}")
    SEMB_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

    DOC_EMBEDS = SEMB_MODEL.encode(
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

    # Fine-tuned 모델 로드 (LoRA adapter)
    if FINETUNED_MODEL_PATH is not None and Path(FINETUNED_MODEL_PATH).exists():
        try:
            try:
                from peft import PeftModel  # type: ignore
            except ImportError:
                print(f"[!] peft 라이브러리가 없어 Fine-tuned 모델을 로드할 수 없습니다.")
                print(f"[!] pip install peft 로 설치하세요.")
            else:
                print(f"[+] Fine-tuned adapter 로딩 중: {FINETUNED_MODEL_PATH}")
                LLM = PeftModel.from_pretrained(LLM, FINETUNED_MODEL_PATH)
                print(f"[+] Fine-tuned adapter 로딩 완료")
        except Exception as e:
            print(f"[!] Fine-tuned 모델 로드 실패: {e}")
            print(f"[!] 기본 모델을 사용합니다.")

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
    모델은 이미 로드되어 있어야 합니다 (SEMB_MODEL이 있어야 함).
    """
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS, MODEL_INFO, SEMB_MODEL
    
    if SEMB_MODEL is None:
        # 모델이 아직 초기화되지 않았으면 전체 초기화
        _init_models_if_needed()
        return
    
    print("[+] 문서 재로딩 중...")
    
    # 문서 다시 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()
    
    # 키워드 인덱스 재구축 (해시 테이블)
    _build_keyword_index()
    
    # 임베딩 재생성
    DOC_EMBEDS = SEMB_MODEL.encode(
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
# 3) 동의어 매핑 로딩
# ==============================

def load_synonyms(path: Path = SYNONYMS_PATH):
    """동의어 그룹을 로드하고 매핑을 생성합니다."""
    global SYNONYM_GROUPS, SYNONYM_MAP
    
    if not path.exists():
        print(f"[!] 동의어 파일이 없습니다: {path}. 기본 동의어만 사용합니다.")
        SYNONYM_GROUPS = []
        SYNONYM_MAP = {}
        return
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        SYNONYM_GROUPS = data.get("synonym_groups", [])
        
        # 각 그룹의 첫 번째 키워드를 정규화 키워드로 사용
        SYNONYM_MAP = {}
        for group in SYNONYM_GROUPS:
            if not group:
                continue
            normalized = group[0].lower()
            for keyword in group:
                SYNONYM_MAP[keyword.lower()] = normalized
        
        print(f"[+] 동의어 매핑 로드 완료: {len(SYNONYM_GROUPS)}개 그룹")
    except Exception as e:
        print(f"[!] 동의어 로드 실패: {e}")
        SYNONYM_GROUPS = []
        SYNONYM_MAP = {}


def normalize_keyword(keyword: str) -> str:
    """키워드를 정규화된 형태로 변환합니다."""
    if not keyword:
        return ""
    normalized = keyword.lower().strip()
    return SYNONYM_MAP.get(normalized, normalized)


def are_synonyms(keyword1: str, keyword2: str) -> bool:
    """두 키워드가 동의어인지 확인합니다."""
    return normalize_keyword(keyword1) == normalize_keyword(keyword2)


# ==============================
# 3-2) 피드백 기반 점수 조정 로딩
# ==============================

def load_feedback_scores(path: Path = FEEDBACK_PATH, force_reload: bool = False):
    """
    피드백 데이터를 로드하여 문서별 점수 조정 가중치를 계산합니다.
    좋은 피드백을 받은 문서는 점수 증가, 나쁜 피드백은 감소.
    
    Args:
        path: 피드백 파일 경로
        force_reload: True이면 캐시를 무시하고 강제로 다시 로드
    """
    global FEEDBACK_SCORES, FEEDBACK_LAST_MODIFIED
    
    # DOCS가 아직 로드되지 않았으면 스킵
    if not DOCS:
        FEEDBACK_SCORES = {}
        return
    
    if not path.exists():
        FEEDBACK_SCORES = {}
        FEEDBACK_LAST_MODIFIED = 0.0
        return
    
    # 캐싱: 파일이 변경되지 않았고 강제 리로드가 아니면 스킵
    try:
        current_mtime = os.path.getmtime(path)
        if not force_reload and FEEDBACK_SCORES and current_mtime == FEEDBACK_LAST_MODIFIED:
            return  # 캐시된 값 사용
        FEEDBACK_LAST_MODIFIED = current_mtime
    except OSError:
        FEEDBACK_SCORES = {}
        return
    
    try:
        with path.open("r", encoding="utf-8") as f:
            feedback_data = json.load(f)
        
        feedbacks = feedback_data.get("feedbacks", [])
        if not feedbacks:
            FEEDBACK_SCORES = {}
            return
        
        # 문서 텍스트 -> 인덱스 매핑 (효율적인 조회를 위해)
        doc_text_to_idx = {}
        for idx, doc_text in enumerate(DOCS):
            normalized_text = doc_text.strip()
            # 정규화된 텍스트로 매핑 (중복 방지)
            if normalized_text not in doc_text_to_idx:
                doc_text_to_idx[normalized_text] = []
            doc_text_to_idx[normalized_text].append(idx)
        
        # 각 문서에 대한 피드백 점수 누적
        doc_feedback_scores = {}  # 인덱스 -> 누적 점수
        
        for feedback in feedbacks:
            if feedback.get("feedback") not in ["like", "dislike"]:
                continue
            
            contexts = feedback.get("contexts", [])
            feedback_type = feedback.get("feedback")
            
            # 좋은 피드백: +1, 나쁜 피드백: -1
            score_adjustment = 1.0 if feedback_type == "like" else -1.0
            
            # 각 컨텍스트에 대해 해당하는 문서 인덱스 찾기
            for ctx in contexts:
                ctx_text = str(ctx.get("text", "")).strip()
                if not ctx_text:
                    continue
                
                # 효율적인 매칭: 정규화된 텍스트로 직접 조회
                ctx_normalized = ctx_text
                
                # 정확한 매칭 우선
                if ctx_normalized in doc_text_to_idx:
                    for idx in doc_text_to_idx[ctx_normalized]:
                        if idx not in doc_feedback_scores:
                            doc_feedback_scores[idx] = 0.0
                        doc_feedback_scores[idx] += score_adjustment
                else:
                    # 부분 매칭 (정확한 매칭이 없는 경우만)
                    for doc_text, idx_list in doc_text_to_idx.items():
                        if ctx_normalized in doc_text or doc_text in ctx_normalized:
                            for idx in idx_list:
                                if idx not in doc_feedback_scores:
                                    doc_feedback_scores[idx] = 0.0
                                doc_feedback_scores[idx] += score_adjustment * 0.5  # 부분 매칭은 가중치 0.5
        
        # 점수를 정규화 (최대 ±0.2 범위로 제한)
        if doc_feedback_scores:
            max_abs_score = max(abs(s) for s in doc_feedback_scores.values())
            if max_abs_score > 0:
                FEEDBACK_SCORES = {
                    idx: (score / max_abs_score) * 0.2  # 최대 ±0.2 보정
                    for idx, score in doc_feedback_scores.items()
                }
            else:
                FEEDBACK_SCORES = {}
        else:
            FEEDBACK_SCORES = {}
        
    except Exception as e:
        print(f"[!] 피드백 점수 로드 실패: {e}")
        FEEDBACK_SCORES = {}


# ==============================
# 4) 키워드 통계
# ==============================

def get_keyword_stats() -> Dict[str, int]:
    _init_models_if_needed()
    stats: Dict[str, int] = {}
    for kw in DOC_KEYWORDS:
        stats[kw] = stats.get(kw, 0) + 1
    return stats


# ==============================
# 5) 검색 (Retrieval) - 하이브리드 검색
# ==============================

def extract_keywords_from_query(query: str) -> List[str]:
    """쿼리에서 키워드를 추출합니다."""
    query_lower = query.lower()
    found_keywords = []
    
    # 모든 키워드 확인
    for keyword in set(DOC_KEYWORDS):
        if keyword.lower() in query_lower:
            found_keywords.append(keyword)
    
    # 동의어도 확인
    for keyword in set(DOC_KEYWORDS):
        normalized_kw = normalize_keyword(keyword)
        # 쿼리의 단어들을 확인
        query_words = query_lower.split()
        for word in query_words:
            if normalize_keyword(word) == normalized_kw:
                if keyword not in found_keywords:
                    found_keywords.append(keyword)
    
    return found_keywords


def compute_keyword_score(doc_keyword: str, query: str) -> float:
    """문서 키워드와 쿼리 간의 키워드 매칭 점수를 계산합니다."""
    doc_keyword_lower = doc_keyword.lower()
    query_lower = query.lower()
    
    # 1. 정확한 키워드 매칭
    if doc_keyword_lower in query_lower:
        return 1.0
    
    # 2. 동의어 매칭
    query_words = query_lower.split()
    for word in query_words:
        if are_synonyms(word, doc_keyword):
            return 0.8
    
    # 3. 부분 매칭 (키워드가 쿼리에 포함)
    if doc_keyword_lower in query_lower or query_lower in doc_keyword_lower:
        return 0.5
    
    return 0.0


def retrieve(
    query: str,
    top_k: int = 3,
    filter_keyword: Optional[str] = None,
    hybrid_weight: float = 0.3,  # 키워드 점수 가중치 (0.0 = 의미 검색만, 1.0 = 키워드 검색만)
):
    """
    하이브리드 검색: 키워드 매칭 + 의미 기반 검색 (해시 테이블 최적화)
    
    Args:
        query: 검색 쿼리
        top_k: 상위 K개 결과 반환
        filter_keyword: 키워드 필터 (선택)
        hybrid_weight: 키워드 점수 가중치 (0.0-1.0)
    """
    _init_models_if_needed()

    # 1. 해시 테이블을 사용한 키워드 기반 후보 문서 필터링 (O(1) 조회)
    candidate_idxs = None
    
    # 쿼리에서 키워드 추출
    query_keywords = extract_keywords_from_query(query)
    
    if query_keywords or filter_keyword:
        candidate_idxs_set = set()
        
        # 쿼리 키워드로 후보 문서 찾기
        for kw in query_keywords:
            # 정확한 키워드 매칭
            if kw in KEYWORD_INDEX:
                candidate_idxs_set.update(KEYWORD_INDEX[kw])
            # 정규화된 키워드 매칭
            normalized_kw = normalize_keyword(kw)
            if normalized_kw in KEYWORD_INDEX_NORMALIZED:
                candidate_idxs_set.update(KEYWORD_INDEX_NORMALIZED[normalized_kw])
        
        # 필터 키워드 적용
        if filter_keyword:
            filter_idxs_set = set()
            # 정확한 키워드 매칭
            if filter_keyword in KEYWORD_INDEX:
                filter_idxs_set.update(KEYWORD_INDEX[filter_keyword])
            # 정규화된 키워드 매칭
            filter_normalized = normalize_keyword(filter_keyword)
            if filter_normalized in KEYWORD_INDEX_NORMALIZED:
                filter_idxs_set.update(KEYWORD_INDEX_NORMALIZED[filter_normalized])
            
            if candidate_idxs_set:
                # 교집합: 쿼리 키워드와 필터 키워드 모두 만족
                candidate_idxs_set = candidate_idxs_set & filter_idxs_set
            else:
                # 필터만 적용
                candidate_idxs_set = filter_idxs_set
        
        if candidate_idxs_set:
            candidate_idxs = list(candidate_idxs_set)
    
    # 필터 키워드만 있고 쿼리 키워드가 없는 경우
    if filter_keyword and not query_keywords and candidate_idxs is None:
        candidate_idxs_set = set()
        if filter_keyword in KEYWORD_INDEX:
            candidate_idxs_set.update(KEYWORD_INDEX[filter_keyword])
        filter_normalized = normalize_keyword(filter_keyword)
        if filter_normalized in KEYWORD_INDEX_NORMALIZED:
            candidate_idxs_set.update(KEYWORD_INDEX_NORMALIZED[filter_normalized])
        if candidate_idxs_set:
            candidate_idxs = list(candidate_idxs_set)
    
    # 해당 keyword 문서가 없는 경우
    if (filter_keyword or query_keywords) and (candidate_idxs is None or len(candidate_idxs) == 0):
        return []
    
    # 검색 대상 문서 인덱스 결정
    search_idxs = candidate_idxs if candidate_idxs is not None else list(range(len(DOCS)))

    # 2. 의미 기반 검색 (후보 문서에 대해서만 수행)
    q_emb = SEMB_MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")

    # 후보 문서의 임베딩만 사용
    candidate_embeds = DOC_EMBEDS[search_idxs]
    semantic_scores_candidate = np.dot(candidate_embeds, q_emb)
    
    # 전체 문서에 대한 점수 배열 생성
    semantic_scores = np.full(len(DOCS), -1.0, dtype=np.float32)
    for i, idx in enumerate(search_idxs):
        semantic_scores[idx] = semantic_scores_candidate[i]
    
    # 정규화: 0-1 범위로 변환 (후보 문서만 고려)
    valid_scores = semantic_scores[search_idxs]
    semantic_min, semantic_max = valid_scores.min(), valid_scores.max()
    if semantic_max > semantic_min:
        semantic_scores_normalized = np.full(len(DOCS), 0.0, dtype=np.float32)
        for idx in search_idxs:
            semantic_scores_normalized[idx] = (semantic_scores[idx] - semantic_min) / (semantic_max - semantic_min)
    else:
        semantic_scores_normalized = np.full(len(DOCS), 0.5, dtype=np.float32)

    # 3. 키워드 매칭 점수 계산 (후보 문서에 대해서만)
    keyword_scores = np.zeros(len(DOCS), dtype=np.float32)
    for idx in search_idxs:
        keyword_scores[idx] = compute_keyword_score(DOC_KEYWORDS[idx], query)

    # 4. 하이브리드 점수 결합
    hybrid_scores = (1 - hybrid_weight) * semantic_scores_normalized + hybrid_weight * keyword_scores
    
    # 5. 피드백 기반 점수 조정 적용
    load_feedback_scores()  # 최신 피드백 반영
    for idx in search_idxs:
        if idx in FEEDBACK_SCORES:
            hybrid_scores[idx] += FEEDBACK_SCORES[idx]
            # 점수를 0-1 범위로 클리핑
            hybrid_scores[idx] = max(0.0, min(1.0, hybrid_scores[idx]))

    # 6. 상위 K 선택 (후보 문서 중에서만)
    sorted_idx = sorted(search_idxs, key=lambda i: -hybrid_scores[i])[:top_k]

    return [
        (DOCS[i], float(hybrid_scores[i]), DOC_KEYWORDS[i])
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
        "반도체 ALD 전문가로서 답변하세요.\n"
        "문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n"
        "반드시 한국어로만 답변해야 합니다."
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
