# backend/app.py

import sys
from pathlib import Path
from typing import List, Optional

# ============================================
# 0) 상위 폴더를 import 경로에 추가
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent  # ~/ald-rag-lab
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# rag_core 함수들 로딩
from rag_core import generate_answer, get_keyword_stats, reload_documents

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================
# FastAPI 기본 세팅
# ============================================

app = FastAPI(
    title="반도체 ALD RAG API",
    description="반도체 지식 기반 RAG 챗봇 API (LLaMA 기반)",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request / Response 모델
# ============================================

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    max_new_tokens: int = 256
    filter_keyword: Optional[str] = None
    context_only: bool = False
    debug: bool = False


class ContextItem(BaseModel):
    text: str
    score: float
    keyword: str


class ChatResponse(BaseModel):
    answer: str
    contexts: List[ContextItem]
    used_keyword: str


# ============================================
# 기본 건강 체크 엔드포인트
# ============================================

@app.get("/")
def root():
    try:
        # 모델 초기화 (문서 로딩)
        from rag_core import _init_models_if_needed, MODEL_INFO
        _init_models_if_needed()
        
        keywords = get_keyword_stats()
        num_docs = MODEL_INFO.get("num_docs", 0)
        keyword_list = MODEL_INFO.get("keywords", [])
    except Exception as e:
        keywords = {}
        num_docs = 0
        keyword_list = []
        print(f"[WARN] get_keyword_stats() 실패: {e}")
    
    return {
        "message": "ALD RAG API 정상 동작 중",
        "keywords": keywords,
        "num_docs": num_docs,
        "keyword_list": keyword_list,
        "note": "POST /chat 으로 질문할 수 있습니다.",
        "endpoints": {
            "chat": "POST /chat - 질문하기",
            "keywords": "GET /keywords - 키워드 통계",
            "docs": "GET /docs - Swagger UI"
        }
    }


# ============================================
# 핵심: /chat RAG 엔드포인트
# ============================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    RAG 챗봇 엔드포인트
    
    - question: 질문 내용 (필수)
    - top_k: 검색할 상위 문장 개수 (기본: 3)
    - max_new_tokens: LLM이 생성할 최대 토큰 수 (기본: 256)
    - filter_keyword: 특정 키워드만 검색 (선택)
    - context_only: 답변 생성 없이 컨텍스트만 반환 (기본: False)
    - debug: 디버그 로그 출력 (기본: False)
    """
    try:
        # generate_answer 호출 (모든 파라미터 전달)
        answer, retrieved = generate_answer(
            query=req.question,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            filter_keyword=req.filter_keyword,
            context_only=req.context_only,
            debug=req.debug
        )
    except Exception as e:
        # 모델 에러 / 파일 누락 등 상세한 에러 정보
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] generate_answer 실패:\n{error_detail}")
        
        return ChatResponse(
            answer=f"서버 오류 발생: {str(e)}\n\n자세한 내용은 서버 로그를 확인하세요.",
            contexts=[],
            used_keyword=req.filter_keyword or ""
        )

    # retrieved 보호 — None이면 빈 리스트 처리
    retrieved = retrieved or []

    # ContextItem 변환 (rag_core는 (text, score, keyword) tuple 반환)
    context_items: List[ContextItem] = []
    for item in retrieved:
        try:
            if isinstance(item, tuple) and len(item) >= 3:
                text, score, keyword = item[0], item[1], item[2]
            elif isinstance(item, dict):
                # dict 형태도 지원 (안전장치)
                text = item.get("text", "")
                score = float(item.get("score", 0.0))
                keyword = item.get("keyword", "")
            else:
                # 예상치 못한 형태
                print(f"[WARN] 예상치 못한 retrieved item 형태: {type(item)}")
                text = str(item)
                score = 0.0
                keyword = ""

            context_items.append(ContextItem(
                text=str(text),
                score=float(score),
                keyword=str(keyword) if keyword else ""
            ))
        except Exception as e:
            print(f"[WARN] ContextItem 변환 실패: {e}, item={item}")
            continue

    # 실제 사용된 키워드 결정 (filter_keyword가 있으면 그것, 없으면 컨텍스트에서 추출)
    used_keyword = req.filter_keyword or ""
    if not used_keyword and context_items:
        # 가장 많이 나타난 키워드 사용
        keyword_counts = {}
        for ctx in context_items:
            if ctx.keyword:
                keyword_counts[ctx.keyword] = keyword_counts.get(ctx.keyword, 0) + 1
        if keyword_counts:
            used_keyword = max(keyword_counts.items(), key=lambda x: x[1])[0]

    return ChatResponse(
        answer=answer or "답변이 생성되지 않았습니다.",
        contexts=context_items,
        used_keyword=used_keyword
    )


# ============================================
# 전체 키워드 목록 제공 (프론트에서 Select 박스 용)
# ============================================

@app.get("/keywords")
def keywords():
    """
    키워드별 문서 개수 통계 반환
    
    프론트엔드에서 키워드 필터 선택 박스를 채우는 데 사용됩니다.
    """
    try:
        stats = get_keyword_stats()
        return stats if stats else {}
    except Exception as e:
        print(f"[ERROR] get_keyword_stats() 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================
# 문서 재로드 엔드포인트 (docs_ald.json 변경 후 사용)
# ============================================

@app.post("/reload-docs")
def reload_docs():
    """
    docs_ald.json 파일을 다시 로드하고 임베딩을 재생성합니다.
    
    docs_ald.json 파일을 수정한 후 이 엔드포인트를 호출하면
    서버를 재시작하지 않고도 새로운 문서가 검색에 반영됩니다.
    """
    try:
        num_docs = reload_documents()
        stats = get_keyword_stats()
        return {
            "success": True,
            "message": f"문서 재로딩 완료: {num_docs}개 문서",
            "num_docs": num_docs,
            "keywords": stats
        }
    except Exception as e:
        print(f"[ERROR] reload_documents() 실패: {e}")
        import traceback
        error_detail = traceback.format_exc()
        print(error_detail)
        return {
            "success": False,
            "error": str(e),
            "detail": error_detail
        }
