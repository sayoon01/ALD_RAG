# backend/app.py

import sys
from pathlib import Path
from typing import List

# ==============================
# 0) 상위 폴더(ald-rag-lab)를 import 경로에 추가
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent  # ~/ald-rag-lab
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from rag_llama import generate_answer  

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# FastAPI 설정
# ==============================

app = FastAPI(
    title="반도체 ALD RAG API",
    description="반도체 ALD 공정 관련 RAG 챗봇 API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로
    redoc_url="/redoc"  # ReDoc 경로
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Request/Response 모델
# ==============================

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    max_new_tokens: int = 256

class ContextItem(BaseModel):
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    contexts: List[ContextItem]

# ==============================
# API 엔드포인트
# ==============================

@app.get("/")
def root():
    return {"message": "ALD RAG API 정상 동작 중"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    answer, retrieved = generate_answer(
        req.question,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
    )

    contexts = []
    for item in retrieved:
        if isinstance(item, tuple):
            text, score = item
        else:
            text = item.get("text", "")
            score = float(item.get("score", 0.0))
        contexts.append(ContextItem(text=text, score=score))

    return ChatResponse(answer=answer, contexts=contexts)
