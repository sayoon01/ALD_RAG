# 반도체 ALD RAG 챗봇

LLaMA-3 기반 RAG (Retrieval Augmented Generation) 시스템으로 반도체 ALD 공정 관련 질의응답을 제공합니다.

## 📁 프로젝트 구조

```
ald-rag-lab/
├── backend/           # FastAPI 백엔드
│   └── app.py        # API 서버
├── frontend/         # 웹 프론트엔드
│   ├── index.html
│   ├── main.js
│   └── style.css
├── docs/             # 문서 및 데이터
│   ├── docs_ald.json # RAG 지식 베이스
│   └── README_*.md   # 데이터 수집 가이드
├── scripts/          # 유틸리티 스크립트
│   ├── rag_llama.py          # CLI 챗봇
│   ├── manage_docs.py        # 문서 관리
│   ├── generate_docs.py      # 데이터 생성
│   ├── extract_from_docs.py  # 문서에서 추출
│   ├── migrate_docs_format.py # 형식 변환
│   ├── embedding_test.py     # 임베딩 테스트
│   └── start_server.sh       # 서버 시작 스크립트
├── logs/             # 로그 파일
├── rag_core.py       # 핵심 RAG 로직
└── torch-env/        # Python 가상환경
```

## 🚀 빠른 시작

### 1. 가상환경 활성화

```bash
source torch-env/bin/activate
```

### 2. 서버 시작

```bash
# 방법 1: 스크립트 사용
./scripts/start_server.sh

# 방법 2: 직접 실행
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. 프론트엔드 접속

브라우저에서 접속:
- 로컬: `http://127.0.0.1:8000`
- 네트워크: `http://<서버IP>:8000` (ifconfig로 확인)

## 📚 주요 기능

### CLI 챗봇

```bash
# 대화 모드
python scripts/rag_llama.py --mode chat

# 한 번만 질문
python scripts/rag_llama.py --mode once -q "ALD에서 purge는 왜 필요한가요?"

# 통계 보기
python scripts/rag_llama.py --mode stats
```

### 문서 관리

```bash
# 키워드별 그룹화 보기
python scripts/manage_docs.py group

# 통계 보기
python scripts/manage_docs.py stats

# 문서 추가
python scripts/manage_docs.py add --keyword ALD --text "새로운 문장"
```

### 데이터 생성

```bash
# LLM 기반 생성 (주의: 전문가 검토 필요)
python scripts/generate_docs.py llm --keyword ALD --count 5

# 템플릿 기반 생성
python scripts/generate_docs.py template --keyword MFC --count 3

# 문서에서 추출 (권장)
python scripts/extract_from_docs.py text --file manual.txt --keywords ALD,Precursor
```

## 📖 상세 문서

- **데이터 수집 가이드**: `docs/README_DATA_COLLECTION.md`
- **데이터 생성 가이드**: `docs/README_DATA_GENERATION.md`

## 🔧 기술 스택

- **LLM**: Meta LLaMA-3-8B-Instruct
- **Embedding**: thenlper/gte-small
- **Backend**: FastAPI
- **Frontend**: HTML/CSS/JavaScript
- **Python**: 3.12+

## 📝 라이선스

MIT License


