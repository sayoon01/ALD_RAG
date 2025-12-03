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
├── docs/                      # 문서 및 데이터
│   ├── docs_ald.json          # RAG 지식 베이스 (키워드 개수별 정리됨)
│   ├── README_DOCS_ORGANIZATION.md  # 문서 정렬 규칙 가이드
│   ├── QUICK_START.md         # 빠른 시작 가이드
│   ├── system/                # 시스템 관련 문서
│   │   ├── EFFICIENCY_CHECK.md
│   │   ├── PERFORMANCE_OPTIMIZATION.md
│   │   └── SYSTEM_VERIFICATION.md
│   └── performance/           # 성능 관련 문서
├── feedback/                  # 피드백 시스템
│   ├── feedback_data.json     # 피드백 데이터
│   └── README.md              # 피드백 시스템 설명
├── finetuning/                # Fine-tuning 관련
│   ├── data/                  # 학습 데이터
│   ├── models/                # Fine-tuned 모델
│   └── scripts/               # Fine-tuning 스크립트
│       ├── prepare_finetuning_data.py
│       ├── finetune_llama.py
│       ├── run_finetuning.sh
│       └── README.md
├── scripts/                   # 유틸리티 스크립트
│   ├── doc_management/        # 문서 관리 스크립트
│   │   ├── merge_docs.py      # 문서 병합/정렬/중복제거
│   │   ├── manage_docs.py     # 문서 관리 (그룹화/통계/추가)
│   │   ├── generate_docs.py   # 문서 생성 (LLM/템플릿)
│   │   ├── extract_from_docs.py  # 문서에서 추출
│   │   ├── migrate_docs_format.py # 문서 형식 변환
│   │   ├── README_DATA_COLLECTION.md  # 데이터 수집 가이드
│   │   └── README_DATA_GENERATION.md  # 데이터 생성 가이드
│   ├── server/                # 서버 관리 스크립트
│   │   ├── run_servers.sh     # 서버 실행 (백엔드+프론트엔드)
│   │   ├── start_server.sh    # 백엔드 서버 시작
│   │   ├── stop_servers.sh    # 서버 종료
│   │   └── 실행방법.md        # 상세 실행 가이드
│   └── test/                  # 테스트/개발 스크립트
│       ├── rag_llama.py       # CLI 챗봇
│       └── embedding_test.py  # 임베딩 테스트
├── logs/             # 로그 파일
├── rag_core.py       # 핵심 RAG 로직
└── torch-env/        # Python 가상환경
```

## 🚀 빠른 시작

### 가장 간단한 방법 (추천!)

```bash
# 프로젝트 루트에서 실행
./scripts/server/run_servers.sh
```

브라우저에서 접속:
- **프론트엔드**: http://localhost:8080
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

### 수동 실행 방법

#### 터미널 1: 백엔드 서버
```bash
cd /home/keti_spark1/ald-rag-lab
source torch-env/bin/activate
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### 터미널 2: 프론트엔드 서버
```bash
cd /home/keti_spark1/ald-rag-lab/frontend
python3 -m http.server 8080 --bind 0.0.0.0
```

### 서버 종료

```bash
# 자동 스크립트 사용
./scripts/server/stop_servers.sh

# 또는 수동 종료
killall uvicorn && killall python3
```

### 상세 실행 가이드

더 자세한 내용은 **[docs/실행방법.md](docs/실행방법.md)** 파일을 참고하세요!

## 📚 주요 기능

### CLI 챗봇

```bash
# 대화 모드
python scripts/test/rag_llama.py --mode chat

# 한 번만 질문
python scripts/test/rag_llama.py --mode once -q "ALD에서 purge는 왜 필요한가요?"

# 통계 보기
python scripts/test/rag_llama.py --mode stats
```

### 문서 관리

```bash
# 키워드별 그룹화 보기
python scripts/doc_management/manage_docs.py group

# 통계 보기
python scripts/doc_management/manage_docs.py stats

# 문서 추가
python scripts/doc_management/manage_docs.py add --keyword ALD --text "새로운 문장"
```

### 데이터 생성

```bash
# LLM 기반 생성 (주의: 전문가 검토 필요)
python scripts/doc_management/generate_docs.py llm --keyword ALD --count 5

# 템플릿 기반 생성
python scripts/doc_management/generate_docs.py template --keyword MFC --count 3

# 문서에서 추출 (권장)
python scripts/doc_management/extract_from_docs.py text --file manual.txt --keywords ALD,Precursor
```

### 문서 관리 (통합)

```bash
# 정렬만 수행 (NEW_DOCS가 빈 리스트인 경우)
python scripts/doc_management/merge_docs.py

# 새 문서 추가 (merge_docs.py의 NEW_DOCS 리스트에 문서 추가 후 실행)
python scripts/doc_management/merge_docs.py
```

**통합된 기능:**
- ✅ 중복 제거 (기존 문서 내 + 새 문서 vs 기존)
- ✅ 키워드 개수별, 종류별 정렬
- ✅ ID 재할당
- ✅ 새 문서 추가 (선택적)

#### 📋 문서 정렬 규칙

`docs_ald.json`의 문서는 다음 규칙에 따라 자동으로 정렬됩니다:

1. **키워드 개수별 정렬**
   - 1개 키워드 → 2개 키워드 → 3개 키워드 → 4개 이상 순서

2. **키워드 종류별 정렬** (같은 개수 내)
   - 키워드를 알파벳/한글 순서로 정렬
   - 예: `["ALD", "Precursor"]` → `["ALD", "Purge"]` → `["MFC", "Flow"]`
   - 모든 키워드를 고려하여 정렬 (첫 번째, 두 번째, ...)

3. **텍스트별 정렬**
   - 키워드가 같으면 텍스트로 정렬

#### 📊 정렬 결과 예시

```
ID 1-36:   1개 키워드 (ALD, Flow, MFC, Precursor, Pressure, Purge, ...)
ID 37-84:  2개 키워드 (ALD+Precursor, ALD+Purge, MFC+Flow, ...)
ID 85-170: 3개 키워드 (플라즈마+압력+챔버, ...)
ID 171+:   4개 이상 키워드
```

#### 🔄 자동 정렬 적용 시점

- `reorganize_docs.py` 실행 시: 전체 문서 재정렬
- `merge_docs.py` 실행 시: 새 문서 추가 후 자동 정렬
- 문서가 항상 정렬된 상태로 유지됨

## 📖 상세 문서

- **데이터 수집 가이드**: `docs/README_DATA_COLLECTION.md`
- **데이터 생성 가이드**: `docs/README_DATA_GENERATION.md`
- **문서 정렬 및 관리 가이드**: `docs/README_DOCS_ORGANIZATION.md`

## 🔧 기술 스택

- **LLM**: Meta LLaMA-3-8B-Instruct
- **Embedding**: thenlper/gte-small
- **Backend**: FastAPI
- **Frontend**: HTML/CSS/JavaScript
- **Python**: 3.12+

## 📝 라이선스

MIT License




