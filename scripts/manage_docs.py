# manage_docs.py
"""
docs_ald.json 관리 스크립트

- stats : 키워드별 문장 개수 통계 출력
- group : 키워드별로 그룹화해서 문서 보기
- add   : 키워드 + 문장 추가 (인터랙티브 OR 옵션 입력)

사용 예시:
    cd ~/ald-rag-lab
    python manage_docs.py stats
    python manage_docs.py group
    python manage_docs.py add
    python manage_docs.py add --keyword Precursor --text "새로 추가할 문장..."
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent  # ~/ald-rag-lab
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"


# ==============================
# 1) 파일 유틸
# ==============================

def load_raw_docs() -> List[Dict[str, Any]]:
    """
    docs_ald.json을 읽어서 리스트 구조를 반환.
    - 새 포맷: { "documents": [{"id": 1, "keywords": ["ALD"], "text": "..."}, ...] }
    - 기존 포맷 호환 지원
    """
    if not DOCS_PATH.exists():
        print(f"[WARN] {DOCS_PATH} 가 아직 없음. 새로 생성할 예정.")
        return []

    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "documents" in data:
        documents = data["documents"]
        
        # 새 구조 (리스트 형태)
        if isinstance(documents, list):
            return documents
        
        # 기존 구조 (키워드별 그룹화) - 새 구조로 변환
        elif isinstance(documents, dict):
            result = []
            next_id = 1
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
                    result.append({
                        "id": next_id,
                        "keywords": [keyword],
                        "text": text
                    })
                    next_id += 1
            return result

    print("[WARN] docs_ald.json 구조가 예상과 다름. 빈 리스트 반환.")
    return []


def save_raw_docs(docs: List[Dict[str, Any]]) -> None:
    """
    리스트 구조로 저장:
    {
      "documents": [
        {"id": 1, "keywords": ["ALD"], "text": "..."},
        ...
      ]
    }
    """
    DOCS_PATH.parent.mkdir(exist_ok=True, parents=True)
    wrapper = {"documents": docs}
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 완료: {DOCS_PATH} (총 {len(docs)} 문서)")


def get_next_id(docs: List[Dict[str, Any]]) -> int:
    """다음 ID 번호 계산"""
    max_id = 0
    for item in docs:
        try:
            item_id = int(item.get("id", 0))
            if item_id > max_id:
                max_id = item_id
        except (ValueError, TypeError):
            continue
    return max_id + 1


# ==============================
# 2) stats 모드
# ==============================

def run_stats():
    if not DOCS_PATH.exists():
        print(f"[stats] {DOCS_PATH} 가 없음.")
        return

    docs = load_raw_docs()
    if not docs:
        print("[stats] 문서가 비어 있음.")
        return

    # 키워드별 카운트
    keyword_counts: Dict[str, int] = {}
    for item in docs:
        keywords = item.get("keywords", [])
        if isinstance(keywords, list):
            for kw in keywords:
                kw = str(kw).strip()
                if kw:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        elif isinstance(keywords, str):
            kw = keywords.strip()
            if kw:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    print("\n[키워드 통계] (docs_ald.json 기준)")
    for kw in sorted(keyword_counts.keys()):
        print(f"- {kw}: {keyword_counts[kw]} 문장")


# ==============================
# 2-1) group 모드: 키워드별로 그룹화해서 보기
# ==============================

def run_group():
    """
    키워드별로 문서를 그룹화해서 보기 좋게 출력
    """
    if not DOCS_PATH.exists():
        print(f"[group] {DOCS_PATH} 가 없음.")
        return

    docs = load_raw_docs()
    if not docs:
        print("[group] 문서가 비어 있음.")
        return

    # 키워드별로 그룹화
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in docs:
        keywords = item.get("keywords", [])
        if isinstance(keywords, list):
            for kw in keywords:
                kw = str(kw).strip()
                if kw:
                    if kw not in grouped:
                        grouped[kw] = []
                    grouped[kw].append(item)
        elif isinstance(keywords, str):
            kw = keywords.strip()
            if kw:
                if kw not in grouped:
                    grouped[kw] = []
                grouped[kw].append(item)

    print("\n" + "=" * 80)
    print("[키워드별 문서 그룹]")
    print("=" * 80)

    total_count = len(docs)
    for kw in sorted(grouped.keys()):
        items = grouped[kw]
        print(f"\n📌 keyword = {kw} ({len(items)}개)")
        print("-" * 80)
        for idx, item in enumerate(items, 1):
            text = str(item.get("text", "")).strip()
            item_id = item.get("id", "?")
            keywords_str = ", ".join(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else str(item.get("keywords", ""))
            print(f"  {idx}. [ID:{item_id}] [{keywords_str}] {text}")
    
    print("\n" + "=" * 80)
    print(f"총 {total_count}개 문서, {len(grouped)}개 키워드")
    print("=" * 80)


# ==============================
# 3) add 모드
# ==============================

def run_add(keyword: str, text: str):
    docs = load_raw_docs()

    # 인터랙티브 모드
    if not keyword:
        keyword = input("키워드 입력 (예: Precursor, Purge, Plasma, MFC, Flow...): ").strip()
    if not text:
        text = input("문장 입력: ").strip()

    if not keyword or not text:
        print("[ERROR] 키워드와 문장은 비어있으면 안 됨.")
        return

    # 키워드를 배열로 변환 (쉼표로 구분된 경우 처리)
    keywords_list = [kw.strip() for kw in keyword.split(",") if kw.strip()]

    # 새 항목 추가
    next_id = get_next_id(docs)
    new_item = {
        "id": next_id,
        "keywords": keywords_list,
        "text": text
    }
    docs.append(new_item)
    save_raw_docs(docs)

    print("\n[추가된 항목]")
    print(f"- id      : {next_id}")
    print(f"- keywords: {', '.join(keywords_list)}")
    print(f"- text    : {text}")


# ==============================
# 4) 메인
# ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="docs_ald.json 관리 스크립트")
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    sub.add_parser("stats", help="키워드별 문장 개수 통계 출력")

    # group
    sub.add_parser("group", help="키워드별로 그룹화해서 문서 보기")

    # add
    p_add = sub.add_parser("add", help="문장 추가")
    p_add.add_argument("--keyword", type=str, default="", help="키워드 (없으면 인터랙티브)")
    p_add.add_argument("--text", type=str, default="", help="문장 내용 (없으면 인터랙티브)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "stats":
        run_stats()
    elif args.command == "group":
        run_group()
    elif args.command == "add":
        run_add(args.keyword, args.text)
