# manage_docs.py
"""
docs_ald.json 관리 스크립트

- stats : 키워드별 문장 개수 통계 출력
- add   : 키워드 + 문장 추가 (인터랙티브 OR 옵션 입력)

사용 예시:
    cd ~/ald-rag-lab
    python manage_docs.py stats
    python manage_docs.py add
    python manage_docs.py add --keyword Precursor --text "새로 추가할 문장..."
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import rag_core  # 키워드 통계 함수를 재사용


BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"


# ==============================
# 1) 파일 유틸
# ==============================

def load_raw_docs() -> List[Dict[str, Any]]:
    if not DOCS_PATH.exists():
        print(f"[WARN] {DOCS_PATH} 가 아직 없음. 새로 생성할 예정.")
        return []

    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미 리스트라 가정. 만약 dict 구조면 여기서 추가 처리하면 됨.
    if isinstance(data, list):
        return data

    print("[WARN] docs_ald.json 구조가 리스트가 아님. 강제로 리스트로 변환.")
    return list(data)


def save_raw_docs(docs: List[Dict[str, Any]]) -> None:
    DOCS_PATH.parent.mkdir(exist_ok=True, parents=True)
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 완료: {DOCS_PATH} (총 {len(docs)} 문장)")


def get_next_id(docs: List[Dict[str, Any]]) -> int:
    max_id = 0
    for item in docs:
        try:
            val = int(item.get("id", 0))
            if val > max_id:
                max_id = val
        except Exception:
            continue
    return max_id + 1


# ==============================
# 2) stats 모드
# ==============================

def run_stats():
    if not DOCS_PATH.exists():
        print(f"[stats] {DOCS_PATH} 가 없음.")
        return

    # rag_core 쪽 get_keyword_stats 재사용
    if hasattr(rag_core, "get_keyword_stats"):
        stats = rag_core.get_keyword_stats()
        print("\n[키워드 통계] (rag_core 기준)")
        if not stats:
            print("- 통계 없음 (키워드가 없거나 문서가 없음)")
            return
        for kw in sorted(stats.keys()):
            print(f"- {kw}: {stats[kw]} 문장")
    else:
        # 혹시 rag_core에 함수 없으면 직접 계산
        raw = load_raw_docs()
        counter: Dict[str, int] = {}
        for item in raw:
            kw = item.get("keyword", "unknown")
            counter[kw] = counter.get(kw, 0) + 1

        print("\n[키워드 통계] (manage_docs 직접 계산)")
        if not counter:
            print("- 통계 없음 (키워드가 없거나 문서가 없음)")
            return
        for kw in sorted(counter.keys()):
            print(f"- {kw}: {counter[kw]} 문장")


# ==============================
# 3) add 모드
# ==============================

def run_add(keyword: str, text: str):
    docs = load_raw_docs()
    next_id = get_next_id(docs)

    # 인터랙티브 모드
    if not keyword:
        keyword = input("키워드 입력 (예: Precursor, Purge, Pressure...): ").strip()
    if not text:
        text = input("문장 입력: ").strip()

    if not keyword or not text:
        print("[ERROR] 키워드와 문장은 비어있으면 안 됨.")
        return

    new_item = {
        "id": next_id,
        "keyword": keyword,
        "text": text,
    }

    docs.append(new_item)
    save_raw_docs(docs)

    print("\n[추가된 항목]")
    print(f"- id      : {next_id}")
    print(f"- keyword: {keyword}")
    print(f"- text   : {text}")


# ==============================
# 4) 메인
# ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="docs_ald.json 관리 스크립트")
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    sub.add_parser("stats", help="키워드별 문장 개수 통계 출력")

    # add
    p_add = sub.add_parser("add", help="문장 추가")
    p_add.add_argument("--keyword", type=str, default="", help="키워드 (없으면 인터랙티브)")
    p_add.add_argument("--text", type=str, default="", help="문장 내용 (없으면 인터랙티브)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "stats":
        run_stats()
    elif args.command == "add":
        run_add(args.keyword, args.text)
