"""
Curiosity Loop (semi-auto)

- 기록: 부족한 개념/주제를 요청으로 남긴다 (--request).
- 주입: 사람이 검수한 파일을 지정 개념으로 먹인다 (--ingest).
  -> data/curiosity/ingested.jsonl 에 저장 (concept, path, text).

의존성: 없음(표준 라이브러리).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CUR_DIR = Path("data/curiosity")
CUR_DIR.mkdir(parents=True, exist_ok=True)
REQ_PATH = CUR_DIR / "requests.jsonl"
INGEST_PATH = CUR_DIR / "ingested.jsonl"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def do_request(concept: str, detail: str | None) -> None:
    rec = {"concept": concept, "detail": detail or "", "status": "pending"}
    append_jsonl(REQ_PATH, rec)
    print(f"[curiosity] request recorded for concept='{concept}'")


def do_ingest(concept: str, file_path: Path) -> None:
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    rec = {"concept": concept, "file": str(file_path), "text": text}
    append_jsonl(INGEST_PATH, rec)
    print(f"[curiosity] ingested file for concept='{concept}' from {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curiosity Loop helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_req = sub.add_parser("request", help="record a missing concept/topic")
    p_req.add_argument("--concept", required=True, help="부족한 개념/주제")
    p_req.add_argument("--detail", default="", help="추가 설명(선택)")

    p_ing = sub.add_parser("ingest", help="ingest a reviewed file for a concept")
    p_ing.add_argument("--concept", required=True, help="채워줄 개념/주제")
    p_ing.add_argument("--file", required=True, type=Path, help="검수된 텍스트 파일 경로")

    args = parser.parse_args()

    if args.cmd == "request":
        do_request(args.concept, args.detail)
    elif args.cmd == "ingest":
        do_ingest(args.concept, args.file)


if __name__ == "__main__":
    main()
