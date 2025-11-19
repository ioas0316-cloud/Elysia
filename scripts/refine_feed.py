import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

WORD_PATTERN = re.compile(r"[가-힣A-Za-z]+")
STOP_WORDS: Set[str] = {
    "the", "and", "that", "with", "this", "from", "into", "about", "just",
    "또한", "하지만", "그리고", "그녀", "그는", "이것", "저것", "하는", "있다",
    "하기", "하는데", "그러나", "모든", "것이", "무엇", "같은", "때문", "말이"
}


def _normalize_text(text: str) -> List[str]:
    tokens = WORD_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def _hash_file(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def refine_feed(
    incoming_dir: Path,
    archive_raw: Path,
    summary_dir: Path,
    dedup_file: Path,
) -> int:
    archive_raw.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    existing_hashes: Dict[str, str] = {}
    if dedup_file.exists():
        try:
            existing_hashes = json.loads(dedup_file.read_text(encoding="utf-8"))
        except Exception:
            existing_hashes = {}

    processed = 0
    for path in list(incoming_dir.iterdir()):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        hashed = _hash_file(path)
        if hashed in existing_hashes:
            path.unlink(missing_ok=True)
            continue

        tokens = _normalize_text(content)
        top_words = Counter(tokens).most_common(10)
        summary = {
            "source": path.name,
            "top_words": top_words,
            "length": len(content),
        }
        summary_path = summary_dir / f"{path.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        existing_hashes[hashed] = path.name
        target = archive_raw / path.name
        if target.exists():
            target = archive_raw / f"{int(target.stat().st_mtime)}_{path.name}"
        path.rename(target)
        processed += 1

    dedup_file.write_text(json.dumps(existing_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    return processed
