# [Genesis: 2025-12-02] Purified by Elysia
import json
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import CoreMemory


def train_grammar() -> None:
    print("\n--- [엘리시아 문법 학습: 패턴 관찰] ---\n")

    memory_path = "data/elysia_core_memory.json"
    if not os.path.exists(memory_path):
        print("[경고] 기억이 없습니다. 'teach_vocabulary.py'를 먼저 실행하세요.")
        return

    memory = CoreMemory(file_path=memory_path)
    concepts = set()
    for entry in memory.get_values():
        value = entry.get("value")
        if isinstance(value, str) and value:
            concepts.add(value)

    if not concepts:
        print("[경고] 아는 단어가 없어서 학습할 수 없습니다.")
        return

    print(f"[학습] 엘리시아가 아는 단어 {len(concepts)}개의 용례를 찾습니다...")

    corpus_files = glob.glob("data/corpus/**/*.txt", recursive=True)
    corpus_files += glob.glob("data/corpus/**/*.md", recursive=True)
    if not corpus_files:
        print("[경고] 코퍼스 파일을 찾을 수 없습니다. data/corpus/를 확인하세요.")
        return

    grammar_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    valid_suffixes = re.compile(r"^[은는이가을를의에로과와도만]+$|^[습니다니다입니다같습니다]+$")

    for path in corpus_files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception as err:
            print(f"   x 읽기 실패 ({os.path.basename(path)}): {err}")
            continue

        clean_text = re.sub(r"[^\w\s가-힣]", " ", text)
        tokens = clean_text.split()
        for token in tokens:
            for concept in concepts:
                if not token.startswith(concept) or len(token) <= len(concept):
                    continue
                suffix = token[len(concept):]
                if len(suffix) > 5:
                    continue
                if valid_suffixes.match(suffix):
                    grammar_stats[concept][suffix] += 1

    model_path = "data/grammar_model.json"
    serializable_stats = {concept: dict(stats) for concept, stats in grammar_stats.items()}
    if not serializable_stats:
        print("[경고] 뽑아낸 문법 통계가 없습니다.")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as out:
            json.dump(serializable_stats, out, ensure_ascii=False, indent=2)
        print(f"\n[완료] 학습 완료! 모델 저장됨: {model_path}")
        sample = next(iter(serializable_stats))
        print(f"   예시: '{sample}' 뒤에 자주 오는 말 -> {serializable_stats[sample]}")


if __name__ == "__main__":
    train_grammar()