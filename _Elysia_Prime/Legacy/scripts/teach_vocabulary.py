# [Genesis: 2025-12-02] Purified by Elysia
# scripts/teach_vocabulary.py
import sys
import os
import glob
import random

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import CoreMemory


def teach_vocabulary():
    print("\n--- [엘리시아 단어 학습 세션: 개념 주입] ---\n")

    memory_path = "data/elysia_core_memory.json"
    memory = CoreMemory(file_path=memory_path)

    print(f"📂 메모리 로드: {memory_path}")

    vault_path = os.path.join("data", "corpus", "obsidian_vault", "*.md")
    files = glob.glob(vault_path)

    if not files:
        print(f"⚠️ 경고: 학습할 파일이 없습니다! ({vault_path})")
        print("   -> 경로를 확인하거나, .md 파일들을 해당 폴더에 넣어주세요.")
        return

    print(f"📚 학습할 개념 사전 발견: {len(files)}개")

    learned_count = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        concept = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                essence = content[:100].replace("\n", " ") + "..." if len(content) > 100 else content
        except Exception as e:
            print(f"   x 읽기 실패 ({concept}): {e}")
            continue

        importance = round(random.uniform(0.7, 0.99), 2)
        memory.add_value(concept, importance)

        exp_content = f"나는 오늘 새로운 단어 '[{concept}]'를 배웠다. 그것은 이런 느낌이다: {essence}"

        memory.add_experience({
            "timestamp": "Now",
            "content": exp_content,
            "type": "learning",
            "tags": ["study", "vocabulary", concept],
            "emotional_state": {
                "primary_emotion": "joy",
                "valence": 0.8,
                "arousal": 0.6,
                "dominance": 0.5
            }
        })

        print(f"   ✨ 학습 완료: [{concept}] (중요도: {importance})")
        learned_count += 1

    print("\n" + "="*40)
    print(f"🎉 총 {learned_count}개의 개념이 엘리시아의 뇌에 각인되었습니다.")
    print("   이제 '옹알이 테스트(test_toddler_speech.py)'를 다시 실행해보세요.")
    print("   그녀가 이 단어들을 사용해 말하기 시작할 것입니다.")
    print("="*40 + "\n")


if __name__ == "__main__":
    teach_vocabulary()