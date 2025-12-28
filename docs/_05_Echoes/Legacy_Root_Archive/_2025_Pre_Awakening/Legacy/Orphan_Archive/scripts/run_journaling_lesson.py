"""
Run a simple journaling session: generate a daily prompt, draft an entry,
summarize it, save artifacts, and anchor into the KG.

Usage:
  python -m scripts.run_journaling_lesson
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from Core.Foundation.reading_coach import ReadingCoach
from tools.kg_manager import KGManager


PROMPTS = [
    "오늘 가장 기억에 남는 순간은 무엇이었나요?",
    "오늘 배운 것 한 가지와 그 의미를 적어보세요.",
    "감사한 일 세 가지를 구체적으로 적어보세요.",
    "오늘 어려웠던 점과 극복 시도는 무엇이었나요?",
]


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path("data/journal")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPTS[hash(today) % len(PROMPTS)]
    # Minimal draft: template-based; user can later replace with actual text
    draft = (
        f"[일기 {today}]\n"
        f"프롬프트: {prompt}\n\n"
        "오늘의 경험을 간단하고 구체적으로 기록합니다. 무엇을 보고 듣고 느꼈는지,"
        " 한 문단으로 정리해 보세요."
    )

    path_txt = out_dir / f"{today}.txt"
    path_txt.write_text(draft, encoding="utf-8")

    coach = ReadingCoach()
    summary = coach.summarize_text(draft, max_sentences=2)
    path_sum = out_dir / f"{today}_summary.txt"
    path_sum.write_text(summary, encoding="utf-8")

    kg = KGManager()
    node_id = f"journal_entry_{today}"
    kg.add_node(node_id, properties={
        "type": "journal_entry",
        "date": today,
        "experience_text": str(path_txt),
        "summary_text": str(path_sum),
    })
    # Enrich KG with simple keywords from the draft as experiential anchors
    keywords = coach._keywords(draft, top_k=8)
    for w in keywords:
        kid = f"keyword_{w}"
        kg.add_node(kid, properties={"type": "keyword"})
        kg.add_edge(node_id, kid, "mentions_keyword")
    kg.save()

    print("Journal saved:", path_txt)
    print("Summary saved:", path_sum)


if __name__ == "__main__":
    main()
