# [Genesis: 2025-12-02] Purified by Elysia
import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

OUTPUT_LOG = Path("logs/language_progress.jsonl")


def run_watch_session() -> str:
    completed = subprocess.run(
        ["python", "scripts/watch_elysia_socialize.py"],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"watch_elysia_socialize failed: {completed.stderr}")
    return completed.stdout


def parse_reasoning(text: str) -> List[Dict]:
    reasoning_pattern = re.compile(
        r"\[reasoning intent=(?P<intent>[^ ]+) subject=(?P<subject>[^ ]+) path=(?P<path>[^\]]+)\]"
    )
    law_pattern = re.compile(
        r"\[law_focus=(?P<focus>\w+) scores=(?P<scores>\{.*\})\]"
    )

    entries = []
    lines = text.splitlines()
    last_reasoning = {}
    for line in lines:
        reasoning_match = reasoning_pattern.search(line)
        if reasoning_match:
            last_reasoning = {
                "intent": reasoning_match.group("intent"),
                "subject": reasoning_match.group("subject"),
                "path": [segment.strip() for segment in reasoning_match.group("path").split("->")],
            }
            entries.append(last_reasoning.copy())
            continue
        law_match = law_pattern.search(line)
        if law_match and last_reasoning:
            try:
                scores = ast.literal_eval(law_match.group("scores"))
            except Exception:
                scores = {}
            last_reasoning["law_focus"] = law_match.group("focus")
            last_reasoning["law_scores"] = scores
    return entries


def summarize(entries: List[Dict]) -> Dict:
    counter = Counter()
    law_counter = Counter()
    law_scores_total: Dict[str, float] = {}
    law_scores_count: Dict[str, int] = {}
    path_lengths = []

    for entry in entries:
        counter[(entry["intent"], entry["subject"])] += 1
        if "law_focus" in entry:
            lf = entry["law_focus"]
            law_counter[lf] += 1
            for axis, value in (entry.get("law_scores") or {}).items():
                law_scores_total[axis] = law_scores_total.get(axis, 0.0) + float(value)
                law_scores_count[axis] = law_scores_count.get(axis, 0) + 1
        path_lengths.append(len(entry.get("path", [])))

    average_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0

    avg_scores = {axis: law_scores_total[axis] / law_scores_count[axis] for axis in law_scores_total}

    return {
        "turns": len(entries),
        "intent_subject_variants": len(counter),
        "law_focus_counts": dict(law_counter),
        "average_path_length": round(average_path, 2),
        "average_law_scores": {k: round(v, 3) for k, v in avg_scores.items()},
    }


def append_log(summary: Dict, details: List[Dict]) -> None:
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"summary": summary, "details": details}
    with OUTPUT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(runs: int = 2) -> None:
    aggregate: Dict[str, float] = {"total_turns": 0, "runs": 0}
    for idx in range(runs):
        print(f"=== Session {idx+1}/{runs} ===")
        output = run_watch_session()
        entries = parse_reasoning(output)
        if not entries:
            print("No reasoning entries captured.")
            continue
        summary = summarize(entries)
        append_log(summary, entries)
        aggregate["total_turns"] += summary["turns"]
        aggregate["runs"] += 1
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if aggregate["runs"]:
        print(f"Average turns per run: {aggregate['total_turns'] / aggregate['runs']:.2f}")


if __name__ == "__main__":
    main(runs=2)