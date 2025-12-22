"""
Causal episode extraction from WORLD logs (2025-11-16).

Purpose
- Build human-like "원인-행동-결과" 에피소드 단위를 만들어,
  Elysia가 패턴이 아니라 인과 구조를 먼저 볼 수 있게 돕는다.

Design
- Read `logs/world_events.jsonl` (WORLD telemetry).
- Read a Character view (id, alignment, notoriety, job 등).
- For 일부 이벤트(KILL, SPELL, EAT/DRINK)에서
  - 행동자/대상자의 상태 요약 (pre_state)
  - 사건 자체 (action)
  - 즉시 관찰 가능한 결과(result_type)
  로 묶어 JSONL로 기록한다.

Notes
- 여기서는 "관찰된 사실"만 기록하고, alignment 변화 같은 것은
  별도의 추론/학습 단계에서 다루도록 남겨둔다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


def _ensure_repo_root_on_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_ensure_repo_root_on_path()

from scripts.character_model import Character
from scripts.relationship_events import load_events_from_log


@dataclass
class CausalEpisode:
    """Minimal causal episode unit for Elysia."""

    timestamp: int
    event_type: str
    actor_id: Optional[str]
    target_id: Optional[str]
    # Local world snapshot (only what we can observe directly here)
    actor_state: Dict[str, Any]
    target_state: Dict[str, Any]
    # World context (shallow)
    context: Dict[str, Any]
    # Immediate, observable outcome type (no long-term inference here)
    result_type: str


@dataclass
class SymbolEpisode:
    """
    Episode unit for 문자/기호 학습.

    - symbol_type: "letter", "digit", "shape", "word" 등.
    - symbol: 실제 기호 (예: "ㄱ", "가", "1").
    - modality: "text", "image" 등 제시 방식.
    - teacher_label: 교사가 정답으로 제시한 기호/이름.
    - learner_guess: 학습자가 시도한 답.
    - correct: 정답 여부.
    - score: 부분 점수나 유사도 (0.0~1.0 권장).
    """

    timestamp: int
    event_type: str
    symbol_type: str
    symbol: str
    modality: str
    teacher_label: str
    learner_guess: str
    correct: bool
    score: float
    context: Dict[str, Any]


@dataclass
class TextEpisode:
    """
    Episode unit for 문장/문단 학습.

    - text_type: "sentence_ko", "sentence_en", "paragraph_ko" 등.
    - teacher_text: 교사가 제시한 정답 문장/문단.
    - learner_text: 학습자가 생성한 문장/문단.
    - correct: 정답 여부(있다면), 없으면 score 기반으로만 평가.
    - score: 유사도/채점 점수 (0.0~1.0 권장).
    """

    timestamp: int
    event_type: str
    text_type: str
    teacher_text: str
    learner_text: str
    correct: bool
    score: float
    context: Dict[str, Any]


@dataclass
class CodeEpisode:
    """
    Episode unit for CodeWorld learning.

    - entity_id: 대상 코드 개체 id.
    - kind: "function" / "module" / "test" / "bug" / ...
    - event_type: BUG_SPAWNED / TEST_STRENGTHENED / REFACTOR_APPLIED 등.
    - complexity_before/after, stability_before/after, coverage 등 메타데이터.
    """

    timestamp: int
    event_type: str
    entity_id: str
    kind: str
    data: Dict[str, Any]


def _character_state(ch: Optional[Character]) -> Dict[str, Any]:
    if ch is None:
        return {}
    return {
        "id": ch.id,
        "race": getattr(ch, "race", None),
        "faction": ch.faction,
        "power_score": float(getattr(ch, "power_score", 0.0)),
        "alignment_law": float(getattr(ch, "alignment_law", 0.0)),
        "alignment_good": float(getattr(ch, "alignment_good", 0.0)),
        "notoriety": float(getattr(ch, "notoriety", 0.0)),
        "job_id": getattr(ch, "job_id", None),
    }


def _make_context(world) -> Dict[str, Any]:
    """Shallow macro context snapshot for the episode."""
    return {
        "time_step": int(getattr(world, "time_step", 0)),
        "macro_war_pressure": float(getattr(world, "macro_war_pressure", 0.0)),
        "macro_monster_threat": float(getattr(world, "macro_monster_threat", 0.0)),
        "macro_unrest": float(getattr(world, "macro_unrest", 0.0)),
        "macro_population": float(getattr(world, "macro_population", 0.0)),
    }


def build_causal_episodes(
    world,
    chars: List[Character],
    events: Iterable[Dict[str, Any]],
) -> List[CausalEpisode]:
    """
    Build a list of CausalEpisode from WORLD events and Character state.

    - We assume `chars` is a snapshot taken 후반부에; 따라서
      alignment/악명 값은 "현재까지의 결과"에 가깝다.
    - 이 함수는 관찰가능한 정보를 episode 형태로 재구성만 하고,
      어떤 규칙/학습도 적용하지 않는다.
    """
    chars_by_id: Dict[str, Character] = {ch.id: ch for ch in chars}
    ctx = _make_context(world)

    episodes: List[CausalEpisode] = []
    for ev in events:
        etype = ev.get("event_type") or ""
        data = ev.get("data", {}) or {}
        ts = int(ev.get("timestamp", 0))

        actor_id: Optional[str] = None
        target_id: Optional[str] = None
        result_type = "unknown"

        if etype == "KILL":
            actor_id = data.get("killer_id")
            target_id = data.get("victim_id")
            victim_element = str(data.get("victim_element", "") or "")
            if victim_element in ("human", "citizen"):
                result_type = "kill_human"
            else:
                result_type = "kill_monster"
        elif etype == "SPELL":
            actor_id = data.get("caster_id")
            target_id = data.get("target_id")
            spell = str(data.get("spell", "") or "").lower()
            if "heal" in spell:
                result_type = "cast_heal"
            else:
                result_type = "cast_offense"
        elif etype in ("EAT", "DRINK"):
            actor_id = data.get("actor_id") or data.get("cell_id")
            provider = data.get("provider_id")
            target_id = provider
            result_type = "eat_or_drink"
        else:
            # For now, we only turn a subset of events into episodes.
            continue

        actor = chars_by_id.get(str(actor_id)) if actor_id else None
        target = chars_by_id.get(str(target_id)) if target_id else None

        episode = CausalEpisode(
            timestamp=ts,
            event_type=etype,
            actor_id=str(actor_id) if actor_id is not None else None,
            target_id=str(target_id) if target_id is not None else None,
            actor_state=_character_state(actor),
            target_state=_character_state(target),
            context=ctx,
            result_type=result_type,
        )
        episodes.append(episode)

    return episodes


def build_symbol_episodes(
    events: Iterable[Dict[str, Any]],
) -> List[SymbolEpisode]:
    """
    Build SymbolEpisode list from generic lesson/log events.

    Expected event format (JSONL 한 줄 기준 예시):
    {
      "timestamp": 123,
      "event_type": "SYMBOL_LESSON",
      "data": {
        "symbol_type": "letter",
        "symbol": "ㄱ",
        "modality": "text",
        "teacher_label": "ㄱ",
        "learner_guess": "ㄴ",
        "correct": false,
        "score": 0.2,
        "lesson_id": "hangeul_basic_1"
      }
    }

    - event_type는 "SYMBOL_LESSON" 이거나,
      "LETTER_SHOWN" / "DIGIT_SHOWN" / "SHAPE_SHOWN" / "WORD_SHOWN" 등으로 확장 가능하다.
    - 정의되지 않은 필드는 context 안에 그대로 보존한다.
    """
    episodes: List[SymbolEpisode] = []

    for ev in events:
        etype = str(ev.get("event_type") or "")
        data = ev.get("data", {}) or {}
        ts = int(ev.get("timestamp", 0))

        if etype not in (
            "SYMBOL_LESSON",
            "LETTER_SHOWN",
            "DIGIT_SHOWN",
            "SHAPE_SHOWN",
            "WORD_SHOWN",
        ):
            continue

        symbol_type = str(data.get("symbol_type") or "unknown")
        symbol = str(data.get("symbol") or "")
        modality = str(data.get("modality") or "text")
        teacher_label = str(data.get("teacher_label") or symbol)
        learner_guess = str(data.get("learner_guess") or "")
        correct = bool(data.get("correct", False))
        score = float(data.get("score", 1.0 if correct else 0.0))

        # 나머지 필드는 context로 남겨둔다.
        known_keys = {
            "symbol_type",
            "symbol",
            "modality",
            "teacher_label",
            "learner_guess",
            "correct",
            "score",
        }
        context = {k: v for k, v in data.items() if k not in known_keys}

        episodes.append(
            SymbolEpisode(
                timestamp=ts,
                event_type=etype,
                symbol_type=symbol_type,
                symbol=symbol,
                modality=modality,
                teacher_label=teacher_label,
                learner_guess=learner_guess,
                correct=correct,
                score=score,
                context=context,
            )
        )

    return episodes


def build_text_episodes(
    events: Iterable[Dict[str, Any]],
) -> List[TextEpisode]:
    """
    Build TextEpisode list from generic sentence/paragraph lesson events.

    Expected event format (JSONL 한 줄 예시):
    {
      "timestamp": 123,
      "event_type": "SENTENCE_LESSON",
      "data": {
        "text_type": "sentence_ko",
        "teacher_text": "지금은 전쟁의 시대다.",
        "learner_text": "지금은 전쟁의 시대다.",
        "correct": true,
        "score": 1.0,
        "lesson_id": "era_summary_1"
      }
    }

    - event_type는 "SENTENCE_LESSON" 또는 "PARAGRAPH_LESSON" 등을 사용할 수 있다.
    - 정의되지 않은 필드는 context 안에 그대로 보존한다.
    """
    episodes: List[TextEpisode] = []

    for ev in events:
        etype = str(ev.get("event_type") or "")
        data = ev.get("data", {}) or {}
        ts = int(ev.get("timestamp", 0))

        if etype not in ("SENTENCE_LESSON", "PARAGRAPH_LESSON"):
            continue

        text_type = str(data.get("text_type") or "sentence_ko")
        teacher_text = str(data.get("teacher_text") or "")
        learner_text = str(data.get("learner_text") or "")
        correct = bool(data.get("correct", False))
        score = float(data.get("score", 1.0 if correct else 0.0))

        known_keys = {
            "text_type",
            "teacher_text",
            "learner_text",
            "correct",
            "score",
        }
        context = {k: v for k, v in data.items() if k not in known_keys}

        episodes.append(
            TextEpisode(
                timestamp=ts,
                event_type=etype,
                text_type=text_type,
                teacher_text=teacher_text,
                learner_text=learner_text,
                correct=correct,
                score=score,
                context=context,
            )
        )

    return episodes


def export_symbol_episodes(
    events_path: str = "logs/symbol_lessons.jsonl",
    out_path: str = "logs/symbol_episodes.jsonl",
) -> str:
    """
    Convenience entry: read generic symbol/lesson logs and emit SymbolEpisode JSONL.
    """
    events = list(load_events_from_log(events_path))
    episodes = build_symbol_episodes(events)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
    return out_path


def export_text_episodes(
    events_path: str = "logs/text_lessons.jsonl",
    out_path: str = "logs/text_episodes.jsonl",
) -> str:
    """
    Read generic text/sentence lesson logs and emit TextEpisode JSONL.
    """
    events = list(load_events_from_log(events_path))
    episodes = build_text_episodes(events)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
    return out_path


def summarize_symbol_episodes_dicts(
    episodes: Iterable[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Summarise symbol learning performance from a stream of SymbolEpisode dicts.

    Returned structure (요약):
    - total_episodes
    - by_symbol_type: {letter/digit/...: {count, correct, accuracy}}
    - by_symbol: {symbol: {count, correct, accuracy}}
    - hardest_symbols: [{symbol, count, accuracy}] (accuracy 오름차순 상위 N개)
    """
    by_type: Dict[str, Dict[str, float]] = {}
    by_symbol: Dict[str, Dict[str, float]] = {}

    for ep in episodes:
        stype = str(ep.get("symbol_type") or "unknown")
        sym = str(ep.get("symbol") or "")
        correct = bool(ep.get("correct", False))

        type_stats = by_type.setdefault(stype, {"count": 0.0, "correct": 0.0})
        type_stats["count"] += 1.0
        if correct:
            type_stats["correct"] += 1.0

        sym_stats = by_symbol.setdefault(sym, {"count": 0.0, "correct": 0.0})
        sym_stats["count"] += 1.0
        if correct:
            sym_stats["correct"] += 1.0

    # Compute accuracies
    for stats in by_type.values():
        c = stats["count"]
        stats["accuracy"] = (stats["correct"] / c) if c > 0 else 0.0
    for stats in by_symbol.values():
        c = stats["count"]
        stats["accuracy"] = (stats["correct"] / c) if c > 0 else 0.0

    # Hardest symbols: 충분히 시도된 것만 대상으로 정확도 낮은 순으로 상위 N개.
    hardest: List[Dict[str, Any]] = []
    for sym, stats in by_symbol.items():
        if stats["count"] < 3:  # 최소 시도 횟수 기준
            continue
        hardest.append(
            {
                "symbol": sym,
                "count": int(stats["count"]),
                "accuracy": stats["accuracy"],
            }
        )
    hardest.sort(key=lambda x: x["accuracy"])
    hardest = hardest[:top_n]

    total_episodes = sum(int(stats["count"]) for stats in by_type.values())

    return {
        "total_episodes": total_episodes,
        "by_symbol_type": by_type,
        "by_symbol": by_symbol,
        "hardest_symbols": hardest,
    }


def summarize_text_episodes_dicts(
    episodes: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarise text (sentence/paragraph) learning performance.

    Returned structure:
    - total_episodes
    - by_text_type: {sentence_ko: {count, avg_score}, ...}
    - by_scope: {world_era: {count, avg_score}, character_arc: {...}, ...}
    """
    by_type: Dict[str, Dict[str, float]] = {}
    by_scope: Dict[str, Dict[str, float]] = {}

    total = 0
    for ep in episodes:
        total += 1
        ttype = str(ep.get("text_type") or "unknown")
        score = float(ep.get("score", 0.0))
        ctx = ep.get("context", {}) or {}
        scope = str(ctx.get("scope") or "unknown")

        t_stats = by_type.setdefault(ttype, {"count": 0.0, "score_sum": 0.0})
        t_stats["count"] += 1.0
        t_stats["score_sum"] += score

        s_stats = by_scope.setdefault(scope, {"count": 0.0, "score_sum": 0.0})
        s_stats["count"] += 1.0
        s_stats["score_sum"] += score

    for stats in by_type.values():
        c = stats["count"]
        stats["avg_score"] = (stats["score_sum"] / c) if c > 0 else 0.0
    for stats in by_scope.values():
        c = stats["count"]
        stats["avg_score"] = (stats["score_sum"] / c) if c > 0 else 0.0

    return {
        "total_episodes": total,
        "by_text_type": by_type,
        "by_scope": by_scope,
    }


def print_symbol_learning_report(
    episodes_path: str = "logs/symbol_episodes.jsonl",
) -> None:
    """
    Load symbol episodes from JSONL and print a compact learning report.
    """
    if not os.path.exists(episodes_path):
        print(f"[symbol_report] No symbol episodes file at {episodes_path}")
        return

    episodes: List[Dict[str, Any]] = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    summary = summarize_symbol_episodes_dicts(episodes)

    print("[Symbol Learning Report]")
    print(f"- total episodes: {summary['total_episodes']}")

    print("\n- accuracy by symbol_type:")
    for stype, stats in summary["by_symbol_type"].items():
        acc = stats.get("accuracy", 0.0)
        print(f"  * {stype}: {acc:.2%} ({int(stats['correct'])}/{int(stats['count'])})")

    hardest = summary["hardest_symbols"]
    if hardest:
        print("\n- hardest symbols (min 3 attempts):")
        for item in hardest:
            print(
                f"  * '{item['symbol']}' "
                f"acc={item['accuracy']:.2%} "
                f"count={item['count']}"
            )
    else:
        print("\n- hardest symbols: (not enough data yet)")


def print_text_learning_report(
    episodes_path: str = "logs/text_episodes.jsonl",
) -> None:
    """
    Load text episodes from JSONL and print a compact learning report.
    """
    if not os.path.exists(episodes_path):
        print(f"[text_report] No text episodes file at {episodes_path}")
        return

    episodes: List[Dict[str, Any]] = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    summary = summarize_text_episodes_dicts(episodes)

    print("[Text Learning Report]")
    print(f"- total episodes: {summary['total_episodes']}")

    print("\n- avg score by text_type:")
    for ttype, stats in summary["by_text_type"].items():
        avg = stats.get("avg_score", 0.0)
        print(f"  * {ttype}: {avg:.2f} (n={int(stats['count'])})")

    print("\n- avg score by scope:")
    for scope, stats in summary["by_scope"].items():
        avg = stats.get("avg_score", 0.0)
        print(f"  * {scope}: {avg:.2f} (n={int(stats['count'])})")


def export_causal_episodes(
    world,
    chars: List[Character],
    events_path: str = "logs/world_events.jsonl",
    out_path: str = "logs/causal_episodes.jsonl",
) -> str:
    """
    Convenience entry: read events from log, build episodes, write JSONL.
    """
    events = list(load_events_from_log(events_path))
    episodes = build_causal_episodes(world, chars, events)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
    return out_path


def _demo() -> None:
    """
    Small demo: reuse export_world_snapshot_for_godot world builder,
    then emit causal episodes alongside the Godot snapshot.
    """
    # Lazy import to avoid circular deps.
    from scripts.export_world_snapshot_for_godot import _build_world

    world, chars, _macro_states = _build_world(years=200, ticks_per_year=3)
    out_path = export_causal_episodes(world, chars)
    print(f"Wrote causal episodes to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    # 기본 데모: WORLD 인과 에피소드 생성
    _demo()
