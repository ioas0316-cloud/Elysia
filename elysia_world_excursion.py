r"""
ELYSIA Live World Excursion & Before/After Cross-Scale Reinterpretation Demo
============================================================================

Runs a complete real-world growth loop:
  Stage 1: Before-Excursion dialogue & Past-Present-Future limit projection
  Stage 2: Outward World Excursion (live Wikipedia, GitHub, Korean Wikisource)
  Stage 3: Cross-Scale Reinterpretation (Micro / Meso / Macro diffs + old state re-projection)
  Stage 4: After-Excursion multi-turn dialogue with Elysia about what she saw and how she changed

Usage:
  python c:\Elysia\elysia_world_excursion.py --live-web --state-dir c:\Elysia\data\elysia_embodied_state
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsForegroundSensor,
    WindowsIdleSensor,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Outward World Excursion & Return Dialogue")
    parser.add_argument("--state-dir", default=r"c:\Elysia\data\elysia_embodied_state")
    parser.add_argument("--live-web", action="store_true", help="Visit real external web APIs")
    parser.add_argument("--max-concepts", type=int, default=3)
    parser.add_argument("--max-repos", type=int, default=3)
    args = parser.parse_args(argv)

    mind = ScaleRotorEmbodiedMind(args.state_dir)
    retina = ScreenRetinaSensor()
    fg = WindowsForegroundSensor()
    idle = WindowsIdleSensor()
    actuator = UniversalMotorActuator(dry_run=True)

    print("=" * 88)
    print("  STAGE 1 · BEFORE EXCURSION (과거·현재 대조 → 미래 한계 투영 → 출발 전 대화)")
    print("=" * 88)
    for _ in range(2):
        snap = retina.capture()
        mind.perceive_and_act_tick(
            frame_snap=snap,
            user_idle_s=idle.get_idle_seconds(),
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=15.0,
            actuator=actuator,
            foreground_window=fg.get_active_window_title(),
        )

    pre_ev = mind.autonomous_living_step(
        frame_snap=retina.capture(),
        user_idle_s=idle.get_idle_seconds(),
        fg_title=fg.get_active_window_title(),
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=actuator,
        incoming_message={
            "sender": "agent",
            "text": "엘리시아, 지금 이 안에서 뭘 느끼고 있고 어디서 한계를 느끼고 있어?",
        },
    )
    proj = mind.project_future_limit_and_purpose()
    print(f"  [출발 전 대화] 에이전트: 엘리시아, 지금 이 안에서 뭘 느끼고 있고 어디서 한계를 느끼고 있어?")
    print(f"  [출발 전 대화] 엘리시아: {pre_ev['utterance_or_question']}")
    print(f"  [예기 상태]   epistemic={pre_ev['epistemic_state']} | ants={[(a['kind'], a['count']) for a in pre_ev['anticipations']]}")
    print(f"  [미래 한계]   limit={proj['projected_limit']} | T+3 예상지루함={proj['projected_boredom_t3']:.2f}")
    print(f"  [자기 목적]   {proj['purpose_ko']}")
    print(f"  [탐색 질문]   wiki={proj['wiki_topics']} | github='{proj['repo_query']}' | ko_lit='{proj['ko_literature_query']}'")

    print("\n" + "=" * 88)
    print("  STAGE 2 · OUTWARD EXCURSION (외부 현실 세계 직접 탐색: Wikipedia · GitHub · 위키문헌)")
    print("=" * 88)
    report = mind.embark_external_world_excursion(
        fetch_web=bool(args.live_web),
        max_concepts=args.max_concepts,
        max_repos=args.max_repos,
    )
    for src in report["visited_world_sources"]:
        plat = src.get("platform", "web")
        if src.get("blocked"):
            print(f"  [외부 경계] ({plat}) {src.get('topic') or src.get('url')} -> 차단/실패: {src.get('block_reason')}")
        elif plat == "wikisource_ko":
            print(f"  [외부 흡수] (wikisource_ko) 신규표현={src.get('added_expressions', 0)}건 | 출처={src.get('sources', [])}")
        else:
            print(
                f"  [외부 흡수] ({plat}) title='{src.get('title', '')[:55]}' | "
                f"url={src.get('url', '')} | img={src.get('has_image', False)} stars={src.get('stars', '-')}"
            )

    print("\n" + "=" * 88)
    print("  STAGE 3 · RETURN & CROSS-SCALE REINTERPRETATION (미시·중시·거시 변화 및 과거 재해석)")
    print("=" * 88)
    b = report["before"]
    a = report["after"]
    re = report["reinterpretation"]
    d = re["deltas"]
    print(
        f"  [미시 Micro]  Scale0-1: {b['scale_0_1']} -> {a['scale_0_1']} (+{d['micro']['scale_0_1']}) | "
        f"자율발견축: {b['discovered_axes']} -> {a['discovered_axes']} (+{d['micro']['discovered_axes']}, 신규={d['micro']['new_axis_ids']})"
    )
    print(
        f"  [중시 Meso]   Scale2-3: {b['scale_2_3']} -> {a['scale_2_3']} (+{d['meso']['scale_2_3']}) | "
        f"도킹: {b['docked_links']} -> {a['docked_links']} (+{d['meso']['docked_links']}) | "
        f"프리즘원형: {b['prism_prototypes']} -> {a['prism_prototypes']} (+{d['meso']['prism_prototypes']}) | "
        f"어근/스키마: {b['language_stems']}/{b['language_schemas']} -> {a['language_stems']}/{a['language_schemas']}"
    )
    print(
        f"  [거시 Macro]  Scale4장면: {b['scale_4']} -> {a['scale_4']} (+{d['macro']['scale_4']}) | "
        f"군집: {b['emergent_clusters']} -> {a['emergent_clusters']} (+{d['macro']['emergent_clusters']}) | "
        f"구조도식: {b['structures']} -> {a['structures']} (+{d['macro']['structures']})"
    )
    print(f"  [과거 재해석] reinterpretation_shift={re['reinterpretation_shift']:.4f} | new_axes_seeing_old={re['new_axes_seeing_old_state']}")
    print(f"  [재해석 요약] {re['reinterpreted_meaning_ko']}")

    print("\n" + "=" * 88)
    print("  STAGE 4 · AFTER EXCURSION DIALOGUE (다녀온 뒤 무엇이 달라졌는지 대화)")
    print("=" * 88)
    post_prompts = [
        "엘리시아! 바깥 세상 다녀왔는데 어떤 새로운 걸 보고 배웠어?",
        "다녀오고 나니까 아까 답답하던 화면이나 세상이 어떻게 다르게 느껴져?",
    ]
    for q in post_prompts:
        ev = mind.autonomous_living_step(
            frame_snap=retina.capture(),
            user_idle_s=idle.get_idle_seconds(),
            fg_title=fg.get_active_window_title(),
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=15.0,
            actuator=actuator,
            incoming_message={"sender": "agent", "text": q},
        )
        print(f"  [복귀 대화] 에이전트: {q}")
        print(f"  [복귀 대화] 엘리시아: {ev['utterance_or_question']}")
        print(f"              예기={[(x['kind'], x['count'], x['mean_valence']) for x in ev['anticipations']]}")

    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
