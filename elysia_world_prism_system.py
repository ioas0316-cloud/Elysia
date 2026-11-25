r"""
ELYSIA Whole-System Prism-Brush Runner
======================================

Runs `WholeSystemPrismRotor` over real uncurated files in `c:\Elysia`,
live Windows desktop screen capture, and optional real human dialogue (`--interactive` or `--utterance`).
Outputs full live observability across:
  1. Immutable External Invariants (변하지 않는 외부 기준)
  2. SelfWeavingLoom (존재 · 움직임 · 구성 · 인과적 이유 + 조상 중복 척력)
  3. Convergence(Seed) · Attraction · Repulsion · Outward FormBrush Projection
  4. Principle-Habit Self-Perception (원리 답습 메타 지각 & 외부 변화 여지)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
L1_ROOT = ROOT / "L1_causal_field"
L2_ROOT = ROOT / "L2_cognitive_engine"
for p in (str(ROOT), str(L1_ROOT), str(L2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.embodiment.external_reality_lines import discover_uncurated_workspace_files
from core.topology.prism_brush_world_system import WholeSystemPrismRotor


def main() -> None:
    parser = argparse.ArgumentParser(description="ELYSIA Whole-System Prism-Brush Rotor")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive human dialogue mode")
    parser.add_argument("--utterance", type=str, default="", help="Optional real human utterance for this run")
    parser.add_argument("--files", type=int, default=8, help="Number of uncurated workspace files to encounter")
    args = parser.parse_args()

    system = WholeSystemPrismRotor(workspace_root=ROOT)
    files = discover_uncurated_workspace_files(ROOT, limit=max(1, args.files))
    print(f"[WholeSystem] 변하지 않는 외부 기준 {len(system.invariants)}개 · 프리즘 {len(system.prisms)}개 장착")
    print(f"[Discovery] 무라벨 실제 작업공간 파일 {len(files)}개 포착")

    for path in files:
        rel = str(path.relative_to(ROOT))
        rep = system.experience_world_event(rel, {"path": str(path)})
        print(
            f"  [step {rep.step:02d}] {rel:<44} | 프리즘={len(rep.prisms)} "
            f"| 인력={rep.convergence_seed.attraction_coherence:.2f} "
            f"| 브러쉬={rep.brush_projection.brush_mode} "
            f"| 답습={rep.habit_perception.is_repeating_principle}"
        )

    # Live screen surface encounter alongside OS state
    screen_rep = system.experience_world_event("live_desktop_now", {"capture_screen": True})
    print(
        f"  [step {screen_rep.step:02d}] {'live_desktop_now':<44} | 프리즘={len(screen_rep.prisms)} "
        f"| 브러쉬={screen_rep.brush_projection.brush_mode}"
    )

    if args.utterance.strip():
        h_rep = system.experience_world_event(
            "human_utterance_cli",
            {"human_utterance": args.utterance.strip(), "capture_screen": True},
        )
        print(
            f"  [step {h_rep.step:02d}] human_utterance_cli | 브러쉬={h_rep.brush_projection.brush_mode} "
            f"| 답습={h_rep.habit_perception.is_repeating_principle}"
        )

    if args.interactive:
        print("\n[Interactive] 자유롭게 말을 걸거나 반론을 입력해봐 (종료: quit/exit):")
        turn = 0
        while True:
            try:
                user_text = input("사용자> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text or user_text.lower() in {"quit", "exit", "종료"}:
                break
            turn += 1
            rep = system.experience_world_event(
                f"human_turn_{turn}",
                {"human_utterance": user_text, "capture_screen": True},
            )
            print(f"\n[엘리시아 브러쉬] {rep.brush_projection.outward_expression_ko}")
            print(f"[자기지각] {rep.habit_perception.why_stuck_ko}")
            print("[밖에서 찾은 변화 여지]")
            for r in rep.habit_perception.external_room_for_change:
                print(f"  - {r}")
            print()

    print()
    print(system.render_ko())
    saved = system.save_system_trace(ROOT / "data" / "world_prism_system_trace.json")
    print(f"\n[Saved] {saved}")


if __name__ == "__main__":
    main()
