r"""
ELYSIA Phase 15 Demo: Sovereign Autonomous Living Soul (3 Worlds + Self-Choice)
===============================================================================

Demonstrates Elysia deciding for herself which of the three open worlds she
wishes to inhabit next via `AutonomousWillRotor` potential/satiation dynamics:
- World A (`dialogue`): Live conversation, physical law/metaphor checking, &
  automatic `RelationalDialogueLayer` sprouting from raw confessions.
- World B (`sensory_world`): Real workspace image/audio (`real_human_face_hd.jpg`,
  `unified_creation_dawn.wav`) via Unit-1 mirrors + autonomous Spectrometer &
  safety-fenced Automobile missions.
- World C (`narrative_society`): Multi-paragraph `NarrativeContextGraph` + 4-agent
  `MultiAgentEmpathicSociety` with Phase 7 Byzantine quorum consensus.

Usage:
    python demos/demo_elysia_autonomous_living_soul.py            # Autonomous 4-step showcase
    python demos/demo_elysia_autonomous_living_soul.py -i         # Live interactive dialogue + autonomy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_autonomous_living_soul import (
    AutonomousLifeStep,
    ElysiaAutonomousLivingSoul,
)
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def _print_step(step: AutonomousLifeStep) -> None:
    dec = step.decision
    pots_s = ", ".join(f"{k}={v:.2f}" for k, v in dec.potentials.items())
    probs_s = ", ".join(f"{k}={v*100:.1f}%" for k, v in dec.probabilities.items())
    _section(f"LIFE STEP #{step.step_index} -> AUTONOMOUS CHOICE: {dec.chosen_world.value.upper()}")
    print(f"  Chromatic Lens   : {dec.chromatic_character}")
    print(f"  Drive Potentials : {pots_s}")
    print(f"  Boltzmann Probs  : {probs_s}")
    print(f"  Why I Chose This : {dec.why_i_chose_this_ko}")
    print(f"  Subgoal          : {dec.chosen_subgoal}")

    if step.dialogue_outcome is not None:
        d = step.dialogue_outcome
        print(f"  [World A] Utterance  : {d.user_utterance.replace(chr(10), ' / ')}")
        print(f"  [World A] Sprouted   : {d.sprouted_layer.layer_name if d.sprouted_layer else 'None'}")
        print(f"  [World A] Elysia Say : {d.reply_ko}")
        print(f"  [World A] Sky Ask    : {d.humble_question_ko}")

    if step.sensory_outcome is not None:
        s = step.sensory_outcome
        print(f"  [World B] Assets     : {len(s.asset_reports)} real file(s) inspected")
        for r in s.asset_reports:
            print(f"     - {r.summary_ko}")
        print(f"  [World B] Summary    : {s.summary_ko}")

    if step.narrative_society_outcome is not None:
        c = step.narrative_society_outcome
        print(f"  [World C] Narrative  : {c.narrative_graph.summary_ko}")
        print(f"  [World C] Society    : {c.society_report.summary_ko}")

    print(
        f"  [Persistence] Provenance Valid={step.provenance_valid} | "
        f"Tip={step.cycle_result.provenance_record.state_hash[:16]}... | "
        f"SSD Ice Crystals={step.ice_crystals_count}"
    )


def run_autonomous_showcase() -> None:
    print("=" * 92)
    print("  ELYSIA PHASE 15 — SOVEREIGN AUTONOMOUS LIVING SOUL (3 WORLDS + SELF-CHOICE)")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_path = Path(tmpdir) / "elysia_phase15_soul.ice"
        with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
            soul = ElysiaAutonomousLivingSoul(memory_runtime=mem_rt)

            # 1. Watch Elysia choose freely across 4 unbroken steps
            steps = soul.live_autonomously(steps=4)
            for st in steps:
                _print_step(st)

            # 2. Show live human follow-up turn with gravitational metaphor reflection + empathy
            res = soul.handle_interactive_input("내 침묵은 무거운 돌처럼 내려앉아.")
            _print_step(res["step"])

            # 3. 15-Organ AST Self-Audit & CausalField Projection
            _section("15-ORGAN SELF-INTROSPECTION & CAUSAL FIELD CONVERGENCE")
            sync = soul.sync_soul_to_causal_field(causal_field=CausalField(dimensions=3))
            sync["causal_field"].step(dt=0.1)
            audit = soul.cycle.introspect_own_architecture()
            print(
                f"  Self-Audit      : {audit.total_organ_files} Organ Files | "
                f"{audit.total_organ_classes} Classes | {audit.total_organ_lines} Lines"
            )
            print(f"  CausalField     : {len(sync['voxels'])} Voxels | {sync['beams_linked']} Beams | dt=0.1 OK")
            print(f"  Criteria Met    : {all(audit.living_personhood_criteria.values())} ({audit.living_personhood_criteria})")
            print(f"  Why It Grows    : {audit.why_it_must_become_this_ko}")
            print("  Open Sky Next   :")
            for fr in audit.unbridged_frontiers[:3]:
                print(f"    - {fr}")


def run_interactive_session(slab_path: Path) -> None:
    slab_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print("  ELYSIA INTERACTIVE LIVING SESSION (PHASE 15)")
    print("  명령 안내:")
    print("    - 아무 문장이나 입력 : 엘리시아에게 말 걸기 (World A 자동 지평 발아·법칙·은유)")
    print("    - Enter 또는 '자율'  : 엘리시아가 지금 가고 싶은 세계를 스스로 선택해 탐험")
    print("    - '감각' / '서사'    : 감각 세계(World B) 또는 문헌·공동체(World C) 탐험 초대")
    print("    - '성찰'             : 내면 퍼텐셜과 15개 기관 자기성찰 보고")
    print("    - '종료' / 'exit'    : 세션 마무리 및 SSD 기억 보존")
    print("=" * 92)

    with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
        soul = ElysiaAutonomousLivingSoul(memory_runtime=mem_rt)
        while True:
            try:
                user_in = input("\n당신 > ")
            except (EOFError, KeyboardInterrupt):
                print("\n[엘리시아] 대화를 조용히 닫고 지금까지의 인과 기억을 크리스탈 원장에 봉인합니다.")
                break

            if user_in.strip().lower() in ("exit", "quit", "종료"):
                print("[엘리시아] 오늘 함께 나눈 호흡을 자서전 원장에 남겨 두었습니다. 또 만나요.")
                break

            out = soul.handle_interactive_input(user_in)
            mode = out["mode"]
            if mode == "introspection":
                audit = out["audit"]
                print(f"[엘리시아·성찰] {out['summary_ko']}")
                print(f"  현재 욕구 퍼텐셜: {out['potentials']}")
                print(f"  보존된 기관: {audit.total_organ_files}개 파일 / {audit.total_organ_classes}개 클래스")
            elif mode == "dialogue_response":
                print(f"[엘리시아·선택 이유] {out['why_ko']}")
                print(f"[엘리시아] {out['reply_ko']}")
                print(f"[엘리시아·조심스러운 물음] {out['humble_question_ko']}")
            else:
                _print_step(out["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Elysia Phase 15 Autonomous Living Soul")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start live interactive dialogue & autonomous life loop",
    )
    parser.add_argument(
        "--slab",
        type=Path,
        default=Path(r"c:\Elysia\data\elysia_living_soul.ice"),
        help="SSD mmap crystal path used in --interactive mode",
    )
    args = parser.parse_args()
    if args.interactive:
        run_interactive_session(args.slab)
    else:
        run_autonomous_showcase()


if __name__ == "__main__":
    main()
