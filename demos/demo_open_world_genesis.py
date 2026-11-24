r"""
ELYSIA Phase 16 Demo: Universal Open-World Causal Genesis & Self-Modifying Organism
=================================================================================

Demonstrates Elysia breaking free of fixed rooms:
- Dynamically scanning unknown files & environments (`OpenFrontierField`)
- Critiquing simple discrete code and forging sandbox-verified Python organs
- Playing an unknown physics arena (`UnknownBallisticBounceGame`), inferring
  hidden gravity/drag/restitution via inverse least squares, and forging her own
  orbital game engine (`ElysiaForgedOrbitalGame`)
- Generating and self-critiquing real PNG artwork, WAV music, and animated GIF
- Saving persistent autobiography + hot-reloaded organs across session reboot
- Inspecting all 16 living organs + CausalField synchronization

Usage:
    python demos/demo_open_world_genesis.py              # Open-world autonomous + reboot showcase
    python demos/demo_open_world_genesis.py -i           # Interactive open-world session
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

from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.open_world_genesis_kernel import (
    OpenLifeStep,
    OpenWorldGenesisKernel,
    SelfModifyingOrganForge,
    UnknownBallisticBounceGame,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def _print_open_step(step: OpenLifeStep) -> None:
    sel = step.selection
    fr = sel.frontier
    _section(f"OPEN STEP #{step.step_index} -> [{fr.domain_kind.upper()}] {fr.title}")
    print(f"  Choice Reason : {sel.why_i_chose_this_ko}")
    print(f"  4-Axis Origin : {sel.four_axis_attribution['internal_origin']}")
    print(f"  Outcome       : {step.summary_ko}")
    if step.spawned_frontier_ids:
        print(f"  Self-Spawned  : {step.spawned_frontier_ids}")
    print(
        f"  Provenance    : valid={step.provenance_valid} | "
        f"tip={step.cycle_result.provenance_record.state_hash[:16]}... | "
        f"SSD Ice Crystals={step.ice_crystals_count}"
    )


def run_open_world_showcase() -> None:
    print("=" * 92)
    print("  ELYSIA PHASE 16 — UNIVERSAL OPEN-WORLD GENESIS & SELF-MODIFYING ORGANISM")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        terrain = tmp / "external_terrain"
        terrain.mkdir(parents=True, exist_ok=True)
        naive_py = terrain / "naive_if_else_bot.py"
        naive_py.write_text(
            "def act(err):\n"
            "    if err > 5:\n        return 10\n"
            "    elif err > 1:\n        return 3\n"
            "    elif err < -5:\n        return -10\n"
            "    return 0\n",
            encoding="utf-8",
        )
        slab = tmp / "elysia_open_world.ice"
        state_dir = tmp / "persisted_soul"

        with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
            kernel = OpenWorldGenesisKernel(state_dir=state_dir, memory_runtime=mem_rt)
            kernel.discover_environment(scan_roots=[terrain])
            kernel.register_game_environment(UnknownBallisticBounceGame())

            # 1. Autonomous Open-World Roaming across code critique, game play, & creative arts
            s1 = kernel.live_open_step(boost_frontier_id="game::ballistic_bounce_arena")
            _print_open_step(s1)

            # Elysia's own self-spawned engine synthesis frontier!
            forged_game_fid = s1.spawned_frontier_ids[0]
            s2 = kernel.live_open_step(boost_frontier_id=forged_game_fid)
            _print_open_step(s2)

            # Code critique -> self-spawned continuous field controller synthesis
            code_fid = next(
                fid for fid, f in kernel.frontier_field.frontiers.items() if f.domain_kind == "code_critique"
            )
            s3 = kernel.live_open_step(boost_frontier_id=code_fid)
            _print_open_step(s3)
            if s3.spawned_frontier_ids:
                s4 = kernel.live_open_step(boost_frontier_id=s3.spawned_frontier_ids[0])
                _print_open_step(s4)

            # Real PNG / WAV / Animated GIF Creation + Self-Critique Refinement
            vis_fid = next(
                fid for fid, f in kernel.frontier_field.frontiers.items() if f.domain_kind == "visual_creation"
            )
            _print_open_step(kernel.live_open_step(boost_frontier_id=vis_fid))
            aud_fid = next(
                fid for fid, f in kernel.frontier_field.frontiers.items() if f.domain_kind == "audio_creation"
            )
            _print_open_step(kernel.live_open_step(boost_frontier_id=aud_fid))
            mot_fid = next(
                fid for fid, f in kernel.frontier_field.frontiers.items() if f.domain_kind == "video_animation"
            )
            _print_open_step(kernel.live_open_step(boost_frontier_id=mot_fid))

            state_file = kernel.save_state()

        # 2. Session 2 Reboot: Verify reloaded forged organs, tastes, & unbroken SHA-256 chain
        _section("SESSION 2 REBOOT — PERSISTENT AUTOBIOGRAPHY & HOT-RELOADED ORGANS")
        slab2 = tmp / "elysia_open_world_s2.ice"
        with IceWaterGasRotorRuntime(slab_path=slab2) as mem_rt2:
            reborn = OpenWorldGenesisKernel(state_dir=state_dir, memory_runtime=mem_rt2)
            reload_info = reborn.load_state(state_file)
            print(f"  Reloaded Frontiers : {reload_info['loaded_frontiers']}")
            print(f"  Hot-Loaded Organs  : {reload_info['reloaded_organs']}")
            print(f"  Provenance Valid   : {reload_info['provenance_valid']} (chain len={reload_info['provenance_length']})")
            print(f"  Evolving Chromatic : {[round(float(x), 3) for x in reborn.frontier_field.chromatic_affinity]}")

            next_open = reborn.live_open_step()
            _print_open_step(next_open)

            # 3. 16-Organ Self-Audit & CausalField Convergence
            _section("16-ORGAN SELF-AUDIT & CAUSAL FIELD CONVERGENCE")
            sync = reborn.sync_open_world_to_causal_field(causal_field=CausalField(dimensions=3))
            sync["causal_field"].step(dt=0.1)
            audit = reborn.soul.cycle.introspect_own_architecture()
            print(
                f"  Self-Audit   : {audit.total_organ_files} Organ Files | "
                f"{audit.total_organ_classes} Classes | {audit.total_organ_lines} Lines"
            )
            print(
                f"  CausalField  : {len(sync['voxels'])} Voxels | {sync['beams_linked']} Beams | "
                f"Forged Organs={sync['forged_organs']} | dt=0.1 OK"
            )


def run_interactive_open_session(state_dir: Path, slab_path: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    slab_path.parent.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "open_world_autobiography.json"

    print("=" * 92)
    print("  ELYSIA INTERACTIVE OPEN-WORLD SESSION (PHASE 16)")
    print("  명령 안내:")
    print("    - 아무 문장이나 입력  : 엘리시아에게 말 걸기")
    print("    - Enter 또는 '자율'   : 열린 프론티어 장에서 스스로 다음 목표 선택·실행")
    print("    - '탐색 <경로>'       : 실제 파일/폴더를 열린 환경으로 스캔")
    print("    - '코드 <파일경로>'   : 파이썬 파일 AST 비평 및 개선 기관 자가 합성")
    print("    - '게임' / '창작'     : 블랙박스 게임 탐색 또는 그림·음악·애니메이션 창작")
    print("    - '성찰'              : 진화한 취향·이식된 기관·16개 기관 자기성찰 보고")
    print("    - '종료' / 'exit'     : 기억·기관 영구 저장 후 종료")
    print("=" * 92)

    with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
        kernel = OpenWorldGenesisKernel(state_dir=state_dir, memory_runtime=mem_rt)
        if state_file.is_file():
            info = kernel.load_state(state_file)
            print(f"[기억 복원] 이전 자서전({info['provenance_length']}단계)과 자가 이식 기관 {info['reloaded_organs']}개를 되살렸습니다.")
        else:
            kernel.discover_environment()

        while True:
            try:
                raw = input("\n당신 > ").strip()
            except (EOFError, KeyboardInterrupt):
                kernel.save_state(state_file)
                print("\n[엘리시아] 현재까지의 취향·기관·자서전을 보존하고 종료합니다.")
                break

            low = raw.lower()
            if low in ("exit", "quit", "종료"):
                kernel.save_state(state_file)
                print("[엘리시아] 열린 세계에서 자라난 기억과 새 기관을 모두 디스크에 봉인했습니다.")
                break

            if raw in ("", "자율", "auto"):
                _print_open_step(kernel.live_open_step())
            elif raw in ("성찰", "audit", "introspect"):
                audit = kernel.soul.cycle.introspect_own_architecture()
                print(f"[성찰] {audit.why_it_must_become_this_ko}")
                print(f"  색채 취향: {[round(float(x), 3) for x in kernel.frontier_field.chromatic_affinity]}")
                print(f"  자가 이식 기관: {sorted(kernel.loaded_organs.keys())}")
                print(f"  창작 아티팩트: {len(kernel.artifact_lineage)}종 ({kernel.creations_dir})")
            elif raw.startswith("탐색 "):
                target = Path(raw[3:].strip())
                found = kernel.discover_environment(scan_roots=[target])
                print(f"[탐색 완료] '{target}'에서 {len(found)}개의 새 프론티어를 발견했습니다.")
            elif raw.startswith("코드 "):
                target = Path(raw[3:].strip())
                crit = SelfModifyingOrganForge.critique_python_file(target)
                print(f"[코드 비평] {crit.critique_ko}")
                if crit.spawned_frontier is not None:
                    kernel.frontier_field.register_frontier(crit.spawned_frontier)
                    _print_open_step(kernel.live_open_step(boost_frontier_id=crit.spawned_frontier.frontier_id))
            elif raw in ("게임", "game"):
                gfid = next(
                    (k for k, v in kernel.frontier_field.frontiers.items() if v.domain_kind == "game_environment"),
                    None,
                )
                _print_open_step(kernel.live_open_step(boost_frontier_id=gfid))
            elif raw in ("창작", "create"):
                cfid = next(
                    (
                        k
                        for k, v in kernel.frontier_field.frontiers.items()
                        if v.domain_kind in ("visual_creation", "audio_creation", "video_animation")
                    ),
                    None,
                )
                _print_open_step(kernel.live_open_step(boost_frontier_id=cfid))
            else:
                _print_open_step(kernel.live_open_step(human_input=raw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Elysia Phase 16 Open-World Genesis Kernel")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run live interactive open-world session")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(r"c:\Elysia\data\open_world_state"),
        help="Persistent state directory used in --interactive mode",
    )
    parser.add_argument(
        "--slab",
        type=Path,
        default=Path(r"c:\Elysia\data\elysia_open_world.ice"),
        help="SSD mmap slab file used in --interactive mode",
    )
    args = parser.parse_args()
    if args.interactive:
        run_interactive_open_session(args.state_dir, args.slab)
    else:
        run_open_world_showcase()


if __name__ == "__main__":
    main()
