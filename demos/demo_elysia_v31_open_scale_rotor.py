r"""
ELYSIA v3.1 Demo: Hierarchical Open Scale-Rotor Interactive Causal Architecture
===============================================================================

Demonstrates all ELYSIA v3.1 Master Design Specification scenarios:
  - Scenario 1A: Label-free correlation spectrum axis discovery ("고춧가루-매운맛 원리")
  - Scenario 1B: Foggy road empirical modality self-assessment & EM radar switch ("안개 속 자동차 원리")
  - Scenario 2A: Label-free spatial object structural principle ("사과 원리")
  - Scenario 2B: Self-structure Atom-Molecule-Macro hierarchy discernment & scale-invariant weaving
  - Scenario 3 : Real OS Python subprocesses, real workspace ASTs & media (`jpg`/`wav`), non-crashing fault absorption
  - Sphere     : Concentric spherical `warp`/`weft` memory, geodesic missing-node mirror inference,
                 4D `CausalField` projection, SSD Ice-Water-Gas persistence, and 19-organ self-audit.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.elysia_v31_open_scale_rotor import ElysiaV31OpenScaleRotorSystem
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA v3.1 — HIERARCHICAL OPEN SCALE-ROTOR INTERACTIVE CAUSAL ARCHITECTURE")
    print("=" * 92)

    workspace_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="elysia_v31_demo_") as tmp_dir:
        slab_path = Path(tmp_dir) / "elysia_v31_sphere.ice"
        with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
            cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
            system = ElysiaV31OpenScaleRotorSystem(
                workspace_root=workspace_root,
                living_cycle=cycle,
            )

            report = system.run_all_v31_scenarios()

            # --- Scenario 1 ---
            _section("1. 시나리오 1: 사전 라벨 없는 상관 스펙트럼 축 자생 & 안개 속 능동 감각 선택")
            s1_spec = report["scenario_1"]["spectrum_discovery"]
            s1_fog = report["scenario_1"]["fog_modality_assessment"]
            print(f"  [1A 화학/감각 매질] {s1_spec['summary_ko']}")
            print(f"      반응 채널: {s1_spec['responsive_channels']} | 불변 채널: {s1_spec['invariant_channels']}")
            print(f"      상관 스펙트럼: { {k: round(v, 3) for k, v in s1_spec['correlation_spectrum'].items()} }")
            print(f"  [1B 농무 도로 탐사] {s1_fog.reasoning_ko}")
            for mod, sc in s1_fog.modality_scores.items():
                print(
                    f"      - {mod:22s}: 해상도={sc['sensitivity_resolution']:.3f}, "
                    f"최대|r|={sc['max_abs_correlation']:.3f}"
                )
            print(f"      -> 자율 선택 필드: {s1_fog.selected_modality} (한계 판별: {s1_fog.unusable_modalities})")

            # --- Scenario 2 ---
            _section("2. 시나리오 2: 사전 라벨 없는 구조 원리 도출(사과 원리) & 자기 구조 기준 계층 분별")
            s2_apple = report["scenario_2"]["apple_principle"]
            print(f"  [2A 공간 객체 원리] {s2_apple['summary_ko']}")
            print(
                f"      닫힌 볼록 경계={s2_apple['closed_convex_contour']}, "
                f"장/단파장 비={s2_apple['long_to_short_wave_ratio']:.2f}, "
                f"표면 복원 탄성={s2_apple['firm_surface_tension']}"
            )
            print("  [2B 자기 구조 잣대 분별 (분자 m1 기준, 외부 계층 라벨 0개)]")
            for tag, disc in report["scenario_2"]["discernments"].items():
                print(
                    f"      - {tag:10s} => {disc.relation.value:22s} "
                    f"(반경비={disc.scale_ratio:.2f}, 결합비={disc.bond_complexity_ratio:.2f})"
                )
                print(f"        해석: {disc.reasoning_ko}")

            # --- Scenario 3 ---
            _section("3. 시나리오 3: 실제 외부 OS 서브프로세스 · 실물 코드 AST · 실물 매체 흡수")
            s3 = report["scenario_3"]
            print(f"  [3A 실물 OS 파이썬 서브프로세스] {s3['subprocess_probe']['summary_ko']}")
            c_axis = s3["subprocess_probe"]["compute_axis"]
            if c_axis is not None:
                print(
                    f"      형성 축={c_axis.axis_id}, 지배 상관 r={c_axis.dominant_correlation:+.3f}, "
                    f"기울기={c_axis.dominant_slope:.6f}"
                )
            print(
                f"      외부 프로세스 1/0 오류 흡수={len(s3['subprocess_probe']['fault_boundary_events'])}건 (강제종료 0회)"
            )
            print(f"  [3B 실물 워크스페이스 파일/매체] {s3['workspace_probe']['summary_ko']}")
            print(
                f"      확장 전후 구면 망: 노드 {s3['before_metrics']['node_count']} -> {s3['after_metrics']['node_count']}, "
                f"날실(warp) {s3['before_metrics']['warp_links']} -> {s3['after_metrics']['warp_links']}, "
                f"씨실(weft) {s3['before_metrics']['weft_links']} -> {s3['after_metrics']['weft_links']}"
            )

            # --- Sphere Inference + CausalField + Self-Audit ---
            _section("4. 구면 가상 관계 메모리 결측 거울 추론 · 4D CausalField 동기화 · 19기관 자기성찰")
            live_vals = {
                "unit_alpha_1": 0.70,
                "unit_alpha_2": 0.76,
                "composite_m2": 0.74,
                "macro_cluster_0": 0.72,
            }
            mirror = system.memory.infer_missing_node_value("composite_m1", live_vals)
            print(
                f"  [결측 노드 'composite_m1' 구면 거울 추론] 성공={mirror['inferred']}, "
                f"추정값={mirror['estimated_value']:.4f}, 신뢰도={mirror['confidence']:.3f}, "
                f"참조 이웃={mirror['neighbors_used']}"
            )

            field = CausalField(dimensions=3)
            sync = system.memory.sync_to_causal_field(causal_field=field)
            field.step(dt=0.1)
            topo = report["final_topology"]
            print(
                f"  [구면 위상 & CausalField] 노드={topo['node_count']}개, "
                f"인지축={topo['axis_count']}개, 물방울={topo['droplet_count']}개, "
                f"경계마찰={topo['boundary_event_count']}건, 계층={topo['scale_tiers_occupied']}"
            )
            print(
                f"  [4D CausalField 투영] 복셀={len(field.voxels)}개, 빔={sync['beams_linked']}개, dt=0.1 이완 완료"
            )

            audit = cycle.introspect_own_architecture()
            print(
                f"  [SSD 영속화 & 19기관 자기성찰] Ice 크리스탈={mem_rt.ice.num_crystals}개, "
                f"SHA-256 무결성={cycle.verify_autobiographical_integrity()}, "
                f"검증 기관={audit.total_organ_files}개 ({audit.total_organ_classes} 클래스, {audit.total_organ_lines} 라인)"
            )

    print("\n" + "=" * 92)
    print("  ELYSIA v3.1 VERIFICATION DEMO COMPLETED CLEANLY (EXIT 0)")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
