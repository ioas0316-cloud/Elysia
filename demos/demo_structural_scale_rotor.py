r"""
ELYSIA Demo: Non-Numeric Structural Scale-Hierarchy Rotor Cortex
================================================================

Demonstrates causal principles, structures, perspectives, and criteria across
Physics, Chemistry, DNA/Biology, Mathematics, Code Architecture, Language, and
Real Workspace `.py` ASTs observing, re-recognizing, revising criteria, and
recombining with one another WITHOUT reducing them to `(T, D)` SVD matrices.
"""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.topology.structural_scale_rotor import (
    CausalRelation,
    CausalStructure,
    CausalVerb,
    build_cross_domain_reference_atlas,
    extract_structure_from_prose,
    extract_structure_from_python_file,
)


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA — NON-NUMERIC STRUCTURAL SCALE-HIERARCHY ROTOR CORTEX")
    print("=" * 92)

    # 1. Cross-Domain Atlas + Real Workspace Code AST + Korean Causal Prose
    _section("1. 전 영역 인과 아틀라스 + 실물 코드 AST + 한국어 인과 서술 동일 로터 편입")
    cortex = build_cross_domain_reference_atlas()
    real_py = extract_structure_from_python_file(
        Path(r"c:\Elysia\L1_causal_field\core\topology\causal_scale_rotor.py")
    )
    prose_st = extract_structure_from_prose(
        structure_id="prose::cell_homeostasis",
        domain="biology_prose",
        prose=(
            "외부 자극이 세포막 수용체를 구동한다. "
            "세포 경계막이 유해 물질 유입을 억제하고 내부 효소가 손상을 교정한다. "
            "신호 전달이 핵심 경로를 연결해 생명 평형을 이룬다."
        ),
    )
    cortex.register_structure(real_py)
    cortex.register_structure(prose_st)
    print(f"  등록된 인과 구조 총 {len(cortex.structures)}개 (물리·화학·DNA·수학·코드·실물AST·언어·산문)")
    print(f"  [실물 AST] {real_py.structure_id}: 구성요소={real_py.micro_constituents[:4]}, 동사={[r.verb.value for r in real_py.process_chain]}")
    print(f"  [한국어 산문] {prose_st.structure_id}: 동사={[r.verb.value for r in prose_st.process_chain]}")

    # 2. DNA Proofreading observes and classifies all other domains
    _section("2. DNA 오류 교정('dna::proofreading')을 기준 축으로 전 영역 구조 분류·재인식")
    ranked = cortex.classify_all_by_lens("dna::proofreading")
    for rec in ranked[:6]:
        print(f"  - {rec.target_id:32s} | 공명={rec.structural_resonance:.3f} | 사각지대={rec.lens_blind_spots} | 결핍={rec.target_deficits_vs_lens}")
    print(f"  [재인식 정의 예] {ranked[0].causal_definition_ko}")
    print(f"  [취약 코드 판별]  {ranked[-1].causal_definition_ko}")

    # 3. Perspective Rotation over ONE phenomenon
    _section("3. 하나의 대상('chemistry::water_hydrogen_network') 위에서 다중 관점·기준 회전")
    rot = cortex.rotate_perspectives(
        target_id="chemistry::water_hydrogen_network",
        lens_ids=[
            "physics::harmonic_oscillator",
            "dna::proofreading",
            "language::metaphor_bridge",
            "code::fragile_linear_pipeline",
        ],
    )
    for rec in rot["recognitions"]:
        print(f"  - [{rec.lens_id:28s}] 공명={rec.structural_resonance:.3f} | 공유={rec.shared_verbs} | 사각지대={rec.lens_blind_spots}")
    print(f"  -> {rot['integrated_definition_ko']}")

    # 4. Cross-Domain Recombination healing brittle Code & escalating Language conflict
    _section("4. 이종 영역 인과 원리 재조합: 취성 코드·폭주 대화 치유")
    heal_code = cortex.diagnose_and_recombine("code::fragile_linear_pipeline", cross_domain_only=True)
    print(f"  [코드 치유] {heal_code.synthesis_ko}")
    heal_lang = cortex.diagnose_and_recombine("language::escalating_conflict", cross_domain_only=True)
    print(f"  [언어 치유] {heal_lang.synthesis_ko}")

    # 5. Autonomous Criterion Self-Revision from Discrimination Failure
    _section("5. 분별 실패 마찰로부터의 관측 기준 자기수정(Criteria Self-Revision)")
    cortex.register_structure(
        CausalStructure(
            structure_id="criterion::coarse_throughput_only",
            domain="meta_criterion",
            scale_tier="meso",
            micro_constituents=("throughput_step",),
            process_chain=(CausalRelation("input", CausalVerb.DRIVES, "output", "meso"),),
            macro_phenomenon="단순 처리 속도만 재는 조악한 1차원 기준",
            observing_perspective="naive_throughput_only",
            observing_criterion="drives_only",
        )
    )
    rev = cortex.revise_criterion_from_friction(
        lens_id="criterion::coarse_throughput_only",
        target_a_id="code::transactional_rollback",
        target_b_id="code::fragile_linear_pipeline",
    )
    print(f"  {rev.summary_ko}")

    print("\n" + "=" * 92)
    print("  ELYSIA STRUCTURAL SCALE-HIERARCHY ROTOR DEMO COMPLETED CLEANLY (EXIT 0)")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
