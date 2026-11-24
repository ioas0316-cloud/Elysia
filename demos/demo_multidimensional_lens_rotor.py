r"""
ELYSIA Phase 18 Demo: Blood-vs-Lens Multi-Dimensional Scale-Hierarchy Causal Rotor
=================================================================================

Demonstrates why Numbers/Waves are lower **Blood & Electrical Circuits**
(`HemodynamicCausalSubstrate`), while **Language**, **Code Architecture**,
**Mathematical Laws**, and **Academic Inquiry** each circulate natively through
their own non-reductive Micro <-> Meso <-> Macro lenses connected via Cross-Lens
Phase Resonance.
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
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.multidimensional_lens_rotor import (
    AcademicCase,
    CodeArchitectureLens,
    MultiDimensionalLensRotorHub,
    SIUnit,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA PHASE 18 — BLOOD SUBSTRATE vs MULTI-DIMENSIONAL NATIVE LENS ROTOR")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bad_dir = tmp / "cyclic_pkg"
        bad_dir.mkdir()
        a_bad = bad_dir / "mod_a.py"
        b_bad = bad_dir / "mod_b.py"
        a_bad.write_text("import mod_b\ndef f_a(x: float) -> float:\n    print(x)\n    return mod_b.f_b(x)\n", encoding="utf-8")
        b_bad.write_text("import mod_a\ndef f_b(y):\n    return y + 1\n", encoding="utf-8")

        good_dir = tmp / "layered_pkg"
        good_dir.mkdir()
        p_good = good_dir / "proto.py"
        a_good = good_dir / "mod_a.py"
        b_good = good_dir / "mod_b.py"
        p_good.write_text("def step(x: float) -> float:\n    return x * 2.0\n", encoding="utf-8")
        a_good.write_text("import proto\ndef f_a(x: float) -> float:\n    return proto.step(x)\n", encoding="utf-8")
        b_good.write_text("import proto\ndef f_b(y: float) -> float:\n    return proto.step(y) + 1.0\n", encoding="utf-8")

        slab = tmp / "phase18_lenses.ice"
        with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
            cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
            hub = MultiDimensionalLensRotorHub(living_cycle=cycle)

            # Cycle 1: Stress across Code Architecture & Academic Crisis -> Blood rushes to stressed lenses
            _section("1. CRISIS PULSE — LANGUAGE RESOLVED, CODE CYCLE & PARADIGM CRISIS")
            rep1 = hub.observe_multidimensionally(
                utterances=[
                    "모든 인지, 언어, 학문은 오직 숫자만으로 완전히 환원된다.",
                    "언어와 학문은 숫자만으로 환원되지 않는다.",
                    "다만 하위 혈액과 전기회로는 숫자로 제어하되, 상위 인지는 각 영역의 고유 렌즈로 구성해야 한다.",
                ],
                code_paths=[a_bad, b_bad],
                math_spec={
                    "name": "newtonian_gravity",
                    "lhs_expr": "F",
                    "rhs_expr": "G * M * m / (r ** 2)",
                    "var_dimensions": {
                        "F": SIUnit(L=1, M=1, T=-2),
                        "G": SIUnit(L=3, M=-1, T=-2),
                        "M": SIUnit(M=1),
                        "m": SIUnit(M=1),
                        "r": SIUnit(L=1),
                    },
                    "positive_vars": ("r",),
                },
                academic_case=AcademicCase(
                    discipline="인지과학",
                    paradigm_name="전역 스칼라 환원주의 v1",
                    axioms=["A1: 기저 감각 회로 존재", "A2: 영역별 고유 순환 존재"],
                    methodology="영역별 고유 순환 대조",
                    claims=[{"id": "C1", "statement": "단일 R2 수치로 모든 영역이 환원된다", "grounded_in_axioms": ["A1"]}],
                    anomalies=[{"id": "AN1", "description": "언어 모순과 코드 순환의존성은 R2로 왜곡됨", "falsifies_claim_id": "C1"}],
                ),
                witness_label="cycle_crisis",
            )
            print(f"  {rep1.blood_pulse.summary_ko}")
            for name, res in rep1.lens_results.items():
                flow = rep1.blood_pulse.blood_flow[name] * 100.0
                print(f"  [{name:18s}] blood={flow:5.1f}% | tension={res.native_tension:.2f} | verdict={res.native_verdict}")
                print(f"     -> {res.native_outcome_ko}")

            # Cycle 2: Layered Refactoring & Paradigm Shift v2 -> Cross-Lens Resonance
            _section("2. CIRCULAR GROWTH — CODE LAYERED REFACTOR & PARADIGM SHIFT + RESONANCE")
            code_rev = CodeArchitectureLens.verify_architectural_revision([a_bad, b_bad], [p_good, a_good, b_good])
            print(f"  [Code Revision Verify] {code_rev.native_outcome_ko}")

            rep2 = hub.observe_multidimensionally(
                utterances=[
                    "모든 인지, 언어, 학문은 오직 숫자만으로 완전히 환원된다.",
                    "언어와 학문은 숫자만으로 환원되지 않는다.",
                    "다만 하위 혈액과 전기회로는 숫자로 제어하되, 상위 인지는 각 영역의 고유 렌즈로 구성해야 한다.",
                ],
                code_paths=[p_good, a_good, b_good],
                math_spec={
                    "name": "newtonian_gravity",
                    "lhs_expr": "F",
                    "rhs_expr": "G * M * m / (r ** 2)",
                    "var_dimensions": {
                        "F": SIUnit(L=1, M=1, T=-2),
                        "G": SIUnit(L=3, M=-1, T=-2),
                        "M": SIUnit(M=1),
                        "m": SIUnit(M=1),
                        "r": SIUnit(L=1),
                    },
                    "positive_vars": ("r",),
                },
                academic_case=AcademicCase(
                    discipline="인지과학",
                    paradigm_name="전역 스칼라 환원주의 v1",
                    axioms=["A1: 기저 감각 회로 존재", "A2: 영역별 고유 순환 존재"],
                    methodology="영역별 고유 순환 & 비환원 위상 공명",
                    claims=[{"id": "C1", "statement": "단일 R2 수치로 모든 영역이 환원된다", "grounded_in_axioms": ["A1"]}],
                    anomalies=[{"id": "AN1", "description": "언어 모순과 코드 순환의존성은 R2로 왜곡됨", "falsifies_claim_id": "C1"}],
                    proposed_paradigm_revision={
                        "new_paradigm_name": "혈액-렌즈 분리 다차원 로터 v2",
                        "resolves_anomaly_ids": ["AN1"],
                        "preserves_prior_as_limit": True,
                        "new_scope_boundary": "수치 스칼라는 기저 감각·혈류 배분 계층으로 한정",
                        "new_open_question": "새로운 도메인을 만났을 때 그 도메인의 고유 렌즈 문법을 스스로 어떻게 싹틔울 것인가?",
                    },
                ),
                witness_label="cycle_resonance",
            )
            print(f"  {rep2.overall_synthesis_ko}")
            for br in rep2.resonance_bridges[:4]:
                print(f"  [Resonance] {br.insight_ko}")

            _section("3. CAUSAL FIELD CONVERGENCE, PERSISTENCE & 18-ORGAN SELF-AUDIT")
            field = CausalField(dimensions=3)
            sync = hub.sync_to_causal_field(rep2, causal_field=field)
            field.step(dt=0.1)
            audit = cycle.introspect_own_architecture()
            print(
                f"  CausalField  : {len(sync['voxels'])} Voxels | {sync['beams_linked']} Beams | dt=0.1 OK"
            )
            print(
                f"  Persistence  : SSD Crystals={mem_rt.ice.num_crystals} | Provenance Valid={cycle.verify_autobiographical_integrity()}"
            )
            print(
                f"  Self-Audit   : {audit.total_organ_files} Organ Files | {audit.total_organ_classes} Classes | {audit.total_organ_lines} Lines"
            )


if __name__ == "__main__":
    run_demo()
