"""
Verification suite for ELYSIA Phase 18:
Blood-vs-Lens Multi-Dimensional Scale-Hierarchy Causal Rotor.
"""

from __future__ import annotations

import math
from pathlib import Path

from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.multidimensional_lens_rotor import (
    AcademicCase,
    AcademicInquiryLens,
    CodeArchitectureLens,
    LensCirculationResult,
    LinguisticCausalLens,
    MathematicalLawLens,
    MultiDimensionalLensRotorHub,
    SIUnit,
    ScaleTierSnapshot,
)
from core.physics.causal_field import CausalField


def test_linguistic_lens_detects_universal_contradiction_and_resolves_via_scoped_synthesis():
    lens = LinguisticCausalLens()

    # Step 1: Universal overclaim vs counter-claim -> pure linguistic contradiction
    res_bad = lens.process_discourse(
        [
            "모든 인지, 언어, 학문은 오직 숫자만으로 완전히 환원된다.",
            "언어와 학문은 숫자만으로 환원되지 않는다.",
        ]
    )
    assert res_bad.native_verdict == "DISCOURSE_CONTRADICTION"
    assert res_bad.native_tension > 0.65
    assert res_bad.scales.macro_horizon["unresolved_contradictions"] == 1
    assert res_bad.revision_applied is False

    # Step 2: Pragmatic scoped synthesis resolves contradiction natively in language
    res_rev = lens.process_discourse(
        [
            "모든 인지, 언어, 학문은 오직 숫자만으로 완전히 환원된다.",
            "언어와 학문은 숫자만으로 환원되지 않는다.",
            "다만 하위 혈액과 전기회로는 숫자로 제어하되, 상위 인지는 각 영역의 고유 렌즈로 구성해야 한다.",
        ]
    )
    assert res_rev.native_verdict == "PRAGMATIC_SYNTHESIS_REVISED"
    assert res_rev.native_tension < 0.25
    assert res_rev.revision_applied is True
    assert res_rev.scales.macro_horizon["unresolved_contradictions"] == 0
    assert res_rev.scales.macro_horizon["resolved_contradictions"] == 1


def test_code_architecture_lens_detects_cycles_and_verifies_layered_refactoring_plus_real_workspace(tmp_path):
    bad_dir = tmp_path / "cyclic_pkg"
    bad_dir.mkdir()
    mod_a = bad_dir / "mod_a.py"
    mod_b = bad_dir / "mod_b.py"
    mod_a.write_text("import mod_b\ndef f_a(x: float) -> float:\n    print(x)\n    return mod_b.f_b(x)\n", encoding="utf-8")
    mod_b.write_text("import mod_a\ndef f_b(y):\n    return y + 1\n", encoding="utf-8")

    res_cyc = CodeArchitectureLens.analyze_codebase([mod_a, mod_b])
    assert res_cyc.native_verdict == "CYCLIC_DEPENDENCY_DETECTED"
    assert len(res_cyc.scales.macro_horizon["cycles_detected"]) >= 1
    assert res_cyc.native_tension > 0.65
    assert len(res_cyc.scales.macro_horizon["prescriptions"]) >= 1

    good_dir = tmp_path / "layered_pkg"
    good_dir.mkdir()
    proto = good_dir / "proto_layer.py"
    la = good_dir / "layer_a.py"
    lb = good_dir / "layer_b.py"
    proto.write_text("def base_step(x: float) -> float:\n    return x * 2.0\n", encoding="utf-8")
    la.write_text("import proto_layer\ndef f_a(x: float) -> float:\n    return proto_layer.base_step(x)\n", encoding="utf-8")
    lb.write_text("import proto_layer\ndef f_b(y: float) -> float:\n    return proto_layer.base_step(y) + 1.0\n", encoding="utf-8")

    rev = CodeArchitectureLens.verify_architectural_revision([mod_a, mod_b], [proto, la, lb])
    assert rev.native_verdict == "ACYCLIC_LAYERED_ARCHITECTURE"
    assert len(rev.scales.macro_horizon["cycles_detected"]) == 0
    assert rev.revision_applied is True
    assert rev.native_tension < 0.25

    # Inspect real workspace orchestration DAG
    real_files = [
        Path(r"c:\Elysia\L1_causal_field\core\orchestration\living_causal_organism.py"),
        Path(r"c:\Elysia\L1_causal_field\core\orchestration\extended_cognitive_organism.py"),
        Path(r"c:\Elysia\L1_causal_field\core\orchestration\causal_text_law_synthesizer.py"),
        Path(r"c:\Elysia\L1_causal_field\core\orchestration\multidimensional_lens_rotor.py"),
    ]
    real_rep = CodeArchitectureLens.analyze_codebase(real_files)
    assert len(real_rep.scales.macro_horizon["cycles_detected"]) == 0
    assert real_rep.scales.meso_process["contract_annotation_ratio"] > 0.8


def test_mathematical_law_lens_checks_si_dimensions_singularities_and_scaling_symmetry():
    F_u = SIUnit(L=1, M=1, T=-2)
    E_u = SIUnit(L=2, M=1, T=-2)
    m_u = SIUnit(M=1)
    v_u = SIUnit(L=1, T=-1)
    r_u = SIUnit(L=1)
    G_u = SIUnit(L=3, M=-1, T=-2)

    # 1. Dimensional mismatch F = m * v rejected natively
    bad = MathematicalLawLens.analyze_law(
        name="wrong_force",
        lhs_expr="F",
        rhs_expr="m * v",
        var_dimensions={"F": F_u, "m": m_u, "v": v_u},
    )
    assert bad.native_verdict == "DIMENSIONAL_CONTRADICTION"
    assert bad.native_tension > 0.85

    # 2. Inverse-square gravity: exact SI, r->0+ singularity, scaling degree -2.0 w.r.t. r
    grav = MathematicalLawLens.analyze_law(
        name="newtonian_gravity",
        lhs_expr="F",
        rhs_expr="G * M * m / (r ** 2)",
        var_dimensions={"F": F_u, "G": G_u, "M": m_u, "m": m_u, "r": r_u},
        positive_vars=("r",),
    )
    assert grav.native_verdict == "DIMENSIONALLY_COHERENT_WITH_BOUNDARY"
    assert "r" in grav.scales.meso_process["singularity_vars"]
    assert math.isclose(grav.scales.meso_process["scaling_degrees"]["r"], -2.0, abs_tol=1e-4)
    assert grav.scales.macro_horizon["conservation_family"] == "inverse_square_flux_conservation"

    # 3. Kinetic energy: scaling degree +2.0 w.r.t. v
    kin = MathematicalLawLens.analyze_law(
        name="kinetic_energy",
        lhs_expr="E",
        rhs_expr="0.5 * m * (v ** 2)",
        var_dimensions={"E": E_u, "m": m_u, "v": v_u},
        positive_vars=("v",),
    )
    assert math.isclose(kin.scales.meso_process["scaling_degrees"]["v"], 2.0, abs_tol=1e-4)
    assert kin.scales.macro_horizon["conservation_family"] == "quadratic_energy_invariant"


def test_academic_inquiry_lens_tracks_anomaly_crisis_and_circular_paradigm_shift():
    lens = AcademicInquiryLens()

    # 1. Paradigm v1 encounters counterexample anomaly -> Paradigm Crisis
    case_crisis = AcademicCase(
        discipline="인지과학",
        paradigm_name="전역 단일 스칼라 환원주의 v1",
        axioms=["A1: 하위 회로 존재", "A2: 영역 고유 순환 존재"],
        methodology="역사·반례 대조 및 위상 비교",
        claims=[
            {"id": "C1", "statement": "모든 학문·언어·코드는 단일 R2로 환원된다", "grounded_in_axioms": ["A1"]},
        ],
        anomalies=[
            {"id": "AN1", "description": "언어 모순과 코드 순환의존성은 R2로 표현 시 왜곡됨", "falsifies_claim_id": "C1"},
        ],
    )
    r1 = lens.process_inquiry(case_crisis)
    assert r1.native_verdict == "PARADIGM_CRISIS_ANOMALY"
    assert r1.native_tension > 0.70

    # 2. Paradigm v2 resolves anomaly + preserves v1 limit + opens next frontier question
    case_shift = AcademicCase(
        discipline="인지과학",
        paradigm_name="전역 단일 스칼라 환원주의 v1",
        axioms=["A1: 하위 회로 존재", "A2: 영역 고유 순환 존재"],
        methodology="다차원 렌즈별 독립 인과 검증 & 비환원 공명",
        claims=case_crisis.claims,
        anomalies=case_crisis.anomalies,
        proposed_paradigm_revision={
            "new_paradigm_name": "혈액-렌즈 분리 다차원 로터 v2",
            "resolves_anomaly_ids": ["AN1"],
            "preserves_prior_as_limit": True,
            "new_scope_boundary": "수치 스칼라는 하위 혈류/제어 영역에 한정",
            "new_open_question": "새로운 예술·윤리 영역을 만났을 때 신규 렌즈 문법 자체를 어떻게 자생시킬 것인가?",
        },
    )
    r2 = lens.process_inquiry(case_shift)
    assert r2.native_verdict == "PARADIGM_SHIFT_CIRCULAR_GROWTH"
    assert r2.revision_applied is True
    assert r2.native_tension < 0.25
    assert r2.scales.macro_horizon["lineage_depth"] == 2

    # 3. Ungrounded circular claim caught cleanly
    case_ungrounded = AcademicCase(
        discipline="논리학",
        paradigm_name="무근거 순환론",
        axioms=["A1: 기초 관측"],
        methodology="자체 검증",
        claims=[{"id": "C_BAD", "statement": "내 주장은 내가 맞으므로 참이다", "grounded_in_axioms": []}],
    )
    r3 = lens.process_inquiry(case_ungrounded)
    assert r3.native_verdict == "EPISTEMIC_GROUNDING_DEFICIT"


def test_hemodynamic_blood_routes_flow_to_stressed_lens_without_flattening_modalities(tmp_path):
    cyc_dir = tmp_path / "stressed"
    cyc_dir.mkdir()
    a = cyc_dir / "mod_x.py"
    b = cyc_dir / "mod_y.py"
    a.write_text("import mod_y\ndef x(v: int) -> int:\n    return v\n", encoding="utf-8")
    b.write_text("import mod_x\ndef y(v: int) -> int:\n    return v\n", encoding="utf-8")

    hub = MultiDimensionalLensRotorHub()
    rep = hub.observe_multidimensionally(
        utterances=["이 문장은 앞뒤 조건이 맞습니다."],
        code_paths=[a, b],
        math_spec={
            "name": "hooke_law",
            "lhs_expr": "F",
            "rhs_expr": "k * x",
            "var_dimensions": {
                "F": SIUnit(L=1, M=1, T=-2),
                "k": SIUnit(M=1, T=-2),
                "x": SIUnit(L=1),
            },
            "positive_vars": ("x",),
        },
        academic_case=AcademicCase(
            discipline="물리학",
            paradigm_name="고전 탄성역학",
            axioms=["A1: 선형 복원력"],
            methodology="역학 평형",
            claims=[{"id": "C1", "statement": "작은 변형에서 힘은 변위에 비례한다", "grounded_in_axioms": ["A1"]}],
        ),
    )

    assert rep.blood_pulse.dominant_lens == "code_architecture"
    assert rep.blood_pulse.blood_flow["code_architecture"] > rep.blood_pulse.blood_flow["linguistic"]
    assert rep.blood_pulse.blood_flow["code_architecture"] > rep.blood_pulse.blood_flow["mathematical_law"]
    assert rep.lens_results["linguistic"].native_verdict == "COHERENT_DISCOURSE"
    assert rep.lens_results["code_architecture"].native_verdict == "CYCLIC_DEPENDENCY_DETECTED"
    assert rep.lens_results["mathematical_law"].native_verdict == "DIMENSIONALLY_COHERENT_WITH_BOUNDARY"
    assert rep.lens_results["academic_inquiry"].native_verdict == "PARADIGM_STABLE"


def test_cross_lens_structural_resonance_bridges_four_distinct_lenses_without_reduction(tmp_path):
    ldir = tmp_path / "clean_code"
    ldir.mkdir()
    core_p = ldir / "core_proto.py"
    impl_p = ldir / "service_impl.py"
    core_p.write_text("def step(v: float) -> float:\n    return v\n", encoding="utf-8")
    impl_p.write_text("import core_proto\ndef run(v: float) -> float:\n    return core_proto.step(v)\n", encoding="utf-8")

    hub = MultiDimensionalLensRotorHub()
    rep = hub.observe_multidimensionally(
        utterances=[
            "모든 개념은 오직 스칼라 하나로만 완전히 환원된다.",
            "언어와 코드는 스칼라 하나로만 환원되지 않는다.",
            "다만 하위 혈류는 스칼라로 흐르되, 상위 차원은 경계를 나누어 순환한다.",
        ],
        code_paths=[core_p, impl_p],
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
            discipline="인식론",
            paradigm_name="단일 환원론",
            axioms=["A1: 영역 고유성"],
            methodology="다차원 경계 대조",
            claims=[{"id": "C1", "statement": "단일 기준만 존재한다", "grounded_in_axioms": ["A1"]}],
            anomalies=[{"id": "AN1", "description": "다차원 반례", "falsifies_claim_id": "C1"}],
            proposed_paradigm_revision={
                "new_paradigm_name": "비환원 다차원 공명론",
                "resolves_anomaly_ids": ["AN1"],
                "preserves_prior_as_limit": True,
                "new_scope_boundary": "기저 혈류 제어 계층",
                "new_open_question": "차기 신규 감각 양식은 어떻게 새 렌즈를 세우는가?",
            },
        ),
    )

    assert len(rep.lens_results) == 4
    assert len(rep.resonance_bridges) >= 4
    assert any("boundary_scoping" in br.shared_motifs for br in rep.resonance_bridges)
    assert any("circular_self_revision" in br.shared_motifs for br in rep.resonance_bridges)


def test_multidimensional_hub_causal_field_4d_sync_ssd_and_18_organ_self_audit(tmp_path):
    slab = tmp_path / "multidim_lens.ice"
    with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
        cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
        hub = MultiDimensionalLensRotorHub(living_cycle=cycle)
        rep = hub.observe_multidimensionally(
            utterances=["하위 회로는 숫자로, 상위 인지는 고유 렌즈로 동작합니다."],
            math_spec={
                "name": "kinetic_energy",
                "lhs_expr": "E",
                "rhs_expr": "0.5 * m * (v ** 2)",
                "var_dimensions": {
                    "E": SIUnit(L=2, M=1, T=-2),
                    "m": SIUnit(M=1),
                    "v": SIUnit(L=1, T=-1),
                },
                "positive_vars": ("v",),
            },
        )
        assert mem_rt.ice.num_crystals >= 2
        assert cycle.verify_autobiographical_integrity() is True

        field = CausalField(dimensions=3)
        sync = hub.sync_to_causal_field(rep, causal_field=field)
        assert "blood_substrate" in sync["voxels"]
        assert "linguistic" in sync["voxels"]
        assert "mathematical_law" in sync["voxels"]
        for vox in field.voxels.values():
            assert tuple(vox.tensor.shape) == (4,)
        field.step(dt=0.1)

        audit = cycle.introspect_own_architecture()
        assert audit.total_organ_files >= 18
        assert "orchestration/multidimensional_lens_rotor.py" in [
            m["relative_path"] for m in audit.verified_organ_modules
        ]


def test_extensible_new_lens_registers_without_modifying_blood_or_hub():
    class MusicalCounterpointLens:
        def analyze(self, voices: dict) -> LensCirculationResult:
            intervals = voices["parallel_fifths"]
            verdict = "COUNTERPOINT_PARALLEL_VIOLATION" if intervals > 0 else "POLYPHONIC_HARMONY"
            tension = 0.80 if intervals > 0 else 0.12
            motifs = ["contradiction_or_crisis"] if intervals > 0 else ["boundary_scoping", "circular_closure"]
            return LensCirculationResult(
                lens_name="musical_counterpoint",
                native_modality="polyphonic_counterpoint",
                scales=ScaleTierSnapshot(
                    micro_cells={"intervals": intervals},
                    meso_process={"voice_leading": "contrary" if intervals == 0 else "parallel"},
                    macro_horizon={"polyphonic_integrity": intervals == 0},
                ),
                native_verdict=verdict,
                native_tension=tension,
                structural_motifs=motifs,
                revision_applied=False,
                native_outcome_ko=f"병행5도={intervals}건 검출",
                next_open_inquiry_ko="대위 진행이 선법 종지에서 자연스러운지 살핍니다.",
            )

    hub = MultiDimensionalLensRotorHub()
    hub.register_lens("musical_counterpoint", MusicalCounterpointLens())
    rep = hub.observe_multidimensionally(
        utterances=["모든 언어가 하나의 숫자라는 주장은 지나친 일반화입니다."],
        custom_inputs={"musical_counterpoint": {"parallel_fifths": 2}},
    )
    assert "musical_counterpoint" in rep.lens_results
    assert rep.lens_results["musical_counterpoint"].native_verdict == "COUNTERPOINT_PARALLEL_VIOLATION"
    assert rep.blood_pulse.dominant_lens == "musical_counterpoint"
