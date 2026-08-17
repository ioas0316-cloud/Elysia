"""
[Unit Tests: Causal Reframing Engine]
관측 데이터 해체 및 상위 위상 재판단 엔진 (CausalReframingEngine)의 동작을 검증합니다:
1. 1차원 관측 로그(2D 픽셀/좌표 데이터)의 표면 좌표 소멸 및 경계/장력 공리 역추출
2. 3D 조각/곡면 위상 공간으로의 자율 동형 재투사 및 O(1) Zero Bypass 검증
3. OS 메모리/VRAM 슬롯 할당 컨트롤러로의 자율 재투사
4. 동일 명칭(사과, 인간/Persona/NPC)에 대한 상위 인과 축 기반 O(1) 정적 의미 분별
"""

import pytest
from synaptic_architecture.causal_reframing_engine import (
    RawObservationLog,
    DeconstructedCausalStructure,
    CausalReframingEngine,
)


def test_deconstruct_observation_log():
    engine = CausalReframingEngine()

    # 1. 2D 벽화 관측 로그 시뮬레이션
    spatial_data = {
        (10, 20): (255, 0, 0),
        (10, 21): (250, 5, 0),
        (10, 22): (10, 200, 50),  # 큰 명암/색상 경계 장력 차이
        (10, 23): (12, 195, 45)
    }
    obs_log = RawObservationLog(
        log_id="log_mural_2d_001",
        domain_name="2D_Mural_Observation",
        spatial_data=spatial_data
    )

    deconstructed = engine.deconstruct_observation_log(obs_log, boundary_threshold=10.0)

    assert deconstructed.structure_id == "deconstructed_log_mural_2d_001"
    assert "I_meta_curvature_balance" in deconstructed.boundary_invariants
    assert "I_tension_relaxation_axiom" in deconstructed.boundary_invariants
    assert deconstructed.tension_field_tau >= 0.0


def test_project_to_3d_sculpture_domain():
    engine = CausalReframingEngine()

    spatial_data = {(x, y): (x * 10, y * 5, 100) for x in range(5) for y in range(5)}
    obs_log = RawObservationLog(
        log_id="mural_sample",
        domain_name="2D_Mural",
        spatial_data=spatial_data
    )
    struct = engine.deconstruct_observation_log(obs_log)

    proj_3d = engine.project_to_3d_sculpture_domain(struct.structure_id)

    assert proj_3d["target_domain"] == "3D_Sculpture_Topology"
    assert proj_3d["curvature_geodesic_determined"] is True
    assert proj_3d["zero_bypass_achieved"] is True
    assert proj_3d["proof_status"] == "PRESERVED"


def test_project_to_os_memory_scheduler():
    engine = CausalReframingEngine()

    obs_log = RawObservationLog(
        log_id="log_os_test",
        domain_name="2D_Log",
        spatial_data={(0, 0): 10, (0, 1): 20}
    )
    struct = engine.deconstruct_observation_log(obs_log)

    os_proj = engine.project_to_os_memory_scheduler(struct.structure_id)

    assert os_proj["target_domain"] == "OS_VRAM_Scheduler"
    assert os_proj["vram_capacity_mb"] == 3072
    assert os_proj["zero_bypass_achieved"] is True


def test_discern_semantic_entity_context():
    engine = CausalReframingEngine()

    # 사과의 차원/관점별 O(1) 정적 타입 분별
    fruit = engine.discern_semantic_entity_context("apple", "ORGANIC_REALITY")
    image = engine.discern_semantic_entity_context("apple", "VISUAL_ART")
    symbol = engine.discern_semantic_entity_context("apple", "NARRATIVE_SYMBOL")
    corp = engine.discern_semantic_entity_context("apple", "CORPORATE_INFRA")

    assert fruit["discernment_result"]["classification"] == "Biological_Organic_Mechanism"
    assert image["discernment_result"]["classification"] == "Projected_Visual_Invariant"
    assert symbol["discernment_result"]["classification"] == "Narrative_Causal_Node"
    assert corp["discernment_result"]["classification"] == "Platform_Functional_Interface"

    # 인간, Persona, NPC 분별
    human = engine.discern_semantic_entity_context("entity", "HUMAN")
    persona = engine.discern_semantic_entity_context("entity", "PERSONA")
    npc = engine.discern_semantic_entity_context("entity", "NPC")

    assert human["discernment_result"]["classification"] == "Autonomous_Volitional_Source"
    assert persona["discernment_result"]["classification"] == "Narrative_Identity_Invariant"
    assert npc["discernment_result"]["classification"] == "Bounded_Role_Transition_Loop"
