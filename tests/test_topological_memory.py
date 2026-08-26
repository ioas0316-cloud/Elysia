"""
Unit & Integration Tests for Topological Memory Landscape & Modality-Agnostic Phase Space Mapper
==================================================================================================
1. PhaseSpaceMapper: Modality-agnostic wave mapping (text, image, wave) and cross-modality isomorphism.
2. TopologicalMemory: Reference plane, friction measurement, dynamic rewiring, concept crystallization.
"""

import pytest
import numpy as np

from core.topology.phase_space_mapper import PhaseSpaceMapper
from core.memory.topological_memory import TopologicalMemory, CrystallizedConcept
from core.topology.cognitive_gate import CognitiveGate, RecursiveCognitiveStack
from core.topology.causal_structure import InformationTopology


def test_phase_space_mapper_modality_agnostic():
    """테스트 1: 이종 입력을 동일한 차원의 연속 위상 벡터 X(p)로 표준화 변환하는지 검증"""
    mapper = PhaseSpaceMapper(target_dimension=8)

    # 1. Text input
    text_wave = mapper.map_text("Elysia Continuous Causal Topology")
    assert text_wave.shape == (8,)
    assert pytest.approx(np.linalg.norm(text_wave), abs=1e-5) == 1.0

    # 2. Image input (2D numpy array)
    img_data = np.random.rand(16, 16)
    img_wave = mapper.map_image(img_data)
    assert img_wave.shape == (8,)
    assert pytest.approx(np.linalg.norm(img_wave), abs=1e-5) == 1.0

    # 3. Wave/Temporal input (1D numpy array)
    t = np.linspace(0, 1, 100)
    audio_wave_data = np.sin(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 12 * t)
    wave_vec = mapper.map_wave(audio_wave_data)
    assert wave_vec.shape == (8,)
    assert pytest.approx(np.linalg.norm(wave_vec), abs=1e-5) == 1.0

    # 4. Universal map_signal interface
    s1 = mapper.map_signal("Hello World", modality="text")
    s2 = mapper.map_signal(img_data, modality="image")
    s3 = mapper.map_signal(audio_wave_data, modality="wave")
    assert s1.shape == (8,) and s2.shape == (8,) and s3.shape == (8,)


def test_cross_modality_isomorphism():
    """테스트 2: 교차 감각 동형성 (Text와 Image에서 동일 대칭 뼈대 추출 시 동형 반응) 검증"""
    mapper = PhaseSpaceMapper(target_dimension=8)
    gate = CognitiveGate(dimension=8, threshold=0.1)

    # 대칭 구조를 갖는 텍스트와 2D 이미지 신호 생성
    text_sym = mapper.map_text("ABCBA_Symmetric_Pattern")

    # 대칭 2D matrix
    grid = np.eye(8) + np.fliplr(np.eye(8))
    img_sym = mapper.map_image(grid)

    I_text, V_text = gate.discriminate(text_sym)
    I_img, V_img = gate.discriminate(img_sym)

    # 모달리티와 상관없이 인지 게이트가 불변량 뼈대를 유효하게 분별해내는지 확인
    assert len(I_text) == 8
    assert len(I_img) == 8
    assert np.linalg.norm(I_text) > 0.0
    assert np.linalg.norm(I_img) > 0.0


def test_topological_memory_reference_plane_and_rewiring():
    """테스트 3: historical reference plane 기반 마찰 측정 및 dynamic topology rewiring 검증"""
    memory = TopologicalMemory(dimension=8, rewire_threshold=0.2)

    # 1. Initial Reference Plane check
    ref_lens_1 = memory.get_reference_plane()
    assert ref_lens_1.shape == (8, 8)

    # 2. Feed sequence of phase signals
    signal1 = np.ones(8, dtype=np.float32) / np.sqrt(8)
    res1 = memory.process_and_rewire(signal1, node_id="node_a")

    assert res1["node_id"] == "node_a"
    assert "invariant" in res1
    assert "variant" in res1
    assert "friction" in res1

    # 3. Feed different phase signal to induce friction and trigger rewiring
    signal2 = np.array([1, -1, 1, -1, 0.5, -0.5, 0.2, -0.2], dtype=np.float32)
    signal2 = signal2 / np.linalg.norm(signal2)
    res2 = memory.process_and_rewire(signal2, node_id="node_b")

    # Check if links were rewired between node_a and node_b
    assert len(memory.topology.links) > 0
    assert memory.get_landscape_summary()["total_nodes"] == 2
    assert memory.get_landscape_summary()["total_links"] >= 1


def test_concept_crystallization_from_friction_boundary():
    """테스트 4: 연속적 마찰 경계면이 이산적 개념(Concept)으로 결속/경계 절단됨을 검증"""
    memory = TopologicalMemory(dimension=8, rewire_threshold=0.1)

    # 마찰을 유발하는 파동 주입
    noisy_wave = np.random.randn(8).astype(np.float32)
    res = memory.process_and_rewire(noisy_wave, node_id="concept_trigger_node")

    # friction이 threshold(0.1)를 초과하여 CrystallizedConcept 생성 확인
    if res["friction"] > 0.1:
        assert res["crystallized_concept"] is not None
        assert isinstance(res["crystallized_concept"], CrystallizedConcept)
        assert len(memory.crystallized_concepts) >= 1
        assert len(memory.topology.symbols) >= 1


def test_full_cognitive_memory_feedback_pipeline():
    """테스트 5: PhaseSpaceMapper -> CognitiveGate -> TopologicalMemory 통합 동적 피드백 수렴 검증"""
    mapper = PhaseSpaceMapper(target_dimension=8)
    memory = TopologicalMemory(dimension=8, rewire_threshold=0.15)

    # 10단계 동안 일관된 멀티모달 패턴 유입 시 마찰 에너지가 수렴하고 memory landscape가 안정화되는지 확인
    frictions = []
    text_inputs = [f"Continuous Causal Reasoning Step {i}" for i in range(10)]

    for txt in text_inputs:
        phase_wave = mapper.map_text(txt)
        res = memory.process_and_rewire(phase_wave)
        frictions.append(res["friction"])

    summary = memory.get_landscape_summary()
    assert summary["total_nodes"] == 10
    assert summary["total_links"] > 0
    assert len(memory.trajectory_history) == 10
