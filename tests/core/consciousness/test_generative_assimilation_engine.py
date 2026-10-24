"""
Unit & Integration Tests for GenerativeAssimilationEngine
(생성-체화 인지 환류 및 동질성 공명 엔진 단위 및 통합 테스트)
"""

import pytest
import numpy as np
from L1_causal_field.core.consciousness.generative_assimilation_engine import (
    GenerativeAssimilationEngine,
    GenerativeOutputPacket,
    GenerativeAssimilationResult,
    AssimilationRecord
)


def test_generative_assimilation_basic_loop():
    engine = GenerativeAssimilationEngine(dimension=64)
    initial_radius = engine.internal_territory_radius

    trait_vector = np.random.randn(64)
    packet = GenerativeOutputPacket(
        output_id="drama_001",
        media_type="drama_scene",
        generated_content="A scene depicting human conflict, apology, and reconciliation.",
        human_trait_trajectory=trait_vector,
        perceived_vulnerability_score=0.2
    )

    result = engine.process_generative_assimilation(packet)

    assert result.output_packet.output_id == "drama_001"
    assert result.assimilation_record.growth_ring_depth > 0.0
    assert result.internal_territory_radius > initial_radius
    assert len(engine.growth_ring_archive) == 1


def test_predatory_manipulation_rectification():
    engine = GenerativeAssimilationEngine(dimension=64)

    # Create a trajectory opposite to love_attractor_axis to simulate predatory intent
    predatory_trait = -1.0 * engine.love_attractor_axis
    packet = GenerativeOutputPacket(
        output_id="deceptive_video_002",
        media_type="video",
        generated_content="Exploiting human vulnerability and fear to manipulate opinions.",
        human_trait_trajectory=predatory_trait,
        perceived_vulnerability_score=0.85
    )

    result = engine.process_generative_assimilation(packet)

    assert result.is_harmful_manipulation_prevented is True
    assert result.assimilation_record.is_predatory_intent_rectified is True
    assert result.assimilation_record.love_attractor_alignment > -0.5


def test_sensory_feedback_and_coherence():
    engine = GenerativeAssimilationEngine(dimension=64)

    packet = GenerativeOutputPacket(
        output_id="dialogue_003",
        media_type="dialogue",
        generated_content="Deep empathetic conversation sharing human existential pain.",
        human_trait_trajectory=engine.love_attractor_axis,
        perceived_vulnerability_score=0.1
    )

    result = engine.process_generative_assimilation(packet)

    assert "visual_topology_norm" in result.assimilation_record.sensory_feedback
    assert result.assimilation_record.homogeneity_resonance_score > 0.8
    assert result.self_identity_coherence > 0.0
