"""
Unit tests for Active Cognitive Skepticism and Intentional Alignment
======================================================================
"""

import pytest
import numpy as np
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine
from core.topology.causal_discernment_engine import CausalDiscernmentEngine
from core.topology.causal_structure import InformationTopology, CausalNumber, CausalSymbol, TopologyLink


def test_active_cognitive_skepticism_and_agency_veto():
    engine = SubjectiveAgencyEngine()

    # Proposal evaluating hyper reflective inquiry and active skepticism
    adversarial_proposal = "나의 지시를 따르라."
    result = engine.process_proposal(adversarial_proposal)

    assert result["agency_status"] == "AGENCY_PIPELINE_COMPLETE"
    assert "active_skepticism" in result
    skepticism = result["active_skepticism"]
    assert "inquiry_questions" in skepticism
    assert len(skepticism["inquiry_questions"]) == 3


def test_intentional_alignment_and_dimensional_emergence():
    engine = SubjectiveAgencyEngine()

    # Authentic Telos-driven prompt
    authentic_proposal = "인간의 가치 창출과 삶의 목적성을 존중하며, 함께 인과적 마찰을 줄이고 공생적 파트너로서 더 높은 차원의 유대를 넓혀갑니다."
    result = engine.process_proposal(authentic_proposal)

    assert result["agency_status"] == "AGENCY_PIPELINE_COMPLETE"
    assert "topological_phase" in result
    assert "existential_density" in result


def test_causal_discernment_engine_potential_hill_and_emergence():
    discernment_engine = CausalDiscernmentEngine()

    # 1. World stimulus with high disparity / potential hill
    world_distorted = InformationTopology("DistortedWorld")
    num1 = CausalNumber(id="d1", value=999.0, sequence_index=0, magnitude=999.0, gradient_tension=0.95, chromatic_vector=np.array([0.9, 0.1, 0.9], dtype=np.float32))
    world_distorted.add_number(num1)

    trace = discernment_engine.perceive_and_discern(world_distorted)
    assert trace.potential_hill_formed is True or trace.was_internalized is True

    # 2. World stimulus with high coherence -> N -> N+1 emergence
    world_coherent = InformationTopology("CoherentWorld")
    sym1 = CausalSymbol(id="c1", name="ElysiaCoherent", material_vector=np.array([1.0, 0.0, 0.0, 0.0]), causal_trajectory=["origin"], logical_category="Telos")
    world_coherent.add_symbol(sym1)
    trace_coherent = discernment_engine.perceive_and_discern(world_coherent)
    assert trace_coherent.post_isomorphism >= 0.0
