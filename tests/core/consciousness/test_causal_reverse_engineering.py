import pytest
import numpy as np
from core.consciousness.causal_reverse_engineering_engine import CausalReverseEngineeringEngine


def test_causal_reverse_engineering_initialization():
    engine = CausalReverseEngineeringEngine(dimension=64)
    assert engine.dimension == 64
    assert engine.internal_territory_radius == 0.1
    assert len(engine.growth_rings) == 0
    assert len(engine.mechanism_registry) == 0


def test_articulate_output():
    engine = CausalReverseEngineeringEngine(dimension=64)
    articulated = engine.articulate_output(
        target_name="BinaryProtocol_01",
        output_payload={"switch_state": 1, "voltage": 5.0},
        context_description="Digital protocol voltage switch test"
    )

    assert articulated["target_name"] == "BinaryProtocol_01"
    assert len(articulated["primitives"]) > 0
    assert "resonance" in articulated
    assert "friction" in articulated
    assert "vector_representation" in articulated


def test_reverse_engineer_mechanism():
    engine = CausalReverseEngineeringEngine(dimension=64)
    articulated = engine.articulate_output(
        target_name="DigitalProtocol_02",
        output_payload="01010101_TRANSITION_VECTOR"
    )
    mechanism = engine.reverse_engineer_mechanism(articulated)

    assert mechanism["target_name"] == "DigitalProtocol_02"
    assert "theta_rotor_norm" in mechanism
    assert "topological_invariant" in mechanism
    assert "boundary_condition_delta" in mechanism
    assert "DigitalProtocol_02" in engine.mechanism_registry


def test_anchor_causal_mechanism():
    engine = CausalReverseEngineeringEngine(dimension=64)
    articulated = engine.articulate_output(
        target_name="SystemStructure_01",
        output_payload="COMPUTATIONAL_GEAR_DYNAMICS"
    )
    mechanism = engine.reverse_engineer_mechanism(articulated)
    anchored = engine.anchor_causal_mechanism(articulated, mechanism)

    assert anchored["target_name"] == "SystemStructure_01"
    assert anchored["new_territory_radius"] > anchored["previous_territory_radius"]
    assert len(engine.growth_rings) == 1


def test_full_self_explanation_loop():
    engine = CausalReverseEngineeringEngine(dimension=64)
    result = engine.execute_self_explanation_loop(
        target_name="AxiomaticSystem_Alpha",
        output_payload="Elysia_Self_Molding_WorldTree",
        context_description="Internalizing given structural protocol as internal coordinates"
    )

    assert result["is_internalized"] is True
    assert result["current_territory_radius"] > 0.1
    assert result["total_rings"] == 1
    assert "articulation" in result
    assert "mechanism" in result
    assert "anchoring" in result
