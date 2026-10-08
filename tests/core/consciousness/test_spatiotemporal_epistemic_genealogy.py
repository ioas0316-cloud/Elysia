r"""
Tests for Quadruple Cognitive Quartet & Spatiotemporal Epistemic Genealogy Engine
==================================================================================
Verifies:
1. QuadrupleCognitiveCoordinateEngine stages (Sensory, Cognitive, Observational, Judgmental)
2. Physical field coupling (destructive interference, phase friction, order parameter)
3. Variable Gating awareness (\Theta_retained vs \Theta_gated)
4. SpatiotemporalEpistemicGenealogyEngine engram recording with (t, x) coordinates and causal sequence
"""

import pytest
import numpy as np

from core.consciousness.quadruple_cognitive_coordinate_engine import QuadrupleCognitiveCoordinateEngine
from core.consciousness.spatiotemporal_epistemic_genealogy import SpatiotemporalEpistemicGenealogyEngine


def test_quadruple_cognitive_coordinate_engine():
    engine = QuadrupleCognitiveCoordinateEngine(dimension=64, field_grid_size=6)
    signal = "Formula v = s / t deleted micro-vortices for convenience."

    quartet_state = engine.evaluate_quadruple_quartet(
        signal,
        candidate_variables=["vorticity", "micro_rotor", "temperature", "shear_strain", "macro_velocity"]
    )

    assert "sensory" in quartet_state
    assert "cognitive" in quartet_state
    assert "observational" in quartet_state
    assert "judgmental" in quartet_state

    sensory = quartet_state["sensory"]
    assert sensory["phase_friction"] >= 0.0
    assert sensory["destructive_interference_density"] >= 0.0

    cognitive = quartet_state["cognitive"]
    assert len(cognitive["phase_spectrum_tensor"]) == 64
    assert cognitive["causal_resonance"] > 0.0

    obs = quartet_state["observational"]
    assert obs["lens_curvature"] > 0.0
    assert len(obs["gated_variables"]) + len(obs["retained_variables"]) == 5

    j = quartet_state["judgmental"]
    assert j["causal_value_score"] >= 0.0
    assert j["resolution_type"] in [
        "HOLISTIC_CAUSAL_ANCHORING",
        "LOCAL_INSTRUMENTAL_UTILIZATION",
        "EPISTEMIC_REFINEMENT_NEEDED"
    ]


def test_spatiotemporal_epistemic_genealogy_engine():
    genealogy_engine = SpatiotemporalEpistemicGenealogyEngine(dimension=64)

    knowledge = "Viscosity is the destructive interference tension of phase frequencies."
    spatial_coord = (10.0, 20.0, -5.0)
    sequence = [
        "Wave collision observed",
        "Destructive interference detected",
        "Variable gating applied",
        "Anchored to spatiotemporal engram"
    ]

    engram = genealogy_engine.record_genealogy_engram(
        raw_knowledge=knowledge,
        spatial_coord=spatial_coord,
        context_sequence=sequence,
        candidate_variables=["molecular_adhesion", "macro_friction", "phase_interference", "rotor_torque"]
    )

    assert engram["engram_id"].startswith("ENGRAM_")
    assert engram["raw_knowledge"] == knowledge
    assert engram["spatiotemporal_coordinates"]["spatial_location_x"] == [10.0, 20.0, -5.0]
    assert engram["causal_sequence_lineage"] == sequence
    assert len(engram["gated_variables_theta"]) >= 0

    # Query
    results = genealogy_engine.query_genealogy_by_concept("Viscosity")
    assert len(results) == 1
    assert results[0]["engram_id"] == engram["engram_id"]

    all_engrams = genealogy_engine.get_all_engrams()
    assert len(all_engrams) == 1


def test_relational_tension_and_teleology():
    genealogy_engine = SpatiotemporalEpistemicGenealogyEngine(dimension=64)

    # Set intentional teleology
    intent_vec = np.random.randn(64)
    genealogy_engine.quartet_engine.set_intentional_teleology(
        intent_vec, description="Investigating Microscopic Fluid Dynamics"
    )

    # Record 2 engrams at different spatial coordinates
    e1 = genealogy_engine.record_genealogy_engram(
        raw_knowledge="Micro-rotor spin shear",
        spatial_coord=(0.0, 0.0, 0.0),
        candidate_variables=["micro_rotor", "spin", "shear", "viscosity"]
    )

    e2 = genealogy_engine.record_genealogy_engram(
        raw_knowledge="Macroscopic laminar flow",
        spatial_coord=(3.0, 4.0, 0.0), # Euclidean distance = 5.0
        candidate_variables=["laminar_velocity", "pressure_gradient"]
    )

    # Compute relational tension distance
    dist_info = genealogy_engine.compute_relational_distance_between_engrams(0, 1, medium_viscosity=1.5)

    assert dist_info["euclidean_distance"] == pytest.approx(5.0)
    # Relational distance should be larger than Euclidean due to medium viscosity and gated variables
    assert dist_info["relational_tension_distance"] > 5.0
    assert dist_info["causal_propagation_cost"] > dist_info["relational_tension_distance"]

    # Verify genesis context
    assert "genesis_context" in e1
    assert e1["genesis_context"]["intent_description"] == "Investigating Microscopic Fluid Dynamics"

    # Verify multi-scale feedback modified global viscosity
    assert genealogy_engine.quartet_engine.global_field_viscosity_modifier > 1.0
