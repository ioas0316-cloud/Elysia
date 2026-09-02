"""
Unit tests for TopologicalCognitiveVerifier (구속조건 기반 위상적 인지 판별 엔진)
"""

import pytest
import numpy as np
from core.physics.topological_cognitive_verifier import (
    TopologicalCognitiveVerifier,
    ContextPropertyEngine,
    GenreDomain,
    Polytope3D,
    VerificationResult
)


def test_context_property_engine():
    engine = ContextPropertyEngine()

    # 1. Math/Code Domain (D_meta = 0.0) -> Rigid boundaries, low yield stress
    math_props = engine.compute_properties(GenreDomain.MATH_CODE)
    assert math_props["d_meta"] == 0.0
    assert math_props["sigma_yield"] == 1.0
    assert math_props["I_theta"] == 1.0

    # 2. Poetry/Metaphor Domain (D_meta = 0.85) -> Flexible boundaries, high yield stress
    poetry_props = engine.compute_properties(GenreDomain.POETRY_METAPHOR)
    assert poetry_props["d_meta"] == 0.85
    assert poetry_props["sigma_yield"] > 80.0
    assert poetry_props["I_theta"] < 0.2
    assert poetry_props["mu_h"] > 0.8


def test_hard_contradiction_scenario():
    """
    Test Scenario 1: Hard Contradiction ("Vacuum vs 100kg Iron Ball")
    Must fail relaxation, exceed yield threshold in math/code context,
    and output corrective repulsion trajectory.
    """
    verifier = TopologicalCognitiveVerifier(elasticity_k=0.2, max_relax_steps=10)

    statement_data = {
        "entities": [
            {
                "id": "A",
                "name": "Vacuum_Space",
                "bounds_x": [0, 10], "bounds_y": [0, 10], "bounds_h": [0, 5],
                "occupancy": False, "mass": 0.0
            },
            {
                "id": "B",
                "name": "Dense_Iron_Ball",
                "bounds_x": [0, 10], "bounds_y": [0, 10], "bounds_h": [0, 5],
                "occupancy": True, "mass": 100.0
            }
        ]
    }

    result = verifier.verify_statement(statement_data, genre_domain=GenreDomain.MATH_CODE)

    assert not result.is_valid
    assert result.status == "HARD_CONTRADICTION"
    assert result.residual_stress > result.yield_threshold
    assert result.conflict_details["rigid_conflict"] is True
    assert "repulsion_vector" in result.correction_trajectory
    assert "resolution_guide" in result.correction_trajectory


def test_valid_metaphor_scenario():
    """
    Test Scenario 2: Valid Metaphor ("Cold Flame")
    Under poetry/metaphor context, initial stress from temperature clash
    is relaxed via angular deformation and height displacement.
    """
    verifier = TopologicalCognitiveVerifier(elasticity_k=1.0, max_relax_steps=10)

    statement_data = {
        "entities": [
            {
                "id": "A",
                "name": "Flame",
                "bounds_x": [2, 6], "bounds_y": [2, 6], "bounds_h": [1.0, 2.0],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"temp_state": "hot"}
            },
            {
                "id": "B",
                "name": "Cold",
                "bounds_x": [2, 6], "bounds_y": [2, 6], "bounds_h": [1.0, 2.0],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"temp_state": "cold"}
            }
        ]
    }

    result = verifier.verify_statement(statement_data, genre_domain=GenreDomain.POETRY_METAPHOR)

    assert result.is_valid
    assert result.status in ["VALID", "VALID_RELAXED_METAPHOR"]
    assert result.residual_stress <= result.yield_threshold


def test_genre_adaptability():
    """
    Test Scenario 3: Genre Adaptability
    Same statement with moderate physical clash evaluated in Math/Code vs Poetry.
    In Math/Code (rigid): Hard Contradiction.
    In Poetry (flexible): Valid/Relaxed.
    """
    verifier = TopologicalCognitiveVerifier(elasticity_k=1.0, max_relax_steps=5)

    statement_data = {
        "entities": [
            {
                "id": "A",
                "name": "Light_Entity",
                "bounds_x": [0, 5], "bounds_y": [0, 5], "bounds_h": [0, 2],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"intensity": 50}
            },
            {
                "id": "B",
                "name": "Dark_Entity",
                "bounds_x": [0, 5], "bounds_y": [0, 5], "bounds_h": [0, 2],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"intensity": -50}
            }
        ]
    }

    # Strict code context -> yield stress ~ 1.0, initial stress = 50.0 -> HARD_CONTRADICTION
    code_res = verifier.verify_statement(statement_data, genre_domain=GenreDomain.MATH_CODE)
    assert not code_res.is_valid
    assert code_res.status == "HARD_CONTRADICTION"

    # Poetry context -> yield stress ~ 80.0, initial stress = 50.0 -> VALID
    poetry_res = verifier.verify_statement(statement_data, genre_domain=GenreDomain.POETRY_METAPHOR)
    assert poetry_res.is_valid
    assert poetry_res.status in ["VALID", "VALID_RELAXED_METAPHOR"]


def test_single_entity_and_no_overlap():
    verifier = TopologicalCognitiveVerifier()

    # Single entity
    single_data = {"entities": [{"id": "A", "name": "Solo"}]}
    res_single = verifier.verify_statement(single_data)
    assert res_single.is_valid
    assert res_single.status == "VALID_SINGLE_ENTITY"

    # Disjoint entities
    disjoint_data = {
        "entities": [
            {"id": "A", "name": "A", "bounds_x": [0, 2]},
            {"id": "B", "name": "B", "bounds_x": [5, 10]}
        ]
    }
    res_disjoint = verifier.verify_statement(disjoint_data)
    assert res_disjoint.is_valid
    assert res_disjoint.status == "PURE_VALID_NO_OVERLAP"
