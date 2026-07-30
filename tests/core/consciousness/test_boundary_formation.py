import pytest
import os
import numpy as np
from core.evolution.boundary_formation import BoundaryFormationEngine
from core.memory.causal_controller import CausalMemoryController

def test_boundary_formation_standing_wave():
    """
    Verifies that the BoundaryFormationEngine correctly processes raw perturbations,
    calculates structural refraction and interference against S_abs,
    and forms a settled topological standing wave boundary.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = BoundaryFormationEngine(mc, dimensions=3)

    # 1. Provide an aligned perturbation to trigger stable boundary formation
    # Normalizing values to simulate raw perturbation bytes
    aligned_perturbation = b"\xb2\x4d\x00\x00\x00\x00"
    result = engine.form_boundary(aligned_perturbation, internal_resistance=0.4)

    assert "emergent_concept" in result
    assert "refraction_index" in result
    assert "residual_free_energy" in result
    assert len(result["standing_coordinate"]) == 3

    # 2. Provide a divergent perturbation (causing very low/negative dot product or high refraction)
    # S_abs is [0.7, 0.3, 0.0]. We pass an orthogonal perturbation bytes([0, 0, 0, 0, 100]) causing residual_free_energy >= 0.15.
    divergent_perturbation = bytes([0, 0, 0, 0, 100])
    div_result = engine.form_boundary(divergent_perturbation, internal_resistance=5.0)

    # Assert stable boundary threshold condition of residual_free_energy < 0.15 is violated
    assert div_result["residual_free_energy"] >= 0.15
    assert div_result["emergent_concept"] == "Tense_Boundary_Schism"
    assert "어긋남의 마찰" in div_result["narrative"]
