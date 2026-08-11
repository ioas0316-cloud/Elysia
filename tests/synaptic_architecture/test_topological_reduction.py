import pytest
import numpy as np
from core.physics.topological_reduction import TopologicalReductionEngine

def test_topological_reduction_equivalence():
    """Verify Principle 1: Equivalence of the condensed network."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)
    G_reduced, R_eq = engine.compress()

    assert G_reduced.shape == (2, 2)
    assert R_eq > 0.0

    # Assert Laplacian property holds on condensed matrix: diagonal equals negative off-diagonal
    # We use atol=1e-3 because of the regularization factor added to G_II to prevent singularity
    assert np.isclose(G_reduced[0, 0], -G_reduced[0, 1], atol=1e-3)
    assert np.isclose(G_reduced[1, 1], -G_reduced[1, 0], atol=1e-3)

def test_modality_agnostic_projection():
    """Verify Principle 2: Modality-agnostic mapping of different inputs."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    # Baseline
    _, R_eq_baseline = engine.compress()

    # Linguistic Map
    engine.map_multimodal_to_network({
        "language": "Equilibrium",
        "physical": {"cpu": 0.1, "ram": 0.2}
    })
    _, R_eq_lang = engine.compress()

    # Visual Map
    engine.map_multimodal_to_network({
        "visual": {"red": 0.8, "green": 0.2, "blue": 0.5}
    })
    _, R_eq_vis = engine.compress()

    assert R_eq_lang != R_eq_baseline
    assert R_eq_vis != R_eq_lang

def test_closed_loop_self_refinement():
    """Verify Principle 3: Self-correction convergence loop."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    # Make initial network state fully deterministic for robust test convergence
    engine.conductance_matrix[:, :] = 0.1
    np.fill_diagonal(engine.conductance_matrix, 0.0)
    engine._rebuild_laplacian_diagonals()

    target = 2.0
    res = engine.run_self_refinement_loop(target_potential=target, max_steps=15, lr=0.1)

    assert "potentials_history" in res
    assert "residuals_history" in res

    # Check that residual decreases significantly
    initial_residual = abs(res["residuals_history"][0])
    final_residual = abs(res["residuals_history"][-1])
    assert final_residual < initial_residual
    assert final_residual < 0.1

def test_cross_modal_resonance():
    """Verify Principle 4: Cross-modal translation via the shared latent equivalent."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    source = {
        "language": "Sabbath",
        "physical": {"cpu": 0.05, "ram": 0.05}
    }

    translation = engine.cross_modal_translate(source, target_key="visual")
    assert "intensity" in translation["translated_data"]
    assert "red" in translation["translated_data"]

    translation_lang = engine.cross_modal_translate(source, target_key="language")
    assert "concept" in translation_lang["translated_data"]
    assert "resonance" in translation_lang["translated_data"]
