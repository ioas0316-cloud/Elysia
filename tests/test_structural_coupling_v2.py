"""
Tests for Structural Coupling Runtime v2 (구조결합 런타임 v2)
============================================================
Verifies:
1. Resonant Attractor Coupling (conditionless attraction toward resonant frequency/phase basins)
2. Meta-Plastic Prismatic Lens (dynamic refraction matrix updates via trajectory spectral feedback)
3. Cross-Modal Manifold Projection (unified Clifford rotor manifold projection and cross-modal resonance)
4. Invariant Core & System Stability under phase perturbations
"""

import pytest
import numpy as np
from elysia_core.structural_coupling_runtime import (
    ResonantAttractorCoupling,
    MetaPlasticPrismaticLens,
    CrossModalManifoldProjection,
    StructuralCouplingRuntime,
)


def test_resonant_attractor_coupling():
    coupling = ResonantAttractorCoupling(dimension=5, num_basins=4)
    assert coupling.num_basins == 4
    assert coupling.dimension == 5

    # Input wave matching basin 0's natural frequency (~1.0)
    input_wave_0 = np.array([1.0, 0.0, 0.5, 0.0, 0.2])
    state_0, active_idx_0, resonance_0 = coupling.evaluate_resonance(input_wave_0, input_frequency=1.0)

    assert isinstance(state_0, np.ndarray)
    assert len(state_0) == 5
    assert active_idx_0 == 0
    assert resonance_0 > 0.0

    # Input wave matching basin 3's natural frequency (~4.0)
    input_wave_3 = np.array([0.0, 1.0, 0.0, 0.5, 0.1])
    state_3, active_idx_3, resonance_3 = coupling.evaluate_resonance(input_wave_3, input_frequency=4.0)

    assert active_idx_3 == 3
    assert resonance_3 > 0.0


def test_meta_plastic_prismatic_lens():
    lens = MetaPlasticPrismaticLens(dimension=5, plasticity_rate=0.1)

    initial_matrix = lens.prism_matrix.copy()
    initial_ref_index = lens.refractive_index

    # Pass an incoming wave through the prism
    input_tensor = np.array([1.0, 0.5, -0.2, 0.0, 0.8])
    refracted = lens.refract(input_tensor)
    assert len(refracted) == 5

    # Generate synthetic trajectory samples
    t = np.linspace(0, 2 * np.pi, 20)
    trajectory = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2 * t),
        np.cos(2 * t),
        np.ones_like(t) * 0.5
    ])

    # Adapt lens based on trajectory spectral power
    lens.adapt_lens_from_trajectory(trajectory)

    # Prism matrix should deform along principal trajectory frequency covariance
    assert not np.array_equal(lens.prism_matrix, initial_matrix)
    assert lens.refractive_index != initial_ref_index
    assert len(lens.spectral_history) == 1


def test_cross_modal_manifold_projection():
    projection = CrossModalManifoldProjection(target_dim=5)

    modal_inputs = {
        "physical": np.array([0.5, 0.1, 0.0]),
        "sensory": np.array([0.2, 0.8, 0.3, 0.1]),
        "chromatic": np.array([0.33, 0.33, 0.34]),
    }

    unified_field, coherence = projection.cross_resonate(modal_inputs)

    assert len(unified_field) == 5
    assert pytest.approx(np.linalg.norm(unified_field), abs=1e-5) == 1.0
    assert -1.0 <= coherence <= 1.0


def test_structural_coupling_runtime_integrated_steps():
    runtime = StructuralCouplingRuntime(target_dim=5)

    inputs = {
        "physical": np.array([1.0, 0.0, 0.0]),
        "sensory": np.array([0.0, 1.0, 0.0]),
        "symbolic": np.array([0.5, 0.5, 0.0]),
    }

    # Execute multiple steps to test trajectory buffer and lens adaptation
    for step in range(1, 12):
        record = runtime.process_step(
            multimodal_inputs=inputs,
            input_frequency=1.0 + (step * 0.2),
            dt=0.1
        )

        assert record["step"] == step
        assert "active_basin_idx" in record
        assert "prism_refractive_index" in record
        assert "macro_coherence_score" in record
        assert "invariant_core_stability" in record
        assert record["invariant_core_stability"] > 0.0

    # Ensure lens adapted after multiple steps
    assert len(runtime.trajectory_buffer) > 0
    assert len(runtime.prismatic_lens.spectral_history) > 0


if __name__ == "__main__":
    pytest.main([__file__])
