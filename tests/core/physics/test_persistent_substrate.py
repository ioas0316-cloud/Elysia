import pytest
import numpy as np
from core.physics.topological_reduction import TopologicalReductionEngine

def test_persistent_substrate_and_decay():
    """Verify that persistent potentials are retained and decayed correctly."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    # Initialize persistent potentials to zero
    assert np.allclose(engine.persistent_potentials, 0.0)

    # Run a diffusion process
    latent_target = 1.5
    potentials_before = engine.diffuse(latent_target)

    # Verify the boundary conditions are set
    assert potentials_before[0] == latent_target
    assert potentials_before[1] == 0.0

    # Verify that the persistent potentials now match the settled state
    assert np.allclose(engine.persistent_potentials, potentials_before)

    # Test decay
    engine.decay_substrate(decay_factor=0.8)
    assert np.allclose(engine.persistent_potentials, potentials_before * 0.8)

def test_scalable_lens_resonant_bypass():
    """Verify that mapping identical or extremely close input triggers O(1) lookup bypass."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    modality_data = {
        "language": "Warmth and Love",
        "visual": {"red": 0.9, "green": 0.1, "blue": 0.3},
        "physical": {"cpu": 0.1, "ram": 0.1}
    }

    # Map first time (Reflection Level)
    engine.map_multimodal_to_network(modality_data)

    # Run refinement loop to settle the potentials and store the converged state in memory
    engine.run_self_refinement_loop(target_potential=2.0, max_steps=5, lr=0.1)

    # Clear traces to isolate the bypass trace
    engine.metacognitive_traces.clear()

    # Map the exact same input again
    engine.map_multimodal_to_network(modality_data)

    # Find if bypass trace was triggered
    bypass_triggered = any(
        trace.get("source") == "scalable_lens_bypass"
        for trace in engine.metacognitive_traces
    )

    assert bypass_triggered, "Scalable Lens O(1) bypass was not triggered for an identical input!"

def test_continuous_local_relaxation():
    """Verify that the continuous-time local relaxation ODE solver runs and settles potentials correctly."""
    engine = TopologicalReductionEngine(num_nodes=8, num_boundary=2)

    target = 3.0
    # Run continuous local relaxation solver
    node_potentials = engine.diffuse(target, use_continuous=True, num_steps=30, dt=0.01)

    # Verify boundary conditions are strictly enforced
    assert node_potentials[0] == target
    assert node_potentials[1] == 0.0

    # Verify that internal nodes settled to some continuous distribution
    for i in engine.internal_nodes:
        # Potentials should stay within stable bounds
        assert -10.0 <= node_potentials[i] <= 10.0

    # Verify it saved state into persistent potentials
    assert np.allclose(engine.persistent_potentials, node_potentials)
