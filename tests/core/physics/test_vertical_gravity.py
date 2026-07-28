import numpy as np
import pytest
from core.physics.causal_gravity_engine import CausalGravityEngine

def test_vertical_gravity_and_antenna_resonance():
    """
    Verifies that the enhanced CausalGravityEngine:
    1. Simulates continuous SNN membrane potential accumulation & decay.
    2. Uses a vertical potential gradient (purpose_tensor) to lift nodes vertically.
    3. Triggers antenna resonance spikes and aligns neighbor positions upon threshold break.
    """
    # 9-dimensional engine
    engine = CausalGravityEngine(dimensions=9)

    # Add two nodes
    # Node 1 is perfectly aligned with purpose_tensor (high resonance)
    # Node 2 is poorly aligned with purpose_tensor
    tensor1 = [1.0] * 9  # High resonance with np.ones
    tensor2 = [0.0] * 9  # Zero resonance

    engine.add_node("node_aligned", b"Aligned content", tensor1)
    engine.add_node("node_dissonant", b"Dissonant content", tensor2)

    # Initial vertical positions (last dimension, index -1)
    initial_vertical_1 = engine.node_data["node_aligned"].position[-1]
    initial_vertical_2 = engine.node_data["node_dissonant"].position[-1]

    # Verify that before stepping, membrane potentials are 0
    assert engine.node_data["node_aligned"].membrane_potential == 0.0
    assert engine.node_data["node_dissonant"].membrane_potential == 0.0

    # Step the engine to evolve the field and observe vertical upward pull and potential accumulation
    engine.step(dt=0.2)

    # 1. Aligned node should have higher vertical movement than the dissonant node
    final_vertical_1 = engine.node_data["node_aligned"].position[-1]
    final_vertical_2 = engine.node_data["node_dissonant"].position[-1]

    aligned_lift = final_vertical_1 - initial_vertical_1
    dissonant_lift = final_vertical_2 - initial_vertical_2

    # Since node_aligned has positive dot product with purpose_tensor, it must be pulled higher
    assert aligned_lift > dissonant_lift

    # 2. Membrane potential of the aligned node should have increased
    assert engine.node_data["node_aligned"].membrane_potential > 0.0

    # 3. Test spiking threshold and antenna resonance (alignment)
    # Manually charge node_aligned to exceed spike threshold
    engine.node_data["node_aligned"].membrane_potential = 1.5
    engine.node_data["node_aligned"].spike_threshold = 1.0

    # Before spike, record the vertical distance between the two nodes
    dist_before_spike = abs(engine.node_data["node_aligned"].position[-1] - engine.node_data["node_dissonant"].position[-1])

    # Synchronize field arrays to make sure the step uses the manually modified potential
    engine._synchronize_field()

    # Step again to trigger the spike
    engine.step(dt=0.1)

    # The aligned node should have spiked (exceeded threshold, triggering reset to 0.0)
    assert engine.node_data["node_aligned"].membrane_potential == 0.0
    assert engine.spikes_triggered_count > 0

    # Check antenna resonance: Node 2's vertical level should have been pulled closer to Node 1's vertical level
    dist_after_spike = abs(engine.node_data["node_aligned"].position[-1] - engine.node_data["node_dissonant"].position[-1])
    assert dist_after_spike < dist_before_spike
