"""
Unit Tests for Adaptive Cognitive Lens & Collective Resonance Network
"""

import pytest
import torch
from synaptic_architecture.adaptive_cognitive_lens import AutonomousCognitiveNode, CollectiveResonanceNetwork


def test_autonomous_cognitive_node_retuning():
    torch.manual_seed(42)
    node = AutonomousCognitiveNode(node_id=0, dim=3, hysteresis_rate=0.85)
    initial_posture = node.posture.clone()

    stimulus_x = torch.tensor([1.0, 0.0, 0.0])
    initial_tension, _ = node.perceive(stimulus_x)

    # Execute adaptation step
    node.step(stimulus_x)
    adapted_tension, _ = node.perceive(stimulus_x)

    # Tension should decrease as node posture rotates towards stimulus
    assert adapted_tension < initial_tension
    assert not torch.allclose(node.posture, initial_posture)


def test_orthogonal_stimulus_hysteresis():
    torch.manual_seed(42)
    node = AutonomousCognitiveNode(node_id=0, dim=3, hysteresis_rate=0.85)

    stimulus_x = torch.tensor([1.0, 0.0, 0.0])
    stimulus_y = torch.tensor([0.0, 1.0, 0.0])

    for _ in range(3):
        node.step(stimulus_x)

    # Step orthogonal stimulus Y
    node.step(stimulus_y)

    # Posture should retain residual X component while shifting towards Y (Hysteresis)
    assert float(node.posture[0].item()) > 0.1
    assert float(node.posture[1].item()) > 0.1


def test_collective_resonance_network_coherence():
    torch.manual_seed(42)
    network = CollectiveResonanceNetwork(num_nodes=6, dim=3)
    initial_coherence = network.calculate_global_coherence()

    stimulus = torch.tensor([1.0, 0.0, 0.0])
    history = network.propagate_resonance(target_node_id=0, direct_stimulus=stimulus, steps=5)

    final_coherence = history[-1]["global_coherence"]

    # Global coherence order parameter should increase after wave propagation
    assert final_coherence > initial_coherence
