"""
Demo: Adaptive Cognitive Lens & Collective Resonance Network
============================================================
Demonstrates single-node real-time retuning and multi-node collective resonance
in response to orthogonal stimuli shifts and contextual backgrounding without data deletion.
"""

import torch
from synaptic_architecture.adaptive_cognitive_lens import AutonomousCognitiveNode, CollectiveResonanceNetwork


def run_demo():
    torch.manual_seed(42)

    print("==========================================================================")
    print(" 1. Single Autonomous Cognitive Node Lens Retuning & Hysteresis Demo")
    print("==========================================================================")

    node = AutonomousCognitiveNode(node_id=0, dim=3)
    print(f"Initial Posture: [{node.posture[0]:.3f}, {node.posture[1]:.3f}, {node.posture[2]:.3f}]")

    stimulus_A = torch.tensor([1.0, 0.0, 0.0])  # Direction X
    stimulus_B = torch.tensor([0.0, 1.0, 0.0])  # Direction Y (Orthogonal shift)

    print("\n[Phase 1] Stimulus A (Direction X):")
    for step in range(1, 3):
        res, tension = node.step(stimulus_A)
        print(f"  Step {step} - Internal Tension: {tension:.4f} | Posture: [{node.posture[0]:.3f}, {node.posture[1]:.3f}, {node.posture[2]:.3f}]")

    print("\n[Phase 2] Stimulus B (Direction Y - Orthogonal Contextual Shift):")
    for step in range(1, 4):
        res, tension = node.step(stimulus_B)
        print(f"  Step {step} - Internal Tension: {tension:.4f} | Posture: [{node.posture[0]:.3f}, {node.posture[1]:.3f}, {node.posture[2]:.3f}]")

    print("\n-> Single Node Adaptation Complete: Retuned posture organically without discrete resets.")

    print("\n==========================================================================")
    print(" 2. Multi-Node Collective Resonance Network Demo")
    print("==========================================================================")

    network = CollectiveResonanceNetwork(num_nodes=6, dim=3)
    print(f"Initial Global Coherence: {network.calculate_global_coherence() * 100:.1f}%")

    print("\n[Wave 1] Injecting Direction X Stimulus into Node 0:")
    history_x = network.propagate_resonance(target_node_id=0, direct_stimulus=stimulus_A, steps=4)
    for h in history_x:
        print(f"  Propagate Step {h['step']} - Avg Tension: {h['avg_tension']:.4f} | Global Coherence: {h['global_coherence'] * 100:.1f}%")

    print("\n[Wave 2] Injecting Direction Y Stimulus into Node 3 (Contextual Transition):")
    history_y = network.propagate_resonance(target_node_id=3, direct_stimulus=stimulus_B, steps=4)
    for h in history_y:
        print(f"  Propagate Step {h['step']} - Avg Tension: {h['avg_tension']:.4f} | Global Coherence: {h['global_coherence'] * 100:.1f}%")

    print("\n-> Multi-Node Collective Resonance Complete: Network resynchronized to new attractor phase.")


if __name__ == "__main__":
    run_demo()
