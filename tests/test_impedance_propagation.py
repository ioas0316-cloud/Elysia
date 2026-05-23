"""
[VERIFICATION: IMPEDANCE-DRIVEN CONVERGENCE]
Verifies unsupervised shortest path mapping via phase coherence and impedance matching.
"""

import math
import sys
import os

# Ensure import paths are resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.impedance_network import ImpedancePropagationNetwork

def test_unsupervised_path_learning():
    """Verify that the network automatically maps out and lowers resistance along the coherent path."""
    net = ImpedancePropagationNetwork()

    # 1. Setup Nodes (Input -> Hidden -> Output)
    net.add_node("IN", layer=0, initial_phase=0.0)
    net.add_node("H1", layer=1, initial_phase=0.05)   # Coherent path A (near 0.0)
    net.add_node("H2", layer=1, initial_phase=math.pi) # Dissonant path B (anti-phase)
    net.add_node("OUT", layer=2, initial_phase=0.0)

    # 2. Setup Links (Initial Resistance R = 10.0 for all)
    link_in_h1 = net.connect_nodes("IN", "H1", initial_R=10.0)
    link_in_h2 = net.connect_nodes("IN", "H2", initial_R=10.0)
    link_h1_out = net.connect_nodes("H1", "OUT", initial_R=10.0)
    link_h2_out = net.connect_nodes("H2", "OUT", initial_R=10.0)

    # Verify initial currents are equal
    net.forward_propagate({"IN": 1.0})
    assert abs(link_in_h1.I - 0.5) < 1e-5
    assert abs(link_in_h2.I - 0.5) < 1e-5

    # 3. Inject pulses and tune network (Simulate 30 cycles of wave propagation)
    for _ in range(30):
        # Propagate current
        net.forward_propagate({"IN": 1.0})
        # Tune impedances based on current and phase diff (dt=0.1, lr=0.8)
        net.tune_network(dt=0.1, lr=0.8)

    # 4. Verification
    # Coherent Path A (H1) should have carved a low resistance highway
    assert link_in_h1.R < 5.0
    assert link_h1_out.R < 5.0

    # Dissonant Path B (H2) should have blocked current flow by maintaining/increasing resistance
    assert link_in_h2.R > 10.0
    assert link_h2_out.R > 10.0

    # Final forward propagation should show current heavily biased/routed through H1 path
    net.forward_propagate({"IN": 1.0})
    print(f"  - Route A (H1) Current: {link_in_h1.I:.4f} | Resistance: {link_in_h1.R:.4f}")
    print(f"  - Route B (H2) Current: {link_in_h2.I:.4f} | Resistance: {link_in_h2.R:.4f}")

    assert link_in_h1.I > 0.8  # Over 80% of current routed through the coherent highway
    assert link_in_h2.I < 0.2  # Dissonant highway successfully blocked
