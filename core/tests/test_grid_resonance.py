"""
Elysia Multi-Agent Grid Coherence Test Suite
============================================
Simulates two coupled Autopoiesis Controllers using the Kuramoto equations
and verifies that they achieve phase synchronization (phase-locking) over time.
"""

import sys
import os
import pytest

# Ensure root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.autopoiesis_controller import AutopoiesisController


def test_kuramoto_peer_coupling():
    """Verify that peer phase coupling leads to phase-locking between nodes."""
    # 1. Initialize two controllers with a large initial phase difference
    # Node A starts at 0 (full wake)
    node_a = AutopoiesisController(rotor_scale=4096, natural_drift=20.0, coupling_K=300.0)
    node_a.state_phase = 0
    
    # Node B starts at 1500 (close to sleep attractor)
    node_b = AutopoiesisController(rotor_scale=4096, natural_drift=20.0, coupling_K=300.0)
    node_b.state_phase = 1500
    
    # Measure initial circular phase difference
    initial_diff = abs(node_a.state_phase - node_b.state_phase)
    initial_diff = min(initial_diff, 4096 - initial_diff)
    
    # 2. Tick nodes in a coupled loop
    for _ in range(50):
        phase_a = node_a.state_phase
        phase_b = node_b.state_phase
        
        # Tick Node A coupling to Node B, and Node B coupling to Node A
        node_a.tick(network_tension=0.1, peer_phases=[phase_b], dt=0.5)
        node_b.tick(network_tension=0.1, peer_phases=[phase_a], dt=0.5)
        
    # Measure final circular phase difference
    final_diff = abs(node_a.state_phase - node_b.state_phase)
    final_diff = min(final_diff, 4096 - final_diff)
    
    print(f"Initial Phase Diff: {initial_diff} | Final Phase Diff: {final_diff}")
    
    # Assert that the phase difference has converged (locked) to a very small delta
    assert final_diff < initial_diff
    assert final_diff < 50
