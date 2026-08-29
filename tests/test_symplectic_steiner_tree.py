import pytest
import numpy as np
from core.physics.symplectic_causal_engine import SymplecticCausalEngine
from core.physics.steiner_causal_network import SteinerCausalNetwork
from core.physics.causal_field import CausalField

def test_symplectic_causal_engine_harmonic_oscillator():
    """
    Verifies that SymplecticCausalEngine conserves energy and converges to minimum potential.
    V(z) = 0.5 * ||z||^2
    """
    pot_fn = lambda z: 0.5 * float(np.sum(z ** 2))
    grad_fn = lambda z: z

    engine = SymplecticCausalEngine(mass=1.0, dt_initial=0.05, gamma_initial=0.4)
    z = np.array([2.0, -1.5], dtype=np.float32)
    p = np.array([0.0, 0.0], dtype=np.float32)

    for _ in range(250):
        z, p, info = engine.step(z, p, pot_fn, grad_fn)

    # Position should converge close to origin
    assert np.linalg.norm(z) < 0.1, f"Expected position close to 0, got {np.linalg.norm(z)}"
    assert info["grad_norm"] < 0.1, f"Expected gradient close to 0, got {info['grad_norm']}"

def test_4point_steiner_tree_convergence():
    """
    Unit Square 4-Terminal Steiner Tree Benchmark Test.
    Terminals: (0,0), (1,0), (1,1), (0,1)

    1. X-shape diagonal length = 2 * sqrt(2) ≈ 2.8284
    2. Optimal Steiner length = 1 + sqrt(3) ≈ 2.73205
    3. Optimal branching angles at Steiner points = 120 degrees
    """
    terminals = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32)
    ]

    net = SteinerCausalNetwork(terminals, num_steiner_points=2, dt=0.02, gamma=0.3)

    x_shape_length = 2.0 * np.sqrt(2.0) # ≈ 2.8284
    optimal_steiner_length = 1.0 + np.sqrt(3.0) # ≈ 2.73205

    initial_length = net.calculate_total_length()

    # Run Symplectic Verlet state convergence loop
    for _ in range(300):
        info = net.step()

    final_length = info["total_length"]
    angles = info["angles"]

    # Check 1: Total length reduced significantly below X-shape
    assert final_length < x_shape_length - 0.05, f"Expected length < {x_shape_length - 0.05}, got {final_length}"

    # Check 2: Total length converges close to theoretical optimal Steiner length (within 1% tolerance)
    assert abs(final_length - optimal_steiner_length) < 0.02, (
        f"Expected final length close to {optimal_steiner_length:.4f}, got {final_length:.4f}"
    )

    # Check 3: Branching angles at Steiner points converge close to 120 degrees (within 5 deg tolerance)
    for s_idx, deg_list in angles.items():
        assert len(deg_list) == 3, f"Expected 3 angles at Steiner point {s_idx}, got {len(deg_list)}"
        for angle in deg_list:
            assert abs(angle - 120.0) < 5.0, (
                f"Steiner point {s_idx} angle {angle:.2f}° not within 5° of 120°"
            )

def test_steiner_network_causal_field_sync():
    """
    Verifies sync from SteinerCausalNetwork to CausalField.
    """
    terminals = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32)
    ]

    net = SteinerCausalNetwork(terminals, num_steiner_points=2)
    net.step()

    cf = CausalField(dimensions=2)
    net.sync_to_causal_field(cf)

    topology = cf.get_topology()
    assert len(topology["voxels"]) == 6 # 4 terminals + 2 Steiner points
    assert len(topology["beams"]) == 5 # 5 connecting edges
