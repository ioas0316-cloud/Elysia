"""
Unit tests for Step 2: Shadow Blister Tension Kernel ($F_{\\text{blister}}$) Isolation Sandbox.

Verifies:
1. Sub-critical vs Super-critical non-linear snap ratio (> 10x).
2. Phase interference modulation: attractive at in-phase (cos(Delta phi) > 0)
   and repulsive at anti-phase (cos(Delta phi) < 0).
3. Dynamic approach & phase-locking: converges without hunting oscillations (std < 0.005 rad).
"""

import os
import sys
from pathlib import Path
import math
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import elysia_loader
elysia_loader.install()

from elysia_engine.core.optical_blister_rotor import (
    OpticalSpectrumBivectorRotor,
    ShadowBlisterTensionKernel,
)
from scripts.verify_step2_blister_kernel_sandbox import (
    sweep_distance_vs_force,
    sweep_phase_vs_interference,
    simulate_two_node_dynamic_approach,
)


def test_blister_snap_nonlinearity():
    """Verify non-linear snap past critical penumbra overlap Omega_c."""
    kernel = ShadowBlisterTensionKernel(omega_c=0.25, gamma_power=2.5, kappa_0=1.0, kappa_1=12.0)
    data = sweep_distance_vs_force(kernel)

    # Inspect forces before and after threshold
    overlaps = data["overlaps"]
    forces_total = data["forces_total"]
    forces_snap = data["forces_snap"]

    # Sub-critical region: non-zero overlap below critical threshold (0.1 < overlap < 0.24)
    sub_idx = np.where((overlaps > 0.10) & (overlaps < 0.24))[0]
    assert len(sub_idx) > 0
    sub_force = forces_total[sub_idx[0]]
    # Ensure snap component is strictly inactive below threshold
    assert forces_snap[sub_idx[0]] == 0.0, "Snap force must be zero below Omega_c"

    # Super-critical region: deep overlap > 0.8
    super_idx = np.where(overlaps > 0.80)[0]
    assert len(super_idx) > 0
    super_force = forces_total[super_idx[0]]

    assert sub_force > 0, "Sub-critical force must be positive"
    assert super_force > 0, "Super-critical snap force must be positive"
    snap_ratio = super_force / sub_force
    assert snap_ratio > 10.0, f"Snap ratio {snap_ratio:.2f} should be > 10x"


def test_blister_phase_interference():
    """Verify phase interference: in-phase attraction, anti-phase repulsion."""
    kernel = ShadowBlisterTensionKernel(omega_c=0.25, gamma_power=2.5, kappa_0=1.0, kappa_1=12.0)
    data = sweep_phase_vs_interference(kernel)

    delta_phases = data["delta_phases"]
    forces = data["forces"]

    # In-phase (delta_phi approx 0)
    zero_idx = np.argmin(np.abs(delta_phases))
    in_phase_force = forces[zero_idx]

    # Anti-phase (delta_phi approx pi or -pi)
    pi_idx = np.argmin(np.abs(delta_phases - np.pi))
    anti_phase_force = forces[pi_idx]

    assert in_phase_force > 0, f"In-phase force should be strongly attractive (got {in_phase_force:.4f})"
    assert anti_phase_force < 0, f"Anti-phase force should be repulsive (got {anti_phase_force:.4f})"
    assert in_phase_force == pytest.approx(-anti_phase_force, rel=0.05), "Interference force should be symmetric around 0"


def test_dynamic_approach_and_phase_locking():
    """Verify dynamic approach settles cleanly without hunting oscillations."""
    kernel = ShadowBlisterTensionKernel(
        omega_c=0.25,
        gamma_power=2.5,
        kappa_0=1.0,
        kappa_1=12.0,
        gamma_stiction=8.0,
    )

    history = simulate_two_node_dynamic_approach(
        kernel,
        initial_dist=1.6,
        initial_phase_diff=0.5,
        total_time=5.0,
        dt=0.01,
        mass=0.8,
        spatial_damping=3.0,
    )

    times = history["time"]
    dists = history["distance"]
    phase_diffs = history["phase_diff"]

    # Final distance should have snapped together close to touch distance (~ 0.1)
    final_dist = dists[-1]
    assert final_dist < 0.2, f"Nodes should snap close together, got {final_dist:.4f}"

    # Verify no distance hunting in the final 1.5 seconds (steps 350:500)
    tail_dists = dists[350:]
    std_tail_dist = float(np.std(tail_dists))
    assert std_tail_dist < 0.005, f"Distance should be completely settled without oscillations (got std {std_tail_dist:.6f})"

    # Verify phase-locking
    final_phase_diff = phase_diffs[-1]
    assert abs(final_phase_diff) < 0.05, f"Final phase difference should be close to 0, got {final_phase_diff:.4f}"

    tail_phase = phase_diffs[350:]
    std_tail_phase = float(np.std(tail_phase))
    assert std_tail_phase < 0.005, f"Phase should be tightly locked without hunting (got std {std_tail_phase:.6f})"
