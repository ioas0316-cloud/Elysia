"""
Unit tests for synaptic_architecture/volitional_gated_architecture.py
"""

import torch
import pytest
from synaptic_architecture.volitional_gated_architecture import (
    VolitionalGatedArchitecturePyTorch,
    DistributedKuramotoPhaseField,
    matrix_exponential_so_n
)


def test_matrix_exponential_so_n_orthogonality():
    dim = 16
    torch.manual_seed(42)
    A = torch.randn(dim, dim)
    Omega = A - A.T  # Skew-symmetric
    g_exp = matrix_exponential_so_n(Omega)

    # Verify g_exp @ g_exp.T == I
    ortho_diff = torch.norm(torch.matmul(g_exp, g_exp.T) - torch.eye(dim)).item()
    assert ortho_diff < 1e-5, f"Matrix exponential in SO(N) failed orthogonality test: diff={ortho_diff}"


def test_volitional_gated_architecture_pytorch_layers():
    dim = 16
    arch = VolitionalGatedArchitecturePyTorch(topology_dim=dim, c_max=0.35, tau_min=0.05)
    internal_state = torch.randn(dim)

    # 1. Passive early exit (x_input == internal_state -> E = 0 < tau_min)
    res_passive = arch(internal_state.clone(), internal_state)
    assert res_passive["passive_gated"]
    assert not res_passive["volitional_active"]
    assert res_passive["compute_cost"] == 0.0

    # 2. Dissonance triggers Volitional Gate (E >= tau_min)
    distorted_input = internal_state + torch.randn(dim) * 1.5
    res_active = arch(distorted_input, internal_state)
    assert not res_active["passive_gated"]
    assert res_active["volitional_active"]
    assert res_active["orthogonality_error"] < 1e-5


def test_distributed_kuramoto_phase_locking_transition():
    dim = 16
    field = DistributedKuramotoPhaseField(
        num_agents=12,
        topology_dim=dim,
        tau_min=0.10,
        c_max=0.50,
        k_0=10.0,
        delta=0.02
    )

    internal_state = torch.randn(dim)

    # 1. Low discrepancy E < tau_min -> Low coupling K(E) -> Decoherent exploration R ~ low
    res_low = field.step(internal_state + torch.randn(dim) * 0.01, internal_state, dt=0.01)
    assert res_low["coupling_strength"] < 1.0

    # 2. High discrepancy E >> tau_min -> High coupling K(E) -> Kuramoto Phase Locking (R -> 1)
    large_shock = internal_state + torch.randn(dim) * 2.5
    for _ in range(30):
        res_high = field.step(large_shock, internal_state, dt=0.05)

    assert res_high["coupling_strength"] > 8.0
    assert res_high["order_parameter_R"] > 0.75, f"Phase locking failed: R={res_high['order_parameter_R']}"
    assert res_high["is_phase_locked"]


if __name__ == "__main__":
    test_matrix_exponential_so_n_orthogonality()
    test_volitional_gated_architecture_pytorch_layers()
    test_distributed_kuramoto_phase_locking_transition()
    print("ALL VOLITIONAL ARCHITECTURE PYTORCH TESTS PASSED!")
