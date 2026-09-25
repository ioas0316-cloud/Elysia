import pytest
import torch
import numpy as np

from elysia_engine.core.cmplr import CliffordPhaseLockKernel
from elysia_engine.core.cmplr_fallback import CliffordPhaseLockKernelFallback

# -------------------------------------------------------------------
# Helper: CSR graph and tensor generation utilities
# -------------------------------------------------------------------
def generate_mock_csr_graph(num_nodes=64, avg_degree=4, device="cpu"):
    """Generates mock CSR graph, K-tensors, and initial multivector state tensor."""
    torch.manual_seed(42)

    # Generate edges per node
    edges = []
    for a in range(num_nodes):
        num_neighbors = torch.randint(1, avg_degree * 2, (1,)).item()
        neighbors = torch.randperm(num_nodes)[:num_neighbors]
        neighbors = neighbors[neighbors != a]  # Exclude self-loops
        for b in neighbors:
            edges.append((a, b.item()))

    row_ptr = [0]
    col_ind = []
    curr_count = 0
    for a in range(num_nodes):
        a_edges = [b for (src, b) in edges if src == a]
        col_ind.extend(a_edges)
        curr_count += len(a_edges)
        row_ptr.append(curr_count)

    row_ptr_t = torch.tensor(row_ptr, dtype=torch.int32, device=device)
    col_ind_t = torch.tensor(col_ind, dtype=torch.int32, device=device)
    num_edges = len(col_ind)

    # K-Tensor: Identity coupling tensor scaled for stable alignment
    K_tensors_t = torch.eye(3, dtype=torch.float32, device=device).repeat(num_edges, 1, 1).view(num_edges, 9) * 0.1

    # Multivector State: [N, 8]
    Psi_t = torch.randn(num_nodes, 8, dtype=torch.float32, device=device)

    # Apply initial Spin(3) gauge projection using Clifford reversal <Psi * ~Psi>_0 = 1
    rev_mask = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32, device=device)
    norm_sq = torch.abs(torch.sum(Psi_t * (Psi_t * rev_mask), dim=-1, keepdim=True))
    Psi_t = Psi_t / torch.sqrt(norm_sq + 1e-8)

    return Psi_t, row_ptr_t, col_ind_t, K_tensors_t


def compute_bivector_phase_disagreement(Psi, row_ptr, col_ind, fallback_inst):
    """Computes total bivector phase mismatch norm across graph edges."""
    rev_mask = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=Psi.dtype, device=Psi.device)
    rev_Psi = Psi * rev_mask
    num_nodes = Psi.shape[0]
    total_mismatch = 0.0

    for a in range(num_nodes):
        start, end = row_ptr[a].item(), row_ptr[a + 1].item()
        if start == end:
            continue
        neighbors = col_ind[start:end]
        delta_ab = fallback_inst._geometric_product_grade2(Psi[neighbors], rev_Psi[a].unsqueeze(0))
        total_mismatch += torch.sum(torch.norm(delta_ab, dim=-1)).item()

    return total_mismatch


# -------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------

def test_spin3_gauge_projection_precision():
    """1) Spin(3) gauge projection precision O(1) verification (<Psi * ~Psi>_0 = 1.0)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Psi, row_ptr, col_ind, K_tensors = generate_mock_csr_graph(num_nodes=128, device=device)

    kernel = CliffordPhaseLockKernel(Psi, row_ptr, col_ind, K_tensors)

    # Perform 500 relaxation steps
    for _ in range(500):
        Psi_next = kernel.step(dt=0.01)

    # Verify scalar norm error <Psi * ~Psi>_0
    rev_mask = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32, device=device)
    scalar_norms = torch.abs(torch.sum(Psi_next * (Psi_next * rev_mask), dim=-1))
    drift_errors = torch.abs(scalar_norms - 1.0)

    max_error = torch.max(drift_errors).item()
    assert max_error < 1e-5, f"Spin(3) Gauge Drift Error exceeded threshold: {max_error:.8e}"


def test_lyapunov_energy_decay():
    """2) Phase mismatch energy decay (Relaxation Decay) and attractor convergence verification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Psi, row_ptr, col_ind, K_tensors = generate_mock_csr_graph(num_nodes=64, device=device)

    kernel = CliffordPhaseLockKernel(Psi, row_ptr, col_ind, K_tensors)

    mismatches = []
    for _ in range(100):
        current_mismatch = compute_bivector_phase_disagreement(kernel.Psi, row_ptr, col_ind, kernel.fallback)
        mismatches.append(current_mismatch)
        kernel.step(dt=0.1)

    initial_m = mismatches[0]
    final_m = mismatches[-1]
    assert final_m <= initial_m + 1e-5, f"Phase mismatch energy failed to decay: Initial {initial_m:.4f} -> Final {final_m:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Extension verification requires CUDA device")
def test_cuda_vs_cpu_equivalence():
    """3) CUDA C++ kernel vs CPU Fallback numerical equivalence comparison (atol=1e-5)."""
    Psi_cpu, row_ptr_cpu, col_ind_cpu, K_cpu = generate_mock_csr_graph(num_nodes=32, device="cpu")

    Psi_cuda = Psi_cpu.clone().cuda()
    row_ptr_cuda = row_ptr_cpu.cuda()
    col_ind_cuda = col_ind_cpu.cuda()
    K_cuda = K_cpu.clone().cuda()

    kernel_cpu = CliffordPhaseLockKernelFallback(Psi_cpu, row_ptr_cpu, col_ind_cpu, K_cpu)
    kernel_cuda = CliffordPhaseLockKernel(Psi_cuda, row_ptr_cuda, col_ind_cuda, K_cuda)

    # Run 10 relaxation steps and measure difference
    for _ in range(10):
        kernel_cpu.step(dt=0.01)
        kernel_cuda.step(dt=0.01)

    max_abs_diff = torch.max(torch.abs(kernel_cpu.Psi - kernel_cuda.Psi.cpu())).item()
    assert max_abs_diff < 1e-5, f"CUDA and CPU Fallback result mismatch. Max Abs Diff: {max_abs_diff:.8e}"


def test_csr_graph_boundary_safety():
    """4) Irregular CSR graph boundary safety with isolated node (Degree 0) and hub node (Degree > 50)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes = 100

    # Node 0: Isolated node (Degree 0)
    # Node 1: Giant Hub node (Connected to all other nodes)
    row_ptr = [0, 0]  # Node 0 has 0 edges
    col_ind = []

    # Node 1 connected to 2..99
    hub_edges = list(range(2, num_nodes))
    col_ind.extend(hub_edges)
    row_ptr.append(len(col_ind))

    # Rest of nodes
    for a in range(2, num_nodes):
        col_ind.append(1)  # connect back to Hub
        row_ptr.append(len(col_ind))

    row_ptr_t = torch.tensor(row_ptr, dtype=torch.int32, device=device)
    col_ind_t = torch.tensor(col_ind, dtype=torch.int32, device=device)
    K_tensors_t = torch.randn(len(col_ind), 9, dtype=torch.float32, device=device)
    Psi_t = torch.randn(num_nodes, 8, dtype=torch.float32, device=device)

    kernel = CliffordPhaseLockKernel(Psi_t, row_ptr_t, col_ind_t, K_tensors_t)

    # Check for Out-of-bounds or NaN exceptions
    try:
        for _ in range(20):
            out = kernel.step(dt=0.01)
        assert not torch.isnan(out).any(), "NaN detected in execution output tensor."
    except Exception as e:
        pytest.fail(f"Boundary condition execution failed with exception: {e}")
