"""
Comprehensive Demonstration Script:
Spatiotemporal Transduction, High-Dimensional Spinor-Riemannian Dynamics,
Static Rotor Aux-Memory Swapping (SRAM-R), and Subconscious SSD Atlas Memory Consolidation.
"""

import math
import torch
import torch.nn as nn

from core.lens.spatiotemporal_transduction_lens import (
    StructuralReceptor,
    TopologicalMapper,
    HighDimSpinorRiemannianLayer,
)
from core.memory.subconscious_rotor_atlas import (
    StaticRotorAuxMemoryUnit,
    SubconsciousSSDAtlas,
)
from core.lens.clifford_spinor_recall_engine import (
    CliffordSpinorRecallEngine,
)


def run_comprehensive_demo():
    torch.manual_seed(42)
    print("=" * 70)
    print("      ELYSIUS SPINOR-RIEMANNIAN SUBCONSCIOUS MEMORY DEMO")
    print("=" * 70)

    # Dimensionality settings
    batch_size = 4
    ext_dim = 64      # External raw physical stimulus space
    bound_dim = 32    # Transduced boundary language
    state_dim = 16    # Internal system state space (N=16)
    num_anchors = 8   # Local memory anchors

    # -------------------------------------------------------------
    # 1. Structural Receptor Transduction & Topological Mapping
    # -------------------------------------------------------------
    print("\n[Step 1] Receptor Boundary Transduction & Tangent Mapping")
    receptor = StructuralReceptor(ext_dim, bound_dim, threshold=0.05)
    mapper = TopologicalMapper(bound_dim, state_dim)

    Xi_ext = torch.randn(batch_size, ext_dim) * 2.0
    S_bound = receptor(Xi_ext)
    v_shift = mapper(S_bound)

    print(f" -> Raw External Stimulus Xi_ext  : {Xi_ext.shape}")
    print(f" -> Transduced Signal S_bound     : {S_bound.shape}")
    print(f" -> Mapped Tangent Shift v_shift   : {v_shift.shape}")

    # -------------------------------------------------------------
    # 2. High-Dimensional Spinor-Riemannian Autograd Layer Step
    # -------------------------------------------------------------
    print("\n[Step 2] High-Dimensional Spinor-Riemannian Autograd Forward & Backward")
    layer = HighDimSpinorRiemannianLayer(state_dim=state_dim, num_anchors=num_anchors, dt=0.05)

    x_init = torch.randn(batch_size, state_dim, requires_grad=True)
    Q_init = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1).requires_grad_(True)
    target_attractor = torch.randn(batch_size, state_dim)

    x_next, Q_next = layer(x_init, Q_init, target_attractor)

    print(f" -> Updated Position State x_next : {x_next.shape}")
    print(f" -> Updated Spinor Frame Q_next    : {Q_next.shape}")

    # Autograd VJP Verification
    loss = (x_next - target_attractor).pow(2).sum() + Q_next.pow(2).sum()
    loss.backward()

    print(f" -> Loss Value                     : {loss.item():.6f}")
    print(f" -> x_init Gradient Norm           : {x_init.grad.norm().item():.6f}")
    print(f" -> Q_init Gradient Norm           : {Q_init.grad.norm().item():.6f}")
    print(f" -> Layer Anchors Gradient Norm    : {layer.anchors.grad.norm().item():.6f}")

    # -------------------------------------------------------------
    # 3. Static Rotor Aux-Memory Unit (SRAM-R) Phase-Preserving Swapping
    # -------------------------------------------------------------
    print("\n[Step 3] Static Rotor (SRAM-R) Cache-RAM Swap Phase Preservation")
    sram_r = StaticRotorAuxMemoryUnit(state_dim=state_dim)

    tag_id = "cache_block_001"
    # Eviction: Freeze Phase (extract Lie algebra bivector Delta Q -> Omega)
    Omega_frozen = sram_r.freeze_phase(tag_id, Q_next.detach())
    print(f" -> [Freeze Phase] Evicted tag '{tag_id}', extracted Omega norm: {Omega_frozen.norm().item():.6f}")

    # Background idle drift simulation
    sram_r.update_background_drift(dt=0.02)

    # Fetch: Sync Phase (re-apply preserved phase Delta R -> Q_synced)
    Q_global = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1)
    Q_synced = sram_r.sync_phase(tag_id, Q_global, dt_step=0.05)
    print(f" -> [Sync Phase] Fetched tag '{tag_id}', restored Q_synced shape: {Q_synced.shape}")

    # -------------------------------------------------------------
    # 4. Subconscious SSD Atlas Event Consolidation & Natural Decay
    # -------------------------------------------------------------
    print("\n[Step 4] Subconscious Memory Atlas Consolidation & Decay")
    ssd_atlas = SubconsciousSSDAtlas(state_dim=state_dim)

    # Event consolidation
    x_evt = x_next[0].detach()
    v_evt = v_shift[0].detach()
    Omega_evt = Omega_frozen[0].detach()

    chart = ssd_atlas.consolidate_event(x_evt, v_evt, Omega_evt, plasticity_rate=0.1)
    print(f" -> Consolidated Event to Chart ID : {chart.chart_id}")
    print(f" -> Chart Lock Count               : {chart.lock_count}")
    print(f" -> Chart Metric g_mem Norm        : {torch.norm(chart.g_mem).item():.6f}")

    # Trigger repeated consolidation to test Phase-Locked Hardening
    for _ in range(5):
        ssd_atlas.consolidate_event(x_evt, v_evt, Omega_evt, plasticity_rate=0.1)

    print(f" -> Hardened Chart Status (gamma->eps): is_hardened={chart.is_hardened}, gamma={chart.gamma:.6f}")

    # Background natural decay & diffusion
    ssd_atlas.step_subconscious_decay(dt=0.1)
    print(f" -> Post Decay g_mem Norm         : {torch.norm(chart.g_mem).item():.6f}")

    # -------------------------------------------------------------
    # 5. Geodesic Recall & Energy Relaxation
    # -------------------------------------------------------------
    print("\n[Step 5] Clifford Spinor Geodesic Recall Dynamics")
    recall_engine = CliffordSpinorRecallEngine(state_dim=state_dim)

    x_start = x_evt.unsqueeze(0) + 0.5 * torch.randn(1, state_dim)
    Q_start = torch.eye(state_dim).unsqueeze(0)
    target_mem = x_evt.unsqueeze(0)

    x_traj, Q_traj, energy_hist = recall_engine.recall_dynamics(
        x_start, Q_start, target_mem, chart.g_mem, Omega_evt.unsqueeze(0), steps=20, dt=0.05
    )

    print(f" -> Recall Trajectory Steps        : {x_traj.shape[0]}")
    print(f" -> Initial Riemannian Distance    : {energy_hist[0]:.6f}")
    print(f" -> Final Geodesic Distance        : {energy_hist[-1]:.6f}")
    print(" -> Energy Relaxation Curve        :", [round(e, 4) for e in energy_hist[:5]], "...")

    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_demo()
