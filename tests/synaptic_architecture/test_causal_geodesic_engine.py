"""
Unit tests for synaptic_architecture/causal_geodesic_engine.py
Verifies:
1. Heterogeneous Memory Manager (System RAM & VRAM Ring Buffer Eviction with 2.0GB VRAM Limit).
2. Potential Tension Field τ & Chromatic Order/Entropy calculations.
3. 4-Stage Causal Geodesic Convergence (Initial convergence, Singularity trigger, LCA backtracking, Reframing I_meta -> I'_meta with failure curvature absorption, and Final resolution).
4. Cascading Relaxation & Complete Raw Tensor Purge (Zero residual tensor footprint).
5. Inverse Mechanism Extraction (Θ) upon geodesic convergence.
"""

import pytest
import torch
import math

from synaptic_architecture.mechanism_tensor import CausalLineage, TopologicalInvariant
from synaptic_architecture.inverse_mechanism_engine import BoundaryCondition
from synaptic_architecture.causal_geodesic_engine import (
    CausalGeodesicEngine,
    MetaInvariantBoundary,
    HeterogeneousMemoryManager,
    ChromaticSignature
)


def test_heterogeneous_memory_manager():
    mgr = HeterogeneousMemoryManager(vram_ring_buffer_capacity=2, max_vram_bytes=1000)
    lineage1 = CausalLineage(node_id="node1")
    lineage2 = CausalLineage(node_id="node2")
    lineage3 = CausalLineage(node_id="node3")

    tensor1 = torch.tensor([1.0, 2.0])
    tensor2 = torch.tensor([3.0, 4.0])
    tensor3 = torch.tensor([5.0, 6.0])

    mgr.register_node("node1", tensor1, lineage1)
    mgr.register_node("node2", tensor2, lineage2)

    assert "node1" in mgr.system_ram_dag
    assert not mgr.system_ram_dag["node1"].is_vram_resident

    # Dispatch node1 and node2 to VRAM
    v_t1 = mgr.dispatch_to_vram("node1")
    v_t2 = mgr.dispatch_to_vram("node2")

    assert len(mgr.vram_active_buffer) == 2
    assert mgr.system_ram_dag["node1"].is_vram_resident

    # Dispatch node3 -> should trigger eviction & complete purge of oldest node (node1)
    mgr.register_node("node3", tensor3, lineage3)
    mgr.dispatch_to_vram("node3")

    assert len(mgr.vram_active_buffer) == 2
    assert "node1" not in mgr.vram_active_buffer
    assert mgr.system_ram_dag["node1"].relaxed_status == "Relaxed_Purged"
    assert mgr.system_ram_dag["node1"].raw_tensor_host is None


def test_chromatic_entropy_tension():
    engine = CausalGeodesicEngine()
    tensor_coherent = torch.tensor([1.0, 1.0, 1.0])
    tensor_noisy = torch.tensor([0.0, 50.0, -50.0])

    tau_coherent, chrom_c = engine.compute_chromatic_entropy_tension(tensor_coherent)
    tau_noisy, chrom_n = engine.compute_chromatic_entropy_tension(tensor_noisy)

    assert tau_noisy > tau_coherent
    assert chrom_n.blue_entropy > chrom_c.blue_entropy


def test_causal_geodesic_convergence_full_flow():
    engine = CausalGeodesicEngine(singularity_threshold=10.0, epsilon_equilibrium=0.1)

    lineage_start = CausalLineage(
        node_id="Start_Point",
        parent_ids=["Root_LCA"],
        transformation_history=["Origin", "Phase1"]
    )
    lineage_competing = CausalLineage(
        node_id="Competing_Point",
        parent_ids=["Root_LCA"],
        transformation_history=["Origin", "Divergent_Phase"]
    )

    tensor_start = torch.tensor([5.0, 8.0, 10.0])
    target_invariant = TopologicalInvariant(name="Equilibrium_Target", target_value=1.0)
    boundary_cond = BoundaryCondition(condition_id="Env_Standard", friction=0.5, gravity=9.81)

    result = engine.execute_geodesic_convergence(
        node_start_id="Start_Point",
        tensor_start=tensor_start,
        lineage_start=lineage_start,
        target_invariant=target_invariant,
        boundary_cond=boundary_cond,
        competing_impasse_lineage=lineage_competing,
        max_steps=20
    )

    assert result["singularity_detected"] is True
    assert result["reframed"] is True
    assert result["lca_branch_id"] == "Root_LCA"
    assert result["final_tau"] < result["initial_tau"] or result["final_tau"] <= 0.5
    assert result["memory_status"] == "Relaxed_Purged"
    assert result["raw_tensor_purged"] is True
    assert result["failure_curvature_count"] > 0
    assert result["extracted_mechanism"] is not None


def test_cascading_relaxation_and_failure_curvature():
    boundary = MetaInvariantBoundary(boundary_id="Curvature_Test_Boundary")
    engine = CausalGeodesicEngine(meta_boundary=boundary)

    # Large tensor to test cascading relaxation
    large_tensor = torch.randn(100000)
    target_invariant = TopologicalInvariant(name="Zero_Target", target_value=0.0)
    lineage = CausalLineage(node_id="Large_Node")
    boundary_cond = BoundaryCondition(condition_id="Env_Large")

    result = engine.execute_geodesic_convergence(
        node_start_id="Large_Node",
        tensor_start=large_tensor,
        lineage_start=lineage,
        target_invariant=target_invariant,
        boundary_cond=boundary_cond,
        max_steps=5
    )

    assert result["memory_status"] == "Relaxed_Purged"
    assert result["raw_tensor_purged"] is True
