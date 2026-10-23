"""
[Causal Geodesic Engine with Heterogeneous Memory Architecture]
Implementation of Causal & Variational Geodesic Convergence (최소 작용 측지선 수렴)
governed by Common Causal Structure (공통 인과 구조):
1. Lineage Trajectory (Lineage DAG with O(1) LCA tracking)
2. Potential Tension Field (Potential Tension Field τ & Chromatic Order/Entropy)
3. Lowest Common Ancestor (LCA) Backtracking & Singularity/Impasse Detection (τ -> ∞)
4. Meta-Invariant Boundary (I_meta) & Reframing/Elevation (I_meta -> I'_meta) with Failure Curvature Absorption

Also incorporates Heterogeneous Memory Architecture (Host System RAM + VRAM Ring Buffer Eviction):
- System RAM: Lineage DAG nodes, I_meta metadata, historical archive (Lineage IDs & hashes only after relaxation)
- PCIe Bus: Spike Dispatch Event Buffers
- VRAM: Active Working Set (Active CellNode Raw Tensors where τ > ε with 2.0GB VRAM Hard Limit)
- Tension-Guided Eviction & Cascading Relaxation:
  - Strict VRAM hard limit prevents memory overflow (simulating GTX 1060 3GB limits).
  - Tensors exceeding chunk thresholds undergo Cascading Relaxation (sliced/chunked processing).
  - Purges raw tensors completely upon relaxation (τ <= ε), retaining only topological Lineage DAG history.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import math
import torch

from synaptic_architecture.mechanism_tensor import CausalLineage, TopologicalInvariant
from synaptic_architecture.inverse_mechanism_engine import InverseMechanismEngine, BoundaryCondition, ObservedTrajectory, GeneratingMechanism


@dataclass
class ChromaticSignature:
    """
    Chromatic Interconnectedness signature (Flux, Order, Entropy).
    R: Flux (Energy/Velocity)
    G: Order (Coherence/Symmetry)
    B: Entropy (Disorder/Noise)
    """
    red_flux: float = 1.0
    green_order: float = 1.0
    blue_entropy: float = 0.0

    def compute_chromatic_tension(self) -> float:
        """Tension increases with high entropy B and low order G."""
        return max(0.0, self.blue_entropy - self.green_order + self.red_flux * 0.1)


@dataclass
class MetaInvariantBoundary:
    """
    Higher Meta-Invariant Boundary (I_meta).
    Defines the top-level topological fence preventing lower tensor state explosion.
    Absorbs and preserves failure curvature (tension divergence trajectories) upon reframing.
    """
    boundary_id: str
    target_symmetries: List[str] = field(default_factory=lambda: ["Flux_Conservation", "Phase_Continuity"])
    max_allowed_tension: float = 10.0
    dimensional_scale: float = 1.0
    active_reframing_count: int = 0
    failure_curvature_history: List[Dict[str, Any]] = field(default_factory=list)

    def reframe_boundary(
        self,
        tension_spike: float,
        failure_trajectory: Optional[List[float]] = None
    ) -> 'MetaInvariantBoundary':
        """
        Elevation / Reframing (I_meta -> I'_meta)
        Extends boundary scale and absorbs failure curvature trajectory (장력 발산 궤적)
        to guarantee Informational Continuity without keeping raw failure tensor state.
        """
        self.active_reframing_count += 1
        curvature_record = {
            "reframing_level": self.active_reframing_count,
            "tension_spike": tension_spike,
            "failure_trajectory": list(failure_trajectory) if failure_trajectory else [tension_spike]
        }
        self.failure_curvature_history.append(curvature_record)

        self.dimensional_scale *= (1.0 + math.log1p(tension_spike))
        self.max_allowed_tension *= 1.5
        self.target_symmetries.append(f"Elevated_Symmetry_L{self.active_reframing_count}")
        return self


@dataclass
class LineageDAGNode:
    """
    System RAM Light-weight Node for Lineage DAG.
    Separates the lineage/metadata view from VRAM heavy raw tensor storage.
    Once relaxed, raw_tensor_host is set to None (purged) to ensure zero residual tensor footprint.
    """
    node_id: str
    lineage: CausalLineage
    is_vram_resident: bool = False
    raw_tensor_host: Optional[torch.Tensor] = None
    relaxed_status: str = "Active"  # "Active", "Relaxed_Purged"


class HeterogeneousMemoryManager:
    """
    Manages System RAM (Host) vs VRAM (GPU Ring Buffer) placement
    using Tension-Guided Eviction and 2.0GB VRAM Hard Limit.
    """
    def __init__(
        self,
        vram_ring_buffer_capacity: int = 4,
        max_vram_bytes: int = 2 * 1024 * 1024 * 1024  # 2.0 GB Hard Limit
    ):
        self.system_ram_dag: Dict[str, LineageDAGNode] = {}
        self.vram_active_buffer: Dict[str, torch.Tensor] = {}
        self.vram_capacity = vram_ring_buffer_capacity
        self.max_vram_bytes = max_vram_bytes
        self.current_vram_usage_bytes = 0

    def register_node(self, node_id: str, raw_tensor: torch.Tensor, lineage: CausalLineage) -> LineageDAGNode:
        """Stores lineage view and initial tensor in System RAM."""
        dag_node = LineageDAGNode(
            node_id=node_id,
            lineage=lineage,
            is_vram_resident=False,
            raw_tensor_host=raw_tensor.detach().cpu().clone(),
            relaxed_status="Active"
        )
        self.system_ram_dag[node_id] = dag_node
        return dag_node

    def dispatch_to_vram(self, node_id: str, chunk_size_limit: int = 100000) -> torch.Tensor:
        """
        Slides-in active cell node tensor to VRAM ring buffer for computation.
        Enforces 2.0GB VRAM hard limit and triggers cascading allocation/eviction.
        """
        dag_node = self.system_ram_dag[node_id]
        if dag_node.is_vram_resident and node_id in self.vram_active_buffer:
            return self.vram_active_buffer[node_id]

        if dag_node.raw_tensor_host is None:
            raise ValueError(f"Cannot dispatch node {node_id}: raw tensor has been purged from memory.")

        tensor_bytes = dag_node.raw_tensor_host.element_size() * dag_node.raw_tensor_host.nelement()

        # Check VRAM hard limit
        while (len(self.vram_active_buffer) >= self.vram_capacity or
               self.current_vram_usage_bytes + tensor_bytes > self.max_vram_bytes) and self.vram_active_buffer:
            # Evict oldest active node
            evict_id = next(iter(self.vram_active_buffer))
            self.purge_from_vram_and_ram(evict_id)

        tensor_vram = dag_node.raw_tensor_host.clone()
        self.vram_active_buffer[node_id] = tensor_vram
        self.current_vram_usage_bytes += tensor_bytes
        dag_node.is_vram_resident = True
        return tensor_vram

    def purge_from_vram_and_ram(self, node_id: str):
        """
        Complete Eviction (Purge):
        Frees tensor from VRAM AND sets raw_tensor_host to None in System RAM.
        Only Lineage DAG topology and hashes remain.
        """
        if node_id in self.vram_active_buffer:
            tensor_vram = self.vram_active_buffer.pop(node_id)
            tensor_bytes = tensor_vram.element_size() * tensor_vram.nelement()
            self.current_vram_usage_bytes = max(0, self.current_vram_usage_bytes - tensor_bytes)
            del tensor_vram

        if node_id in self.system_ram_dag:
            dag_node = self.system_ram_dag[node_id]
            dag_node.raw_tensor_host = None
            dag_node.is_vram_resident = False
            dag_node.relaxed_status = "Relaxed_Purged"


class CausalGeodesicEngine:
    """
    [Causal Geodesic Engine]
    Computes the unique, frictionless minimal action path (Geodesic)
    connecting S_start to S_target on the multi-dimensional spacetime potential field.

    Integrates:
    - O(1) LCA tracking on Lineage DAG
    - Potential Tension Field τ & Chromatic Order/Entropy
    - Impasse/Singularity detection and LCA backtracking
    - Meta-Invariant Boundary I_meta Reframing (I_meta -> I'_meta) with Failure Curvature Absorption
    - Inverse Mechanism Extraction (InverseMechanismEngine)
    - Heterogeneous Memory Manager with 2.0GB VRAM Limit and Cascading Relaxation
    - Complete Raw Tensor Purging upon Equilibrium Convergence
    """

    def __init__(
        self,
        meta_boundary: Optional[MetaInvariantBoundary] = None,
        singularity_threshold: float = 15.0,
        epsilon_equilibrium: float = 1e-3,
        max_vram_bytes: int = 2 * 1024 * 1024 * 1024
    ):
        self.meta_boundary = meta_boundary or MetaInvariantBoundary("Default_I_meta")
        self.singularity_threshold = singularity_threshold
        self.epsilon_equilibrium = epsilon_equilibrium
        self.memory_mgr = HeterogeneousMemoryManager(max_vram_bytes=max_vram_bytes)
        self.inverse_engine = InverseMechanismEngine()
        self.geodesic_history: List[Dict[str, Any]] = []

    def compute_chromatic_entropy_tension(self, tensor: torch.Tensor) -> Tuple[float, ChromaticSignature]:
        """
        Maps tensor variance and flux to Chromatic Signature (RGB).
        Calculates Chromatic Entropy / Order metric for potential tension τ.
        """
        if tensor.numel() == 0:
            return 0.0, ChromaticSignature()

        mean_val = float(tensor.mean().item())
        var_val = float(tensor.var().item()) if tensor.numel() > 1 else 0.0

        red_flux = abs(mean_val)
        green_order = 1.0 / (1.0 + var_val)
        blue_entropy = math.log1p(var_val)

        chromatic = ChromaticSignature(
            red_flux=red_flux,
            green_order=green_order,
            blue_entropy=blue_entropy
        )

        tau_chromatic = blue_entropy * 2.0 + (1.0 - green_order) * 3.0
        return tau_chromatic, chromatic

    def compute_potential_tension(
        self,
        current_tensor: torch.Tensor,
        target_invariant: TopologicalInvariant
    ) -> float:
        """
        Calculates total potential tension field τ:
        τ = Topological Mismatch Error + Chromatic Entropy Tension
        """
        topological_err, _ = target_invariant.compute_error(current_tensor)
        tau_topo = float(topological_err.item())
        tau_chromatic, _ = self.compute_chromatic_entropy_tension(current_tensor)

        total_tau = tau_topo + tau_chromatic
        return total_tau

    def cascading_relaxation_step(
        self,
        vram_tensor: torch.Tensor,
        target_val: float,
        learning_rate: float = 0.5,
        chunk_size: int = 50000
    ) -> torch.Tensor:
        """
        Cascading Relaxation (폭포수 이완):
        Processes tensor in chunks/slices to avoid memory spikes under strict VRAM bounds.
        """
        num_elements = vram_tensor.numel()
        if num_elements <= chunk_size:
            vram_tensor += (target_val - vram_tensor) * learning_rate
        else:
            flat_tensor = vram_tensor.view(-1)
            for start_idx in range(0, num_elements, chunk_size):
                end_idx = min(start_idx + chunk_size, num_elements)
                chunk = flat_tensor[start_idx:end_idx]
                chunk += (target_val - chunk) * learning_rate
        return vram_tensor

    def execute_geodesic_convergence(
        self,
        node_start_id: str,
        tensor_start: torch.Tensor,
        lineage_start: CausalLineage,
        target_invariant: TopologicalInvariant,
        boundary_cond: BoundaryCondition,
        competing_impasse_lineage: Optional[CausalLineage] = None,
        max_steps: int = 15
    ) -> Dict[str, Any]:
        """
        Full 4-Stage Causal Geodesic Process with Type State Transition:
        1. Initial Convergence: τ reduction towards S_target.
        2. Impasse/Singularity Trigger: τ -> ∞ when encountering topological conflict.
        3. LCA Backtracking & Reframing: Tracing split node via Lineage DAG LCA -> I_meta -> I'_meta (absorbs failure curvature).
        4. Geodesic Path Resolution: Cascading relaxation to τ <= ε, inverse mechanism extraction, and physical tensor purge.
        """
        # Step 0: Register in System RAM & Dispatch to VRAM
        self.memory_mgr.register_node(node_start_id, tensor_start, lineage_start)
        vram_tensor = self.memory_mgr.dispatch_to_vram(node_start_id)

        current_lineage = lineage_start
        geodesic_trajectory: List[str] = [f"Start({node_start_id})"]
        tau_history: List[float] = []

        step = 0
        singularity_detected = False
        reframed = False
        lca_branch_id = None

        while step < max_steps:
            tau = self.compute_potential_tension(vram_tensor, target_invariant)
            tau_history.append(tau)

            # Check for Equilibrium
            if tau <= self.epsilon_equilibrium:
                geodesic_trajectory.append(f"EquilibriumReached(step={step}, tau={tau:.4f})")
                break

            # Stage 2: Singularity / Impasse Trigger Simulation if competing impasse provided
            if step == 2 and competing_impasse_lineage is not None and not singularity_detected:
                # Inject contradictory constraint causing τ -> ∞
                vram_tensor = vram_tensor * 10.0 + 25.0
                tau = self.compute_potential_tension(vram_tensor, target_invariant)
                tau_history.append(tau)
                singularity_detected = True
                geodesic_trajectory.append(f"SingularityTriggered(tau={tau:.2f} > {self.singularity_threshold})")

            # Handle Singularity
            if tau > self.singularity_threshold and not reframed:
                # Stage 3: LCA Backtracking & Reframing
                if competing_impasse_lineage:
                    lca_branch_id, split_depth = current_lineage.find_lowest_common_ancestor(competing_impasse_lineage)
                    geodesic_trajectory.append(
                        f"LCABacktrack(LCA_ID={lca_branch_id}, split_depth={split_depth})"
                    )

                # Reframing: I_meta -> I'_meta with Failure Curvature Absorption
                self.meta_boundary.reframe_boundary(tau, failure_trajectory=tau_history)
                reframed = True
                geodesic_trajectory.append(
                    f"Reframing(I_meta->I'_meta, scale={self.meta_boundary.dimensional_scale:.2f}, failure_curvature_absorbed=True)"
                )

                # Relax tension energy physically under reframed scale
                vram_tensor = vram_tensor / self.meta_boundary.dimensional_scale
                continue

            # Stage 1 / Stage 4: Cascading Relaxation step towards geodesic
            vram_tensor = self.cascading_relaxation_step(
                vram_tensor=vram_tensor,
                target_val=target_invariant.target_value,
                learning_rate=0.5
            )

            current_lineage.transformation_history.append(f"GeodesicStep_{step}(tau={tau:.4f})")
            geodesic_trajectory.append(f"Step_{step}(tau={tau:.4f})")

            step += 1

        # Final Tension
        final_tau = self.compute_potential_tension(vram_tensor, target_invariant)

        # Offload and Complete Purge of Raw Tensor (Zero Residual Tensor Footprint)
        self.memory_mgr.purge_from_vram_and_ram(node_start_id)

        # Inverse Mechanism Extraction (Generating Equation Θ)
        obs_traj = ObservedTrajectory(
            trajectory_id=f"Geodesic_Obs_{node_start_id}",
            boundary_id=boundary_cond.condition_id,
            states=[[t] for t in tau_history],
            intent_tag="Causal_Geodesic_Convergence"
        )
        extracted_mechanism = self.inverse_engine.extract_generating_mechanism(
            mechanism_id=f"Mech_{node_start_id}",
            observations=[obs_traj],
            boundaries={boundary_cond.condition_id: boundary_cond}
        )

        result = {
            "node_id": node_start_id,
            "geodesic_trajectory": geodesic_trajectory,
            "tau_history": tau_history,
            "initial_tau": tau_history[0] if tau_history else 0.0,
            "final_tau": final_tau,
            "singularity_detected": singularity_detected,
            "reframed": reframed,
            "lca_branch_id": lca_branch_id,
            "meta_boundary_scale": self.meta_boundary.dimensional_scale,
            "failure_curvature_count": len(self.meta_boundary.failure_curvature_history),
            "extracted_mechanism": extracted_mechanism,
            "memory_status": self.memory_mgr.system_ram_dag[node_start_id].relaxed_status,
            "raw_tensor_purged": self.memory_mgr.system_ram_dag[node_start_id].raw_tensor_host is None
        }
        self.geodesic_history.append(result)
        return result
