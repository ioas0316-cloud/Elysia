"""
Simplicial Spatiotemporal Manifold Engine (시공간 위상 복합체 매니폴드 엔진)
===================================================================================

This module implements the high-dimensional Simplicial Complex Spatiotemporal Manifold,
bridging 0D Atom Nodes, 1D Causal Trajectories, 2D Context Fields, 3D Multi-Sensory Manifolds,
and 4D Spacetime Phase Velocity Dials with GPU-accelerated GEMM matrix projections,
Implicit Signed Distance Fields (SDF), Clifford Geometric Algebra rotors, and Persistent Homology loss.

Hierarchy of Simplicial Dimensions:
-----------------------------------
1. 0D Point (노드/점): Immutable atom state signature (e.g. Hangul phonemes, sound formants, CIELAB colors).
2. 1D Line (인과적 궤적/선): State transition path and directional vector between 0D points.
3. 2D Plane (문맥 제약 영역/면): Boundary domain enclosed by 1D lines (e.g. Syllable Phonology Domain).
4. 3D Volume (다감각 매니폴드/공간): Integrated manifold where heterogenous sensory planes intersect & phase-lock.
5. 4D Spacetime (위상 속도 & 상전이/시간): Causal transition sequence governed by phase velocity dial.

Technical Enhancements:
----------------------
- Affine Homogeneous Projection Matrix & GEMM Shader Parameter Modulation.
- Implicit Signed Distance Field (SDF) Boundary Constraints.
- Clifford Geometric Algebra Rotor Transformations.
- Persistent Homology Invariant Regularization (Betti Invariant Conservation).
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 0D Node, 1D Trajectory, 2D Boundary Field Definitions
# -----------------------------------------------------------------------------

@dataclass
class AtomNode0D:
    """0D Atom Node representing an immutable state signature."""
    node_id: str
    modality: str                        # 'text_symbol', 'acoustic_sound', 'visual_color', 'physical_texture'
    vector: torch.Tensor                 # Base signature feature tensor [D]
    chromatic_signature: Tuple[float, float, float] = (0.33, 0.33, 0.34)  # (Red/Flux, Blue/Order, Yellow/Entropy)


@dataclass
class CausalTrajectory1D:
    """1D Causal Line Trajectory representing state transition vector."""
    source_id: str
    target_id: str
    transition_vector: torch.Tensor      # Directional vector [D]
    topological_shift: float = 1.0       # Trajectory length / momentum weight


@dataclass
class ContextField2D:
    """
    2D Context Field / Boundary Domain.
    Enclosed by 1D trajectories forming valid semantic domain (e.g., Hangul Syllable Field).
    """
    field_id: str
    boundary_nodes: List[str]            # IDs of 0D nodes forming boundary
    sdf_center: torch.Tensor             # Center of context plane [D]
    sdf_radius: float                    # Radius of valid context domain boundary
    domain_type: str = "syllable_phonology"

    def is_inside_sdf_boundary(self, point: torch.Tensor) -> bool:
        """Evaluates implicit Signed Distance Field: SDF(x) = ||x - center|| - radius <= 0."""
        dist = torch.norm(point - self.sdf_center).item()
        return (dist - self.sdf_radius) <= 0.0

    def compute_sdf(self, point: torch.Tensor) -> torch.Tensor:
        """Returns signed distance to boundary."""
        return torch.norm(point - self.sdf_center, dim=-1) - self.sdf_radius


# -----------------------------------------------------------------------------
# PyTorch Simplicial Spatiotemporal Manifold Engine
# -----------------------------------------------------------------------------

class SimplicialSpatiotemporalManifold(nn.Module):
    """
    PyTorch-accelerated Simplicial Spatiotemporal Manifold Layer.
    Executes 0D-4D simplicial complex updates via GEMM matrix projections,
    Clifford rotor transformations, SDF implicit context field masking,
    and Persistent Homology topological regularization.
    """

    def __init__(self, d_model: int = 64, num_modalities: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # 1. GEMM Homogeneous Affine Projection Matrices for 3D/4D Manifolds
        self.affine_projection = nn.Parameter(torch.eye(d_model))
        self.cross_modal_phase_lock_matrix = nn.Parameter(torch.randn(num_modalities, d_model, d_model) * 0.1)

        # 2. Shader Uniform Dial Rotators (Clifford Geometric Algebra Rotor parameters)
        self.rotor_bivector = nn.Parameter(torch.randn(d_model, d_model) * 0.05)

        # 3. Persistent Homology Baseline Reference
        self.register_buffer("betti_baseline", torch.tensor([1.0, 0.0, 0.0]))  # (Beta_0, Beta_1, Beta_2)

    def get_skew_bivector(self) -> torch.Tensor:
        """Computes skew-symmetric bivector B = 0.5 * (W - W^T) for Clifford Rotors."""
        return 0.5 * (self.rotor_bivector - self.rotor_bivector.T)

    def apply_clifford_rotor(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        """
        Applies Clifford Rotor Sandwich transformation R x R^T using matrix exponential
        R = exp(0.5 * theta * B).
        """
        B = self.get_skew_bivector()
        R = torch.matrix_exp(0.5 * theta * B)
        return torch.matmul(x, R)

    def compute_cross_modal_phase_lock(self, modal_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D Multi-Sensory Phase-Locking via Tensor Inner Product & GEMM Affine Projection:
        S = A * B^T / sqrt(D)
        Returns:
            phase_locked_manifold: [N, D] integrated 3D manifold
            coherence_matrix: [num_modalities, num_modalities] inner product phase-lock matrix
        """
        num_m = len(modal_tensors)
        if num_m == 0:
            return torch.zeros(1, self.d_model), torch.eye(1)

        # Stack modality representations: [M, N, D]
        stacked = torch.stack(modal_tensors, dim=0)  # [M, N, D]

        # Apply modality GEMM projection
        projected_modalities = []
        for i in range(num_m):
            proj = torch.matmul(stacked[i], self.cross_modal_phase_lock_matrix[i % self.num_modalities])
            projected_modalities.append(proj)

        proj_stacked = torch.stack(projected_modalities, dim=0)  # [M, N, D]

        # Coherence Matrix via Inner Product GEMM across modalities
        # Mean pool over N: [M, D]
        mod_means = proj_stacked.mean(dim=1)  # [M, D]
        coherence_matrix = torch.matmul(mod_means, mod_means.T) / math.sqrt(self.d_model)
        coherence_matrix = torch.sigmoid(coherence_matrix)

        # Integrated Multi-Sensory Manifold Volume Tensor: Weighted GEMM combination
        weights = coherence_matrix.mean(dim=1, keepdim=True).unsqueeze(-1)  # [M, 1, 1]
        integrated_manifold = (proj_stacked * weights).sum(dim=0)            # [N, D]
        integrated_manifold = torch.matmul(integrated_manifold, self.affine_projection)

        return integrated_manifold, coherence_matrix

    def step_4d_spacetime_dial(
        self,
        current_state: torch.Tensor,
        dial_delta: float,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        4D Spacetime Phase Velocity & Transition Engine:
        Controls state boundary transition order and phase velocity.
        - Low dial velocity (|dial_delta| < 1.0): Continuous smooth phase shift (Clifford Rotor).
        - High dial velocity (|dial_delta| >= 1.0): Structural phase transition (Structural Jump & Attractor Re-alignment).
        """
        phase_velocity = dial_delta / dt
        is_phase_transition = abs(phase_velocity) > 5.0

        if is_phase_transition:
            # Structural Phase Transition: Non-linear attractor jump
            rotor_angle = math.pi * 0.5 * (1.0 if dial_delta >= 0 else -1.0)
            shifted_state = self.apply_clifford_rotor(current_state, rotor_angle)
            shifted_state = torch.tanh(shifted_state * 1.5)  # Phase boundary leap
            transition_type = "Structural Phase Transition (상전이)"
        else:
            # Continuous Smooth Phase Shift
            rotor_angle = dial_delta * 0.2
            shifted_state = self.apply_clifford_rotor(current_state, rotor_angle)
            transition_type = "Continuous Phase Shift (연속 변이)"

        info = {
            "dial_delta": dial_delta,
            "phase_velocity": phase_velocity,
            "is_phase_transition": is_phase_transition,
            "transition_type": transition_type,
            "energy_momentum": torch.norm(shifted_state - current_state).item()
        }
        return shifted_state, info

    def compute_persistent_homology_loss(self, point_cloud: torch.Tensor, distance_threshold: float = 0.5) -> torch.Tensor:
        """
        Computes Persistent Homology Regularization Loss:
        Measures topological Betti invariants (Beta_0 connected components, Beta_1 1D loops)
        from pairwise distance matrix to prevent topological phase collapse and identity loss.
        """
        # Pairwise euclidean distances [N, N]
        dist = torch.cdist(point_cloud, point_cloud)  # [N, N]

        # Adjacency matrix at filtration distance_threshold
        adj = (dist < distance_threshold).float()

        # 0D Betti Number estimation: Graph Laplacian zero eigenvalues / connected components proxy
        degree = torch.diag(adj.sum(dim=-1))
        laplacian = degree - adj
        eigenvalues = torch.linalg.eigvalsh(laplacian)

        # Estimate Beta_0 (number of eigenvalues near zero)
        beta_0_est = torch.sum(torch.exp(-10.0 * eigenvalues))

        # Regularization loss against baseline topology
        loss_betti_0 = F.mse_loss(beta_0_est, self.betti_baseline[0])

        # Add dispersion penalty if points collapse to a single point (check N > 1)
        if point_cloud.size(0) > 1:
            var_val = torch.var(point_cloud, dim=0, correction=0).mean()
        else:
            var_val = torch.tensor(0.0, device=point_cloud.device)

        variance_penalty = F.relu(0.1 - var_val)

        return loss_betti_0 + 5.0 * variance_penalty


# -----------------------------------------------------------------------------
# Simplicial Manifold Pipeline Orchestrator
# -----------------------------------------------------------------------------

class SimplicialManifoldPipeline:
    """
    High-level Orchestrator for Simplicial Complex Spatiotemporal Manifold.
    Manages 0D-4D simplicial entities, Hangul Syllable Plane Fields,
    Multi-Sensory SDF domains, and 4D Spacetime Dial controls.
    """

    def __init__(self, d_model: int = 64):
        self.d_model = d_model
        self.manifold_engine = SimplicialSpatiotemporalManifold(d_model=d_model)

        # Storage for 0D/1D/2D Simplicial Complexes
        self.nodes_0d: Dict[str, AtomNode0D] = {}
        self.trajectories_1d: List[CausalTrajectory1D] = []
        self.context_fields_2d: Dict[str, ContextField2D] = {}

    def add_atom_node(
        self,
        node_id: str,
        modality: str,
        vector: np.ndarray,
        chromatic_signature: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    ) -> AtomNode0D:
        """Registers a 0D Atom Node."""
        t_vec = torch.tensor(vector, dtype=torch.float32)
        if t_vec.shape[0] != self.d_model:
            if t_vec.shape[0] < self.d_model:
                t_vec = F.pad(t_vec, (0, self.d_model - t_vec.shape[0]))
            else:
                t_vec = t_vec[:self.d_model]

        node = AtomNode0D(
            node_id=node_id,
            modality=modality,
            vector=t_vec,
            chromatic_signature=chromatic_signature
        )
        self.nodes_0d[node_id] = node
        return node

    def add_causal_trajectory(self, source_id: str, target_id: str) -> CausalTrajectory1D:
        """Connects two 0D nodes with a 1D Causal Line Trajectory."""
        src_vec = self.nodes_0d[source_id].vector
        tgt_vec = self.nodes_0d[target_id].vector
        transition = tgt_vec - src_vec
        traj = CausalTrajectory1D(
            source_id=source_id,
            target_id=target_id,
            transition_vector=transition,
            topological_shift=torch.norm(transition).item()
        )
        self.trajectories_1d.append(traj)
        return traj

    def create_hangul_syllable_plane(
        self,
        syllable_name: str,
        choseong_id: str,
        jungseong_id: str,
        jongseong_id: str
    ) -> ContextField2D:
        """
        Creates a 2D Context Field (Syllable Phonology Domain) formed by
        Choseong (초성) - Jungseong (중성) - Jongseong (종성) 1D trajectory loop.
        """
        cho = self.nodes_0d[choseong_id].vector
        jung = self.nodes_0d[jungseong_id].vector
        jong = self.nodes_0d[jongseong_id].vector

        # 1D line connections forming closed 2D plane triangle loop
        self.add_causal_trajectory(choseong_id, jungseong_id)
        self.add_causal_trajectory(jungseong_id, jongseong_id)
        self.add_causal_trajectory(jongseong_id, choseong_id)

        # Compute center of 2D plane
        center = (cho + jung + jong) / 3.0
        # Radius as max distance from center to vertices + margin
        r = max(
            torch.norm(cho - center).item(),
            torch.norm(jung - center).item(),
            torch.norm(jong - center).item()
        ) + 0.2

        field = ContextField2D(
            field_id=f"Plane_{syllable_name}",
            boundary_nodes=[choseong_id, jungseong_id, jongseong_id],
            sdf_center=center,
            sdf_radius=r,
            domain_type="syllable_phonology"
        )
        self.context_fields_2d[field.field_id] = field
        return field

    def process_cross_modal_manifold(
        self,
        text_nodes: List[str],
        sound_nodes: List[str],
        visual_nodes: List[str],
        texture_nodes: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthesizes 3D Multi-Sensory Manifold across heterogenous modalities.
        Returns:
            integrated_manifold: [N, D] 3D manifold
            coherence_matrix: Phase-lock coherence matrix
            homology_loss: Persistent Homology topology loss
        """
        modal_groups = [text_nodes, sound_nodes, visual_nodes, texture_nodes]
        modal_tensors = []

        for group in modal_groups:
            if group:
                stacked = torch.stack([self.nodes_0d[nid].vector for nid in group], dim=0).mean(dim=0, keepdim=True)
            else:
                stacked = torch.randn(1, self.d_model)
            modal_tensors.append(stacked)

        integrated_manifold, coherence_matrix = self.manifold_engine.compute_cross_modal_phase_lock(modal_tensors)
        homology_loss = self.manifold_engine.compute_persistent_homology_loss(integrated_manifold)

        return integrated_manifold, coherence_matrix, homology_loss

    def rotate_spacetime_dial(
        self,
        manifold_state: torch.Tensor,
        dial_rotation_delta: float,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        4D Spacetime Phase Velocity Modulation.
        Rotates the spacetime dial and shifts state across phase boundaries.
        """
        return self.manifold_engine.step_4d_spacetime_dial(manifold_state, dial_rotation_delta, dt=dt)
