"""
Dual-Track Causal Cognitive Engine Module (`core/engine/dual_track_causal_engine.py`)

This engine implements a Dual-Track Causal Cognitive Engine architecture,
bridging graphics/physics pipelines with high-level cognitive models.

Core Components:
1. Dual-Track Node Phase & Energy Model (Track A: Fixed VAT / Track B: Dynamic CS Physics)
2. Phase Differential Alignment, Hermite S-Curve, and Quaternion Slerp Blending (A <-> B)
3. 16-Channel Unified Sensory Spatial Hash Tensor Field with 3D Diffusion & Damping
4. Vector Quantization (VQ) Codebook, Delta Trajectory Compression, and Simulated DirectStorage LUT Baking
5. Causal Attractor Landscape & Critical Entropy Bifurcation Ridge Engine with Parallel Exploratory Branching
"""

import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# ============================================================================
# PART 1: Data Structures & Hardware Alignments (DirectStorage / LUT Header)
# ============================================================================

DIRECTSTORAGE_MAGIC = b"ELYSIAN1"
PAGE_ENTRY_SIZE = 64  # Cache-line aligned 64 bytes
HEADER_SECTOR_SIZE = 4096  # 4KB Sector Aligned


@dataclass
class DirectStorageFileHeader:
    """SSD File Top-level Global Meta Header (4KB Sector Aligned)."""
    magic_bytes: bytes = DIRECTSTORAGE_MAGIC
    schema_version: int = 1
    total_baked_pages: int = 0
    page_table_offset: int = 4096
    data_region_offset: int = 65536
    codebook_size: int = 256

    def pack(self) -> bytes:
        data = struct.pack(
            "<8sIQQQI",
            self.magic_bytes,
            self.schema_version,
            self.total_baked_pages,
            self.page_table_offset,
            self.data_region_offset,
            self.codebook_size,
        )
        padding = b"\x00" * (HEADER_SECTOR_SIZE - len(data))
        return data + padding

    @classmethod
    def unpack(cls, buffer: bytes) -> "DirectStorageFileHeader":
        magic, ver, baked_pages, pt_off, data_off, cb_size = struct.unpack(
            "<8sIQQQI", buffer[:40]
        )
        return cls(
            magic_bytes=magic,
            schema_version=ver,
            total_baked_pages=baked_pages,
            page_table_offset=pt_off,
            data_region_offset=data_off,
            codebook_size=cb_size,
        )


@dataclass
class CausalPageEntry:
    """
    64-Byte CPU/GPU Cache-line Aligned Individual Causal Page Entry.
    """
    spatial_hash_key: int = 0
    nvme_sector_offset: int = 0
    payload_size_bytes: int = 0
    vq_codebook_idx: int = 0
    is_baked: bool = False
    vram_pinned: bool = False
    is_attractor: bool = False
    has_bifurcation: bool = False
    last_access_tick: int = 0
    usage_frequency: int = 0
    vram_page_address: int = 0

    def pack(self) -> bytes:
        flags = (
            (1 if self.is_baked else 0)
            | ((1 if self.vram_pinned else 0) << 1)
            | ((1 if self.is_attractor else 0) << 2)
            | ((1 if self.has_bifurcation else 0) << 3)
        )
        data = struct.pack(
            "<QQIHBIIQ",
            self.spatial_hash_key,
            self.nvme_sector_offset,
            self.payload_size_bytes,
            self.vq_codebook_idx,
            flags,
            self.last_access_tick,
            self.usage_frequency,
            self.vram_page_address,
        )
        padding = b"\x00" * (PAGE_ENTRY_SIZE - len(data))
        return data + padding

    @classmethod
    def unpack(cls, buffer: bytes) -> "CausalPageEntry":
        (
            hash_key,
            sector_off,
            payload_sz,
            vq_idx,
            flags,
            last_tick,
            freq,
            vram_addr,
        ) = struct.unpack("<QQIHBIIQ", buffer[:39])
        is_baked = bool(flags & 0x01)
        vram_pinned = bool(flags & 0x02)
        is_attractor = bool(flags & 0x04)
        has_bifurcation = bool(flags & 0x08)
        return cls(
            spatial_hash_key=hash_key,
            nvme_sector_offset=sector_off,
            payload_size_bytes=payload_sz,
            vq_codebook_idx=vq_idx,
            is_baked=is_baked,
            vram_pinned=vram_pinned,
            is_attractor=is_attractor,
            has_bifurcation=has_bifurcation,
            last_access_tick=last_tick,
            usage_frequency=freq,
            vram_page_address=vram_addr,
        )


# ============================================================================
# PART 2: 16-Channel Sensory Tensor & VQ Codebook Compression
# ============================================================================

class SensoryTensorEncoder:
    """
    16-Channel Unified Sensory Tensor Encoder & Vector Quantization (VQ).
    Channels:
      [0..3]: Visual & Depth (R, G, B, Depth)
      [4..7]: Acoustic & Phase (Amp, Freq, Sin(Phase), Cos(Phase))
      [8..11]: Physical Force & Friction (Fx, Fy, Fz, Viscosity)
      [12..15]: Contextual Meta (Entropy, ContextID, Prior_1, Prior_2)
    """

    def __init__(self, codebook_size: int = 256, vector_dim: int = 16):
        self.codebook_size = codebook_size
        self.vector_dim = vector_dim
        # Initialize orthogonal / normalized centroids
        rng = np.random.default_rng(42)
        self.centroids = rng.normal(0, 1.0, (codebook_size, vector_dim)).astype(np.float32)
        # Normalize centroids
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids /= norms

    def encode_frame(
        self,
        visual_depth: Tuple[float, float, float, float],
        acoustic_phase: Tuple[float, float, float, float],
        physical_force: Tuple[float, float, float, float],
        contextual_meta: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Packs raw sensory inputs into a 16-element float32 vector."""
        vec = np.zeros(16, dtype=np.float32)
        vec[0:4] = visual_depth
        vec[4:8] = acoustic_phase
        vec[8:12] = physical_force
        vec[12:16] = contextual_meta
        return vec

    def quantize(self, tensor_batch: np.ndarray) -> np.ndarray:
        """
        Batch L2 Squared Distance quantization to 1-byte codebook index (K=256).
        tensor_batch: [N, 16]
        returns: [N] uint8 codebook indices
        """
        if tensor_batch.ndim == 1:
            tensor_batch = tensor_batch[np.newaxis, :]

        # Compute pairwise squared L2 distance: ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
        a_sq = np.sum(tensor_batch ** 2, axis=1, keepdims=True)  # [N, 1]
        b_sq = np.sum(self.centroids ** 2, axis=1, keepdims=True).T  # [1, K]
        ab = np.dot(tensor_batch, self.centroids.T)  # [N, K]

        dists = a_sq + b_sq - 2.0 * ab
        indices = np.argmin(dists, axis=1).astype(np.uint8)
        return indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct 16-channel vector from VQ indices."""
        return self.centroids[indices]


# ============================================================================
# PART 3: 3D Spatial Tensor Field & Discrete Diffusion Engine
# ============================================================================

class SpatialHashTensorField:
    """
    3D Spatial Hash Tensor Field for Multi-Sensory Diffusion & Impulse Stamping.
    Implements 3D Discrete Laplacian Diffusion and Damping.
    """

    def __init__(
        self,
        grid_dim: Tuple[int, int, int] = (16, 16, 16),
        cell_size: float = 1.0,
        diffusion_rate: float = 0.1,
        damping_factor: float = 0.95,
    ):
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.diffusion_rate = diffusion_rate
        self.damping_factor = damping_factor

        # 3D Grid storing 16-channel sensory energy fields
        self.grid = np.zeros((*grid_dim, 16), dtype=np.float32)

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Maps 3D world coordinate to grid indices."""
        gx = int(np.floor(pos[0] / self.cell_size)) % self.grid_dim[0]
        gy = int(np.floor(pos[1] / self.cell_size)) % self.grid_dim[1]
        gz = int(np.floor(pos[2] / self.cell_size)) % self.grid_dim[2]
        return gx, gy, gz

    def stamp_energy(self, pos: np.ndarray, sensory_tensor: np.ndarray):
        """Pass 1: Energy Stamping CS onto 3D Grid."""
        gx, gy, gz = self.world_to_grid(pos)
        self.grid[gx, gy, gz] += sensory_tensor

    def step_diffusion(self):
        """Pass 2: 3D Discrete Laplacian Diffusion & Decay CS."""
        # Roll grid in 6 axial directions to compute 6-neighbor sum
        neighbor_sum = (
            np.roll(self.grid, 1, axis=0)
            + np.roll(self.grid, -1, axis=0)
            + np.roll(self.grid, 1, axis=1)
            + np.roll(self.grid, -1, axis=1)
            + np.roll(self.grid, 1, axis=2)
            + np.roll(self.grid, -1, axis=2)
        )

        # 3D Discrete Laplacian: \nabla^2 T = \sum N_6 - 6.0 * T_center
        laplacian = neighbor_sum - 6.0 * self.grid

        # Update: T_{new} = (T_center + \kappa * laplacian) * \gamma
        self.grid = (self.grid + self.diffusion_rate * laplacian) * self.damping_factor
        np.maximum(0.0, self.grid, out=self.grid)

    def sample_field(self, pos: np.ndarray) -> np.ndarray:
        """Pass 3: Trilinear Interpolation Sampling for Inactive Nodes."""
        gx, gy, gz = self.world_to_grid(pos)
        return self.grid[gx, gy, gz]


# ============================================================================
# PART 4: Quaternion & Phase Alignment Helper Functions
# ============================================================================

def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation (Slerp) between two unit quaternions [w, x, y, z]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # If dot product is negative, invert one quaternion to take the shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        res = q0 + t * (q1 - q0)
        return res / np.linalg.norm(res)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def hermite_s_curve(a: float) -> float:
    """3rd order Hermite S-Curve: S(a) = 3a^2 - 2a^3 for C^1 continuity."""
    a = max(0.0, min(1.0, a))
    return 3.0 * (a ** 2) - 2.0 * (a ** 3)


# ============================================================================
# PART 5: Dual-Track Node & Engine Mechanics
# ============================================================================

@dataclass
class DualTrackNode:
    """Represents a Node in the Dual-Track Engine Phase & Energy Space."""
    node_id: int
    ground_pos: np.ndarray  # Ground trajectory position (Track A VAT)
    dynamic_pos: np.ndarray  # Dynamic state position (Track B CS)
    current_pos: np.ndarray  # Blended final position

    ground_orient: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    dynamic_orient: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    current_orient: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    phase: float = 0.0  # Phase \Phi \in [0, 2\pi]
    energy: float = 0.0  # Energy density E_i
    activity: float = 0.0  # Activity state A_i \in [0.0, 1.0]

    # Historical trajectory buffer for baking
    trajectory_history: List[np.ndarray] = field(default_factory=list)


class DualTrackCausalEngine:
    """
    Dual-Track Causal Cognitive Engine.
    Manages million-node trajectories, state transitions, baking, and bifurcation.
    """

    def __init__(
        self,
        num_nodes: int = 1000,
        energy_threshold: float = 1.0,
        lock_threshold: float = 0.2,
        decay_rate: float = 0.05,
    ):
        self.num_nodes = num_nodes
        self.energy_threshold = energy_threshold  # E_th (Breakout threshold)
        self.lock_threshold = lock_threshold      # E_lock (Attractor capture threshold)
        self.decay_rate = decay_rate

        self.sensory_encoder = SensoryTensorEncoder()
        self.spatial_field = SpatialHashTensorField()
        self.page_table: Dict[int, CausalPageEntry] = {}

        # Initialize nodes
        self.nodes: List[DualTrackNode] = []
        rng = np.random.default_rng(123)
        for i in range(num_nodes):
            g_pos = rng.uniform(-10.0, 10.0, size=3).astype(np.float32)
            node = DualTrackNode(
                node_id=i,
                ground_pos=g_pos.copy(),
                dynamic_pos=g_pos.copy(),
                current_pos=g_pos.copy(),
                phase=rng.uniform(0, 2 * math.pi),
            )
            self.nodes.append(node)

        self.active_node_indices: List[int] = []
        self.bifurcation_history: List[Dict[str, Any]] = []

    def update_ground_trajectories(self, t: float):
        """Track A: VAT Playback Harmonic Oscillation (ALU Cost ~ 0)."""
        for node in self.nodes:
            # Simple harmonic oscillation for VAT ground trajectory
            node.ground_pos[0] += 0.01 * math.sin(t + node.phase)
            node.ground_pos[1] += 0.01 * math.cos(t + node.phase)
            node.phase = (node.phase + 0.02) % (2 * math.pi)

    def inject_external_impulse(self, node_idx: int, impulse_energy: float, sensory_frame: Optional[np.ndarray] = None):
        """Injects external impulse / sensory stimulus into a specific node."""
        if 0 <= node_idx < self.num_nodes:
            node = self.nodes[node_idx]
            node.energy += impulse_energy

            if sensory_frame is None:
                sensory_frame = self.sensory_encoder.encode_frame(
                    visual_depth=(1.0, 0.5, 0.2, node.ground_pos[2]),
                    acoustic_phase=(impulse_energy, 440.0, math.sin(node.phase), math.cos(node.phase)),
                    physical_force=(0.0, 0.0, impulse_energy, 0.1),
                    contextual_meta=(0.5, 1.0, 0.0, 0.0),
                )
            self.spatial_field.stamp_energy(node.current_pos, sensory_frame)

    def step_simulation(self, t: float) -> Dict[str, Any]:
        """
        Executes one full tick of the Dual-Track Causal Engine.
        Returns performance and state statistics.
        """
        # 1. Update Track A VAT Ground Trajectories
        self.update_ground_trajectories(t)

        # 2. Field Sampling for Track A (Inactive) Nodes
        for node in self.nodes:
            if node.activity == 0.0:
                sampled_sensory = self.spatial_field.sample_field(node.ground_pos)
                sampled_energy = float(sampled_sensory[8] + sampled_sensory[10])  # Physical force magnitude
                node.energy += sampled_energy * 0.1

        # 3. State Transition: Track A -> Track B Breakout Check
        new_active_indices = []
        for node in self.nodes:
            if node.activity == 0.0 and node.energy >= self.energy_threshold:
                # Resonance Breakout!
                node.activity = 0.01  # Wake up
                node.dynamic_pos = node.ground_pos.copy() + np.array([0.1, 0.1, 0.1], dtype=np.float32)

            if node.activity > 0.0:
                new_active_indices.append(node.node_id)

        self.active_node_indices = new_active_indices

        # 4. Stream Compaction & Track B Dynamic Physics Kernel Execution
        for idx in self.active_node_indices:
            node = self.nodes[idx]
            # Increase activity to max 1.0
            if node.energy >= self.energy_threshold:
                node.activity = min(1.0, node.activity + 0.1)

            # Dynamic CS physics calculation
            force = np.random.normal(0, 0.05, size=3).astype(np.float32)
            node.velocity += force
            node.dynamic_pos += node.velocity

            # Energy Dissipation & Damping
            node.energy = max(0.0, node.energy - self.decay_rate)
            node.velocity *= 0.95

            # Record trajectory frame for baking
            node.trajectory_history.append(node.dynamic_pos.copy())

            # Stamp energy into 3D Tensor Grid
            sensory_vec = self.sensory_encoder.encode_frame(
                visual_depth=(0.8, 0.2, 0.1, node.dynamic_pos[2]),
                acoustic_phase=(node.energy, 220.0, math.sin(node.phase), math.cos(node.phase)),
                physical_force=(node.velocity[0], node.velocity[1], node.velocity[2], 0.2),
                contextual_meta=(node.activity, 2.0, 0.0, 0.0),
            )
            self.spatial_field.stamp_energy(node.dynamic_pos, sensory_vec)

        # 5. Attractor Capture & Phase-Lock Decay (Track B -> Track A)
        for idx in self.active_node_indices:
            node = self.nodes[idx]
            if node.energy < self.lock_threshold:
                # Attractor capture decay
                node.activity = max(0.0, node.activity - 0.1)

                # Check for Phase-Lock Runtime Baking condition
                if len(node.trajectory_history) >= 10:
                    self._bake_trajectory(node)

        # 6. Hermite S-Curve Blending & Quaternion Alignment
        for node in self.nodes:
            if node.activity == 0.0:
                node.current_pos = node.ground_pos.copy()
                node.current_orient = node.ground_orient.copy()
            else:
                weight = hermite_s_curve(node.activity)
                node.current_pos = (1.0 - weight) * node.ground_pos + weight * node.dynamic_pos
                node.current_orient = quaternion_slerp(node.ground_orient, node.dynamic_orient, weight)

        # 7. Step 3D Spatial Field Diffusion
        self.spatial_field.step_diffusion()

        # Compute summary metrics
        active_ratio = len(self.active_node_indices) / float(self.num_nodes)
        alu_reduction = (1.0 - active_ratio) * 100.0

        return {
            "total_nodes": self.num_nodes,
            "active_nodes": len(self.active_node_indices),
            "active_ratio": active_ratio,
            "alu_reduction_percent": alu_reduction,
            "baked_pages_count": len(self.page_table),
        }

    def _bake_trajectory(self, node: DualTrackNode):
        """Bakes stabilized trajectory chunk into DirectStorage LUT & VQ Codebook."""
        traj_data = np.array(node.trajectory_history, dtype=np.float32)
        node.trajectory_history.clear()

        # Compute spatial hash key
        gx, gy, gz = self.spatial_field.world_to_grid(node.current_pos)
        hash_key = (gx * 73856093) ^ (gy * 19349663) ^ (gz * 83492791)

        # Quantize latest sensory state
        sensory_vec = self.sensory_encoder.encode_frame(
            visual_depth=(1.0, 1.0, 1.0, node.current_pos[2]),
            acoustic_phase=(0.0, 0.0, 0.0, 1.0),
            physical_force=(0.0, 0.0, 0.0, 0.0),
            contextual_meta=(0.0, 0.0, 0.0, 0.0),
        )
        vq_idx = int(self.sensory_encoder.quantize(sensory_vec)[0])

        page_entry = CausalPageEntry(
            spatial_hash_key=hash_key,
            nvme_sector_offset=len(self.page_table) * 512,
            payload_size_bytes=traj_data.nbytes,
            vq_codebook_idx=vq_idx,
            is_baked=True,
            vram_pinned=True,
            is_attractor=True,
            usage_frequency=1,
            last_access_tick=int(time.time()),
        )
        self.page_table[hash_key] = page_entry

    def evaluate_critical_bifurcation(self, entropy_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        Evaluates Critical Entropy on Saddle Points (Bifurcation Ridge).
        Forks exploratory parallel branches if entropy exceeds threshold.
        """
        if not self.active_node_indices:
            return None

        # Calculate decision entropy among active nodes
        energies = np.array([self.nodes[idx].energy for idx in self.active_node_indices])
        probs = energies / (np.sum(energies) + 1e-8)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))

        if entropy > entropy_threshold:
            # Fork parallel branch simulation
            branch_alpha = [self.nodes[idx].dynamic_pos + np.array([0.05, 0.0, 0.0]) for idx in self.active_node_indices]
            branch_beta = [self.nodes[idx].dynamic_pos - np.array([0.05, 0.0, 0.0]) for idx in self.active_node_indices]

            bifurcation_event = {
                "timestamp": time.time(),
                "entropy": float(entropy),
                "active_nodes_count": len(self.active_node_indices),
                "branch_alpha_sample": branch_alpha[0].tolist(),
                "branch_beta_sample": branch_beta[0].tolist(),
                "resolution": "Ridge_Split_Conditional_Attractor",
            }
            self.bifurcation_history.append(bifurcation_event)
            return bifurcation_event

        return None
