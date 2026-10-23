"""
Scale-Coupling Cognitive System (원리 문서 v0.1 수직 슬라이스)

Implements the fundamental principles of scale-coupled perception:
1. Three States: Gas (distribution), Liquid (active PDE field), Ice (mmap disk-frozen static block).
2. Structural Nodes: Structure(n+1) = CouplingFunction(Components = Structure(n)s).
3. Structural Temperature: T_struct = Tr(Cov(dPhi/dt)) observing micro-rearrangement rate.
4. Hysteresis State Transitions: Separate freezing threshold (eps_freeze) and thawing threshold (eps_thaw).
5. Bottom-Up Scale Propagation: When all children are Ice, parent applies coupling function and crystallizes.
6. Limit Map Recording: Unexplainable reconstruction errors > eps_limit are recorded as Ice limit blocks.
"""

from enum import Enum
import os
import tempfile
import time
import numpy as np
import torch


class NodeState(Enum):
    GAS = "GAS"
    LIQUID = "LIQUID"
    ICE = "ICE"


class DefaultCouplingFunction:
    """
    Default coupling function for 2D Quadtree structure.
    Synthesizes 4 child patches (each S x S) into 1 parent patch (S x S)
    via spatial downsampling / mean aggregation, and reconstructs via upsampling.
    """
    def __init__(self, patch_shape=(8, 8)):
        self.patch_shape = patch_shape

    def couple(self, child_nodes):
        """
        Combines 4 child node values and level-sets into 1 parent node patch.
        child_nodes: list of 4 StructureNodes (TL, TR, BL, BR)
        Returns:
            parent_phi: torch.Tensor of shape patch_shape
            parent_val: torch.Tensor of shape patch_shape
        """
        if len(child_nodes) != 4:
            raise ValueError("Quadtree coupling expects exactly 4 child nodes")

        # Stack child phi and val tensors into 2x2 grid
        top_phi = torch.cat([child_nodes[0].phi, child_nodes[1].phi], dim=1)
        bottom_phi = torch.cat([child_nodes[2].phi, child_nodes[3].phi], dim=1)
        grid_phi = torch.cat([top_phi, bottom_phi], dim=0)

        top_val = torch.cat([child_nodes[0].val, child_nodes[1].val], dim=1)
        bottom_val = torch.cat([child_nodes[2].val, child_nodes[3].val], dim=1)
        grid_val = torch.cat([top_val, bottom_val], dim=0)

        # Downsample 2x grid to parent patch_shape using area average
        grid_phi_b = grid_phi.unsqueeze(0).unsqueeze(0)
        grid_val_b = grid_val.unsqueeze(0).unsqueeze(0)

        parent_phi = torch.nn.functional.interpolate(grid_phi_b, size=self.patch_shape, mode='area').squeeze()
        parent_val = torch.nn.functional.interpolate(grid_val_b, size=self.patch_shape, mode='area').squeeze()

        return parent_phi, parent_val

    def reconstruct(self, parent_phi, parent_val, child_shape=(8, 8)):
        """
        Reconstructs 4 child nodes from parent patch via bilinear upsampling.
        Returns:
            recon_children_phi: list of 4 child phi tensors
            recon_children_val: list of 4 child val tensors
        """
        parent_phi_b = parent_phi.unsqueeze(0).unsqueeze(0)
        parent_val_b = parent_val.unsqueeze(0).unsqueeze(0)

        grid_shape = (child_shape[0] * 2, child_shape[1] * 2)
        grid_phi = torch.nn.functional.interpolate(parent_phi_b, size=grid_shape, mode='bilinear', align_corners=False).squeeze()
        grid_val = torch.nn.functional.interpolate(parent_val_b, size=grid_shape, mode='bilinear', align_corners=False).squeeze()

        H, W = child_shape
        recon_phi = [
            grid_phi[:H, :W],
            grid_phi[:H, W:],
            grid_phi[H:, :W],
            grid_phi[H:, W:]
        ]
        recon_val = [
            grid_val[:H, :W],
            grid_val[:H, W:],
            grid_val[H:, :W],
            grid_val[H:, W:]
        ]

        return recon_phi, recon_val


class LimitRecord:
    """
    Record of a failure / boundary limit when reconstruction error exceeds threshold.
    Recorded as a static Ice block as mandated by Section 5 of Principle Doc v0.1.
    """
    def __init__(self, node_id, scale, recon_error, threshold, input_phi, input_val):
        self.node_id = node_id
        self.scale = scale
        self.recon_error = float(recon_error)
        self.threshold = float(threshold)
        self.input_phi = input_phi.clone().detach()
        self.input_val = input_val.clone().detach()
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "scale": self.scale,
            "recon_error": self.recon_error,
            "threshold": self.threshold,
            "timestamp": self.timestamp
        }


class StructureNode:
    """
    Structural Node operating at Scale (n).
    Contains:
      - Boundary Level-set Phi
      - Internal Field Tensor Val
      - State (GAS, LIQUID, ICE)
      - Structural Temperature T_struct
      - Freezing & Thawing Hysteresis logic
      - mmap disk storage for frozen ICE blocks
    """
    def __init__(
        self,
        node_id: str,
        scale: int,
        patch_shape=(8, 8),
        storage_dir=None,
        eps_freeze=0.01,
        eps_thaw=0.05,
        freeze_persistence=3,
        eps_limit=0.25,
        coupling_fn=None
    ):
        self.node_id = node_id
        self.scale = scale
        self.patch_shape = patch_shape
        self.storage_dir = storage_dir or tempfile.mkdtemp(prefix=f"ice_store_{node_id}_")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.eps_freeze = eps_freeze
        self.eps_thaw = eps_thaw
        self.freeze_persistence = freeze_persistence
        self.eps_limit = eps_limit
        self.coupling_fn = coupling_fn or DefaultCouplingFunction(patch_shape=patch_shape)

        self.state = NodeState.GAS
        self.children = []
        self.parent = None

        # Level-set boundary field Phi and internal field Val
        self.phi = torch.zeros(patch_shape, dtype=torch.float32)
        self.val = torch.zeros(patch_shape, dtype=torch.float32)

        # Gas state distribution parameters (mean and variance)
        self.gas_mean = torch.zeros(patch_shape, dtype=torch.float32)
        self.gas_var = torch.ones(patch_shape, dtype=torch.float32)

        # Dynamic state tracking
        self.dphi_dt = torch.zeros(patch_shape, dtype=torch.float32)
        self.temperature = float('inf')
        self.low_temp_counter = 0

        # Memory Mapped Storage info
        self.mmap_path = os.path.join(self.storage_dir, f"{self.node_id}_frozen.bin")

        # Limit Map storage
        self.limit_records = []

        # Benchmark / FLOPs counter
        self.flops_count = 0

    def add_children(self, children_list):
        self.children = children_list
        for child in children_list:
            child.parent = self

    def condense_from_gas(self, observation_phi: torch.Tensor, observation_val: torch.Tensor):
        """
        Gas -> Liquid Transition rule:
        External observation V_ext arrives, reducing distribution variance.
        """
        self.phi = observation_phi.clone().detach()
        self.val = observation_val.clone().detach()
        self.dphi_dt = torch.zeros_like(self.phi)
        self.temperature = float('inf')
        self.low_temp_counter = 0
        self.state = NodeState.LIQUID

    def compute_structural_temperature(self, dphi_dt: torch.Tensor) -> float:
        """
        Calculates structural temperature T_struct = Tr(Cov(dPhi/dt)).
        Measures the micro-rearrangement fluctuation rate of the structure.
        """
        dphi_flat = dphi_dt.reshape(-1)
        if dphi_flat.numel() <= 1:
            return 0.0
        # Trace of Covariance is equal to total variance = sum of squared dev from mean
        var_val = torch.var(dphi_flat, unbiased=False)
        # Mean squared rate of change is also a direct scalar measure of kinetic rearrangement
        mean_sq = torch.mean(dphi_flat ** 2)
        t_struct = (var_val + mean_sq).item()
        return t_struct

    def step_pde(self, dt=0.05, gamma=0.1, external_stimulus: torch.Tensor = None):
        """
        Advances the Liquid state PDE field.
        In Gas state or Ice state:
          - Gas: Awaits lower scale crystallization / coupling or observation.
          - Ice: No internal computations are run (FLOPs = 0), only thawing check is active.
        """
        if self.state == NodeState.GAS:
            return

        if self.state == NodeState.ICE:
            # Check thawing condition if external stimulus touches boundary
            if external_stimulus is not None:
                stimulus_norm = torch.norm(external_stimulus).item()
                if stimulus_norm > self.eps_thaw:
                    self.thaw()
            return

        # FLOPs counted for Liquid update
        # 1. Calculate Mean Curvature / Laplacian flow for level set smoothing
        laplacian_phi = (
            torch.roll(self.phi, 1, 0) + torch.roll(self.phi, -1, 0) +
            torch.roll(self.phi, 1, 1) + torch.roll(self.phi, -1, 1) - 4 * self.phi
        )
        self.flops_count += self.phi.numel() * 5

        # 2. Compute rate of change
        self.dphi_dt = gamma * laplacian_phi
        if external_stimulus is not None:
            self.dphi_dt += external_stimulus
            self.flops_count += self.phi.numel()

        # 3. Update level-set and internal field
        self.phi = self.phi + dt * self.dphi_dt
        self.val = self.val + dt * torch.tanh(self.dphi_dt)
        self.flops_count += self.phi.numel() * 3

        # 4. Measure structural temperature
        self.temperature = self.compute_structural_temperature(self.dphi_dt)

        # 5. Check crystallization (Freezing) condition
        if self.temperature <= self.eps_freeze:
            self.low_temp_counter += 1
            if self.low_temp_counter >= self.freeze_persistence:
                self.freeze()
        else:
            self.low_temp_counter = 0

    def freeze(self):
        """
        Liquid -> Ice Transition rule:
        Crystallizes structure block to disk via np.memmap when temperature <= eps_freeze.
        State transitions to ICE.
        """
        # Save phi and val to memory-mapped binary file
        data_to_store = np.stack([self.phi.numpy(), self.val.numpy()], axis=0).astype(np.float32)
        mmap_arr = np.memmap(self.mmap_path, dtype='float32', mode='w+', shape=data_to_store.shape)
        mmap_arr[:] = data_to_store[:]
        mmap_arr.flush()
        del mmap_arr

        self.state = NodeState.ICE
        self.low_temp_counter = 0
        self.dphi_dt = torch.zeros_like(self.phi)
        self.temperature = 0.0

        # Notify parent if parent exists
        if self.parent is not None:
            self.parent.check_children_crystallization()

    def check_children_crystallization(self):
        """
        Bottom-up crystallization rule (Section 3 & T3):
        When all children of this parent node are ICE,
        parent applies coupling function f_coupling to synthesize parent structure and freezes.
        """
        if not self.children:
            return

        all_children_frozen = all(child.state == NodeState.ICE for child in self.children)
        if all_children_frozen and self.state != NodeState.ICE:
            # Couple children to synthesize parent phi and val
            parent_phi, parent_val = self.coupling_fn.couple(self.children)
            self.phi = parent_phi
            self.val = parent_val

            # Deconstruct / Reconstruct test to evaluate reconstruction error
            recon_phis, recon_vals = self.coupling_fn.reconstruct(self.phi, self.val)
            total_recon_error = 0.0
            total_norm = 0.0

            for idx, child in enumerate(self.children):
                err_phi = torch.norm(child.phi - recon_phis[idx])
                err_val = torch.norm(child.val - recon_vals[idx])
                total_recon_error += (err_phi + err_val).item()
                total_norm += (torch.norm(child.phi) + torch.norm(child.val)).item() + 1e-8

            rel_recon_error = total_recon_error / total_norm

            # Check Limit Map threshold (Section 5 & T4)
            if rel_recon_error > self.eps_limit:
                limit_rec = LimitRecord(
                    node_id=self.node_id,
                    scale=self.scale,
                    recon_error=rel_recon_error,
                    threshold=self.eps_limit,
                    input_phi=self.phi,
                    input_val=self.val
                )
                self.limit_records.append(limit_rec)

            # Freeze parent node
            self.freeze()


    def thaw(self):
        """
        Ice -> Liquid Transition rule:
        When external boundary stimulus exceeds eps_thaw (eps_thaw > eps_freeze),
        loads static block from mmap disk back into memory and sets state to LIQUID.
        """
        if os.path.exists(self.mmap_path):
            shape = (2, self.patch_shape[0], self.patch_shape[1])
            mmap_arr = np.memmap(self.mmap_path, dtype='float32', mode='r', shape=shape)
            self.phi = torch.from_numpy(np.array(mmap_arr[0])).clone()
            self.val = torch.from_numpy(np.array(mmap_arr[1])).clone()

        self.state = NodeState.LIQUID
        self.low_temp_counter = 0
        self.temperature = float('inf')



class ScaleCouplingSystem:
    """
    Top-level multiscale hierarchy manager (Quadtree).
    Manages 2D hierarchical grid of StructureNodes across scales n=0, 1, ...
    """
    def __init__(self, depth=2, patch_shape=(8, 8), storage_dir=None, eps_freeze=0.01, eps_thaw=0.05, eps_limit=0.25):
        self.depth = depth
        self.patch_shape = patch_shape
        self.storage_dir = storage_dir or tempfile.mkdtemp(prefix="scale_coupling_sys_")
        self.eps_freeze = eps_freeze
        self.eps_thaw = eps_thaw
        self.eps_limit = eps_limit

        self.nodes = {}  # id -> StructureNode
        self.root = None
        self.leaf_nodes = []

        self._build_quadtree()

    def _build_quadtree(self):
        """
        Builds a 2-level Quadtree (Scale 0 leaves -> Scale 1 root).
        """
        # Create root at Scale 1
        self.root = StructureNode(
            node_id="root_s1",
            scale=1,
            patch_shape=self.patch_shape,
            storage_dir=os.path.join(self.storage_dir, "s1"),
            eps_freeze=self.eps_freeze,
            eps_thaw=self.eps_thaw,
            eps_limit=self.eps_limit
        )
        self.nodes[self.root.node_id] = self.root

        # Create 4 leaf children at Scale 0 (TL, TR, BL, BR)
        leaf_ids = ["leaf_s0_tl", "leaf_s0_tr", "leaf_s0_bl", "leaf_s0_br"]
        for lid in leaf_ids:
            leaf = StructureNode(
                node_id=lid,
                scale=0,
                patch_shape=self.patch_shape,
                storage_dir=os.path.join(self.storage_dir, "s0"),
                eps_freeze=self.eps_freeze,
                eps_thaw=self.eps_thaw,
                eps_limit=self.eps_limit
            )
            self.nodes[lid] = leaf
            self.leaf_nodes.append(leaf)

        self.root.add_children(self.leaf_nodes)

    def inject_observation(self, leaf_observations: list):
        """
        Injects observations into leaf nodes (Gas -> Liquid transition).
        leaf_observations: list of 4 tuples (obs_phi, obs_val) for leaves.
        """
        if len(leaf_observations) != len(self.leaf_nodes):
            raise ValueError(f"Expected {len(self.leaf_nodes)} leaf observations")

        for leaf, (obs_phi, obs_val) in zip(self.leaf_nodes, leaf_observations):
            leaf.condense_from_gas(obs_phi, obs_val)

    def step(self, external_stimuli: dict = None, dt=0.05):
        """
        Advances all nodes in the system by 1 time step.
        """
        external_stimuli = external_stimuli or {}
        for node_id, node in self.nodes.items():
            stimulus = external_stimuli.get(node_id, None)
            node.step_pde(dt=dt, external_stimulus=stimulus)
