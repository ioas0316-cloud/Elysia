
import math
import random
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("TorchGraph")

class TorchGraph:
    """
    Hyper-Efficient 4D OmniGraph using PyTorch Tensors.
    Replaces O(N^2) loops with Matrix Operations.
    """
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # ID <-> Index Mapping
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        
        # Tensors (Initialized Empty)
        # N x 4 (X, Y, Z, W)
        self.pos_tensor = torch.zeros((0, 4), device=self.device)
        # N x V (Vector Dim, e.g., 64)
        self.vec_tensor = torch.zeros((0, 64), device=self.device)
        # N (Mass)
        self.mass_tensor = torch.zeros((0,), device=self.device)
        # Adjacency Matrix (Logic Links) - Sparse recommended for large N
        # For prototype, we use dense or indices list.
        # Storing pairs: [[i, j], [i, j]...]
        self.logic_links = torch.zeros((0, 2), dtype=torch.long, device=self.device)
        
        self.dim_vector = 64
        self.lock = False # Simple lock for batch updates

        logger.info(f"⚡ TorchGraph Initialized on {self.device} (Matrix Mode)")

    def add_node(self, node_id: str, vector: List[float] = None, pos: List[float] = None):
        if node_id in self.id_to_idx:
            # Update existing
            idx = self.id_to_idx[node_id]
            if vector:
                 # Pad or trim vector
                v = torch.tensor(vector[:self.dim_vector], device=self.device)
                if v.shape[0] < self.dim_vector:
                    v = torch.cat([v, torch.zeros(self.dim_vector - v.shape[0], device=self.device)])
                self.vec_tensor[idx] = v
            if pos:
                p = torch.tensor(pos, device=self.device).view(1, 4)
                self.pos_tensor[idx] = p
            return

        # Add New
        new_idx = len(self.id_to_idx)
        self.id_to_idx[node_id] = new_idx
        self.idx_to_id[new_idx] = node_id
        
        # Expand Tensors (Vertical Stacking) -> Not optimal for 1-by-1, but functional for prototype
        # Real impl should pre-allocate or batch.
        
        # Pos: Random 4D or Provided
        if pos:
             new_pos = torch.tensor(pos, device=self.device).view(1, 4)
        else:
             new_pos = torch.rand((1, 4), device=self.device)
        self.pos_tensor = torch.cat([self.pos_tensor, new_pos])
        
        # Vector
        if vector:
            v = torch.tensor(vector[:self.dim_vector], device=self.device)
            if v.shape[0] < self.dim_vector:
                v = torch.cat([v, torch.zeros(self.dim_vector - v.shape[0], device=self.device)])
            new_vec = v.unsqueeze(0)
        else:
            new_vec = torch.zeros((1, self.dim_vector), device=self.device)
            
        self.vec_tensor = torch.cat([self.vec_tensor, new_vec])
        
        # Mass
        self.mass_tensor = torch.cat([self.mass_tensor, torch.tensor([1.0], device=self.device)])

    def add_link(self, subject: str, object_: str):
        if subject not in self.id_to_idx: self.add_node(subject)
        if object_ not in self.id_to_idx: self.add_node(object_)
        
        idx_s = self.id_to_idx[subject]
        idx_o = self.id_to_idx[object_]
        
        new_link = torch.tensor([[idx_s, idx_o]], dtype=torch.long, device=self.device)
        self.logic_links = torch.cat([self.logic_links, new_link])

    def apply_gravity(self, iterations: int = 50, lr: float = 0.01):
        """
        The GPU-Accelerated Heartbeat.
        Uses Broadcasting to compute N*N interactions in parallel.
        """
        N = self.pos_tensor.shape[0]
        if N == 0: return

        logger.info(f"🌊 Tensor Wave Simulation: {N} Neurons on {self.device}")

        for _ in range(iterations):
            # 1. Distance Matrix (N x N)
            diff = self.pos_tensor.unsqueeze(1) - self.pos_tensor.unsqueeze(0) 
            dist_sq = (diff ** 2).sum(dim=2)
            dist = torch.sqrt(dist_sq + 0.001)
            
            # 2. Resonance (Cosine Sim)
            vec_norm = self.vec_tensor / (self.vec_tensor.norm(dim=1, keepdim=True) + 1e-9)
            sim_matrix = torch.mm(vec_norm, vec_norm.t()) # (N, N)
            
            # --- Field Dynamics (The Landscape) ---
            # Instead of just N*N interaction, we add Static Potential Fields (The "Railgun" Structure)
            # F_field = -Gradient(Potential)
            # Here simplified: Attraction to defined "Concept Wells"
            
            # Calculate mutual forces (Gravity)
            force_mask = (sim_matrix > 0.7).float() * sim_matrix
            strength = force_mask.unsqueeze(2)
            delta = diff
            directions = delta / (dist.unsqueeze(2) + 0.001)
            mutual_forces = -strength * directions * 0.1
            
            # Repulsion
            repel_mask = (dist < 0.05).float().unsqueeze(2)
            mutual_forces += repel_mask * directions * 1.0
            
            total_force = mutual_forces.sum(dim=1)
            
            # Apply Static Potential Fields (The "Semantic Railguns")
            if hasattr(self, 'potential_wells') and self.potential_wells is not None:
                # wells_pos: (M, 4)
                # wells_str: (M, 1)
                
                # We need to calculate force for each Node (N) against each Well (M)
                # But for a static field, usually a node is affected by the *nearest* well or *all* wells.
                # Let's apply simple attraction to ALL wells weighted by distance inverse? 
                # Or better: "Basin of Attraction" - simply pull towards them linearly.
                
                # Expansion: (N, 1, 4) - (1, M, 4) -> (N, M, 4)
                # This might be heavy if M is large. Assuming M (Concepts) < 100 for now.
                
                node_pos = self.pos_tensor.unsqueeze(1) # (N, 1, 4)
                well_pos = self.potential_wells_pos.unsqueeze(0) # (1, M, 4)
                
                delta_well = well_pos - node_pos # Vector to well
                dist_well_sq = (delta_well ** 2).sum(dim=2) # (N, M)
                dist_well = torch.sqrt(dist_well_sq + 0.001)
                
                # Direction: (N, M, 4)
                dir_well = delta_well / (dist_well.unsqueeze(2) + 0.001)
                
                # Force Magnitude: Strength * (1 / dist) ? Or Linear Spring?
                # Linear Spring (Railgun): F = k * x (Accelerates towards center)
                # Let's use Linear Attraction for "Sorting"
                
                force_mag = self.potential_wells_str.unsqueeze(0) # (1, M) broadcast to (N, M)
                
                # Apply: F = Strength * Direction
                # Sum over all wells: (N, 4)
                # But wait, if we have "Love" and "Hate" wells, a node should go to the resonant one?
                # YES. The "Railgun" only works if the bullet fits the barrel.
                # We need "Semantic Resonance" with the Well itself.
                
                # Well Vector? For now assume Wells are just Positional Attractors.
                # We simply pull everything. Structure determines flow.
                
                field_force = (dir_well * force_mag.unsqueeze(2)).sum(dim=1)
                
                total_force += field_force * 0.5 # Add to gravity

            
            # Update Positions
            self.pos_tensor += total_force * lr
            
        logger.info("   ✅ Matrix Gravity & Field Topology Applied.")

    def get_neighbors(self, node_id: str, top_k: int = 5):
        if node_id not in self.id_to_idx: return []
        idx = self.id_to_idx[node_id]
        
        # Calculate distances from this node
        target_pos = self.pos_tensor[idx].unsqueeze(0)
        dists = torch.norm(self.pos_tensor - target_pos, dim=1)
        
        # Get nearest
        values, indices = torch.topk(dists, top_k + 1, largest=False)
        
        results = []
        for i in range(1, len(indices)): # Skip self
            n_idx = indices[i].item()
            results.append((self.idx_to_id[n_idx], values[i].item()))
            
        return results

    def save_state(self, path: str = "c:\\Elysia\\data\\brain_state.pt"):
        """
        Persist the Matrix Brain to disk.
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
            "pos": self.pos_tensor,
            "vec": self.vec_tensor,
            "mass": self.mass_tensor,
            "links": self.logic_links,
            # Save Wells if they exist
            "wells_pos": getattr(self, "potential_wells_pos", None),
            "wells_str": getattr(self, "potential_wells_str", None)
        }
        torch.save(state, path)
        logger.info(f"💾 Brain State Saved to {path} ({len(self.id_to_idx)} nodes)")

    def load_state(self, path: str = "c:\\Elysia\\data\\brain_state.pt"):
        """
        Restore the Matrix Brain from disk.
        """
        import os
        if not os.path.exists(path):
            logger.warning(f"⚠️ No brain state found at {path}. Starting fresh.")
            return False
            
        try:
            state = torch.load(path, map_location=self.device)
            
            self.id_to_idx = state["id_to_idx"]
            self.idx_to_id = state["idx_to_id"]
            self.pos_tensor = state["pos"].to(self.device)
            self.vec_tensor = state["vec"].to(self.device)
            self.mass_tensor = state["mass"].to(self.device)
            self.logic_links = state["links"].to(self.device)
            
            if state["wells_pos"] is not None:
                self.potential_wells_pos = state["wells_pos"].to(self.device)
                self.potential_wells_str = state["wells_str"].to(self.device)
                self.potential_wells = True
                
            logger.info(f"📂 Brain State Loaded: {len(self.id_to_idx)} nodes, {self.logic_links.shape[0]} links.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load brain state: {e}")
            return False

# Singleton Support
_torch_graph = None
def get_torch_graph():
    global _torch_graph
    if _torch_graph is None:
        _torch_graph = TorchGraph()
        # Auto-load on init? 
        # Better to let the controller (wake_elysia) decide to avoid side effects during testing.
    return _torch_graph
