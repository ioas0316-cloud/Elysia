
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
        
        # Knowledge Store (Metadata/Principles)
        self.node_metadata: Dict[str, Dict] = {} # {id: { "principle": "...", "mechanism": "..." }}

        
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
        # Weights (Synaptic Strength)
        self.link_weights = torch.zeros((0,), dtype=torch.float, device=self.device)
        
        # [Neural Link] Default to SBERT (384) dimension
        self.dim_vector = 384
        # Re-init vec_tensor with correct dim
        self.vec_tensor = torch.zeros((0, self.dim_vector), device=self.device)
        self.lock = False # Simple lock for batch updates
        
        # [The Kidney]
        from Core.Foundation.concept_sanitizer import get_sanitizer
        self.sanitizer = get_sanitizer()

        logger.info(f"⚡ TorchGraph Initialized on {self.device} (Matrix Mode)")

    def add_node(self, node_id: str, vector: List[float] = None, pos: List[float] = None):
        # ... (Existing add_node logic unchanged) ...
        # (For brevity, assuming replace_file_content replaces only the target range reliably)
        # Actually I need to be careful not to delete add_node if I request a replacement around it.
        # So I will target __init__ and add_link separately if they are far apart.
        # They are at lines 38 and 105.
        pass
         
    # RE-TARGETING __init__ only first

    def add_node(self, node_id: str, vector: List[float] = None, pos: List[float] = None, metadata: Dict = None):
        # [Sanitization]
        if not self.sanitizer.is_valid(node_id):
            logger.debug(f"🛑 Rejecting toxic node: {node_id}")
            return
            
        if node_id in self.id_to_idx:
            # Update existing
            idx = self.id_to_idx[node_id]
            if metadata:
                self.node_metadata.setdefault(node_id, {}).update(metadata)
            
            if vector:
                 # Pad or trim vector
                v = torch.tensor(vector, device=self.device).view(-1)
                v = v[:self.dim_vector]
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
        if metadata:
            self.node_metadata[node_id] = metadata
        
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
             # Pad or trim vector
            v = torch.tensor(vector, device=self.device).view(-1)
            v = v[:self.dim_vector]
            if v.shape[0] < self.dim_vector:
                v = torch.cat([v, torch.zeros(self.dim_vector - v.shape[0], device=self.device)])
            new_vec = v.unsqueeze(0)
        else:
            new_vec = torch.zeros((1, self.dim_vector), device=self.device)
            
        self.vec_tensor = torch.cat([self.vec_tensor, new_vec])
        
        # Mass
        self.mass_tensor = torch.cat([self.mass_tensor, torch.tensor([1.0], device=self.device)])

    def update_node_vector(self, idx: int, vector: torch.Tensor):
        """
        [Digestion Protocol]
        Updates the vector of an existing node (e.g. adding Visual Frequencies).
        """
        if idx < 0 or idx >= self.vec_tensor.shape[0]: return
        
        # Ensure dimension match
        v = vector[:self.dim_vector]
        if v.shape[0] < self.dim_vector:
            v = torch.cat([v, torch.zeros(self.dim_vector - v.shape[0], device=self.device)])
            
        self.vec_tensor[idx] = v
        # Increase mass slightly to represent "Weight of Knowledge"
        self.mass_tensor[idx] += 0.1

    def add_link(self, subject: str, object_: str, weight: float = 1.0):
        if subject not in self.id_to_idx: self.add_node(subject)
        if object_ not in self.id_to_idx: self.add_node(object_)
        
        idx_s = self.id_to_idx[subject]
        idx_o = self.id_to_idx[object_]
        
        new_link = torch.tensor([[idx_s, idx_o]], dtype=torch.long, device=self.device)
        self.logic_links = torch.cat([self.logic_links, new_link])
        
        new_weight = torch.tensor([weight], dtype=torch.float, device=self.device)
        self.link_weights = torch.cat([self.link_weights, new_weight])

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

    def find_hollow_nodes(self, limit: int = 10) -> List[str]:
        """
        [The Sovereign Loop]
        Identifies concepts that are 'Heavy' (High Mass/Connectivity) 
        but 'Hollow' (Lack Wisdom/Metadata).
        """
        hollows = []
        # Sort by Mass (Importance) descending
        # mass_tensor is (N,)
        if self.mass_tensor.shape[0] == 0: return []
        
        # Get top indices by mass
        sorted_indices = torch.argsort(self.mass_tensor, descending=True)
        
        for idx in sorted_indices:
            if len(hollows) >= limit: break
            
            i = idx.item()
            concept_id = self.idx_to_id.get(i)
            if not concept_id: continue
            
            # Check Wisdom
            # If metadata is missing or sparse, it's hollow.
            meta = self.node_metadata.get(concept_id, {})
            if not meta or "principle" not in meta:
                hollows.append(concept_id)
                
        return hollows

    def apply_metabolism(self, decay_rate: float = 0.001, prune_threshold: float = 0.5):
        """
        [Optimization Protocol]
        Applies entropy to the brain. Nodes that are not reinforced will fade.
        1. Decay Mass.
        2. Prune weak nodes.
        """
        if self.mass_tensor.shape[0] == 0: return
        
        # 1. Decay
        self.mass_tensor -= decay_rate
        # Clamp to 0
        self.mass_tensor = torch.max(self.mass_tensor, torch.zeros_like(self.mass_tensor))
        
        # 2. Identify Dead Nodes
        # Condition: Mass < Threshold AND Locked=False
        # For prototype, we just check Mass.
        # We need to be careful not to delete indices that shift others...
        # Deletion in Tensor is expensive (copy).
        # Strategy: Mark as dead (Mass=0) and periodically compact.
        
        dead_indices = (self.mass_tensor <= prune_threshold).nonzero(as_tuple=True)[0]
        
        if len(dead_indices) > 0:
            count = len(dead_indices)
            logger.info(f"💀 Metabolism: {count} weak concepts are fading... (Mass < {prune_threshold})")
            
            # Real Deletion (Compaction) - Expensive, maybe run rarely.
            # For now, just remove from Logic Links so they drift away?
            # Or actually delete. Let's actually delete for "Optimization".
            
            # Inverse mask
            keep_mask = self.mass_tensor > prune_threshold
            
            # create new mapping
            old_idx_to_id = self.idx_to_id.copy()
            self.id_to_idx = {}
            self.idx_to_id = {}
            
            # Filter tensors
            self.pos_tensor = self.pos_tensor[keep_mask]
            self.vec_tensor = self.vec_tensor[keep_mask]
            self.mass_tensor = self.mass_tensor[keep_mask]
            
            # Rebuild Mapping
            kept_indices = keep_mask.nonzero(as_tuple=True)[0]
            for new_i, old_i in enumerate(kept_indices):
                old_id = old_idx_to_id[old_i.item()]
                self.id_to_idx[old_id] = new_i
                self.idx_to_id[new_i] = old_id
                
            # Filter Links (This is hard because indices shift)
            # Brute force rebuild for prototype
            # Or just drop all links for now (Loss of structure!) -> BAD.
            # Correct way: Remap links.
            
            # Remap Map: Old -> New
            remap = torch.full((len(old_idx_to_id),), -1, dtype=torch.long, device=self.device)
            remap[kept_indices] = torch.arange(len(kept_indices), device=self.device)
            
            # Update links
            if self.logic_links.shape[0] > 0:
                src = self.logic_links[:, 0]
                tgt = self.logic_links[:, 1]
                
                new_src = remap[src]
                new_tgt = remap[tgt]
                
                # Keep valid links (both src and tgt survived)
                valid_link_mask = (new_src != -1) & (new_tgt != -1)
                
                self.logic_links = torch.stack((new_src[valid_link_mask], new_tgt[valid_link_mask]), dim=1)
                if self.link_weights is not None and self.link_weights.shape[0] > 0:
                     self.link_weights = self.link_weights[valid_link_mask]

            logger.info(f"🗑️ Pruned {count} nodes. New Brain Size: {len(self.id_to_idx)}")

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
            "link_weights": self.link_weights,
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
            
            # Load Weights (Backward compat: if missing, ones)
            if "link_weights" in state:
                self.link_weights = state["link_weights"].to(self.device)
            else:
                 self.link_weights = torch.ones((self.logic_links.shape[0],), device=self.device)
            
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
