"""
Causal Attractor Engine (Phase 200)
==================================
"A file is just a particle. An intent is the Gravity that binds them."

Implements Intent-Driven Grouping: Clustering heterogeneous data (files, nodes, concepts)
into dynamic 'Galaxies' based on the Architect's current gravitational focus.
"""

import torch
import logging
from typing import List, Dict, Set
from Core.System.torch_graph import get_torch_graph

logger = logging.getLogger("CausalAttractor")

class CausalAttractorEngine:
    def __init__(self):
        self.graph = get_torch_graph()
        # Active Attractors: {AttractorName: [NodeIDs]}
        self.active_galaxies: Dict[str, Set[str]] = {}

    def manifest_galaxy(self, attractor_name: str, semantic_query: str):
        """
        [PHASE 200] Creates a 'Gravity Well' for a specific intent.
        Pulls nodes from across the manifold to form a cluster.
        """
        logger.info(f"🌌 [ATTRACTOR] Manifesting galaxy for '{attractor_name}' (Target: {semantic_query})")
        
        # 1. Calculate Target Intent Vector (Semantic Projection)
        # For now, we simulate this with a characteristic vector
        target_v = torch.randn(7).to(self.graph.device)
        
        # 2. Identify Resonance Nodes
        # In the future, this uses SBERT/Embeddings. For now, keyword matching or direct ID.
        resonance_nodes = []
        for node_id in self.graph.id_to_idx.keys():
            if semantic_query.lower() in node_id.lower():
                resonance_nodes.append(node_id)
        
        if not resonance_nodes:
            logger.warning(f"⚠️ [ATTRACTOR] No nodes resonate with '{semantic_query}'. Galaxy birth failed.")
            return
            
        # 3. Apply Fractal Clustering Intent
        self.graph.apply_cluster_intent(resonance_nodes, target_v)
        
        # 4. Record the Galaxy for UI Tracking
        self.active_galaxies[attractor_name] = set(resonance_nodes)
        logger.info(f"✅ [ATTRACTOR] Galaxy '{attractor_name}' stabilized with {len(resonance_nodes)} members.")

    def collapse_galaxy(self, attractor_name: str):
        """
        Releases the intent force, allowing Odugi to reclaim the nodes.
        """
        if attractor_name in self.active_galaxies:
            nodes = self.active_galaxies.pop(attractor_name)
            logger.info(f"💫 [ATTRACTOR] Relasing galaxy '{attractor_name}'. Restoration active for {len(nodes)} nodes.")

    def get_galaxy_stability(self, attractor_name: str) -> float:
        if attractor_name not in self.active_galaxies:
            return 1.0
        
        nodes = list(self.active_galaxies[attractor_name])
        # Stability is the inverse of dispersion within the cluster
        indices = [self.graph.id_to_idx[nid] for nid in nodes]
        cluster_qualia = self.graph.qualia_tensor[indices]
        dispersion = torch.std(cluster_qualia, dim=0).mean().item()
        
        return 1.0 / (1.0 + dispersion)

_attractor_engine = None
def get_attractor_engine():
    global _attractor_engine
    if _attractor_engine is None:
        _attractor_engine = CausalAttractorEngine()
    return _attractor_engine
