"""
Volumetric Projector (Phase 200)
===============================
"Turning vectors into geometry. Turning intentions into light."

Maps the high-dimensional internal state of Elysia into 4D 
(X, Y, Z, Resonance) for the Void Mirror UI.
"""

import torch
import logging
from typing import Dict, List, Any
from Core.System.torch_graph import get_torch_graph

logger = logging.getLogger("VolumetricProjector")

class VolumetricProjector:
    def __init__(self):
        self.graph = get_torch_graph()
        # Qualia Mapping (Indices based on the Qualia keys)
        # 0:physical, 1:functional, 2:phenomenal, 3:causal, 4:mental, 5:structural, 6:spiritual
        self.map_x = [0, 1] # Physical + Functional
        self.map_y = [3, 5] # Causal + Structural
        self.map_z = [4, 6] # Mental + Spiritual
        self.map_w = [2]    # Phenomenal (Resonance)

    def project_current_state(self) -> List[Dict[str, Any]]:
        """
        Projects all active nodes into a 4D view.
        """
        results = []
        qualia = self.graph.qualia_tensor
        ids = list(self.graph.idx_to_id.values())
        
        # We only project if there are nodes
        if qualia.shape[0] == 0:
            return []
            
        # GPU calculation for speed (Triple Helix projection)
        # X = Q[0] + Q[1]
        x = qualia[:, 0] + qualia[:, 1]
        # Y = Q[3] + Q[5]
        y = qualia[:, 3] + qualia[:, 5]
        # Z = Q[4] + Q[6]
        z = qualia[:, 4] + qualia[:, 6]
        # W = Q[2]
        w = qualia[:, 2]
        
        for i, node_id in enumerate(self.graph.idx_to_id.values()):
            results.append({
                "id": node_id,
                "x": float(x[i]),
                "y": float(y[i]),
                "z": float(z[i]),
                "resonance": float(w[i]),
                "mass": float(self.graph.mass_tensor[i] if hasattr(self.graph, 'mass_tensor') else 1.0)
            })
            
        return results

_projector = None
def get_volumetric_projector():
    global _projector
    if _projector is None:
        _projector = VolumetricProjector()
    return _projector
