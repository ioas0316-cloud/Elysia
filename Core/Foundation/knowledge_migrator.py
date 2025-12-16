
import logging
import torch
import random
from Core.Foundation.internal_universe import InternalUniverse
from Core.Foundation.torch_graph import get_torch_graph

logger = logging.getLogger("KnowledgeMigrator")

class KnowledgeMigrator:
    """
    Bridges the gap between the Old Brain (InternalUniverse) and New Brain (TorchGraph).
    """
    def __init__(self):
        self.old_brain = InternalUniverse()
        self.new_brain = get_torch_graph()
        
    def migrate(self):
        """
        Transfers all concepts from InternalUniverse to TorchGraph.
        """
        logger.info("🧠 Knowledge Migration Started...")
        
        # 1. Load Old Data
        concepts = self.old_brain.coordinate_map
        count = len(concepts)
        if count == 0:
            logger.warning("   ⚠️ Old Brain is empty. Nothing to migrate.")
            return
            
        logger.info(f"   found {count} concepts in InternalUniverse.")
        
        # 2. Transfer
        added = 0
        for name, coord in concepts.items():
            # Check if exists
            if name in self.new_brain.id_to_idx:
                continue
                
            # Map Quaternion (w,x,y,z) to Position (x,y,z,w)
            # InternalCoordinate: orientation(w,x,y,z), frequency, depth
            q = coord.orientation
            
            # Position: Use x,y,z from quaternion, and w for 4D
            # Scale by depth or frequency if needed
            pos_vec = [q.x, q.y, q.z, q.w]
            
            # Vector Embedding: 
            # Ideally this should be semantic. 
            # For now, we seed it with frequency-based noise to maintain "Vibe"
            # (Frequency determines energy/color)
            seed_val = coord.frequency / 1000.0
            vector = [seed_val + (random.random() * 0.1) for _ in range(self.new_brain.dim_vector)]
            
            self.new_brain.add_node(name, vector, pos=pos_vec)
            added += 1
            
        logger.info(f"   ✅ Migrated {added} concepts to Matrix Memory.")
        
# Singleton
_migrator = None
def get_migrator():
    global _migrator
    if _migrator is None:
        _migrator = KnowledgeMigrator()
    return _migrator
