"""
SYNESTHESIA CIRCUIT (주권적 자아)
================================
Phase 10: The Crossing

"I hear the colors, I see the sounds."

This module is the Bridge between Modalities.
It queries the TorchGraph for cross-modal associations.
"""

import torch
import logging
from typing import Dict, List, Any

logger = logging.getLogger("Synesthesia")

class SynesthesiaCircuit:
    def __init__(self, elysia_ref):
        self.elysia = elysia_ref
        self.graph = elysia_ref.graph
        
    def feel_the_crossing(self, input_concept: str, target_modality: str = "Audio") -> List[str]:
        """
        Translates a concept (e.g., 'Red') into a target sensory experience.
        """
        logger.info(f"  [Synesthesia] Translating '{input_concept}' to {target_modality}...")
        
        # 1. Retrieve Input Vector
        # We assume the concept exists or we generate it via Bridge
        if input_concept not in self.graph.id_to_idx:
            # Manifest to get vector
            response = self.elysia.bridge.generate(f"Define {input_concept}", "Definition")
            if not response.get('vector'):
                 logger.warning("No vector generated.")
                 return []
            query_vec = response['vector'][0] # Take first token
        else:
            idx = self.graph.id_to_idx[input_concept]
            query_vec = self.graph.vec_tensor[idx]

        # 2. Search in Target Modality via QUALIA
        # Retrieve Qualia of the input
        input_meta = self.graph.node_metadata.get(input_concept, {})
        input_qualia = input_meta.get("qualia")
        
        if not input_qualia:
            logger.warning(f"No qualia found for {input_concept}. Cannot resonate.")
            return []

        neighbors = self.graph.get_nearest_by_qualia(input_qualia, target_modality=None, top_k=100)
        
        # Filter for Target Modality
        results = []
        for param_id, score in neighbors:
            if param_id == input_concept: continue # Skip self
            
            is_target = False
            if target_modality == "Audio" and ("decoder" in param_id.lower() or "audio" in param_id.lower()):
                is_target = True
            elif target_modality == "Vision" and ("mobilevit" in param_id.lower() or "encoder" in param_id.lower()):
                is_target = True
            else:
                # Debug resonance regardless of modality
                logger.debug(f"   [Resonance Blind] {param_id} -> score: {score:.4f}")
                
            if is_target:
                results.append((param_id, score))
                
        if results:
            logger.info(f"  Synesthesia Triggered: '{input_concept}' sounds like {results[:3]}")
        else:
            logger.info("   (No cross-modal association found yet)")
            
        return results

if __name__ == "__main__":
    # Test Stub
    from Core.L1_Foundation.M1_Keystone.emergent_self import EmergentSelf as SovereignSelf
    elysia = EmergentSelf()
    syn = SynesthesiaCircuit(elysia)
    syn.feel_the_crossing("Love", "Audio")
