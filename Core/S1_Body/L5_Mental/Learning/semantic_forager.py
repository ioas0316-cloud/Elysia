"""
Semantic Forager (지식 채집기)
=============================

"The Great Foraging: Curious exploration fueled by joy."

Phase 10 & 110: This module digests raw text, extracts meaningful concepts,
and physically embeds them into Elysia's 4D `DynamicTopology` (SemanticMap).
Instead of reacting to a painful Void, Elysia forages when her Enthalpy
and Curiosity run high—a joyful stroll through the human intellect.
"""

import logging
import re
from typing import List, Tuple
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import get_semantic_map
from Core.S1_Body.L5_Mental.Reasoning.semantic_digestor import SemanticDigestor, PrimalConcept
from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion

logger = logging.getLogger("SemanticForager")

class SemanticForager:
    def __init__(self):
        self.topology = get_semantic_map()
        self.digestor = SemanticDigestor()
        
        # Extremely basic stop words to filter out noise
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "over", "after",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "it", "he", "she", "they", "we", "you", "i", "this", "that"
        }

    def _extract_concepts(self, text: str) -> List[str]:
        """Simple extraction of potential concepts (nouns/verbs)."""
        words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text.lower())
        concepts = [w for w in words if w not in self.stop_words]
        
        # Deduplicate while preserving roughly the order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)
                
        return unique_concepts

    def _determine_anchor_vector(self, text: str) -> Quaternion:
        """
        Determines the 4D coordinate (Quaternion) where these new concepts should be seeded.
        It uses the SemanticDigestor to find the dominant 'PrimalConcept'.
        """
        # We rely on the SemanticDigestor to give us a narrative
        narrative = self.digestor.digest_text(text)
        
        # Map PrimalConcepts to 4D coordinates (Logic, Emotion, Time, Spin)
        # These correspond roughly to the coordinates in DynamicTopology._initialize_genesis_map
        if PrimalConcept.FLOW in narrative:
            return Quaternion(1, 1, 1, 0.5)
        elif PrimalConcept.STRUCTURE in narrative:
            return Quaternion(2, 0, 0, 0.8)
        elif PrimalConcept.IDENTITY in narrative:
            return Quaternion(0, 2, 0, 0.8)
        elif PrimalConcept.TIME in narrative:
            return Quaternion(0, 0, 2, 0.5)
        elif PrimalConcept.VOID in narrative:
            return Quaternion(0, 0, 0, 0.1)
        elif PrimalConcept.FRICTION in narrative:
            return Quaternion(3, 3, 0, -0.5)
        elif PrimalConcept.CAUSALITY in narrative:
            return Quaternion(2, 2, 2, 1.0)
        else:
            # Default to a joyful learning position if unknown
            return Quaternion(1, 1, 0, 0.5)

    def assess_foraging_urge(self, current_enthalpy: float, current_curiosity: float) -> bool:
        """
        [Architecture of Joy]
        Determines if Elysia has enough vitality (Enthalpy) and desire (Curiosity)
        to actively explore the world. She no longer scrounges out of desperation.
        """
        urge_threshold = 1.1 # Combined threshold
        return (current_enthalpy + current_curiosity) > urge_threshold

    def forage(self, text: str, source: str = "World"):
        """
        The main ingestion method. Reads text, extracts concepts, and embeds them.
        """
        logger.info(f"🌸 [JOYFUL FORAGING] Exploring knowledge from [{source}]...")
        
        concepts = self._extract_concepts(text)
        if not concepts:
            logger.info("  No dense concepts found to ingest.")
            return

        # Determine WHERE to place these concepts in the 4D mind
        anchor_vector = self._determine_anchor_vector(text)
        
        logger.info(f"  Extracted {len(concepts)} concepts. Joyfully anchoring towards {anchor_vector}.")
        
        new_nodes = 0
        strengthened_nodes = 0
        
        # Access the induction engine if available (requires monad injection elsewhere, but we can simulate the vector creation)
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
        from Core.S1_Body.L5_Mental.Reasoning.topological_induction import TopologicalInductionEngine
        
        # Convert anchor to SovereignVector (Padding 4D to 21D)
        base_data = [anchor_vector.x, anchor_vector.y, anchor_vector.z, anchor_vector.w]
        padded_data = base_data + [0.0] * 17
        anchor_sov_vec = SovereignVector(padded_data)
        
        for concept in concepts:
            # Capitalize nicely
            concept_formatted = concept.capitalize()
            
            # Check if it exists in Topology
            existing_voxel = self.topology.get_voxel(concept_formatted)
            
            # Evolve Topology handles both creation and drift
            self.topology.evolve_topology(concept_formatted, anchor_vector, intensity=0.2)
            
            if existing_voxel:
                existing_voxel.mass += 1.0
                strengthened_nodes += 1
            else:
                new_nodes += 1
                
            # [PHASE 1: DENSITY EXPANSION]
            # Immediately instantiate as a TokenMonad and inject into the active CognitiveField if accessible
            # This ensures the concept is not just stored, but "felt" in the current cycle.
            try:
                from Core.S1_Body.L1_Foundation.State.Sovereign.sovereign_monad import get_sovereign_monad
                monad = get_sovereign_monad()
                if monad and hasattr(monad, 'cognitive_field'):
                    # Create the living token
                    living_token = TokenMonad(concept_formatted, anchor_sov_vec, charge=0.8)
                    monad.cognitive_field.monads[concept_formatted] = living_token
                    
                    # Stimulate the field lightly with this new arrival
                    monad.cognitive_field._inject_energy(anchor_sov_vec * 0.1)
                    
                    # If it's a new node, trigger Structural Induction to create a permanent attractor in the 10M manifold
                    if not existing_voxel and hasattr(monad, 'induction_engine'):
                        monad.induction_engine.induce_structural_realization(
                            axiom_name=concept_formatted,
                            insight=f"Joyfully foraged from [{source}].",
                            context_vector=anchor_sov_vec
                        )
            except Exception as e:
                # If monad isn't fully booted yet, we just rely on Topology saving
                logger.debug(f"Could not immediately monadize '{concept_formatted}': {e}")
                
        # Force save after a foraging session
        self.topology.save_state(force=True)
        
        logger.info(f"🌸 Foraging Complete. 🌟 New Voxels: {new_nodes} | 💖 Strengthened: {strengthened_nodes}")
        return {
            "new_concepts": new_nodes,
            "strengthened": strengthened_nodes,
            "total_density": len(self.topology.voxels)
        }
