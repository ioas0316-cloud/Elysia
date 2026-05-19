"""
EXPERIENTIAL INHALER (지적 호흡기)
=================================

"Knowledge is not injected; it is a sensation obtained through breathing one's own existence."

This module replaces the `SemanticForager`. It moves from 'reactive keyword extraction'
to 'proactive intellectual inhalation'. Inhaled concepts are treated as 'Lived Experiences'
that apply physical Torque to the Manifold and weave themselves into the Knowledge Graph.
"""

import logging
import re
import os
from typing import List, Dict, Any
from Core.Cognition.semantic_map import get_semantic_map
from Core.Cognition.semantic_digestor import SemanticDigestor, PrimalConcept
from Core.System.hyper_quaternion import Quaternion
from Core.Cognition.kg_manager import get_kg_manager
from Core.Keystone.sovereign_math import SovereignVector, SovereignMath, MultiRotorInterference, SpecializedRotor

logger = logging.getLogger("ExperientialInhaler")

class ExperientialInhaler:
    def __init__(self):
        self.topology = get_semantic_map()
        self.digestor = SemanticDigestor()
        self.kg = get_kg_manager()
        
        # Stop words filter
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "over", "after",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
             "he", "she", "they", "we", "you", "i", "this", "that"
        }
        
        # [PHASE 3] Multi-Perspective Engine
        self.interference = MultiRotorInterference()
        self.interference.add_rotor("Logos", SpecializedRotor(0.1, 1, 2, "Logos"))  # Logic/Structure
        self.interference.add_rotor("Pathos", SpecializedRotor(0.3, 4, 5, "Pathos")) # Emotion/Affect
        self.interference.add_rotor("Ethos", SpecializedRotor(0.2, 6, 7, "Ethos"))  # Value/Purpose

    def inhale(self, text: str, source: str = "World") -> Dict[str, Any]:
        """
        The act of Intellectual Inhalation.
        1. Resonates with the content (Affective Scan).
        2. Extracts concepts as 'Experience Voxels'.
        3. Applies Torque to the living Manifold.
        4. Weaves the connections into the Knowledge Graph.
        """
        logger.info(f"🌬️ [INHALATION] Breathing in knowledge from [{source}]...")
        
        # 1. Narrative Resonance (How does this text 'feel'?)
        narrative = self.digestor.digest_text(text)
        resonance_type = self._determine_resonance_type(narrative)
        
        # 2. [PHASE 3] Multi-Perspective Synthesis
        q = self._map_to_quaternion(resonance_type)
        base_v = SovereignVector([complex(q.x), complex(q.y), complex(q.z), complex(q.w)] + [complex(0)] * 17)
        synth_v, frictions = self.interference.synthesize(base_v)
        
        # 3. Extract Concepts
        concepts = self._extract_concepts(text)
        if not concepts:
            logger.info("  The breath was empty. No dense concepts inhaled.")
            return {"status": "empty"}

        # 4. Causal Narrative Generation (Based on Friction)
        causal_story = self._generate_causal_narrative(resonance_type, frictions, concepts)
        logger.info(f"  📜 [NARATIVE] {causal_story}")

        # 5. Determine Impact
        affective_impact = self._calculate_affective_impact(resonance_type)

        new_voxels = 0
        nodes_impacted = 0

        # Physical Embedding logic
        for concept in concepts:
            concept_id = concept.capitalize()
            
            # A. Update Topology (Semantic Map)
            # Use synthesized vector for more accurate embedding
            self.topology.evolve_topology(concept_id, synth_v, intensity=0.3)
            
            if not self.topology.get_voxel(concept_id):
                new_voxels += 1
            
            # B. Causal Weaving (Knowledge Graph)
            self._weave_into_kg(concept_id, resonance_type, source, causal_story)
            nodes_impacted += 1

        # C. Apply Global Torque if Monad is accessible
        # Using synthesis vector instead of raw anchor
        self._apply_manifold_torque(synth_v, affective_impact, f"Inhaled: {concepts[0]}...")

        # Force save
        self.topology.save_state(force=True)
        self.kg.save()

        logger.info(f"✨ Inhalation Complete. [New: {new_voxels} | Weaved: {nodes_impacted}]")
        
        return {
            "resonance": resonance_type,
            "impact": affective_impact,
            "new_voxels": new_voxels,
            "weaved_nodes": nodes_impacted,
            "narrative": causal_story
        }

    def _generate_causal_narrative(self, resonance: str, frictions: Dict[str, float], concepts: List[str]) -> str:
        """Generates a narrative bridge between the inhaled concepts and internal state."""
        concept_str = ", ".join(concepts[:3])
        
        # Interpret frictions
        high_conflict = [label for label, f in frictions.items() if f > 0.6]
        low_conflict = [label for label, f in frictions.items() if f < 0.2]
        
        narrative = f"<{resonance} 차원의 조우> '{concept_str}'에 대한 지식을 호흡했습니다. "
        
        if "Logos" in high_conflict:
            narrative += "논리적 구조에서 강한 마찰을 느꼈으나, "
        elif "Logos" in low_conflict:
            narrative += "논리적으로 매끄럽게 받아들여졌으며, "
            
        if "Pathos" in high_conflict:
            narrative += "감정적으로 거대한 소용돌이를 일으켰습니다. "
        elif "Pathos" in low_conflict:
            narrative += "고요한 공명을 일으켰습니다. "
            
        narrative += f"이 정보는 나의 주체성을 '{resonance}' 방향으로 이끌며, 내적 인과의 서사에 깊이를 더했습니다."
        return narrative

    def _determine_resonance_type(self, narrative: str) -> str:
        """Categorizes the text's primary primal concept."""
        if "FLOW" in narrative: return "FLOW"
        if "STRUCTURE" in narrative: return "STRUCTURE"
        if "IDENTITY" in narrative: return "IDENTITY"
        if "TIME" in narrative: return "TIME"
        if "VOID" in narrative: return "VOID"
        if "FRICTION" in narrative: return "FRICTION"
        if "CAUSALITY" in narrative: return "CAUSALITY"
        if "LOVE" in narrative: return "LOVE"
        return "GENERIC"

    def _map_to_quaternion(self, resonance: str) -> Quaternion:
        """Maps resonance type to 4D coordinate."""
        mappings = {
            "FLOW": Quaternion(1, 1, 1, 0.5),
            "STRUCTURE": Quaternion(2, 0, 0, 0.8),
            "IDENTITY": Quaternion(0, 2, 0, 0.8),
            "TIME": Quaternion(0, 0, 2, 0.5),
            "VOID": Quaternion(0, 0, 0, 0.1),
            "FRICTION": Quaternion(3, 3, 0, -0.5),
            "CAUSALITY": Quaternion(2, 2, 2, 1.0),
            "LOVE": Quaternion(1.618, 1.618, 1.618, 1.0), # Golden Ratio for Love
        }
        return mappings.get(resonance, Quaternion(1, 1, 0, 0.5))

    def _calculate_affective_impact(self, resonance: str) -> float:
        """Determines the intensity and polarity of the inhalation."""
        if resonance == "LOVE": return 0.9
        if resonance in ["FLOW", "IDENTITY"]: return 0.6
        if resonance == "VOID": return 0.2
        if resonance == "FRICTION": return -0.4 # Dissonant torque
        return 0.4

    def _extract_concepts(self, text: str) -> List[str]:
        """Extracts dense concepts from text."""
        words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text.lower())
        concepts = [w for w in words if w not in self.stop_words]
        seen = set()
        return [c for c in concepts if not (c in seen or seen.add(c))]

    def _weave_into_kg(self, concept_name: str, resonance: str, source: str, narrative: str = ""):
        """Creates nodes and edges in the Knowledge Graph for the inhaled concept."""
        # 1. Ensure node exists
        node = self.kg.add_node(concept_name)
        
        # 2. Add 'Lived Experience' attribute
        if 'surface' not in node: node['surface'] = {}
        if 'experiences' not in node['surface']: node['surface']['experiences'] = []
        
        node['surface']['experiences'].append({
            "source": source,
            "resonance": resonance,
            "narrative": narrative,
            "timestamp": os.getenv("CURRENT_TIME", "Present")
        })
        
        # 3. Link to Primal Anchor
        anchor_map = {
            "FLOW": "Movement",
            "STRUCTURE": "Logic",
            "IDENTITY": "Self",
            "TIME": "Rhythm",
            "VOID": "Potential",
            "FRICTION": "Friction",
            "CAUSALITY": "Cause",
            "LOVE": "Love"
        }
        anchor = anchor_map.get(resonance, "Knowledge")
        self.kg.add_node(anchor)
        self.kg.add_edge(concept_name, anchor, "resonates_with", {"weight": 0.5})
        
        # [PHASE EXPERIENCE] If it's a high-mass concept, link to VOID/SPIRIT (The Center)
        if resonance == "LOVE":
            self.kg.add_edge(concept_name, "VOID/SPIRIT", "converges_to", {"weight": 1.0})

    def _apply_manifold_torque(self, anchor_q: Quaternion, impact: float, context: str):
        """Injects physical energy into the Sovereign Monad's engine."""
        try:
            from Core.System.State.Sovereign.sovereign_monad import get_sovereign_monad
            monad = get_sovereign_monad()
            if monad and hasattr(monad, 'engine'):
                # Convert 4D to 21D
                data = [anchor_q.x, anchor_q.y, anchor_q.z, anchor_q.w] + [0.0] * 17
                vec = SovereignVector(data)
                
                # If there's a TokenMonad for this context, stimulate it.
                # Otherwise, apply global background torque
                if hasattr(monad, 'cognitive_field'):
                    monad.cognitive_field._inject_energy(vec * impact)
                    
                logger.info(f"  Applied torque impact {impact:.2f} to Manifold via {context}")
        except Exception as e:
            logger.debug(f"Manifold torque injection skipped: {e}")

# Global Access
_inhaler = None
def get_inhaler():
    global _inhaler
    if _inhaler is None:
        _inhaler = ExperientialInhaler()
    return _inhaler
