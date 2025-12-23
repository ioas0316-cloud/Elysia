
import logging
import uuid
from typing import List, Tuple, Dict, Any, Optional
from .causal_narrative_engine import (
    CausalKnowledgeBase, 
    CausalNode, 
    CausalLink, 
    CausalRelationType
)

logger = logging.getLogger("ConceptSynthesizer")

class ConceptSynthesizer:
    """
    Concept Synthesizer (Idea Breeding Engine)
    =========================================
    
    Philosophy: Thesis + Antithesis -> Synthesis.
    Intelligence is the capacity to combine disparate concepts into new emergent wholes.
    """
    
    def __init__(self, knowledge_base: CausalKnowledgeBase):
        self.kb = knowledge_base

    def find_resonant_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find pairs of nodes that resonate but are not directly linked.
        """
        pairs = []
        node_ids = list(self.kb.nodes.keys())
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id_a, id_b = node_ids[i], node_ids[j]
                
                # Rule 1: Must resonate above threshold
                score = self.kb.calculate_resonance(id_a, id_b)
                if score < threshold:
                    continue
                
                # Rule 2: Must not have a direct link already (we want leap-frog breeding)
                # Check outgoing from A to B and B to A
                link_exists = False
                for link_id in self.kb.outgoing.get(id_a, []):
                    link = self.kb.links.get(link_id)
                    if link and link.target_id == id_b:
                        link_exists = True
                        break
                
                if not link_exists:
                    for link_id in self.kb.outgoing.get(id_b, []):
                        link = self.kb.links.get(link_id)
                        if link and link.target_id == id_a:
                            link_exists = True
                            break
                
                if not link_exists:
                    pairs.append((id_a, id_b, score))
                    
        # Sort by resonance score descending
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def synthesize(self, id_a: str, id_b: str, resonance_score: float) -> Optional[CausalNode]:
        """
        Breed two concepts into a new one.
        """
        node_a = self.kb.nodes.get(id_a)
        node_b = self.kb.nodes.get(id_b)
        
        if not node_a or not node_b:
            return None
            
        child_id = f"syn_{uuid.uuid4().hex[:8]}"
        
        # 1. Semantic Synthesis (Simple for now: "A + B")
        # In the future, this could be an LLM-driven synthesis.
        child_description = f"Synthesis of {node_a.description} and {node_b.description}"
        
        # 2. Attribute Inheritance
        child_valence = (node_a.emotional_valence + node_b.emotional_valence) / 2.0
        
        # 3. Sensory Signature Blending
        all_senses = set(node_a.sensory_signature.keys()) | set(node_b.sensory_signature.keys())
        child_signature = {}
        for s in all_senses:
            val_a = node_a.sensory_signature.get(s, 0.0)
            val_b = node_b.sensory_signature.get(s, 0.0)
            child_signature[s] = (val_a + val_b) / 2.0
            
        # 4. Create Child Node
        child_node = CausalNode(
            id=child_id,
            description=child_description,
            is_state=node_a.is_state, # Simplified
            concepts=list(set(node_a.concepts + node_b.concepts)),
            sensory_signature=child_signature,
            emotional_valence=child_valence,
            experience_count=1,
            importance=(node_a.importance + node_b.importance) / 2.0 + 0.1 # Emergent bonus
        )
        
        # 5. Integrate into KB
        self.kb.add_node(child_node)
        
        # 6. Create Parental Links (The "Nervous System" expands)
        self.kb.add_link(
            source_id=id_a,
            target_id=child_id,
            relation=CausalRelationType.ASSOCIATED_WITH,
            strength=resonance_score,
            description="Parent of Synthesis"
        )
        self.kb.add_link(
            source_id=id_b,
            target_id=child_id,
            relation=CausalRelationType.ASSOCIATED_WITH,
            strength=resonance_score,
            description="Parent of Synthesis"
        )
        
        logger.info(f"✨ Synthesized new concept: {child_id} ({child_description}) from {id_a} and {id_b}")
        return child_node

    def run_synthesis_cycle(self, limit: int = 5, threshold: float = 0.7) -> List[str]:
        """
        Execute a single generation of idea breeding.
        """
        pairs = self.find_resonant_pairs(threshold=threshold)
        created_ids = []
        
        for i in range(min(limit, len(pairs))):
            id_a, id_b, score = pairs[i]
            child = self.synthesize(id_a, id_b, score)
            if child:
                created_ids.append(child.id)
                
        return created_ids
