"""
Causal Narrative Engine -          
                                                                              

                        :
-      (         )         
-                
-                                  

                                                                               
            (Dimensional Expansion Structure)                             
                                                                              
                                                                               
    (Point)    (Line)    (Plane)     (Space)     /  (Law)             
                                                                               
   1.   (Point) -                                                         
           : " ", "   ", "  "                                         
                                                                               
   2.   (Line) -                                                          
             : "       "                                               
                                                                        
                                                                               
   3.   (Plane) -   /                                                     
              : "  +   +          +   "                          
                                                                        
                                                                               
   4.    (Space) -    /                                                
              : "     ", "     "                                   
                                                                        
                                                                               
   5.    (Law/Principle) -                                              
                    : "   ", "      "                         
                                                                           
                                                                               

[                  (Thought Universe & Concept Node Mutual Correction)]

                   .
    (Thought Universe)            ,     ,      .

1.       (Bottom-Up Correction)
   -                  
   - "         "   "           " (     )

2.       (Top-Down Correction)
   -                 
   - "      "     "    "        

3.       (Lateral Correction)
   -                  
   - "   "  "   "            

[          (Providential Causal Structure)]

1.         (Cause   Effect)
   -                  
   - "       "      "               "

2.          (Condition   Possibility)
   - "                 "
   -          

3.         (Purpose   Means)
   - "                      "
   -        

4.     (Counterfactual)
   - "                            "
   -        
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import defaultdict
from enum import Enum
import logging
import time
import random
import os

from Core.L5_Mental.M1_Cognition.metacognition import MaturityModel, CognitiveMetrics
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L1_Foundation.Foundation.hyper_cosmos import HyperCosmos, UnifiedMonad, Unified12DVector

logger = logging.getLogger("CausalNarrativeEngine")


# ============================================================================
#       (Dimensional Hierarchy)
#                    
# ============================================================================

class DimensionLevel(Enum):
    """
             
    
           ,        ,     ,             
    """
    POINT = 0      #  :         
    LINE = 1       #  :            (  )
    PLANE = 2      #  :               
    SPACE = 3      #   :                /   
    LAW = 4        #   :                


@dataclass
class DimensionalEntity:
    """
           -                 
    
                           ,
      /                 .
    """
    id: str
    level: DimensionLevel
    description: str = ""
    
    #          (              )
    parent_ids: List[str] = field(default_factory=list)
    
    #          (                 )
    child_ids: List[str] = field(default_factory=list)
    
    #            
    confidence: float = 1.0
    experience_count: int = 0
    last_updated: float = 0.0
    
    #       (                )
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_level_name(self) -> str:
        """        """
        names = {
            DimensionLevel.POINT: " (Point)",
            DimensionLevel.LINE: " (Line)",
            DimensionLevel.PLANE: " (Plane)",
            DimensionLevel.SPACE: "  (Space)",
            DimensionLevel.LAW: "  (Law)",
        }
        return names.get(self.level, "      ")


# ============================================================================
#   (Point) -      
# ============================================================================

@dataclass
class ConceptPoint(DimensionalEntity):
    """
      (Point) -              
    
     : " ", "   ", "  ", " "
    
               ,   /            
    """
    
    #       (8  )
    sensory_signature: Dict[str, float] = field(default_factory=dict)
    
    #       (-1 ~ +1)
    emotional_valence: float = 0.0
    
    #        (       "     ")
    activation: float = 0.0
    
    #       
    concept_type: str = "general"  # "object", "action", "state", "relation", "abstract"
    
    def __post_init__(self):
        self.level = DimensionLevel.POINT


# ============================================================================
#   (Line) -      
# ============================================================================

@dataclass
class CausalLine(DimensionalEntity):
    """
      (Line) -              
    
     : "       ", "        "
    
             ,                   
    """
    
    #         
    source_point_id: str = ""
    target_point_id: str = ""
    
    #      
    relation_type: str = "causes"  # "causes", "enables", "prevents", "follows", etc.
    
    #       (0 ~ 1)
    strength: float = 1.0
    
    #     (                )
    conditions: List[str] = field(default_factory=list)
    
    #        (                )
    exceptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.level = DimensionLevel.LINE


# ============================================================================
#   (Plane) -   /  
# ============================================================================

@dataclass
class ContextPlane(DimensionalEntity):
    """
      (Plane) -              
    
     : "            " ( ,  ,   ,    ,        )
    
                      
    """
    
    #             
    line_ids: List[str] = field(default_factory=list)
    
    #             (      )
    point_ids: List[str] = field(default_factory=list)
    
    #       
    context_type: str = "situation"  # "situation", "scenario", "episode"
    
    #         (             )
    emotional_tone: float = 0.0
    
    #       
    temporal_span: str = "instant"  # "instant", "short", "long", "repeated"
    
    #       /  
    lesson: str = ""
    
    def __post_init__(self):
        self.level = DimensionLevel.PLANE


# ============================================================================
#    (Space) -    /   
# ============================================================================

@dataclass
class SchemaSpace(DimensionalEntity):
    """
       (Space) -               /   
    
     : "         ", "         ", "            "
    
                          
    """
    
    #              
    plane_ids: List[str] = field(default_factory=list)
    
    #        
    schema_type: str = "behavior"  # "behavior", "belief", "emotion", "social"
    
    #            (                )
    core_patterns: List[str] = field(default_factory=list)
    
    #       (            )
    applicability: List[str] = field(default_factory=list)
    
    #     (                        )
    predictive_power: float = 0.0
    
    def __post_init__(self):
        self.level = DimensionLevel.SPACE

@dataclass
class EpistemicSpace(SchemaSpace):
    """
            (Epistemic Space) -               
    
                ,        '  '      .
      (Density)    (Resistance)     ,    (Methodology)               .
    """
    
    #        (   /   ) -                   
    density: float = 1.0
    
    #        (                      )
    #  : "EMPIRICAL" (   ), "LOGICAL" (   ), "INTUITIVE" (   )
    methodologies: List[str] = field(default_factory=list)
    
    #        (                )
    #  :     -> ["Time", "Space", "Mass", "Energy"]
    internal_dimensions: List[str] = field(default_factory=list)



@dataclass
class CognitiveMetrics:
    """
          (Cognitive Metrics) -               
    
         '  '       :
    1. Differentiation (   ):        ? (   ,    )
    2. Integration (   /  ):        ? (     ,      )
    3. Abstraction (   ):                     
    """
    differentiation: float = 0.0  # 0~1:     ->         
    integration: float = 0.0      # 0~1:     ->        
    abstraction: float = 0.0      # 0~1:     ->      
    
    def get_resonance_score(self) -> float:
        """          (                       )"""
        return (self.differentiation * self.integration)

@dataclass
class MaturityModel:
    """
           (Maturity Model) -             
    """
    level_name: str
    required_metrics: CognitiveMetrics
    description: str

    @staticmethod
    def get_standard_model() -> Dict[str, 'MaturityModel']:
        return {
            "CHILD": MaturityModel("CHILD", CognitiveMetrics(0.2, 0.2, 0.1), "Simple linear causality"),
            "ADOLESCENT": MaturityModel("ADOLESCENT", CognitiveMetrics(0.5, 0.4, 0.3), "Beginning to see context but rigid"),
            "ADULT": MaturityModel("ADULT", CognitiveMetrics(0.8, 0.8, 0.7), "Nuanced, Paradox-holding, Principle-based"),
            "SAGE": MaturityModel("SAGE", CognitiveMetrics(0.95, 0.95, 0.95), "Universal resonance")
        }

# ============================================================================
#    (Law) -       
# ============================================================================

@dataclass
class UniversalLaw(DimensionalEntity):
    """
       (Law) -                   
    
     : "   ", "      ", "  -     "
    
                          
    """
    
    #               
    space_ids: List[str] = field(default_factory=list)
    
    #       
    law_type: str = "causal"  # "causal", "conservation", "symmetry", "teleological"
    
    #         (    )
    formulation: str = ""
    
    #          (        ,          )
    is_absolute: bool = False
    
    #    (              )
    supporting_evidence: List[str] = field(default_factory=list)
    
    #    (              )
    counter_examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.level = DimensionLevel.LAW


# ============================================================================
#          (Types of Causal Relations)
# ============================================================================

class CausalRelationType(Enum):
    """        """
    #       
    CAUSES = "causes"           # A  B      (A   B)
    PREVENTS = "prevents"       # A  B     (A   B)
    ENABLES = "enables"         # A  B          
    
    #       
    CONDITIONAL = "conditional"  # A   B (if A then B)
    NECESSARY = "necessary"      # A    B    (B requires A)
    SUFFICIENT = "sufficient"    # A      B    (A is enough for B)
    
    #        
    MEANS_TO = "means_to"        # A  B        (A is means to B)
    PURPOSE_OF = "purpose_of"    # A      B (purpose of A is B)
    
    #       
    PRECEDES = "precedes"        # A  B      (A before B)
    FOLLOWS = "follows"          # A  B    (A after B)
    SIMULTANEOUS = "simultaneous"  # A  B    
    
    #    /       (Experiential/Semantic)
    CORRELATES = "correlates"      # A  B          
    ASSOCIATED_WITH = "associated_with" # A  B      (      )
    MOTIVATES = "motivates"        # A  B(  /  )       
    CONTRASTS_WITH = "contrasts_with" # A  B      (             )




# ============================================================================
#       (Causal Node) -             /  
# ============================================================================

@dataclass
class CausalNode:
    """
          -           '  '    '  '
    
    " "      "        "
    "   "      "        "
    
      (State)    (Event)    :
    -   :         ( :     ,     )
    -   :         ( :      ,   )
    """
    
    id: str
    description: str              #       
    
    #      
    is_state: bool = True         # True:   , False:   
    
    #         (             )
    concepts: List[str] = field(default_factory=list)

    #       :                   (EpistemicSpace)
    inner_space_id: Optional[str] = None
    
    #   /     
    sensory_signature: Dict[str, float] = field(default_factory=dict)
    emotional_valence: float = 0.0  # -1 (  ) ~ +1 ( )
    
    #     (        )
    agent: Optional[str] = None
    
    #      
    timestamp: float = 0.0
    duration: float = 0.0  #          ,             
    
    #    /   
    activation: float = 1.0
    importance: float = 1.0
    
    #      
    experience_count: int = 0
    
    # Epistemic Grounding (Phase 18.5)
    # [DELUSION, HYPOTHESIS, TRUTH]
    epistemic_status: str = "TRUTH" # Default for existing/sensory nodes
    internal_law: Optional[str] = None # The underlying principle/law
    
    def get_valence_description(self) -> str:
        """      """
        if self.emotional_valence > 0.5:
            return "      "
        elif self.emotional_valence > 0:
            return "   "
        elif self.emotional_valence < -0.5:
            return "      "
        elif self.emotional_valence < 0:
            return "   "
        else:
            return "   "


# ============================================================================
#       (Causal Link) -            
# ============================================================================

@dataclass
class CausalLink:
    """
          -               
    
              , " "  "   "    
    """
    
    source_id: str                #   /  /  
    target_id: str                #   /   /  
    relation: CausalRelationType  #      
    
    #       (0-1)
    strength: float = 1.0
    
    #     (              )
    confidence: float = 1.0
    
    #    (                 )
    conditions: List[str] = field(default_factory=list)
    
    #       (         )
    experience_count: int = 1
    
    #            
    counterfactual_tested: bool = False
    counterfactual_confirmed: bool = False
    
    #       
    description: str = ""
    
    def strengthen(self, amount: float = 0.1):
        """         (     )"""
        self.strength = min(1.0, self.strength + amount * (1 - self.strength))
        self.experience_count += 1
    
    def weaken(self, amount: float = 0.1):
        """         (  )"""
        self.strength = max(0.0, self.strength - amount)
    
    def get_description(self) -> str:
        """           """
        relation_descriptions = {
            CausalRelationType.CAUSES: "   ",
            CausalRelationType.PREVENTS: "   ",
            CausalRelationType.ENABLES: "         ",
            CausalRelationType.CONDITIONAL: "  ",
            CausalRelationType.NECESSARY: "         ",
            CausalRelationType.SUFFICIENT: "        ",
            CausalRelationType.MEANS_TO: "     ",
            CausalRelationType.PURPOSE_OF: "     ",
            
            CausalRelationType.CORRELATES: "      ",
            CausalRelationType.ASSOCIATED_WITH: "          ",
            CausalRelationType.MOTIVATES: "          ",
            CausalRelationType.CONTRASTS_WITH: "         (    )",


            CausalRelationType.PRECEDES: "     ",
            CausalRelationType.FOLLOWS: "   ",
            CausalRelationType.SIMULTANEOUS: "     ",
        }
        return self.description or relation_descriptions.get(self.relation, "    ")


# ============================================================================
#       (Causal Chain) -           
# ============================================================================

@dataclass
class CausalChain:
    """
          -                  
    
     :                            
    
            "   "       .
    """
    
    id: str
    name: str = ""
    
    #              (     !)
    node_sequence: List[str] = field(default_factory=list)
    
    #           
    links: List[CausalLink] = field(default_factory=list)
    
    #          
    initial_state: Optional[str] = None  #       (  )
    final_state: Optional[str] = None    #      (  )
    
    #       
    is_goal_directed: bool = False  #         
    goal: Optional[str] = None      #   
    
    #           
    emotional_arc: List[float] = field(default_factory=list)
    
    #      
    experience_count: int = 1
    
    #         (        )
    success_rate: float = 0.0
    
    def get_length(self) -> int:
        return len(self.node_sequence)
    
    def get_emotional_trajectory(self) -> str:
        """        """
        if not self.emotional_arc:
            return "  "
        
        start = self.emotional_arc[0] if self.emotional_arc else 0
        end = self.emotional_arc[-1] if self.emotional_arc else 0
        
        if end > start + 0.3:
            return "   (     )"
        elif end < start - 0.3:
            return "   (     )"
        else:
            return "  "
    """Represents a linear sequence of causal links (1D)."""
    id: str
    node_sequence: List[str]
    links: List[CausalLink]
    confidence_score: float = 0.0

    def __post_init__(self):
        # Calculate average confidence
        if self.links:
            self.confidence_score = sum(link.strength for link in self.links) / len(self.links)

@dataclass
class ContextPlane:
    """
    Represents a 2D plane of reasoning formed by intersecting causal chains.
    It captures a broader 'situation' or 'context' (e.g., 'Rainy Day' context formed by Rain->Wet and Rain->Cold).
    """
    id: str
    anchor_node: str  # The node where chains intersect (e.g., "Rain")
    component_chains: List[CausalChain]
    related_concepts: Set[str] = field(default_factory=set)

    def integrate_chain(self, chain: CausalChain):
        """Adds a chain to this plane and updates related concepts."""
        if chain not in self.component_chains:
            self.component_chains.append(chain)
            self.related_concepts.update(chain.node_sequence)

class CausalKnowledgeBase:
    """
    Causal Knowledge Base
    Stores nodes, links, chains, and context planes.
    """
    def __init__(self, persistence_path: Optional[str] = None):
        self.nodes: Dict[str, CausalNode] = {}
        self.links: Dict[str, CausalLink] = {}
        self.outgoing: Dict[str, List[str]] = defaultdict(list)
        self.incoming: Dict[str, List[str]] = defaultdict(list)
        self.chains: List[CausalChain] = []
        self.planes: List[ContextPlane] = []
        self.persistence_path = persistence_path
        self.spatial_index = HyperCosmos() # Merkava Engine
        
        if self.persistence_path:
            self.load_narrative()

    def save_narrative(self):
        """Persists the causal knowledge base to disk."""
        if not self.persistence_path: return
        
        from dataclasses import asdict
        import json
        
        data = {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "links": {k: {**asdict(v), "relation": v.relation.value} for k, v in self.links.items()},
        }
        
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"📚 [CAUSAL_KB] Narrative saved to {self.persistence_path}")
        except Exception as e:
            logger.error(f"❌ [CAUSAL_KB] Save failed: {e}")

    def load_narrative(self):
        """Loads the causal knowledge base from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path): return
        
        import json
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for nid, n_data in data.get("nodes", {}).items():
                self.nodes[nid] = CausalNode(**n_data)
                
            for lid, l_data in data.get("links", {}).items():
                rel_str = l_data.pop("relation")
                l_data["relation"] = CausalRelationType(rel_str)
                link = CausalLink(**l_data)
                self.links[lid] = link
                self.outgoing[link.source_id].append(lid)
                self.incoming[link.target_id].append(lid)
                
            logger.info(f"📚 [CAUSAL_KB] Narrative loaded. Nodes: {len(self.nodes)}")
        except Exception as e:
            logger.error(f"❌ [CAUSAL_KB] Load failed: {e}")

    
    def add_node(self, node: CausalNode) -> CausalNode:
        """             """
        if node.id in self.nodes:
            #            (     )
            existing = self.nodes[node.id]
            existing.experience_count += 1
            existing.activation = min(1.0, existing.activation + 0.1)
        else:
            self.nodes[node.id] = node
            
            # [PHASE 30: SPATIAL INDEXING]
            # Project the node into the 12D HyperCosmos
            # Dynamic mapping based on concepts to create distinct spatial locations
            
            # Default vector
            v_args = {
                "foundation": 0.5,
                "meaning": node.importance,
                "causal": 0.9, # Default high causal for all narratives
                "phenomena": node.emotional_valence
            }
            
            # Enrich based on concepts
            for c in node.concepts:
                cl = c.lower()
                if "connection" in cl or "love" in cl: 
                    # Amplify Love dimensions significantly
                    v_args["spirit"] = 1.0 
                    v_args["phenomena"] = 1.0
                    v_args["foundation"] = 0.2 # Love is fluid, less grounded
                if "structure" in cl or "logic" in cl: 
                    v_args["mental"] = 1.0 
                    v_args["structure"] = 1.0
                    v_args["causal"] = 1.0
                if "will" in cl or "power" in cl: v_args["will"] = 1.0
                if "self" in cl: v_args["foundation"] = 1.0

            vector_12d = Unified12DVector.create(**v_args)
            self.spatial_index.inhale(UnifiedMonad(node.id, vector_12d))
            logger.info(f"✨ [CAUSAL] Node added & Spatialized: {node.id}")
        
        return self.nodes[node.id]
    
    def add_link(
        self,
        source_id: str,
        target_id: str,
        relation: CausalRelationType,
        strength: float = 1.0,
        conditions: List[str] = None,
        description: str = ""
    ) -> CausalLink:
        """              """
        link_id = f"{source_id}_{relation.value}_{target_id}"
        
        if link_id in self.links:
            #         
            self.links[link_id].strengthen()
        else:
            #        
            link = CausalLink(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                strength=strength,
                conditions=conditions or [],
                description=description
            )
            self.links[link_id] = link
            self.outgoing[source_id].append(link_id)
            self.incoming[target_id].append(link_id)
        
        return self.links[link_id]

    # ============================================================================
    # Dimensional Expansion (Phase 9) - Context Planes
    # ============================================================================

    # ============================================================================
    # Dimensional Expansion (Phase 9) - Context Planes
    # ============================================================================

    def detect_intersections(self, chain: CausalChain) -> List[ContextPlane]:
        """
        Detects if the given chain intersects with existing chains or planes.
        If an intersection is found (shared node), it forms or updates a ContextPlane.
        """
        affected_planes = []
        
        # 1. Check against existing planes first
        for plane in self.planes:
            # Check if any node in the chain exists in the plane's related concepts
            intersection = set(chain.node_sequence) & plane.related_concepts
            if intersection:
                plane.integrate_chain(chain)
                affected_planes.append(plane)
                # Note: We update the anchor if this new intersection is significant? 
                # For now, keep original anchor.

        # 2. Check against other individual chains to form NEW planes
        if not affected_planes: # Only look for new planes if not already integrated? Or always?
            # Let's looking for new intersections effectively.
            for other_chain in self.chains:
                if other_chain.id == chain.id:
                    continue
                
                # Check for shared nodes (excluding potentially generic ones if we had a stop-list, but for now strict)
                intersection = set(chain.node_sequence) & set(other_chain.node_sequence)
                
                if intersection:
                    # Found a common node! Create a new plane.
                    anchor = list(intersection)[0] # Pick the first intersection as anchor for now
                    
                    # Check if these two are already in a plane together (optimization)
                    # For now, simplify: create new plane.
                    
                    new_plane_id = f"plane_{anchor}_{len(self.planes)}"
                    new_plane = ContextPlane(
                        id=new_plane_id,
                        anchor_node=anchor,
                        component_chains=[chain, other_chain],
                        related_concepts=set(chain.node_sequence) | set(other_chain.node_sequence)
                    )
                    self.planes.append(new_plane)
                    affected_planes.append(new_plane)
        
        return affected_planes

    # ============================================================================
    # Phase 30: Spatial Retrieval (Merkava)
    # ============================================================================

    def query_related_nodes(self, concept_vector: List[float], top_k: int = 5) -> List[CausalNode]:
        """
        [PHASE 30: MERKAVA RETRIEVAL]
        Uses spatial resonance in HyperCosmos instead of linear scanning.
        """
        if not self.spatial_index.monads:
            return []
            
        # 1. Create Query Vector
        q_vec = Unified12DVector.create()
        # Map 7D input to 12D query roughly
        if len(concept_vector) >= 4:
             # Input Vector assumed: [Foundation, Metabolism, Phenomena, Causal, Mental, Structure, Spirit]
             # 12D Vector map: 
             # 0: Foundation, 1: Metabolism, 2: Phenomena, 3: Causal
             # 4: Mental, 5: Structure, 6: Spirit, 7: Will, 8: Intent...
             
             q_vec.data[2] = concept_vector[2] # Phenomena (Love)
             q_vec.data[3] = concept_vector[3] # Causal 
             
             # Also map Mental and Structure if available
             if len(concept_vector) > 5:
                 q_vec.data[4] = concept_vector[4] # Mental
                 q_vec.data[5] = concept_vector[5] # Structure
             
        # 2. Spatial Query uses JAX-accelerated resonance if available
        spatial_results = self.spatial_index.query_resonance(q_vec, top_k=top_k)
        
        # 3. Retrieve Nodes
        related_nodes = []
        for nid, score in spatial_results:
            if nid in self.nodes:
                related_nodes.append(self.nodes[nid])
                
        return related_nodes

    # ============================================================================
    # Phase 10: Resonance & Fuzzy Logic
    # ============================================================================

    def calculate_resonance(self, node_id_a: str, node_id_b: str) -> float:
        """
        Calculates the resonance (similarity/affinity) score between two nodes.
        Score range: 0.0 to 1.0
        Based on:
        1. Emotional Valence Similarity
        2. Description/Keyword Overlap
        3. Shared Concepts
        """
        node_a = self.nodes.get(node_id_a)
        node_b = self.nodes.get(node_id_b)
        
        if not node_a or not node_b:
            return 0.0
            
        # 1. Emotional Resonance
        # High resonance if valences are similar.
        valence_diff = abs(node_a.emotional_valence - node_b.emotional_valence)
        emotional_score = max(0.0, 1.0 - (valence_diff / 2.0)) # Normalize diff (max 2.0 -> 0.0)
        
        # 2. Semantic Overlap (Description Words)
        words_a = set(node_a.description.lower().split())
        words_b = set(node_b.description.lower().split())
        
        if not words_a or not words_b:
            semantic_score = 0.0
        else:
            intersection = words_a & words_b
            union = words_a | words_b
            semantic_score = len(intersection) / len(union)
            
        # 3. Concept Overlap
        concepts_a = set(node_a.concepts)
        concepts_b = set(node_b.concepts)
        
        concept_score = 0.0
        if concepts_a or concepts_b:
             union_c = concepts_a | concepts_b
             if union_c:
                concept_score = len(concepts_a & concepts_b) / len(union_c)
                
        # Weighted Total (Adjust weights as needed)
        # Balanced Approach: Emotion (0.5) + Concept (0.3) + Semantic (0.2)
        final_score = (emotional_score * 0.5) + (semantic_score * 0.2) + (concept_score * 0.3)
        
        logger.info(f"   [DEBUG] ResCalc '{node_id_a}'<->'{node_id_b}': E({emotional_score:.2f}*0.5={emotional_score*0.5:.2f}) + S({semantic_score:.2f}*0.2={semantic_score*0.2:.2f}) + C({concept_score:.2f}*0.3={concept_score*0.3:.2f}) = {final_score:.2f}")

        # Boost if very high emotional match AND some concept overlap
        if emotional_score > 0.8 and concept_score > 0.0:
            final_score += 0.1
            
        return min(1.0, final_score)


    def find_resonant_nodes(self, target_node_id: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Finds all nodes that resonate with the target node above a certain threshold.
        Returns list of (node_id, score).
        """
        results = []
        for node_id in self.nodes:
            if node_id == target_node_id:
                continue
            
            score = self.calculate_resonance(target_node_id, node_id)
            if score >= threshold:
                results.append((node_id, score))
                
        # Sort by score desc
        results.sort(key=lambda x: x[1], reverse=True)
        return results


    def infer_contextual_link(self, start_node: str) -> List[str]:
        """
        Performs lateral/spatial inference.
        "Given 'start_node', what else is in this context plane?"
        e.g., Rain -> Wet. Rain -> Cold. 
        Input: Wet. Inferred: Cold (via Rain context).
        """
        inferences = []
        for plane in self.planes:
            if start_node in plane.related_concepts:
                # This node is part of this plane.
                # Return other significant concepts in this plane (siblings).
                # Exclude the node itself and immediate parents/children if possible to find *lateral* links.
                
                for concept in plane.related_concepts:
                    if concept != start_node:
                        inferences.append(f"In the context of '{plane.anchor_node}', '{start_node}' is related to '{concept}'.")
                        
        return inferences
    
    def get_causes_of(self, node_id: str) -> List[Tuple[CausalNode, CausalLink]]:
        """           (         )"""
        results = []
        for link_id in self.incoming.get(node_id, []):
            link = self.links[link_id]
            if link.relation in [CausalRelationType.CAUSES, CausalRelationType.ENABLES]:
                source_node = self.nodes.get(link.source_id)
                if source_node:
                    results.append((source_node, link))
        return results
    
    def get_effects_of(self, node_id: str) -> List[Tuple[CausalNode, CausalLink]]:
        """           (         )"""
        results = []
        for link_id in self.outgoing.get(node_id, []):
            link = self.links[link_id]
            if link.relation in [CausalRelationType.CAUSES, CausalRelationType.ENABLES]:
                target_node = self.nodes.get(link.target_id)
                if target_node:
                    results.append((target_node, link))
        return results
    
    def trace_causal_chain(
        self,
        start_id: str,
        max_depth: int = 5
    ) -> List[CausalChain]:
        """                    """
        chains = []
        
        def dfs(current_id: str, path: List[str], links: List[CausalLink], depth: int):
            if depth >= max_depth:
                if len(path) >= 2:
                    chain = CausalChain(
                        id=f"chain_{len(chains)}",
                        node_sequence=path.copy(),
                        links=links.copy()
                    )
                    chains.append(chain)
                return
            
            for link_id in self.outgoing.get(current_id, []):
                link = self.links[link_id]
                if link.target_id not in path:  #      
                    path.append(link.target_id)
                    links.append(link)
                    dfs(link.target_id, path, links, depth + 1)
                    path.pop()
                    links.pop()
            
            #                (     )
            if len(path) >= 2 and not self.outgoing.get(current_id):
                chain = CausalChain(
                    id=f"chain_{len(chains)}",
                    node_sequence=path.copy(),
                    links=links.copy()
                )
                chains.append(chain)
        
        dfs(start_id, [start_id], [], 0)
        return chains
    
    def find_path(self, source_id: str, target_id: str) -> Optional[CausalChain]:
        """                  (BFS)"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        visited = {source_id}
        queue = [(source_id, [source_id], [])]
        
        while queue:
            current_id, path, links = queue.pop(0)
            
            if current_id == target_id:
                return CausalChain(
                    id=f"path_{source_id}_to_{target_id}",
                    node_sequence=path,
                    links=links
                )
            
            for link_id in self.outgoing.get(current_id, []):
                link = self.links[link_id]
                if link.target_id not in visited:
                    visited.add(link.target_id)
                    queue.append((
                        link.target_id,
                        path + [link.target_id],
                        links + [link]
                    ))
        
        return None
    
    def counterfactual_query(
        self,
        premise_node_id: str,
        premise_negated: bool,
        conclusion_node_id: str
    ) -> Tuple[bool, str]:
        """
               
        
        "   A  (     /    ) B          ?"
        
        Args:
            premise_node_id:      
            premise_negated: True   "    ", False   "    "
            conclusion_node_id:      
        
        Returns:
            (  ,   )
        """
        # A   B           
        path = self.find_path(premise_node_id, conclusion_node_id)
        
        if path is None:
            return (False, f"{premise_node_id}  {conclusion_node_id}             ")
        
        #            
        total_strength = 1.0
        for link in path.links:
            total_strength *= link.strength
        
        if premise_negated:
            # "    "            
            if total_strength > 0.7:
                return (True, f"   {premise_node_id}      , {conclusion_node_id}        (   : {total_strength:.0%})")
            else:
                return (False, f"{premise_node_id}      {conclusion_node_id}                  ")
        else:
            # "    "            
            if total_strength > 0.5:
                return (True, f"   {premise_node_id}      , {conclusion_node_id}        (   : {total_strength:.0%})")
            else:
                return (False, f"{premise_node_id}     {conclusion_node_id}          ")


# ============================================================================
#        (Causal Experience)
# ============================================================================

@dataclass
class CausalExperience:
    """
           -              
    
                    .
    """
    
    id: str
    timestamp: float
    
    #          
    cause_node: CausalNode         #   /     
    effect_node: CausalNode        #   /    
    intermediate_nodes: List[CausalNode] = field(default_factory=list)  #       
    
    #    
    agent_id: str = ""
    
    #   
    success: bool = True           #         
    emotional_outcome: float = 0.0  #        (-1 ~ +1)
    
    #        
    counterfactual_considered: bool = False
    alternative_action: Optional[str] = None
    
    def get_full_sequence(self) -> List[CausalNode]:
        """         """
        return [self.cause_node] + self.intermediate_nodes + [self.effect_node]


# ============================================================================
#           (Causal Narrative Engine)
# ============================================================================

class CausalNarrativeEngine:
    """
             
    
                                      .
                , " "  "   "           .
    
         :
    1.          (Experience-Based Learning)
       -                       
    
    2.         (Counterfactual Reasoning)
       - "   ~   "             
    
    3.           (Goal-Directed Planning)
       -                 
    
    4.        (Emotional Learning)
       -   /                
    """
    
    def __init__(self):
        self.knowledge_base = CausalKnowledgeBase()
        
        #      
        self.experiences: List[CausalExperience] = []
        
        #      
        self.total_experiences = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        #              
        self._initialize_fundamental_causality()
    def synthesize_narrative(self, chain: CausalChain) -> str:
        """
        Synthesizes a narrative paragraph from a CausalChain.
        Transform a list of nodes and links into a coherent story.
        """
        if not chain.node_sequence:
            return "Nothing happened."

        narrative = []
        
        # Start
        start_node = chain.node_sequence[0]
        narrative.append(f"It started with {start_node}.")
        
        # Process links
        for i, link in enumerate(chain.links):
            source = link.source_id
            target = link.target_id
            relation = link.relation
            
            # Transition logic based on relation
            transition = ""
            if relation == CausalRelationType.CAUSES:
                transition = "This caused"
            elif relation == CausalRelationType.ENABLES:
                transition = "This enabled"
            elif relation == CausalRelationType.PREVENTS:
                transition = "However, this prevented"
            elif relation == CausalRelationType.CONDITIONAL:
                transition = "Under these conditions,"
            else:
                transition = "Then,"
            
            # Add sentence
            # "This caused [Target]." or "Then, [Target] happened."
            sentence = f"{transition} {target}."
            narrative.append(sentence)
            
        # Conclusion
        narrative.append(f"Finally, the chain completed at {chain.node_sequence[-1]}.")
        
        return " ".join(narrative)

    def generate_prediction_sentence(self, source: str, target: str) -> str:
        """
        Generates a natural language prediction.
        "If [source]..., then [target]..."
        """
        # Simple template for now
        templates = [
            f"If I encounter '{source}', I expect to see '{target}'.",
            f"'{source}' is likely a precursor to '{target}'.",
            f"Given '{source}', '{target}' should follow.",
            f"The presence of '{source}' suggests '{target}' is near."
        ]
        return random.choice(templates)

    def _initialize_fundamental_causality(self):
        """               """
        #            
        fundamental_causality = [
            #       
            ("fire_contact", "pain", CausalRelationType.CAUSES, "          "),
            ("pain", "avoidance", CausalRelationType.CAUSES, "          "),
            ("avoidance", "safety", CausalRelationType.CAUSES, "         "),
            
            #      
            ("hunger", "seek_food", CausalRelationType.CAUSES, "            "),
            ("seek_food", "find_food", CausalRelationType.ENABLES, "                "),
            ("find_food", "eat", CausalRelationType.ENABLES, "                "),
            ("eat", "satiety", CausalRelationType.CAUSES, "        "),
            ("satiety", "pleasure", CausalRelationType.CAUSES, "           "),
            
            #       
            ("loneliness", "seek_company", CausalRelationType.CAUSES, "                "),
            ("company", "comfort", CausalRelationType.CAUSES, "           "),
            ("help_given", "trust", CausalRelationType.CAUSES, "              "),
            ("trust", "cooperation", CausalRelationType.ENABLES, "                "),
            
            #      
            ("curiosity", "exploration", CausalRelationType.CAUSES, "             "),
            ("exploration", "discovery", CausalRelationType.ENABLES, "             "),
            ("discovery", "knowledge", CausalRelationType.CAUSES, "            "),
            ("knowledge", "prediction", CausalRelationType.ENABLES, "                "),
            
            #        
            ("goal", "planning", CausalRelationType.CAUSES, "            "),
            ("planning", "action", CausalRelationType.ENABLES, "                "),
            ("action", "outcome", CausalRelationType.CAUSES, "           "),
        ]
        
        for source, target, relation, description in fundamental_causality:
            #      
            if source not in self.knowledge_base.nodes:
                self.knowledge_base.add_node(CausalNode(
                    id=source,
                    description=source.replace("_", " "),
                    is_state=True
                ))
            if target not in self.knowledge_base.nodes:
                self.knowledge_base.add_node(CausalNode(
                    id=target,
                    description=target.replace("_", " "),
                    is_state=True
                ))
            
            #         
            self.knowledge_base.add_link(
                source, target, relation,
                strength=0.5,  #       (        )
                description=description
            )
            
        # Seed Experiential Contexts (User Request: Winter -> Hunger/Loneliness)
        self._initialize_experiential_context()

    def _initialize_experiential_context(self):
        """
        Initializes experiential contexts that link disparate concepts through the 'Self'.
        (e.g., Winter -> Hunger, Cold, Loneliness)
        """
        # Simulating a pre-existing experiential plane for "Winter"
        # This allows the system to infer "Hunger" or "Loneliness" from "Winter" 
        # even without a direct physical causal chain, based on 'lived experience'.
        
        winter_plane = ContextPlane(
            id="plane_winter_experiential",
            anchor_node="winter",
            component_chains=[], 
            related_concepts={"winter", "cold", "hunger", "loneliness", "darkness", "need_for_comfort"}
        )
        self.knowledge_base.planes.append(winter_plane)

    def add_node(self, node: CausalNode):
        return self.knowledge_base.add_node(node)
        
    def add_link(self, *args, **kwargs):
        return self.knowledge_base.add_link(*args, **kwargs)

    @property
    def chains(self):
        return self.knowledge_base.chains

    @property
    def planes(self):
        return self.knowledge_base.planes

    def trace_causal_chain(self, start_id: str, max_depth: int = 5) -> List[CausalChain]:
        return self.knowledge_base.trace_causal_chain(start_id, max_depth)

    def detect_intersections(self, chain: CausalChain) -> List[ContextPlane]:
        return self.knowledge_base.detect_intersections(chain)

    def infer_contextual_link(self, start_node: str) -> List[str]:
        return self.knowledge_base.infer_contextual_link(start_node)

    def calculate_resonance(self, node_id_a: str, node_id_b: str) -> float:
        return self.knowledge_base.calculate_resonance(node_id_a, node_id_b)

    def find_resonant_nodes(self, target_node_id: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        return self.knowledge_base.find_resonant_nodes(target_node_id, threshold)

    def find_path(self, source_id: str, target_id: str) -> Optional[CausalChain]:
        return self.knowledge_base.find_path(source_id, target_id)

    def experience_causality(
        self,
        cause_description: str,
        effect_description: str,
        relation: CausalRelationType = CausalRelationType.CAUSES,
        emotional_outcome: float = 0.0,
        success: bool = True,
        agent_id: str = "self"
    ) -> CausalExperience:
        """
                 (         )
        
        Args:
            cause_description:      
            effect_description:      
            relation:         
            emotional_outcome:        (-1 ~ +1)
            success:            
            agent_id:      
        
        Returns:
                CausalExperience
        """
        #              
        cause_id = cause_description.lower().replace(" ", "_")
        effect_id = effect_description.lower().replace(" ", "_")
        
        cause_node = self.knowledge_base.add_node(CausalNode(
            id=cause_id,
            description=cause_description,
            is_state=False,  #   
            agent=agent_id,
            timestamp=time.time()
        ))
        
        effect_node = self.knowledge_base.add_node(CausalNode(
            id=effect_id,
            description=effect_description,
            is_state=True,  #   
            emotional_valence=emotional_outcome,
            agent=agent_id,
            timestamp=time.time()
        ))
        
        #         
        link = self.knowledge_base.add_link(
            cause_id, effect_id, relation,
            description=f"{cause_description}   {effect_description}"
        )
        
        #             
        if success and emotional_outcome > 0:
            link.strengthen(0.2)  #        =      
        elif not success or emotional_outcome < 0:
            link.weaken(0.1)  #        =      
        
        #      
        experience = CausalExperience(
            id=f"exp_{self.total_experiences}",
            timestamp=time.time(),
            cause_node=cause_node,
            effect_node=effect_node,
            agent_id=agent_id,
            success=success,
            emotional_outcome=emotional_outcome
        )
        
        self.experiences.append(experience)
        self.total_experiences += 1
        
        logger.debug(f"  : {cause_description}   {effect_description} "
                    f"(  : {emotional_outcome:.1f},   : {success})")
        
        return experience
    
    def experience_chain(
        self,
        descriptions: List[str],
        emotional_arc: List[float],
        agent_id: str = "self"
    ) -> CausalChain:
        """
                 (            )
        
        Args:
            descriptions:             
            emotional_arc:             
            agent_id:      
        
        Returns:
                CausalChain
        """
        if len(descriptions) < 2:
            raise ValueError("          2     ")
        
        if len(emotional_arc) != len(descriptions):
            emotional_arc = [0.0] * len(descriptions)
        
        #               
        nodes = []
        links = []
        
        for i in range(len(descriptions) - 1):
            exp = self.experience_causality(
                cause_description=descriptions[i],
                effect_description=descriptions[i + 1],
                emotional_outcome=emotional_arc[i + 1],
                agent_id=agent_id
            )
            nodes.append(exp.cause_node.id)
            
            if i == len(descriptions) - 2:
                nodes.append(exp.effect_node.id)
        
        #      
        chain = CausalChain(
            id=f"chain_{len(self.knowledge_base.chains)}",
            name=f"{descriptions[0]}   {descriptions[-1]}",
            node_sequence=nodes,
            initial_state=nodes[0],
            final_state=nodes[-1],
            emotional_arc=emotional_arc
        )
        
        self.knowledge_base.chains[chain.id] = chain
        
        logger.info(f"        : {chain.name} ({len(nodes)}  )")
        
        return chain
    
    def predict_effect(
        self,
        cause_description: str
    ) -> List[Tuple[str, float, CausalRelationType]]:
        """
                    
        
        Args:
            cause_description:      
        
        Returns:
            [(     ,   ,      ), ...]
        """
        cause_id = cause_description.lower().replace(" ", "_")
        
        if cause_id not in self.knowledge_base.nodes:
            return []
        
        effects = self.knowledge_base.get_effects_of(cause_id)
        
        predictions = []
        for effect_node, link in effects:
            predictions.append((
                effect_node.description,
                link.strength * link.confidence,
                link.relation
            ))
        
        predictions.sort(key=lambda x: -x[1])
        return predictions
    
    def find_cause(
        self,
        effect_description: str
    ) -> List[Tuple[str, float, CausalRelationType]]:
        """
                    (         )
        
        Args:
            effect_description:      
        
        Returns:
            [(     ,   ,      ), ...]
        """
        effect_id = effect_description.lower().replace(" ", "_")
        
        if effect_id not in self.knowledge_base.nodes:
            return []
        
        causes = self.knowledge_base.get_causes_of(effect_id)
        
        inferences = []
        for cause_node, link in causes:
            inferences.append((
                cause_node.description,
                link.strength * link.confidence,
                link.relation
            ))
        
        inferences.sort(key=lambda x: -x[1])
        return inferences
    
    def plan_to_achieve(
        self,
        current_state: str,
        goal_state: str
    ) -> Optional[CausalChain]:
        """
                     (        )
        
        Args:
            current_state:      
            goal_state:      
        
        Returns:
                  (  )    None
        """
        current_id = current_state.lower().replace(" ", "_")
        goal_id = goal_state.lower().replace(" ", "_")
        
        return self.knowledge_base.find_path(current_id, goal_id)
    
    def counterfactual_reasoning(
        self,
        premise: str,
        premise_negated: bool,
        conclusion: str
    ) -> Tuple[bool, str]:
        """
               
        
        "   ~   /       , ~   ?"
        
        Args:
            premise:   
            premise_negated: True   "       "
            conclusion:   
        
        Returns:
            (  ,   )
        """
        premise_id = premise.lower().replace(" ", "_")
        conclusion_id = conclusion.lower().replace(" ", "_")
        
        return self.knowledge_base.counterfactual_query(
            premise_id, premise_negated, conclusion_id
        )
    
    def explain_why(self, state: str, depth: int = 3) -> List[str]:
        """
        " ?"          
        
        Args:
            state:           
            depth:         
        
        Returns:
                  
        """
        state_id = state.lower().replace(" ", "_")
        
        if state_id not in self.knowledge_base.nodes:
            return [f"'{state}'                ."]
        
        explanations = []
        
        def trace_back(node_id: str, current_depth: int, path: List[str]):
            if current_depth >= depth:
                return
            
            causes = self.knowledge_base.get_causes_of(node_id)
            for cause_node, link in causes:
                #      
                cause_desc = cause_node.description
                effect_desc = self.knowledge_base.nodes[node_id].description
                
                explanation = f"{effect_desc}      {cause_desc} {link.get_description()}"
                if explanation not in explanations:
                    explanations.append(explanation)
                
                #        
                if cause_node.id not in path:
                    trace_back(cause_node.id, current_depth + 1, path + [cause_node.id])
        
        trace_back(state_id, 0, [state_id])
        
        return explanations if explanations else [f"'{state}'              ."]
    
    def get_statistics(self) -> Dict[str, Any]:
        """     """
        return {
            "total_nodes": len(self.knowledge_base.nodes),
            "total_links": len(self.knowledge_base.links),
            "total_chains": len(self.knowledge_base.chains),
            "total_experiences": self.total_experiences,
            "avg_link_strength": np.mean([
                l.strength for l in self.knowledge_base.links.values()
            ]) if self.knowledge_base.links else 0,
        }
    
    def get_strongest_causalities(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """            """
        causalities = [
            (link.source_id, link.target_id, link.strength)
            for link in self.knowledge_base.links.values()
        ]
        causalities.sort(key=lambda x: -x[2])
        return causalities[:n]


# ============================================================================
#      (Thought Universe) -                 
# ============================================================================

class ThoughtUniverse:
    """
         (Thought Universe)
    
                                          ,
                           .
    
         :
    1.       (Dimensional Expansion)
       -                     
       -              ,             
    
    2.       (Mutual Correction)
       -      :                
       -      :                 
       -      :                 
    
    3.        (Consistency Maintenance)
       -           
       -            
    """
    
    def __init__(self, name: str = "Elysia's Mind"):
        self.name = name
        
        #        
        self.points: Dict[str, ConceptPoint] = {}
        self.lines: Dict[str, CausalLine] = {}
        self.planes: Dict[str, ContextPlane] = {}
        self.spaces: Dict[str, SchemaSpace] = {}
        self.laws: Dict[str, UniversalLaw] = {}
        
        #         (Epistemic Spaces - Fractal Worlds)
        self.epistemic_spaces: Dict[str, EpistemicSpace] = {}

        #           (      )
        self.all_entities: Dict[str, DimensionalEntity] = {}
        
        #      
        self.correction_history: List[Dict[str, Any]] = []
        
        #      
        self.total_points = 0
        self.total_lines = 0
        self.total_planes = 0
        self.total_spaces = 0
        self.total_laws = 0
        self.total_corrections = 0
        
        #         
        self.causal_engine = CausalNarrativeEngine()
        
        logger.info(f"  ThoughtUniverse '{name}' initialized")
    
    # ========================================================================
    #   (Point)   
    # ========================================================================
    
    def add_point(
        self,
        id: str,
        description: str,
        sensory_signature: Dict[str, float] = None,
        emotional_valence: float = 0.0,
        concept_type: str = "general"
    ) -> ConceptPoint:
        """       """
        point = ConceptPoint(
            id=id,
            level=DimensionLevel.POINT,
            description=description,
            sensory_signature=sensory_signature or {},
            emotional_valence=emotional_valence,
            concept_type=concept_type,
            last_updated=time.time()
        )
        
        self.points[id] = point
        self.all_entities[id] = point
        self.total_points += 1
        
        logger.debug(f"    : {description}")
        return point
    
    def get_or_create_point(self, id: str, description: str = None) -> ConceptPoint:
        """            """
        if id in self.points:
            return self.points[id]
        return self.add_point(id, description or id)
    
    # ========================================================================
    #   (Line)    -        
    # ========================================================================
    
    def add_line(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "causes",
        strength: float = 1.0,
        conditions: List[str] = None,
        description: str = ""
    ) -> CausalLine:
        """        (       )"""
        #           
        source = self.get_or_create_point(source_id)
        target = self.get_or_create_point(target_id)
        
        line_id = f"{source_id}__{relation_type}__{target_id}"
        
        if line_id in self.lines:
            #        
            existing = self.lines[line_id]
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.experience_count += 1
            existing.last_updated = time.time()
            return existing
        
        line = CausalLine(
            id=line_id,
            level=DimensionLevel.LINE,
            description=description or f"{source_id} {relation_type} {target_id}",
            source_point_id=source_id,
            target_point_id=target_id,
            relation_type=relation_type,
            strength=strength,
            conditions=conditions or [],
            last_updated=time.time()
        )
        
        #            
        line.child_ids = [source_id, target_id]
        source.parent_ids.append(line_id)
        target.parent_ids.append(line_id)
        
        self.lines[line_id] = line
        self.all_entities[line_id] = line
        self.total_lines += 1
        
        logger.debug(f"    : {source_id}   {target_id}")
        return line
    
    # ========================================================================
    #   (Plane)    -              
    # ========================================================================
    
    def add_plane(
        self,
        id: str,
        description: str,
        line_ids: List[str],
        context_type: str = "situation",
        lesson: str = ""
    ) -> ContextPlane:
        """       """
        #          
        point_ids = set()
        for line_id in line_ids:
            if line_id in self.lines:
                line = self.lines[line_id]
                point_ids.add(line.source_point_id)
                point_ids.add(line.target_point_id)
        
        plane = ContextPlane(
            id=id,
            level=DimensionLevel.PLANE,
            description=description,
            line_ids=line_ids,
            point_ids=list(point_ids),
            context_type=context_type,
            lesson=lesson,
            last_updated=time.time()
        )
        
        #           
        plane.child_ids = line_ids
        for line_id in line_ids:
            if line_id in self.lines:
                self.lines[line_id].parent_ids.append(id)
        
        self.planes[id] = plane
        self.all_entities[id] = plane
        self.total_planes += 1
        
        logger.debug(f"    : {description}")
        return plane
    
    def emerge_plane_from_experience(
        self,
        experience_description: str,
        point_sequence: List[str],
        emotional_arc: List[float] = None
    ) -> ContextPlane:
        """
                (  )   
        
                         ,       
        """
        if len(point_sequence) < 2:
            raise ValueError("   2           ")
        
        emotional_arc = emotional_arc or [0.0] * len(point_sequence)
        
        #      
        line_ids = []
        for i in range(len(point_sequence) - 1):
            line = self.add_line(
                source_id=point_sequence[i],
                target_id=point_sequence[i + 1],
                relation_type="causes"
            )
            line_ids.append(line.id)
        
        #     
        plane_id = f"plane_{len(self.planes)}_{point_sequence[0]}_{point_sequence[-1]}"
        
        #              
        start_emotion = emotional_arc[0]
        end_emotion = emotional_arc[-1]
        if end_emotion > start_emotion + 0.3:
            lesson = "              "
        elif end_emotion < start_emotion - 0.3:
            lesson = "              "
        else:
            lesson = "      "
        
        plane = self.add_plane(
            id=plane_id,
            description=experience_description,
            line_ids=line_ids,
            context_type="experience",
            lesson=lesson
        )
        
        plane.emotional_tone = end_emotion
        
        return plane
    
    # ========================================================================
    #    (Space)    -               
    # ========================================================================
    
    def add_space(
        self,
        id: str,
        description: str,
        plane_ids: List[str],
        schema_type: str = "behavior",
        core_patterns: List[str] = None
    ) -> SchemaSpace:
        """         """
        space = SchemaSpace(
            id=id,
            level=DimensionLevel.SPACE,
            description=description,
            plane_ids=plane_ids,
            schema_type=schema_type,
            core_patterns=core_patterns or [],
            last_updated=time.time()
        )
        
        #           
        space.child_ids = plane_ids
        for plane_id in plane_ids:
            if plane_id in self.planes:
                self.planes[plane_id].parent_ids.append(id)
        
        self.spaces[id] = space
        self.all_entities[id] = space
        self.total_spaces += 1
        
        logger.debug(f"     : {description}")
        return space
    
    def emerge_space_from_planes(
        self,
        plane_ids: List[str],
        min_common_points: int = 2
    ) -> Optional[SchemaSpace]:
        """
                (   )   
        
                                     
        """
        if len(plane_ids) < 2:
            return None
        
        #        
        all_point_sets = []
        for plane_id in plane_ids:
            if plane_id in self.planes:
                all_point_sets.append(set(self.planes[plane_id].point_ids))
        
        if not all_point_sets:
            return None
        
        common_points = all_point_sets[0]
        for point_set in all_point_sets[1:]:
            common_points = common_points.intersection(point_set)
        
        if len(common_points) < min_common_points:
            return None
        
        #         
        core_patterns = list(common_points)
        
        #       
        space_id = f"schema_{len(self.spaces)}"
        description = f"     : {', '.join(core_patterns)}"
        
        # ... (Validation would happen here)
        
        return self.add_space(
            id=space_id,
            description=description,
            plane_ids=plane_ids,
            core_patterns=core_patterns
        )

    # ============================================================================
    # Phase 12: Dimensional Fractals (Universal Principles)
    # ============================================================================

    def extract_principle(self, chain: CausalChain, principle_name: str) -> UniversalLaw:
        """
        Abstracts a concrete causal chain into a Universal Law.
        (e.g., "Winter -> Cold -> Hunger" => "AdverseCondition -> Deprivation -> Distress")
        """
        # simplified abstraction: just structural pattern
        abstract_chain = []
        for link in chain.links:
            # Safely handle relation enum or string
            rel_val = link.relation
            if hasattr(link.relation, 'value'):
                 rel_val = link.relation.value
            abstract_chain.append(f"[{rel_val}]")
            
        law_description = f"Principle of {principle_name}: Sequence " + " -> ".join(abstract_chain)
        
        law = UniversalLaw(
            id=f"law_{principle_name.lower()}",
            level=DimensionLevel.LAW,
            description=law_description,
            law_type="fractal_pattern",
            formulation=str(abstract_chain),
            supporting_evidence=[chain.id]
        )
        
        self.laws[law.id] = law
        self.all_entities[law.id] = law
        self.total_laws += 1
        
        logger.info(f"     Law Extracted: {law.description}")
        return law

    # ========================================================================
    # Phase 13: Epistemic Topology (Fractal Knowledge Worlds)
    # ========================================================================

    def expand_node_into_space(
        self, 
        node_id: str, 
        space_name: str,
        density: float = 1.0,
        methodologies: List[str] = None
    ) -> EpistemicSpace:
        """
        Expands a single node (Point) into an entire Epistemic Space (Fractal World).
        
        This transforms a 'concept' into a 'field of study'.
        e.g., Node 'Science' -> EpistemicSpace 'Physics_World'
        """
        # 1. Get the original point/node
        if node_id not in self.points:
            # Create if not exists
            self.get_or_create_point(node_id)
            
        point = self.points[node_id]
        
        # 2. Create the inner space
        space_id = f"space_{node_id}_internal"
        space = EpistemicSpace(
            id=space_id,
            level=DimensionLevel.SPACE,
            description=f"Internal World of {space_name}",
            schema_type="epistemic_field",
            density=density,
            methodologies=methodologies or ["LOGICAL", "EMPIRICAL"]
        )
        
        # 3. Link them (Fractal Connection)
        # Note: We need to update CausalNode definition to support this, 
        # OR we use a mapping in ThoughtUniverse.
        # Assuming we updated CausalNode or ConceptPoint? 
        # ConceptPoint is defined in this file (lines 105-140 range for DimensionalEntity, Point around 150-200)
        # But wait, CausalNarrativeEngine has CausalNode. ThoughtUniverse uses ConceptPoint.
        # Let's map it in ThoughtUniverse for now or use a dynamic attribute.
        
        # Store in our registry
        self.epistemic_spaces[space_id] = space
        self.all_entities[space_id] = space
        
        # Link physically (meta-physically)
        # We'll treat the point as the "Portal" to the space
        point.child_ids.append(space_id)
        space.parent_ids.append(node_id)
        
        logger.info(f"  Expanded Node '{node_id}' into Epistemic Space '{space_name}' (Density: {density})")
        return space

    def traverse_epistemic_field(
        self, 
        agent_id: str, 
        space_id: str, 
        current_knowledge: List[str],
        target_concept: str
    ) -> Dict[str, Any]:
        """
        Simulates the traversal of an agent through a knowledge space.
        
        Movement is not free; it requires overcoming 'density' using 'methodology'.
        """
        if space_id not in self.epistemic_spaces:
            return {"status": "error", "message": "Space not found"}
            
        space = self.epistemic_spaces[space_id]
        
        # 1. Calculate Resistance
        # Resistance = Space Density * (1 - Knowledge Overlap)
        # If you know nothing, resistance is max.
        
        # Simple simulation:
        # Check if agent has prerequisite concepts to 'move' to target.
        # Prereqs are 'intermediate' concepts in this space.
        
        # Let's assume the space contains internal points.
        # We need to populate the space with points first (done via add_point + linkage).
        
        # Find path from current_knowledge (closest node) to target_concept
        # For this simulation, we'll check if target is 'reachable' given density.
        
        resistance = space.density
        
        # Methodology Check
        # Does the agent use the right methodology? (Mock check)
        agent_methodology = "EMPIRICAL" # default for now
        efficiency = 1.0
        if agent_methodology in space.methodologies:
            efficiency = 1.5 # Bonus
            
        effort_required = resistance / efficiency
        
        result = {
            "space": space.id,
            "target": target_concept,
            "resistance": resistance,
            "effort_required": effort_required,
            "status": "traversing",
            "path_log": []
        }
        
        # Simulation of "steps"
        steps = int(effort_required * 5) # Arbitrary scale
        for i in range(steps):
             result["path_log"].append(f"Step {i+1}: Overcoming conceptual density...")
             
        result["status"] = "arrived"
        result["message"] = f"Successfully traversed {space.description} to reach {target_concept}."

        return result
        
    def apply_principle_to_domain(self, law: UniversalLaw, domain_map: Dict[str, str]) -> List[str]:
        """
        Applies a Universal Law to a new domain using a mapping.
        (Fractal Expansion)
        """
        narrative = [f"Applying {law.id} to new domain..."]
        
        import ast
        try:
             relations = ast.literal_eval(law.formulation)
        except:
             # Fallback if formulation isn't list string
             return ["Failed to parse law formulation."]
             
        # Generate the new narrative
        steps = []
        # We need relation count + 1 items
        count = len(relations) + 1
        
        for i in range(count):
             key = f"Step_{i}"
             if key in domain_map:
                 steps.append(domain_map[key])
             else:
                 steps.append("Unknown")
                 
        for i, relation in enumerate(relations):
            if i+1 < len(steps):
                source = steps[i]
                target = steps[i+1]
                narrative.append(f"{source} --({relation})--> {target}")
                
        return narrative

    # ========================================================================
    # Phase 14: Metacognitive Architecture (Self-Evolution)
    # ========================================================================

    def evaluate_maturity(self, concept_id: str) -> Dict[str, Any]:
        """
        Evaluates the maturity of a specific concept against the 'ADULT' standard.
        analysis: "How dense, differentiated, and integrated is my understanding?"
        """
        metrics = CognitiveMetrics(0.2, 0.2, 0.1) # Default CHILD level
        
        if concept_id in self.points:
            point = self.points[concept_id]
            
            # Simple heuristics for simulation
            # Differentiation: Number of child nodes (which represent details or subtypes in this context)
            diff_score = min(0.9, len(point.child_ids) * 0.2)
            
            # Integration: how many parents? (part of larger structures)
            int_score = min(0.9, len(point.parent_ids) * 0.2)
            
            # Abstraction: Is it linked to a Space or Law?
            abs_score = 0.1
            for pid in point.parent_ids:
                if pid.startswith("law_") or pid.startswith("space_"):
                    abs_score += 0.3
            abs_score = min(0.9, abs_score)
            
            metrics = CognitiveMetrics(diff_score, int_score, abs_score)
            
        # Compare with Standard
        standard = MaturityModel.get_standard_model()["ADULT"]
        gap_report = {
            "concept": concept_id,
            "current_metrics": metrics,
            "target_metrics": standard.required_metrics,
            "gaps": {},
            "status": "IMMATURE"
        }
        
        # Calculate gaps
        if metrics.differentiation < standard.required_metrics.differentiation:
            gap_report["gaps"]["differentiation"] = standard.required_metrics.differentiation - metrics.differentiation
        if metrics.integration < standard.required_metrics.integration:
             gap_report["gaps"]["integration"] = standard.required_metrics.integration - metrics.integration
        if metrics.abstraction < standard.required_metrics.abstraction:
             gap_report["gaps"]["abstraction"] = standard.required_metrics.abstraction - metrics.abstraction
             
        if not gap_report["gaps"]:
            gap_report["status"] = "MATURE"
            
        return gap_report

    def formulate_growth_plan(self, gap_report: Dict[str, Any]) -> List[str]:
        """
        Generates 'Intentions' (Tasks) to bridge the identified gaps.
        Autonomously decides *what to do* to become smarter.
        """
        intentions = []
        concept = gap_report["concept"]
        
        if gap_report["status"] == "MATURE":
            return ["Maintain current understanding."]
            
        gaps = gap_report["gaps"]
        
        # 1. Address Differentiation Gap (Too simple?)
        if "differentiation" in gaps:
            val = gaps["differentiation"]
            intentions.append(f"INTENTION: Deepen differentiation of '{concept}'. Explore nuances and subtypes. (Gap: {val:.2f})")
            
        # 2. Address Integration Gap (Disconnected?)
        if "integration" in gaps:
             val = gaps["integration"]
             intentions.append(f"INTENTION: Increase integration of '{concept}'. Connect to wider contexts and other concepts. (Gap: {val:.2f})")
             
        # 3. Address Abstraction Gap (Too concrete?)
        if "abstraction" in gaps:
             val = gaps["abstraction"]
             intentions.append(f"INTENTION: Lift '{concept}' to higher abstraction. Find universal principles or laws it belongs to. (Gap: {val:.2f})")
             
        return intentions

        
        space = self.add_space(
            id=space_id,
            description=description,
            plane_ids=plane_ids,
            schema_type="emergent",
            core_patterns=core_patterns
        )
        
        return space
    
    # ========================================================================
    #    (Law)    -                
    # ========================================================================
    
    def add_law(
        self,
        id: str,
        description: str,
        space_ids: List[str],
        formulation: str = "",
        law_type: str = "causal"
    ) -> UniversalLaw:
        """         """
        law = UniversalLaw(
            id=id,
            level=DimensionLevel.LAW,
            description=description,
            space_ids=space_ids,
            formulation=formulation,
            law_type=law_type,
            last_updated=time.time()
        )
        
        #            
        law.child_ids = space_ids
        for space_id in space_ids:
            if space_id in self.spaces:
                self.spaces[space_id].parent_ids.append(id)
        
        self.laws[id] = law
        self.all_entities[id] = law
        self.total_laws += 1
        
        logger.info(f"        : {description}")
        return law
    
    def discover_law_from_spaces(
        self,
        space_ids: List[str],
        confidence_threshold: float = 0.8
    ) -> Optional[UniversalLaw]:
        """
                    
        
                                      
        """
        if len(space_ids) < 2:
            return None
        
        #         
        all_pattern_sets = []
        for space_id in space_ids:
            if space_id in self.spaces:
                all_pattern_sets.append(set(self.spaces[space_id].core_patterns))
        
        if not all_pattern_sets:
            return None
        
        common_patterns = all_pattern_sets[0]
        for pattern_set in all_pattern_sets[1:]:
            common_patterns = common_patterns.intersection(pattern_set)
        
        if not common_patterns:
            return None
        
        #      
        law_id = f"law_{len(self.laws)}"
        formulation = "   ".join(common_patterns)  #       
        
        law = self.add_law(
            id=law_id,
            description=f"      : {formulation}",
            space_ids=space_ids,
            formulation=formulation,
            law_type="emergent"
        )
        
        return law
    
    # ========================================================================
    #       (Mutual Correction)
    # ========================================================================
    
    def bottom_up_correct(
        self,
        new_experience: Dict[str, Any],
        affected_entity_id: str
    ) -> Dict[str, Any]:
        """
             :                  
        
         : "         "   "           " (     )
        """
        if affected_entity_id not in self.all_entities:
            return {"success": False, "reason": "entity not found"}
        
        entity = self.all_entities[affected_entity_id]
        
        correction = {
            "type": "bottom_up",
            "timestamp": time.time(),
            "entity_id": affected_entity_id,
            "before": entity.confidence,
            "experience": new_experience,
        }
        
        #                      
        is_consistent = new_experience.get("confirms", True)
        
        if is_consistent:
            #                
            entity.confidence = min(1.0, entity.confidence + 0.05)
            entity.experience_count += 1
            correction["action"] = "strengthen"
        else:
            #                    
            entity.confidence = max(0.0, entity.confidence - 0.1)
            exception = new_experience.get("exception", "")
            if isinstance(entity, CausalLine) and exception:
                entity.exceptions.append(exception)
            correction["action"] = "weaken"
        
        correction["after"] = entity.confidence
        entity.corrections.append(correction)
        entity.last_updated = time.time()
        
        self.correction_history.append(correction)
        self.total_corrections += 1
        
        return correction
    
    def top_down_correct(
        self,
        law_id: str,
        target_entity_id: str
    ) -> Dict[str, Any]:
        """
             :                 
        
         : "      "     "    "        
        """
        if law_id not in self.laws:
            return {"success": False, "reason": "law not found"}
        if target_entity_id not in self.all_entities:
            return {"success": False, "reason": "target not found"}
        
        law = self.laws[law_id]
        target = self.all_entities[target_entity_id]
        
        correction = {
            "type": "top_down",
            "timestamp": time.time(),
            "law_id": law_id,
            "target_id": target_entity_id,
            "before": target.confidence,
        }
        
        #            
        is_consistent = self._check_consistency_with_law(law, target)
        
        if is_consistent:
            #                    
            law.supporting_evidence.append(target_entity_id)
            correction["action"] = "confirmed"
        else:
            #                            
            if target.confidence < law.confidence:
                #                    
                target.confidence *= 0.9
                correction["action"] = "adjusted_down"
            else:
                #                 
                law.counter_examples.append(target_entity_id)
                correction["action"] = "counter_example"
        
        correction["after"] = target.confidence
        target.corrections.append(correction)
        target.last_updated = time.time()
        
        self.correction_history.append(correction)
        self.total_corrections += 1
        
        return correction
    
    def lateral_correct(
        self,
        entity_id_1: str,
        entity_id_2: str
    ) -> Dict[str, Any]:
        """
             :                  
        
         : "   "  "   "            
        """
        if entity_id_1 not in self.all_entities:
            return {"success": False, "reason": "entity 1 not found"}
        if entity_id_2 not in self.all_entities:
            return {"success": False, "reason": "entity 2 not found"}
        
        entity_1 = self.all_entities[entity_id_1]
        entity_2 = self.all_entities[entity_id_2]
        
        #           
        if entity_1.level != entity_2.level:
            return {"success": False, "reason": "different levels"}
        
        correction = {
            "type": "lateral",
            "timestamp": time.time(),
            "entity_1": entity_id_1,
            "entity_2": entity_id_2,
        }
        
        #  (Point)    :         
        if entity_1.level == DimensionLevel.POINT:
            p1, p2 = entity_1, entity_2
            if isinstance(p1, ConceptPoint) and isinstance(p2, ConceptPoint):
                #             ( :          )
                overlap = self._calculate_sensory_overlap(
                    p1.sensory_signature,
                    p2.sensory_signature
                )
                if overlap < -0.5:
                    #               
                    correction["relation"] = "opposites"
                elif overlap > 0.5:
                    #              
                    correction["relation"] = "similar"
                else:
                    correction["relation"] = "independent"
        
        self.correction_history.append(correction)
        self.total_corrections += 1
        
        return correction
    
    def _check_consistency_with_law(
        self,
        law: UniversalLaw,
        entity: DimensionalEntity
    ) -> bool:
        """           """
        #       :            
        if not law.formulation:
            return True
        
        #  (Point)    
        if isinstance(entity, ConceptPoint):
            return entity.id in law.formulation or entity.description in law.formulation
        
        #  (Line)    
        if isinstance(entity, CausalLine):
            return (entity.source_point_id in law.formulation or
                    entity.target_point_id in law.formulation)
        
        return True
    
    def _calculate_sensory_overlap(
        self,
        sig1: Dict[str, float],
        sig2: Dict[str, float]
    ) -> float:
        """             (-1 ~ +1)"""
        common_keys = set(sig1.keys()) & set(sig2.keys())
        if not common_keys:
            return 0.0
        
        total = 0.0
        for key in common_keys:
            #          ,          
            total += sig1[key] * sig2[key]
        
        return total / len(common_keys)
    
    # ========================================================================
    #            
    # ========================================================================
    
    def learn_from_experience(
        self,
        experience_steps: List[str],
        emotional_arc: List[float] = None,
        auto_emergence: bool = True
    ) -> Dict[str, Any]:
        """
                    
        
        1.      /  
        2.       (     )
        3.      (  )
        4.       /     
        
        Args:
            experience_steps:       
            emotional_arc:         
            auto_emergence:               
        
        Returns:
                    
        """
        result = {
            "points_created": 0,
            "lines_created": 0,
            "plane_created": None,
            "space_emerged": None,
            "law_discovered": None,
        }
        
        # 1.      /  
        for step in experience_steps:
            point_id = step.lower().replace(" ", "_")
            if point_id not in self.points:
                self.add_point(point_id, step)
                result["points_created"] += 1
        
        # 2.      (        )
        point_ids = [s.lower().replace(" ", "_") for s in experience_steps]
        plane = self.emerge_plane_from_experience(
            experience_description="   ".join(experience_steps),
            point_sequence=point_ids,
            emotional_arc=emotional_arc
        )
        result["plane_created"] = plane.id
        result["lines_created"] = len(plane.line_ids)
        
        # 3.            
        if auto_emergence:
            #          
            similar_planes = self._find_similar_planes(plane, threshold=0.5)
            if len(similar_planes) >= 2:
                space = self.emerge_space_from_planes(
                    [plane.id] + similar_planes
                )
                if space:
                    result["space_emerged"] = space.id
                    
                    #                
                    similar_spaces = self._find_similar_spaces(space, threshold=0.7)
                    if len(similar_spaces) >= 2:
                        law = self.discover_law_from_spaces(
                            [space.id] + similar_spaces
                        )
                        if law:
                            result["law_discovered"] = law.id
        
        return result
    
    def _find_similar_planes(
        self,
        target_plane: ContextPlane,
        threshold: float = 0.5
    ) -> List[str]:
        """         (          )"""
        similar = []
        target_points = set(target_plane.point_ids)
        
        for plane_id, plane in self.planes.items():
            if plane_id == target_plane.id:
                continue
            
            plane_points = set(plane.point_ids)
            if not plane_points:
                continue
            
            overlap = len(target_points & plane_points) / len(target_points | plane_points)
            if overlap >= threshold:
                similar.append(plane_id)
        
        return similar
    
    def _find_similar_spaces(
        self,
        target_space: SchemaSpace,
        threshold: float = 0.7
    ) -> List[str]:
        """          (           )"""
        similar = []
        target_patterns = set(target_space.core_patterns)
        
        for space_id, space in self.spaces.items():
            if space_id == target_space.id:
                continue
            
            space_patterns = set(space.core_patterns)
            if not space_patterns:
                continue
            
            overlap = len(target_patterns & space_patterns) / len(target_patterns | space_patterns)
            if overlap >= threshold:
                similar.append(space_id)
        
        return similar
    
    # ========================================================================
    #        
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """       """
        return {
            "name": self.name,
            "total_points": self.total_points,
            "total_lines": self.total_lines,
            "total_planes": self.total_planes,
            "total_spaces": self.total_spaces,
            "total_laws": self.total_laws,
            "total_corrections": self.total_corrections,
            "dimension_breakdown": {
                " (Point)": self.total_points,
                " (Line)": self.total_lines,
                " (Plane)": self.total_planes,
                "  (Space)": self.total_spaces,
                "  (Law)": self.total_laws,
            }
        }
    
    def visualize_hierarchy(self, max_items: int = 5) -> str:
        """         """
        lines = [
            f"      : {self.name}",
            "=" * 50,
            "",
            "      (Law) -       ",
            "-" * 30,
        ]
        
        for law_id in list(self.laws.keys())[:max_items]:
            law = self.laws[law_id]
            lines.append(f"    {law.description}")
        
        lines.extend([
            "",
            "     (Space) -    /   ",
            "-" * 30,
        ])
        
        for space_id in list(self.spaces.keys())[:max_items]:
            space = self.spaces[space_id]
            lines.append(f"    {space.description}")
        
        lines.extend([
            "",
            "    (Plane) -   /  ",
            "-" * 30,
        ])
        
        for plane_id in list(self.planes.keys())[:max_items]:
            plane = self.planes[plane_id]
            lines.append(f"    {plane.description[:50]}...")
        
        lines.extend([
            "",
            "     (Line) -      ",
            "-" * 30,
        ])
        
        for line_id in list(self.lines.keys())[:max_items]:
            line = self.lines[line_id]
            lines.append(f"    {line.source_point_id}   {line.target_point_id}")
        
        lines.extend([
            "",
            "    (Point) -      ",
            "-" * 30,
        ])
        
        for point_id in list(self.points.keys())[:max_items]:
            point = self.points[point_id]
            lines.append(f"    {point.description}")
        
        return "\n".join(lines)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """            """
    print("=" * 70)
    print("Causal Narrative Engine -          ")
    print("=" * 70)
    print()
    print("                        ")
    print("            , ' '  '   '       ")
    print()
    
    engine = CausalNarrativeEngine()
    
    # 1.         
    print("-" * 70)
    print("1.         ")
    print("-" * 70)
    
    #         
    engine.experience_chain(
        descriptions=["        ", "        ", "     ", "     "],
        emotional_arc=[0.0, -0.8, -0.3, 0.5],
        agent_id="  "
    )
    print("               ")
    
    #          
    engine.experience_chain(
        descriptions=["      ", "       ", "   ", "      "],
        emotional_arc=[-0.5, 0.0, 0.5, 0.9],
        agent_id="  "
    )
    print("                ")
    
    # 2.      
    print()
    print("-" * 70)
    print("2.       (       )")
    print("-" * 70)
    
    predictions = engine.predict_effect("      ")
    print("  '      '       :")
    for effect, prob, relation in predictions[:3]:
        print(f"      {effect} (  : {prob:.0%})")
    
    # 3.      
    print()
    print("-" * 70)
    print("3.       (       )")
    print("-" * 70)
    
    causes = engine.find_cause("     ")
    print("  '     '    :")
    for cause, prob, relation in causes[:3]:
        print(f"      {cause} (  : {prob:.0%})")
    
    # 4.  ?   
    print()
    print("-" * 70)
    print("4. ' ?'   ")
    print("-" * 70)
    
    explanations = engine.explain_why("      ")
    print("  '      '       :")
    for exp in explanations[:3]:
        print(f"      {exp}")
    
    # 5.        
    print()
    print("-" * 70)
    print("5.        ")
    print("-" * 70)
    
    result, explanation = engine.counterfactual_reasoning(
        premise="        ",
        premise_negated=True,
        conclusion="        "
    )
    print(f"  Q:                 ?")
    print(f"  A: {explanation}")
    
    # 6.         
    print()
    print("-" * 70)
    print("6.         ")
    print("-" * 70)
    
    plan = engine.plan_to_achieve("hunger", "satiety")
    if plan:
        print(f"    :          ")
        print(f"    : {'   '.join(plan.node_sequence)}")
    
    #   
    print()
    print("-" * 70)
    print("7.      ")
    print("-" * 70)
    stats = engine.get_statistics()
    print(f"      : {stats['total_nodes']}")
    print(f"         : {stats['total_links']}")
    print(f"      : {stats['total_experiences']}")
    print(f"          : {stats['avg_link_strength']:.2f}")
    
    print()
    print("=" * 70)
    print("                   !  ")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()