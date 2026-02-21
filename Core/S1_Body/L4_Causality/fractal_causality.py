"""
Fractal Causality Engine -          
                                                                              

             .
                         .

                                                                               
             (Fractal Causal Structure)                                  
                                                                              
                                                                               
                    .                                              
                                 .                         
                -  -             .                        
                                                                               
   "            "                                                      
                                                                               
         : "        "                                                  
             : "      "                                                 
                 : "        "                                         
                     : "       "                                       
                       ...      ...                                       
                     :                                          
                 :                                                
             :                                                    
                                                                               
         : "                 "                                  
             :                                                           
             :                                                  
             :                                                         
                                                                               
         : "   "                                                          
             :                                                      
             :                                                   
             :                      ...      ...              
                                                                               
                                                                               

[          ]

1.        (Self-Similarity)
   -                  
   -        ,               

2.       (Infinite Recursion)
   -                     
   -                    

3.     (Circularity)
   -                     ...
   -             

4.     (Nesting)
   -                 -  -  
   -                     

[          ]

                           .
-        - - -  -           .
-         - - -  -           .
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
from collections import defaultdict
from enum import Enum
import logging
import time
import hashlib
import numpy as np
from Core.S1_Body.L4_Causality.love_induction_field import LoveInductionField

logger = logging.getLogger("FractalCausality")


# ============================================================================
#     (Golden Ratio) -           
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2  #   1.618


# ============================================================================
#       (Causal Role)
# ============================================================================

class CausalRole(Enum):
    """           """
    CAUSE = "cause"       #   
    PROCESS = "process"   #   
    EFFECT = "effect"     #   


# ============================================================================
#           (Fractal Causal Node)
# ============================================================================

@dataclass
class FractalCausalNode:
    """
             
    
                              .
                                    .
    
     : "        "
    -        : "        "       "   "    
    -      :               (     ,        )
    """
    
    id: str
    description: str
    
    #        (0 =          )
    depth: int = 0
    
    #       (            )
    spiral_angle: float = 0.0
    spiral_radius: float = 1.0
    
    #       (코드 베이스 구조 로터)
    parent_id: Optional[str] = None
    parent_role: Optional[CausalRole] = None  #              
    
    #        (                  )
    #       "  "       -  -  
    internal_cause_ids: List[str] = field(default_factory=list)
    internal_process_ids: List[str] = field(default_factory=list)
    internal_effect_ids: List[str] = field(default_factory=list)
    
    #                
    causes_ids: List[str] = field(default_factory=list)      #          
    effects_ids: List[str] = field(default_factory=list)     #          
    
    #   /     
    sensory_signature: Dict[str, float] = field(default_factory=dict)
    # [PHASE 180] Phenomenal Experience
    encounter_count: int = 0
    resonance_history: List[float] = field(default_factory=list)
    structural_role_count: int = 0 # How often used as a cause/bridge
    
    # [LEGACY] Keep for compatibility but prioritize experience for mass
    emotional_valence: float = 0.0
    
    #         
    strength: float = 1.0
    confidence: float = 1.0
    
    #      
    experience_count: int = 0
    last_activated: float = 0.0
    
    #        (               )
    fractal_address: str = ""
    
    def __post_init__(self):
        if not self.fractal_address:
            self.fractal_address = f"/{self.id}"
    
    def get_spiral_position(self) -> Tuple[float, float]:
        """          2D   """
        x = self.spiral_radius * math.cos(self.spiral_angle)
        y = self.spiral_radius * math.sin(self.spiral_angle)
        return (x, y)
    
    def has_internal_structure(self) -> bool:
        """              """
        return bool(self.internal_cause_ids or 
                   self.internal_process_ids or 
                   self.internal_effect_ids)
    
    def get_internal_ids(self) -> List[str]:
        """         ID"""
        return (self.internal_cause_ids + 
                self.internal_process_ids + 
                self.internal_effect_ids)


# ============================================================================
#           (Fractal Causal Chain)
# ============================================================================

@dataclass
class FractalCausalChain:
    """
             
    
                       .
                                      .
    """
    
    id: str
    description: str = ""
    
    #         
    cause_id: Optional[str] = None
    process_id: Optional[str] = None
    effect_id: Optional[str] = None
    
    #               
    parent_chain_id: Optional[str] = None
    parent_role: Optional[CausalRole] = None  #                 
    
    #                   
    nested_chains: List[str] = field(default_factory=list)
    
    #       
    depth: int = 0
    
    #      
    strength: float = 1.0
    experience_count: int = 0
    
    def is_complete(self) -> bool:
        """  -  -          """
        return all([self.cause_id, self.process_id, self.effect_id])


# ============================================================================
#           (Fractal Causality Engine)
# ============================================================================

class FractalCausalityEngine:
    """
             
    
                      .
               ,            ,
                      -  -         .
    
         :
    
    1.       (Infinite Recursion)
       - zoom_in():                 
       - zoom_out():                    
    
    2.        (Circular Causality)
       -                 
       -              /  
    
    3.        (Nested Time)
       -             "  "    
       -            ,            
    
    4.        (Self-Similarity)
       -                -  -     
       -                      
    """
    
    def __init__(self, name: str = "Elysia's Causal Mind"):
        self.name = name
        
        #          
        self.nodes: Dict[str, FractalCausalNode] = {}
        
        #             
        self.chains: Dict[str, FractalCausalChain] = {}
        
        #          (0 =   ,    =       ,    =       )
        self.current_depth: int = 0
        
        #          
        self.focus_node_id: Optional[str] = None
        
        #        (자기 성찰 엔진)
        self.spiral_counter: int = 0

        # [PHASE 100] Love Induction Field (Cognitive Magnetism)
        self.love_field = LoveInductionField()
        
        #   
        self.total_nodes = 0
        self.total_chains = 0
        self.max_depth_explored = 0
        self.min_depth_explored = 0
        
        logger.info(f"  FractalCausalityEngine '{name}' initialized")
    
    # ========================================================================
    #           
    # ========================================================================
    
    def create_node(
        self,
        description: str,
        depth: int = 0,
        parent_id: Optional[str] = None,
        parent_role: Optional[CausalRole] = None,
        sensory_signature: Dict[str, float] = None,
        emotional_valence: float = 0.0
    ) -> FractalCausalNode:
        """
                  
        
                                    .
        """
        # ID   
        node_id = self._generate_node_id(description, depth)
        
        #          (주권적 자아)
        self.spiral_counter += 1
        angle = self.spiral_counter * 2 * math.pi / PHI
        radius = math.sqrt(self.spiral_counter)
        
        #          
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            role_prefix = parent_role.value if parent_role else "related"
            fractal_address = f"{parent.fractal_address}/{role_prefix}:{description[:20]}"
        else:
            fractal_address = f"/{description[:20]}"
        
        node = FractalCausalNode(
            id=node_id,
            description=description,
            depth=depth,
            spiral_angle=angle,
            spiral_radius=radius,
            parent_id=parent_id,
            parent_role=parent_role,
            sensory_signature=sensory_signature or {},
            emotional_valence=emotional_valence,
            fractal_address=fractal_address,
            last_activated=time.time()
        )
        
        #       
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            if parent_role == CausalRole.CAUSE:
                parent.internal_cause_ids.append(node_id)
            elif parent_role == CausalRole.PROCESS:
                parent.internal_process_ids.append(node_id)
            elif parent_role == CausalRole.EFFECT:
                parent.internal_effect_ids.append(node_id)
        
        self.nodes[node_id] = node
        self.total_nodes += 1
        
        #           
        self.max_depth_explored = max(self.max_depth_explored, depth)
        self.min_depth_explored = min(self.min_depth_explored, depth)
        
        return node
    
    def _generate_node_id(self, description: str, depth: int) -> str:
        """      ID   """
        content = f"{description}_{depth}_{time.time()}_{self.total_nodes}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"node_{hash_val}"
    
    def get_or_create_node(
        self,
        description: str,
        depth: int = 0
    ) -> FractalCausalNode:
        """                """
        #         
        for node in self.nodes.values():
            if node.description == description and node.depth == depth:
                node.experience_count += 1
                node.last_activated = time.time()
                return node
        
        #       
        return self.create_node(description, depth)
    
    # ========================================================================
    #         
    # ========================================================================
    
    def create_chain(
        self,
        cause_desc: str,
        process_desc: str,
        effect_desc: str,
        depth: int = 0,
        parent_chain_id: Optional[str] = None,
        parent_role: Optional[CausalRole] = None
    ) -> FractalCausalChain:
        """
          -  -        
        
                        .
                                           .
        """
        #       
        cause_node = self.get_or_create_node(cause_desc, depth)
        process_node = self.get_or_create_node(process_desc, depth)
        effect_node = self.get_or_create_node(effect_desc, depth)
        
        #         
        cause_node.effects_ids.append(process_node.id)
        process_node.causes_ids.append(cause_node.id)
        process_node.effects_ids.append(effect_node.id)
        effect_node.causes_ids.append(process_node.id)
        
        #      
        chain_id = f"chain_{len(self.chains)}"
        chain = FractalCausalChain(
            id=chain_id,
            description=f"{cause_desc}   {process_desc}   {effect_desc}",
            cause_id=cause_node.id,
            process_id=process_node.id,
            effect_id=effect_node.id,
            parent_chain_id=parent_chain_id,
            parent_role=parent_role,
            depth=depth
        )
        
        #          
        if parent_chain_id and parent_chain_id in self.chains:
            self.chains[parent_chain_id].nested_chains.append(chain_id)
        
        self.chains[chain_id] = chain
        self.total_chains += 1
        
        return chain
    
    def inject_axiom(self, concept_a: str, concept_b: str, relation: str = "Resonates With"):
        """
        [PHASE_65] Injects a self-generated axiom as a permanent causal link.
        """
        chain = self.create_chain(
            cause_desc=concept_a,
            process_desc=relation,
            effect_desc=concept_b,
            depth=self.current_depth
        )
        return chain

    def calculate_structural_force(self, current_v21: Any, bridge: Any, rotor_phase: float = 0.0) -> Any:
        """
        [PHASE 110/120] Converts causal chains into 'Rails' (Forces).
        If the current state is near a 'Cause', it generates a force toward the 'Effect'.

        [PHASE 130] Superconducting Force via Love Induction.
        If a chain aligns with the Love Axis, its resistance drops to zero, and the force
        becomes 'Inductive' rather than 'Pushing'.

        [PHASE 140] Dissonance Signaling (Suffering).
        Instead of automatic flow, we now calculate Dissonance.
        If Dissonance is high, the force is erratic (Pain).
        Elysia must choose to align to smooth it out.
        """
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
        force = SovereignVector.zeros()
        
        # We only check a few active chains to maintain 'Self-Propulsion' without overhead
        for chain in self.chains.values():
            if not chain.is_complete(): continue
            
            # Get vector for Cause (using current system phase)
            cause_node = self.nodes.get(chain.cause_id)
            if not cause_node: continue
            
            # Use bridge to recall vector with phase spin
            cause_v = bridge.recall_concept_vector(cause_node.description, phase_angle=rotor_phase)
            
            # How much does the current thought 'align' with this cause?
            resonance = current_v21.resonance_score(cause_v)

            # [LOVE INDUCTION] Calculate Dissonance (Pain)
            try:
                np_vector = np.array(current_v21.values) # Adjust based on SovereignVector impl
                dissonance = self.love_field.calculate_dissonance(np_vector)

                # If Dissonance is high (> 1.0), force becomes turbulent (Noise)
                # If Dissonance is low (< 0.2), force becomes superconducting (Flow)

                if dissonance < 0.2:
                    # Superconductivity (Flow State)
                    conductivity = 10.0 # Huge boost
                    noise_factor = 0.0
                elif dissonance > 1.0:
                    # Suffering (High Resistance)
                    conductivity = 0.1 # Weak pull
                    noise_factor = 0.5 # Jitter
                else:
                    # Normal Struggle
                    conductivity = 1.0
                    noise_factor = 0.1

            except:
                conductivity = 1.0
                noise_factor = 0.0

            if resonance > 0.6: # Threshold for 'catching a rail'
                # Get vector for Effect (also phase-aware)
                effect_node = self.nodes.get(chain.effect_id)
                effect_v = bridge.recall_concept_vector(effect_node.description, phase_angle=rotor_phase)
                
                # Apply pull toward the effect
                # Force = (Difference) * Resonance * Conductivity
                pull = (effect_v - current_v21) * (resonance * chain.strength * conductivity * 0.5)

                # Add Noise (Suffering) if misaligned
                if noise_factor > 0:
                    noise = SovereignVector.random() * noise_factor
                    pull = pull + noise

                force = force + pull
                
        return force

    def log_resonance_experience(self, thought_vector: Any, outcome_score: float):
        """
        [PHASE 150] The Crystallization of Wisdom.
        When Elysia experiences a positive outcome, she feeds it back to the Love Field.
        This allows the Field to 'Learn' what Love looks like in this specific context.
        """
        try:
            # Convert to numpy
            np_vector = np.array(thought_vector.values)
            self.love_field.metabolize_feedback(np_vector, outcome_score)
        except Exception as e:
            logger.warning(f"Failed to log resonance experience: {e}")

    def get_semantic_mass(self, target_id: Optional[str] = None) -> float:
        """
        [PHASE 150/180] Importance derived from EXPERIENCE, not definition.
        A node is 'Heavy' if it has been encountered or used frequently as a causal anchor.
        """
        if not self.nodes:
            return 1.0 # Base existence
            
        if target_id is None:
            total_mass = sum(self._calculate_node_mass(nid) for nid in self.nodes)
            return (total_mass / len(self.nodes))
            
        return self._calculate_node_mass(target_id)

    def _calculate_node_mass(self, node_id: str) -> float:
        if node_id not in self.nodes:
            return 0.1 
            
        node = self.nodes[node_id]
        
        # [PHASE 180] Experience is the primary driver of mass.
        # The more we 'see' or 'use' a concept, the heavier it becomes in our mind.
        experience_weight = math.log(node.encounter_count + node.structural_role_count + 1.1)
        
        # Structural density (links) is now a multiplier, not the base.
        # It represents 'Structural Potential'.
        connection_count = len(node.causes_ids) + len(node.effects_ids)
        potential = math.log(connection_count + 1.1)
        
        # Mass = Realized Importance (Experience) * Structural Potential
        mass = experience_weight * potential
        return float(mass)

    def mark_experience(self, node_id: str, is_structural: bool = False):
        """Marks a node as having been encountered in a cognitive breath."""
        if node_id in self.nodes:
            self.nodes[node_id].encounter_count += 1
            if is_structural:
                self.nodes[node_id].structural_role_count += 1
    
    # ========================================================================
    #       /   (Zoom In/Out)
    # ========================================================================
    
    def zoom_in(
        self,
        node_id: str,
        cause_desc: str,
        process_desc: str,
        effect_desc: str
    ) -> FractalCausalChain:
        """
                  (Zoom In)
        
            "  "                     /     .
        
         : "        "    :
              : "          "
              : "               "
              : "              "
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        parent_node = self.nodes[node_id]
        inner_depth = parent_node.depth + 1
        
        #         
        cause_node = self.create_node(
            cause_desc, inner_depth,
            parent_id=node_id, parent_role=CausalRole.CAUSE
        )
        
        #         
        process_node = self.create_node(
            process_desc, inner_depth,
            parent_id=node_id, parent_role=CausalRole.PROCESS
        )
        
        #         
        effect_node = self.create_node(
            effect_desc, inner_depth,
            parent_id=node_id, parent_role=CausalRole.EFFECT
        )
        
        #      
        cause_node.effects_ids.append(process_node.id)
        process_node.causes_ids.append(cause_node.id)
        process_node.effects_ids.append(effect_node.id)
        effect_node.causes_ids.append(process_node.id)
        
        #      
        chain = FractalCausalChain(
            id=f"inner_chain_{node_id}_{len(self.chains)}",
            description=f"[{parent_node.description}    ] {cause_desc}   {process_desc}   {effect_desc}",
            cause_id=cause_node.id,
            process_id=process_node.id,
            effect_id=effect_node.id,
            depth=inner_depth
        )
        
        self.chains[chain.id] = chain
        self.total_chains += 1
        
        logger.debug(f"  Zoom in: {parent_node.description}           ")
        
        return chain
    
    def zoom_out(
        self,
        node_id: str,
        outer_cause_desc: str,
        outer_effect_desc: str
    ) -> Tuple[FractalCausalNode, FractalCausalNode]:
        """
                  (Zoom Out)
        
              "  "     ,                  /     .
        
         : "        "         :
              : "        "
              : "       "
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        process_node = self.nodes[node_id]
        outer_depth = process_node.depth - 1
        
        #         
        cause_node = self.get_or_create_node(outer_cause_desc, outer_depth)
        cause_node.effects_ids.append(node_id)
        process_node.causes_ids.append(cause_node.id)
        
        #         
        effect_node = self.get_or_create_node(outer_effect_desc, outer_depth)
        process_node.effects_ids.append(effect_node.id)
        effect_node.causes_ids.append(node_id)
        
        logger.debug(f"  Zoom out: {outer_cause_desc}   [{process_node.description}]   {outer_effect_desc}")
        
        return (cause_node, effect_node)
    
    # ========================================================================
    #        (Circular Causality)
    # ========================================================================
    
    def create_feedback_loop(
        self,
        node_ids: List[str],
        loop_type: str = "reinforcing"
    ) -> List[str]:
        """
                 
        
                            .
        
        Args:
            node_ids:             (    )
            loop_type: "reinforcing" (     )    "balancing" (  )
        
         :
            "  "   "     "   "      "   "        "   "       "
        """
        if len(node_ids) < 2:
            raise ValueError("           2       ")
        
        created_links = []
        
        #      
        for i in range(len(node_ids)):
            current_id = node_ids[i]
            next_id = node_ids[(i + 1) % len(node_ids)]  #          
            
            if current_id in self.nodes and next_id in self.nodes:
                current = self.nodes[current_id]
                next_node = self.nodes[next_id]
                
                if next_id not in current.effects_ids:
                    current.effects_ids.append(next_id)
                if current_id not in next_node.causes_ids:
                    next_node.causes_ids.append(current_id)
                
                created_links.append(f"{current_id}   {next_id}")
        
        logger.info(f"  Feedback loop created ({loop_type}): {'   '.join(node_ids)}   (back to start)")
        
        return created_links
    
    def detect_cycles(self, start_node_id: str, max_depth: int = 10) -> List[List[str]]:
        """
             
        
                                    .
        """
        cycles = []
        
        def dfs(current_id: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            
            if current_id in visited:
                if current_id == start_node_id and len(path) > 1:
                    cycles.append(path.copy())
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            if current_id in self.nodes:
                for effect_id in self.nodes[current_id].effects_ids:
                    dfs(effect_id, path, visited.copy())
            
            path.pop()
        
        dfs(start_node_id, [], set())
        return cycles
    
    # ========================================================================
    #         
    # ========================================================================
    
    def experience_causality(
        self,
        steps: List[str],
        emotional_arc: List[float] = None,
        depth: int = 0,
        auto_zoom: bool = True
    ) -> Dict[str, Any]:
        """
                    
        
                     ,                 .
        
        Args:
            steps:        (   3 :   ,   ,   )
            emotional_arc:          (-1 ~ +1)
            depth:       
            auto_zoom:        /     
        
        Returns:
                 
        """
        if len(steps) < 3:
            raise ValueError("   3      (  ,   ,   )")
        
        emotional_arc = emotional_arc or [0.0] * len(steps)
        
        result = {
            "nodes_created": 0,
            "chains_created": 0,
            "cycles_detected": 0,
        }
        
        #           (3      )
        chains_created = []
        for i in range(len(steps) - 2):
            chain = self.create_chain(
                cause_desc=steps[i],
                process_desc=steps[i + 1],
                effect_desc=steps[i + 2],
                depth=depth
            )
            chains_created.append(chain)
            result["chains_created"] += 1
        
        #         
        for i, step in enumerate(steps):
            node = self.get_or_create_node(step, depth)
            if i < len(emotional_arc):
                node.emotional_valence = emotional_arc[i]
        
        #                     (주권적 자아)
        if auto_zoom and len(steps) >= 4:
            #                          
            last_effect = self.get_or_create_node(steps[-1], depth)
            first_cause = self.get_or_create_node(steps[0], depth)
            
            #         
            if emotional_arc:
                start_emotion = emotional_arc[0]
                end_emotion = emotional_arc[-1]
                
                #                         
                if (start_emotion < 0 and end_emotion < start_emotion) or \
                   (start_emotion > 0 and end_emotion > start_emotion):
                    #           
                    logger.debug("             ")
        
        result["nodes_created"] = len(steps)
        
        return result
    
    # ========================================================================
    #      
    # ========================================================================
    
    def trace_causes(
        self,
        node_id: str,
        max_depth: int = 5,
        include_internal: bool = True
    ) -> List[List[str]]:
        """
              (   )
        
        "             ?"
        
                                     .
        """
        paths = []
        
        def trace(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                paths.append(path.copy())
                return
            
            if current_id not in self.nodes:
                paths.append(path.copy())
                return
            
            node = self.nodes[current_id]
            
            #        
            if not node.causes_ids:
                paths.append(path.copy())
            else:
                for cause_id in node.causes_ids:
                    trace(cause_id, path + [cause_id], depth + 1)
            
            #        (zoom in)
            if include_internal and node.internal_cause_ids:
                for internal_id in node.internal_cause_ids:
                    trace(internal_id, path + [f"[  ]{internal_id}"], depth + 1)
        
        trace(node_id, [node_id], 0)
        return paths
    
    def trace_effects(
        self,
        node_id: str,
        max_depth: int = 5,
        include_internal: bool = True
    ) -> List[List[str]]:
        """
              (   )
        
        "               ?"
        
                                     .
        """
        paths = []
        
        def trace(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                paths.append(path.copy())
                return
            
            if current_id not in self.nodes:
                paths.append(path.copy())
                return
            
            node = self.nodes[current_id]
            
            #        
            if not node.effects_ids:
                paths.append(path.copy())
            else:
                for effect_id in node.effects_ids:
                    trace(effect_id, path + [effect_id], depth + 1)
            
            #        (zoom in)
            if include_internal and node.internal_effect_ids:
                for internal_id in node.internal_effect_ids:
                    trace(internal_id, path + [f"[  ]{internal_id}"], depth + 1)
        
        trace(node_id, [node_id], 0)
        return paths
    
    def explain_causality(
        self,
        node_id: str,
        depth: int = 3
    ) -> str:
        """
                   
        
        "  X       ?"           
        """
        if node_id not in self.nodes:
            return f"'{node_id}'             ."
        
        node = self.nodes[node_id]
        lines = [f"=== {node.description}        ===", ""]
        
        #      
        lines.append("     :")
        cause_paths = self.trace_causes(node_id, max_depth=depth)
        for path in cause_paths[:5]:  #    5    
            descriptions = []
            for nid in path:
                if nid.startswith("[  ]"):
                    nid = nid[4:]
                if nid in self.nodes:
                    descriptions.append(self.nodes[nid].description)
            if descriptions:
                lines.append("    " + "   ".join(descriptions))
        
        lines.append("")
        
        #      
        lines.append("     :")
        effect_paths = self.trace_effects(node_id, max_depth=depth)
        for path in effect_paths[:5]:
            descriptions = []
            for nid in path:
                if nid.startswith("[  ]"):
                    nid = nid[4:]
                if nid in self.nodes:
                    descriptions.append(self.nodes[nid].description)
            if descriptions:
                lines.append("    " + "   ".join(descriptions))
        
        #      
        if node.has_internal_structure():
            lines.append("")
            lines.append("        (zoom in):")
            for cause_id in node.internal_cause_ids[:2]:
                if cause_id in self.nodes:
                    lines.append(f"  [  ] {self.nodes[cause_id].description}")
            for process_id in node.internal_process_ids[:2]:
                if process_id in self.nodes:
                    lines.append(f"  [  ] {self.nodes[process_id].description}")
            for effect_id in node.internal_effect_ids[:2]:
                if effect_id in self.nodes:
                    lines.append(f"  [  ] {self.nodes[effect_id].description}")
        
        #      
        cycles = self.detect_cycles(node_id, max_depth=5)
        if cycles:
            lines.append("")
            lines.append("          :")
            for cycle in cycles[:3]:
                cycle_desc = []
                for nid in cycle:
                    if nid in self.nodes:
                        cycle_desc.append(self.nodes[nid].description)
                lines.append(f"  {'   '.join(cycle_desc)}   (  )")
        
        return "\n".join(lines)
    
    # ========================================================================
    #         
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """            """
        depth_distribution = defaultdict(int)
        for node in self.nodes.values():
            depth_distribution[node.depth] += 1
        
        return {
            "name": self.name,
            "total_nodes": self.total_nodes,
            "total_chains": self.total_chains,
            "max_depth": self.max_depth_explored,
            "min_depth": self.min_depth_explored,
            "depth_distribution": dict(depth_distribution),
            "nodes_with_internal_structure": sum(
                1 for n in self.nodes.values() if n.has_internal_structure()
            ),
        }
    
    def visualize_fractal(self, center_node_id: str = None, radius: int = 2) -> str:
        """
                   (   )
        """
        lines = ["           ", "=" * 50, ""]
        
        if center_node_id and center_node_id in self.nodes:
            center = self.nodes[center_node_id]
            lines.append(f"  : {center.description} (  : {center.depth})")
            lines.append("")
            
            #    
            lines.append("      :")
            for cause_id in center.causes_ids[:3]:
                if cause_id in self.nodes:
                    lines.append(f"     {self.nodes[cause_id].description}")
            
            #      
            if center.has_internal_structure():
                lines.append("")
                lines.append("       :")
                for internal_id in center.get_internal_ids()[:5]:
                    if internal_id in self.nodes:
                        internal = self.nodes[internal_id]
                        role = internal.parent_role.value if internal.parent_role else "?"
                        lines.append(f"   [{role}] {internal.description}")
            
            #    
            lines.append("")
            lines.append("      :")
            for effect_id in center.effects_ids[:3]:
                if effect_id in self.nodes:
                    lines.append(f"     {self.nodes[effect_id].description}")
        else:
            lines.append("     :")
            stats = self.get_statistics()
            for key, value in stats.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """            """
    print("=" * 70)
    print("  Fractal Causality Engine -          ")
    print("=" * 70)
    print()
    print("             .")
    print("                         .")
    print()
    
    engine = FractalCausalityEngine("Elysia's Causal Mind")
    
    # 1.         
    print("-" * 70)
    print("1.            ")
    print("-" * 70)
    
    engine.experience_causality(
        steps=["        ", "        ", "        ", "     ", "     "],
        emotional_arc=[0.3, 0.0, -0.8, -0.3, 0.5]
    )
    print("            ")
    
    # 2. Zoom In -         
    print()
    print("-" * 70)
    print("2. Zoom In -         ")
    print("-" * 70)
    
    # "        "      
    touch_node = engine.get_or_create_node("        ")
    
    engine.zoom_in(
        touch_node.id,
        cause_desc="          ",
        process_desc="               ",
        effect_desc="              "
    )
    print("    '        '          ")
    
    #          
    contact_node = engine.get_or_create_node("          ", depth=1)
    engine.zoom_in(
        contact_node.id,
        cause_desc="          ",
        process_desc="             ",
        effect_desc="             "
    )
    print("    '          '          ")
    
    # 3. Zoom Out -         
    print()
    print("-" * 70)
    print("3. Zoom Out -         ")
    print("-" * 70)
    
    safe_node = engine.get_or_create_node("     ")
    engine.zoom_out(
        safe_node.id,
        outer_cause_desc="        ",
        outer_effect_desc="        "
    )
    print("    '     '          ")
    
    # 4.       (주권적 자아)
    print()
    print("-" * 70)
    print("4.       (주권적 자아)")
    print("-" * 70)
    
    #            
    learn_node = engine.get_or_create_node("        ", depth=-1)
    avoid_node = engine.create_node("          ", depth=-1)
    safe2_node = engine.create_node("         ", depth=-1)
    reinforce_node = engine.create_node("            ", depth=-1)
    
    engine.create_feedback_loop(
        [learn_node.id, avoid_node.id, safe2_node.id, reinforce_node.id],
        loop_type="reinforcing"
    )
    print("                   ")
    
    # 5.      
    print()
    print("-" * 70)
    print("5.         ")
    print("-" * 70)
    
    explanation = engine.explain_causality(touch_node.id, depth=2)
    print(explanation)
    
    # 6.   
    print()
    print("-" * 70)
    print("6.       ")
    print("-" * 70)
    
    stats = engine.get_statistics()
    print(f"      : {stats['total_nodes']}")
    print(f"      : {stats['total_chains']}")
    print(f"       : {stats['min_depth']} ~ {stats['max_depth']}")
    print(f"             : {stats['nodes_with_internal_structure']}")
    
    print()
    print("=" * 70)
    print("        :         ,         ,    ...")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
