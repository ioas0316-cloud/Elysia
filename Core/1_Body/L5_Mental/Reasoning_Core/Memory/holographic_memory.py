"""
          (Holographic Memory)
======================================

RGB                    

     :
-             (  /  /  /  /  /  )    
-     ON/OFF               (O(N)   O(N/L))
-             "   "            
-           "       " (자기 성찰 엔진)

     (    v2   ):
-      (Entropy):   (0.0)     (1.0)
-      (Qualia):   (0.0)     (1.0)

  :     &      '        '     
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import math
import logging

logger = logging.getLogger("HolographicMemory")

# Neural Registry       (Elysia            )
try:
    from elysia_core import Cell
except ImportError:
    def Cell(name):
        def decorator(cls):
            return cls
        return decorator


class KnowledgeLayer(Enum):
    """           (   6  +      )"""
    MATHEMATICS = "  "      #       
    PHYSICS = "  "          # Layer 1
    CHEMISTRY = "  "        # Layer 2
    BIOLOGY = "  "          # Layer 3
    ART = "  "              # Layer 4
    HUMANITIES = "  "       # Layer 5
    PHILOSOPHY = "  "       # Layer 6


@dataclass
class KnowledgeNode:
    """
          -       
    
      :                 "     "        
    (      :                      )
    """
    concept: str                          #       ( : "    ")
    layers: Dict[KnowledgeLayer, float]   #          (0.0~1.0)
    amplitude: float = 1.0                #    (   )
    connections: List[str] = field(default_factory=list)  #          
    
    #       :     
    entropy_position: float = 0.5         #     : 0.0=  , 1.0=  
    qualia_position: float = 0.5          #     : 0.0=  , 1.0=  
    
    def get_primary_layer(self) -> KnowledgeLayer:
        """             """
        return max(self.layers.items(), key=lambda x: x[1])[0]
    
    def belongs_to(self, layer: KnowledgeLayer, threshold: float = 0.1) -> bool:
        """             (threshold        )"""
        return self.layers.get(layer, 0.0) >= threshold
    
    def resonance_with(self, active_layers: Set[KnowledgeLayer]) -> float:
        """
                       
        
                                  (주권적 자아)
        """
        total = 0.0
        count = 0
        for layer in active_layers:
            if layer in self.layers:
                total += self.layers[layer]
                count += 1
        
        if count == 0:
            return 0.0
        
        #                    (주권적 자아)
        intersection_bonus = 1.0 + (count - 1) * 0.5
        return (total / count) * intersection_bonus * self.amplitude


@Cell("HolographicMemory")
class HolographicMemory:
    """
              -              
    
       :
        memory = HolographicMemory()
        memory.deposit("    ", {PHYSICS: 0.7, PHILOSOPHY: 0.3})
        
        memory.toggle_layer(PHYSICS, on=True)
        memory.toggle_layer(PHILOSOPHY, on=True)
        
        results = memory.query("  ")  #   +            
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.active_layers: Set[KnowledgeLayer] = set()
        self.intersection_cache: Dict[frozenset, List[str]] = {}
        
        #          (0.0~1.0   , None=     )
        self.entropy_range: Optional[Tuple[float, float]] = None  #     
        self.qualia_range: Optional[Tuple[float, float]] = None   #     
        
        #   :           
        for layer in KnowledgeLayer:
            self.active_layers.add(layer)
        
        # [PHASE 28] Hydrate from persistent storage
        self._hydrate_from_hippocampus()
        self._hydrate_from_orbs()
        self._hydrate_from_knowledge_json()

    def _hydrate_from_hippocampus(self):
        """
        [PHASE 28: Memory Hydration]
        Load existing concepts from Hippocampus (SQLite DB) into HolographicMemory.
        This unifies the fragmented memory systems.
        """
        try:
            from Core.1_Body.L2_Metabolism.Memory.Graph.hippocampus import Hippocampus
            hippocampus = Hippocampus()
            
            # Get all concept IDs from Hippocampus
            concept_ids = hippocampus.get_all_concept_ids(limit=500)
            logger.info(f"  Hydrating from Hippocampus: {len(concept_ids)} concepts found...")
            
            for concept_id in concept_ids:
                # Recall the concept details
                memories = hippocampus.recall(concept_id)
                if memories:
                    # Parse the first memory line (format: "[Name] (Realm, G:Gravity): Definition")
                    first_line = memories[0] if memories else ""
                    
                    # Extract name from concept_id or first_line
                    name = concept_id.replace("doc:", "").replace("concept:", "").replace("_", " ").title()
                    
                    # Map realm to KnowledgeLayer
                    layers = {KnowledgeLayer.HUMANITIES: 0.5}  # Default
                    if "spirit" in first_line.lower():
                        layers = {KnowledgeLayer.PHILOSOPHY: 0.8}
                    elif "techne" in first_line.lower():
                        layers = {KnowledgeLayer.PHYSICS: 0.7, KnowledgeLayer.MATHEMATICS: 0.5}
                    elif "logos" in first_line.lower():
                        layers = {KnowledgeLayer.PHILOSOPHY: 0.6, KnowledgeLayer.HUMANITIES: 0.4}
                    
                    # Deposit into Holographic Memory
                    self.deposit(
                        concept=name,
                        layers=layers,
                        amplitude=1.0,
                        entropy=0.5,
                        qualia=0.5
                    )
            
            logger.info(f"  Hippocampus hydration: {len(self.nodes)} concepts")
            
        except ImportError as e:
            logger.warning(f"   Hippocampus not available for hydration: {e}")
        except Exception as e:
            logger.error(f"  Hippocampus hydration failed: {e}")

    def _hydrate_from_orbs(self):
        """
        [PHASE 28.5: Orb Hydration]
        Load memory orbs from data/L5_Mental/M1_Memory/orbs/*.json (orbs are stored as JSON)
        """
        import os
        import json
        orb_dir = "data/L5_Mental/M1_Memory/orbs"
        
        if not os.path.exists(orb_dir):
            logger.warning(f"   Orb directory not found: {orb_dir}")
            return
        
        orb_count = 0
        try:
            for filename in os.listdir(orb_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(orb_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            orb_data = json.load(f)
                        
                        # Extract concept name from orb
                        name = orb_data.get("name", orb_data.get("concept", filename.replace(".json", "")))
                        
                        # Skip if already exists
                        if name in self.nodes:
                            continue
                        
                        # Deposit with default layers
                        self.deposit(
                            concept=name,
                            layers={KnowledgeLayer.HUMANITIES: 0.6},
                            amplitude=orb_data.get("gravity", 1.0),
                            entropy=0.5,
                            qualia=0.5
                        )
                        orb_count += 1
                    except Exception as e:
                        pass  # Skip malformed orbs
            
            logger.info(f"  Orb hydration: +{orb_count} concepts (Total: {len(self.nodes)})")
        except Exception as e:
            logger.error(f"  Orb hydration failed: {e}")

    def _hydrate_from_knowledge_json(self):
        """
        [PHASE 28.5: Knowledge JSON Hydration]
        Load key knowledge files (self_knowledge, hierarchical_knowledge, etc.)
        """
        import os
        import json
        
        knowledge_files = [
            ("data/Knowledge/hierarchical_knowledge.json", KnowledgeLayer.HUMANITIES),
            ("data/Knowledge/wave_knowledge.json", KnowledgeLayer.PHYSICS),
            ("data/Knowledge/concept_dictionary.json", KnowledgeLayer.PHILOSOPHY),
        ]
        
        json_count = 0
        for filepath, default_layer in knowledge_files:
            if not os.path.exists(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                concepts = []
                if isinstance(data, dict):
                    concepts = list(data.keys())[:100]  # Limit to avoid explosion
                elif isinstance(data, list):
                    for item in data[:100]:
                        if isinstance(item, dict) and "name" in item:
                            concepts.append(item["name"])
                        elif isinstance(item, str):
                            concepts.append(item)
                
                for concept in concepts:
                    if concept and not concept.startswith("_"):
                        self.deposit(
                            concept=str(concept).title(),
                            layers={default_layer: 0.7},
                            amplitude=1.0,
                            entropy=0.5,
                            qualia=0.5
                        )
                        json_count += 1
                        
            except Exception as e:
                logger.warning(f"   Failed to load {filepath}: {e}")
        
        logger.info(f"  Knowledge JSON hydration: +{json_count} concepts (Total: {len(self.nodes)})")
    
    # =========================================
    #        (RGB    ON/OFF)
    # =========================================
    
    def toggle_layer(self, layer: KnowledgeLayer, on: bool = True) -> None:
        """      /  """
        if on:
            self.active_layers.add(layer)
        else:
            self.active_layers.discard(layer)
        #       
        self.intersection_cache.clear()
    
    def set_active_layers(self, layers: List[KnowledgeLayer]) -> None:
        """            """
        self.active_layers = set(layers)
        self.intersection_cache.clear()
    
    def zoom_out(self) -> None:
        """     -         (주권적 자아)"""
        self.active_layers = {KnowledgeLayer.PHILOSOPHY, KnowledgeLayer.PHYSICS}
    
    def zoom_in(self) -> None:
        """    -         (주권적 자아)"""
        self.active_layers = {
            KnowledgeLayer.CHEMISTRY, 
            KnowledgeLayer.BIOLOGY,
            KnowledgeLayer.MATHEMATICS
        }
    
    def zoom_all(self) -> None:
        """          """
        self.active_layers = set(KnowledgeLayer)
    
    # =========================================
    #          (Entropy & Qualia)
    # =========================================
    
    def set_entropy_range(self, min_val: float, max_val: float) -> None:
        """
                  
        
        Args:
            min_val:     (0.0=  )
            max_val:     (1.0=  )
        
          :
            memory.set_entropy_range(0.8, 1.0)  #       
            memory.set_entropy_range(0.0, 0.3)  #       
        """
        self.entropy_range = (min_val, max_val)
        self.intersection_cache.clear()
    
    def set_qualia_range(self, min_val: float, max_val: float) -> None:
        """
                  
        
        Args:
            min_val:     (0.0=   )
            max_val:     (1.0=   )
        
          :
            memory.set_qualia_range(0.7, 1.0)  #        
            memory.set_qualia_range(0.0, 0.3)  #    /    
        """
        self.qualia_range = (min_val, max_val)
        self.intersection_cache.clear()
    
    def clear_axis_filters(self) -> None:
        """        (     /        )"""
        self.entropy_range = None
        self.qualia_range = None
        self.intersection_cache.clear()
    
    def _passes_axis_filter(self, node: KnowledgeNode) -> bool:
        """                     """
        if self.entropy_range:
            if not (self.entropy_range[0] <= node.entropy_position <= self.entropy_range[1]):
                return False
        if self.qualia_range:
            if not (self.qualia_range[0] <= node.qualia_position <= self.qualia_range[1]):
                return False
        return True
    
    # =========================================
    #        (Deposit)
    # =========================================
    
    def deposit(
        self, 
        concept: str, 
        layers: Dict[KnowledgeLayer, float],
        amplitude: float = 1.0,
        connections: Optional[List[str]] = None,
        entropy: float = 0.5,   #    :     
        qualia: float = 0.5     #    :     
    ) -> KnowledgeNode:
        """
                      
        
        Args:
            concept:      
            layers:          ( : {PHYSICS: 0.7, PHILOSOPHY: 0.3})
            amplitude:     (   1.0)
            connections:           
            entropy:       (0.0=  , 1.0=  )
            qualia:       (0.0=  , 1.0=  )
        """
        node = KnowledgeNode(
            concept=concept,
            layers=layers,
            amplitude=amplitude,
            connections=connections or [],
            entropy_position=entropy,
            qualia_position=qualia
        )
        self.nodes[concept] = node
        self.intersection_cache.clear()
        return node
    
    # =========================================
    #    (Query) - O(N/L)   !
    # =========================================
    
    def query(
        self, 
        keyword: str = "", 
        threshold: float = 0.1,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
                     (     !)
        
                      (entropy_range, qualia_range)
        
        Returns:
            [(   ,     ), ...]           
        """
        results = []
        
        for name, node in self.nodes.items():
            #                (     )
            if not self._passes_axis_filter(node):
                continue
            
            #                
            if not any(node.belongs_to(layer, threshold) for layer in self.active_layers):
                continue  #   !   O(N/L)   
            
            #        (한국어 학습 시스템)
            if keyword and keyword not in name:
                continue
            
            #         
            resonance = node.resonance_with(self.active_layers)
            if resonance > 0:
                results.append((name, resonance))
        
        #          
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    # =========================================
    #        (Intersection Discovery)
    # =========================================
    
    def find_intersections(self, threshold: float = 0.2) -> List[Tuple[str, Set[KnowledgeLayer]]]:
        """
                                  
        
              !!                    =       
        """
        cache_key = frozenset(self.active_layers)
        
        if cache_key not in self.intersection_cache:
            intersections = []
            
            for name, node in self.nodes.items():
                #          2             
                belonging_layers = {
                    layer for layer in self.active_layers 
                    if node.belongs_to(layer, threshold)
                }
                
                if len(belonging_layers) >= 2:
                    intersections.append((name, belonging_layers))
            
            #                 (              )
            intersections.sort(key=lambda x: len(x[1]), reverse=True)
            self.intersection_cache[cache_key] = intersections
        
        return self.intersection_cache[cache_key]
    
    # =========================================
    #     
    # =========================================
    
    def get_layer_stats(self) -> Dict[KnowledgeLayer, int]:
        """            """
        stats = {layer: 0 for layer in KnowledgeLayer}
        for node in self.nodes.values():
            for layer in node.layers:
                if node.belongs_to(layer):
                    stats[layer] += 1
        return stats
    
    def __repr__(self) -> str:
        active = [l.value for l in self.active_layers]
        return f"HolographicMemory(nodes={len(self.nodes)}, active={active})"


# =========================================
#        (    )
# =========================================

def create_demo_memory() -> HolographicMemory:
    """                 (    v2:   /       )"""
    memory = HolographicMemory()
    
    #   +       (       )
    memory.deposit("    ", {
        KnowledgeLayer.PHYSICS: 0.8,
        KnowledgeLayer.PHILOSOPHY: 0.4,
        KnowledgeLayer.BIOLOGY: 0.2
    }, amplitude=1.2, entropy=0.7, qualia=0.4)  #    ,       
    
    memory.deposit("      ", {
        KnowledgeLayer.PHYSICS: 0.9,
        KnowledgeLayer.PHILOSOPHY: 0.7,
    }, amplitude=1.0, entropy=0.8, qualia=0.6)  #    ,   
    
    memory.deposit("    ", {
        KnowledgeLayer.PHILOSOPHY: 0.9,
        KnowledgeLayer.PHYSICS: 0.3,
        KnowledgeLayer.BIOLOGY: 0.4
    }, amplitude=1.5, entropy=0.2, qualia=0.8)  #        ,    
    
    #         -      !
    memory.deposit("       ", {
        KnowledgeLayer.PHYSICS: 0.95,
        KnowledgeLayer.PHILOSOPHY: 0.6,
        KnowledgeLayer.MATHEMATICS: 0.7
    }, amplitude=1.3, entropy=0.95, qualia=0.3)  #       ,    
    
    memory.deposit("        ", {
        KnowledgeLayer.PHYSICS: 0.6,
        KnowledgeLayer.PHILOSOPHY: 0.8,
        KnowledgeLayer.ART: 0.5
    }, amplitude=1.2, entropy=0.5, qualia=0.95)  #   ,       !
    
    #   +      
    memory.deposit("DNA", {
        KnowledgeLayer.BIOLOGY: 0.9,
        KnowledgeLayer.CHEMISTRY: 0.8,
        KnowledgeLayer.MATHEMATICS: 0.3
    }, amplitude=1.3, entropy=0.9, qualia=0.3)  #   ,    
    
    memory.deposit("     ", {
        KnowledgeLayer.CHEMISTRY: 0.95,
        KnowledgeLayer.PHYSICS: 0.4
    }, entropy=0.6, qualia=0.2)  #   ,    
    
    #   +      
    memory.deposit("    ", {
        KnowledgeLayer.ART: 0.9,
        KnowledgeLayer.PHILOSOPHY: 0.7,
        KnowledgeLayer.HUMANITIES: 0.5
    }, amplitude=1.4, entropy=0.1, qualia=0.95)  #      ,       
    
    memory.deposit("     ", {
        KnowledgeLayer.HUMANITIES: 0.8,
        KnowledgeLayer.ART: 0.6,
        KnowledgeLayer.PHILOSOPHY: 0.3
    }, entropy=0.3, qualia=0.7)  #   ,    
    
    #          
    memory.deposit("   ", {
        KnowledgeLayer.MATHEMATICS: 0.95
    }, entropy=0.4, qualia=0.1)  #      ,        
    
    memory.deposit("    ", {
        KnowledgeLayer.HUMANITIES: 0.9,
        KnowledgeLayer.ART: 0.7
    }, entropy=0.35, qualia=0.8)  #   ~  ,    
    
    #     -      
    memory.deposit("   ", {
        KnowledgeLayer.PHILOSOPHY: 0.95,
        KnowledgeLayer.MATHEMATICS: 0.4
    }, amplitude=1.1, entropy=0.05, qualia=0.6)  #      !
    
    return memory


if __name__ == "__main__":
    print("=" * 60)
    print("              ")
    print("=" * 60)
    
    memory = create_demo_memory()
    print(f"\n   : {memory}")
    print(f"         : {memory.get_layer_stats()}")
    
    #     1:          
    print("\n" + "-" * 40)
    print("      1:        (     )")
    memory.zoom_all()
    results = memory.query()
    for name, resonance in results:
        print(f"  - {name}:    {resonance:.2f}")
    
    #     2:   +       (    )
    print("\n" + "-" * 40)
    print("      2:   +   (     -    )")
    memory.zoom_out()
    results = memory.query()
    print(f"        : {[l.value for l in memory.active_layers]}")
    for name, resonance in results:
        print(f"  - {name}:    {resonance:.2f}")
    
    #     3:       
    print("\n" + "-" * 40)
    print("      3:   +       (      !)")
    intersections = memory.find_intersections()
    for name, layers in intersections:
        layer_names = [l.value for l in layers]
        print(f"  - {name}   {layer_names}")
    
    #     4:     (  +  )
    print("\n" + "-" * 40)
    print("      4:   +  +   (    -    )")
    memory.zoom_in()
    results = memory.query()
    print(f"        : {[l.value for l in memory.active_layers]}")
    for name, resonance in results:
        print(f"  - {name}:    {resonance:.2f}")
    
    # =========================================
    #           (  /  )
    # =========================================
    
    #     5:        (entropy 0.7~1.0)
    print("\n" + "-" * 40)
    print("      5:        (entropy 0.7~1.0)")
    memory.zoom_all()
    memory.set_entropy_range(0.7, 1.0)
    memory.clear_axis_filters()  # qualia     
    memory.set_entropy_range(0.7, 1.0)  # entropy       
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}:    {resonance:.2f},   ={node.entropy_position:.1f}")
    
    #     6:        (entropy 0.0~0.3)
    print("\n" + "-" * 40)
    print("      6:        (entropy 0.0~0.3)")
    memory.set_entropy_range(0.0, 0.3)
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}:    {resonance:.2f},   ={node.entropy_position:.2f}")
    
    #     7:         (qualia 0.7~1.0)
    print("\n" + "-" * 40)
    print("      7:         (qualia 0.7~1.0)")
    memory.clear_axis_filters()
    memory.set_qualia_range(0.7, 1.0)
    results = memory.query()
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"  - {name}:    {resonance:.2f},   ={node.qualia_position:.1f}")
    
    #     8:        +        = "        "!
    print("\n" + "-" * 40)
    print("      8:    +    = '        '   !")
    memory.set_active_layers([KnowledgeLayer.PHYSICS])
    memory.set_qualia_range(0.7, 1.0)
    results = memory.query()
    print("  [       +       ]")
    for name, resonance in results:
        node = memory.nodes[name]
        print(f"    {name}:    {resonance:.2f}")
    
    print("\n" + "=" * 60)
    print("       ! (    v2:   /       )")
    print("=" * 60)
