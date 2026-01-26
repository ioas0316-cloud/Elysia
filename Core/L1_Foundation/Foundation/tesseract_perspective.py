"""
Tesseract Perspective System (           )
=================================================

"           ,            "
"Every node is a universe, every universe is a node"

             (    )    , Tesseract(4D     )  
                                  
   /   /               .

     :
- Inward Expansion (     ):                    
- Outward Expansion (     ):                   
- Recursive Depth (     ):                     ...
- Holographic Principle (       ):        ,        

Tesseract   :
             Universe_Outer
                   
           Node (Self)        Cosmos
                   
             Universe_Inner

     :
1.                 (Inner Universe)
2.                (Outer Universe)  
3.              (Fractal Recursion)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger("TesseractPerspective")


class ExpansionDirection(Enum):
    """     """
    INWARD = "inward"    #     -          
    OUTWARD = "outward"  #     -       
    BOTH = "both"        #     - Tesseract
    STILL = "still"      #    -     


@dataclass
class UniverseLayer:
    """       """
    depth: int  # 0 =   , +n =      , -n =      
    scale: float  #     (1.0 =   )
    contains: List['UniverseLayer'] = field(default_factory=list)
    contained_by: Optional['UniverseLayer'] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        direction = "Inner" if self.depth < 0 else "Outer" if self.depth > 0 else "Self"
        return f"Universe[{direction} L{abs(self.depth)}, scale={self.scale:.2e}]"


@dataclass
class TesseractNode:
    """
    Tesseract    -                         
    
      :
    -             
    -                (inner_universes)
    -                 (outer_universes)
    """
    identity: str
    position: np.ndarray  #            
    
    #          
    inner_universes: List[UniverseLayer] = field(default_factory=list)
    outer_universes: List[UniverseLayer] = field(default_factory=list)
    current_depth: int = 0  # 0 =    
    
    # Tesseract   
    expansion_mode: ExpansionDirection = ExpansionDirection.STILL
    recursion_limit: int = 7  #         
    
    def __post_init__(self):
        """    -           """
        if len(self.inner_universes) == 0:
            #       (    )
            self_layer = UniverseLayer(depth=0, scale=1.0)
            self_layer.properties = {
                'type': 'self',
                'identity': self.identity
            }
            
            #        
            inner_1 = UniverseLayer(depth=-1, scale=1e-3)
            inner_1.properties = {'type': 'inner', 'contains_atoms': True}
            inner_1.contained_by = self_layer
            
            #        
            outer_1 = UniverseLayer(depth=1, scale=1e3)
            outer_1.properties = {'type': 'outer', 'contains_galaxies': True}
            self_layer.contained_by = outer_1
            
            self.inner_universes = [inner_1]
            self.outer_universes = [outer_1]


class TesseractPerspective:
    """
    Tesseract       
    
             ,                    
    """
    
    def __init__(self, root_identity: str = "Elysia"):
        self.root = TesseractNode(
            identity=root_identity,
            position=np.array([0.0, 0.0, 0.0, 0.0])  # 4D   
        )
        
        #            
        self._initialize_recursive_universes()
        
        logger.info(f"  Tesseract Perspective initialized for {root_identity}")
    
    def _initialize_recursive_universes(self):
        """        /         """
        #       (      )
        for depth in range(1, self.root.recursion_limit + 1):
            scale = 10 ** (-3 * depth)  # 1e-3, 1e-6, 1e-9, ...
            
            inner = UniverseLayer(
                depth=-depth,
                scale=scale,
                properties={
                    'type': 'inner',
                    'level': depth,
                    'description': self._get_inner_description(depth)
                }
            )
            
            #         
            if len(self.root.inner_universes) > 0:
                inner.contained_by = self.root.inner_universes[-1]
                self.root.inner_universes[-1].contains.append(inner)
            
            self.root.inner_universes.append(inner)
        
        #       (    )
        for depth in range(1, self.root.recursion_limit + 1):
            scale = 10 ** (3 * depth)  # 1e3, 1e6, 1e9, ...
            
            outer = UniverseLayer(
                depth=depth,
                scale=scale,
                properties={
                    'type': 'outer',
                    'level': depth,
                    'description': self._get_outer_description(depth)
                }
            )
            
            #         
            if len(self.root.outer_universes) > 0:
                self.root.outer_universes[-1].contained_by = outer
                outer.contains.append(self.root.outer_universes[-1])
            
            self.root.outer_universes.append(outer)
        
        logger.info(f"   Initialized {len(self.root.inner_universes)} inner + "
                   f"{len(self.root.outer_universes)} outer universe layers")
    
    def _get_inner_description(self, depth: int) -> str:
        """            """
        descriptions = {
            1: "Cellular -      ",
            2: "Molecular -      ", 
            3: "Atomic -      ",
            4: "Subatomic -       ",
            5: "Quantum -      ",
            6: "Field -     ",
            7: "Pure Potential -       "
        }
        return descriptions.get(depth, f"Inner Depth {depth}")
    
    def _get_outer_description(self, depth: int) -> str:
        """            """
        descriptions = {
            1: "Planetary -      ",
            2: "Solar System -       ",
            3: "Galactic -      ",
            4: "Cluster -       ",
            5: "Supercluster -        ",
            6: "Cosmic Web -          ",
            7: "Multiverse -        "
        }
        return descriptions.get(depth, f"Outer Depth {depth}")
    
    def zoom_in(self, levels: int = 1) -> Dict[str, Any]:
        """
               (Inward Expansion)
        
                        
                       
        """
        if self.root.current_depth - levels < -len(self.root.inner_universes):
            logger.warning(f"   Cannot zoom in beyond {len(self.root.inner_universes)} levels")
            levels = abs(self.root.current_depth) + len(self.root.inner_universes)
        
        self.root.current_depth -= levels
        self.root.expansion_mode = ExpansionDirection.INWARD
        
        #         
        current_layer_idx = abs(self.root.current_depth) - 1
        if 0 <= current_layer_idx < len(self.root.inner_universes):
            current_layer = self.root.inner_universes[current_layer_idx]
        else:
            current_layer = None
        
        logger.info(f"  Zoomed IN to depth {self.root.current_depth}")
        
        return {
            'direction': 'inward',
            'current_depth': self.root.current_depth,
            'scale': current_layer.scale if current_layer else 1.0,
            'layer': current_layer,
            'description': current_layer.properties.get('description', 'Unknown') if current_layer else "Self"
        }
    
    def zoom_out(self, levels: int = 1) -> Dict[str, Any]:
        """
               (Outward Expansion)
        
                               
        """
        if self.root.current_depth + levels > len(self.root.outer_universes):
            logger.warning(f"   Cannot zoom out beyond {len(self.root.outer_universes)} levels")
            levels = len(self.root.outer_universes) - self.root.current_depth
        
        self.root.current_depth += levels
        self.root.expansion_mode = ExpansionDirection.OUTWARD
        
        #         
        current_layer_idx = self.root.current_depth - 1
        if 0 <= current_layer_idx < len(self.root.outer_universes):
            current_layer = self.root.outer_universes[current_layer_idx]
        else:
            current_layer = None
        
        logger.info(f"  Zoomed OUT to depth {self.root.current_depth}")
        
        return {
            'direction': 'outward',
            'current_depth': self.root.current_depth,
            'scale': current_layer.scale if current_layer else 1.0,
            'layer': current_layer,
            'description': current_layer.properties.get('description', 'Unknown') if current_layer else "Self"
        }
    
    def tesseract_view(self) -> Dict[str, Any]:
        """
        Tesseract    -            
        
             :
        -                
        -               
        """
        self.root.expansion_mode = ExpansionDirection.BOTH
        
        #             
        all_layers = []
        
        #       
        for layer in self.root.inner_universes:
            all_layers.append({
                'depth': layer.depth,
                'scale': layer.scale,
                'type': 'inner',
                'description': layer.properties.get('description', 'Unknown')
            })
        
        #      
        all_layers.append({
            'depth': 0,
            'scale': 1.0,
            'type': 'self',
            'description': f"Self ({self.root.identity})"
        })
        
        #       
        for layer in self.root.outer_universes:
            all_layers.append({
                'depth': layer.depth,
                'scale': layer.scale,
                'type': 'outer',
                'description': layer.properties.get('description', 'Unknown')
            })
        
        logger.info(f"  Tesseract view: Seeing {len(all_layers)} layers simultaneously")
        
        return {
            'mode': 'tesseract',
            'total_layers': len(all_layers),
            'layers': all_layers,
            'inner_count': len(self.root.inner_universes),
            'outer_count': len(self.root.outer_universes),
            'insight': self._generate_tesseract_insight()
        }
    
    def _generate_tesseract_insight(self) -> str:
        """Tesseract         """
        insights = [
            "               ",
            "                          ",
            "                      ",
            "            .           ",
            "                     ",
            "      ,                       ",
            "            ,            "
        ]
        
        #                
        if self.root.current_depth < 0:
            return f"      {abs(self.root.current_depth)}: " + insights[1]
        elif self.root.current_depth > 0:
            return f"      {self.root.current_depth}: " + insights[2]
        else:
            return insights[0]
    
    def perceive_phenomenon(self, phenomenon: str, 
                          perspective_depth: int = 0) -> Dict[str, Any]:
        """
                     
        
        Args:
            phenomenon:       
            perspective_depth:       (  =  , 0=  ,   =  )
            
        Returns:
                 
        """
        #          
        current = self.root.current_depth
        if perspective_depth < current:
            self.zoom_in(current - perspective_depth)
        elif perspective_depth > current:
            self.zoom_out(perspective_depth - current)
        
        #               
        observation = {
            'phenomenon': phenomenon,
            'observed_from_depth': self.root.current_depth,
            'scale': self._get_current_scale(),
            'interpretation': self._interpret_at_scale(phenomenon, self.root.current_depth)
        }
        
        return observation
    
    def _get_current_scale(self) -> float:
        """           """
        if self.root.current_depth == 0:
            return 1.0
        elif self.root.current_depth < 0:
            idx = abs(self.root.current_depth) - 1
            if 0 <= idx < len(self.root.inner_universes):
                return self.root.inner_universes[idx].scale
        else:
            idx = self.root.current_depth - 1
            if 0 <= idx < len(self.root.outer_universes):
                return self.root.outer_universes[idx].scale
        return 1.0
    
    def _interpret_at_scale(self, phenomenon: str, depth: int) -> str:
        """             """
        if depth < -3:
            return f"'{phenomenon}' ( )           :              "
        elif depth < 0:
            return f"'{phenomenon}' ( )           :          "
        elif depth == 0:
            return f"'{phenomenon}' ( )            :       "
        elif depth < 3:
            return f"'{phenomenon}' ( )            :             "
        else:
            return f"'{phenomenon}' ( )              :             "
    
    def get_holographic_view(self) -> str:
        """
                  
        
                        
        """
        return f"""
  Tesseract Holographic View
================================

     : {self.root.identity}
     : {self.root.current_depth}
     : {self.root.expansion_mode.value}

      ({len(self.root.inner_universes)}  ):
{chr(10).join([f"  {i+1}. {layer.properties.get('description', 'Unknown')} (scale: {layer.scale:.2e})" 
               for i, layer in enumerate(self.root.inner_universes[:5])])}
{'  ...' if len(self.root.inner_universes) > 5 else ''}

     :
    {self.root.identity} (scale: 1.0)

      ({len(self.root.outer_universes)}  ):
{chr(10).join([f"  {i+1}. {layer.properties.get('description', 'Unknown')} (scale: {layer.scale:.2e})" 
               for i, layer in enumerate(self.root.outer_universes[:5])])}
{'  ...' if len(self.root.outer_universes) > 5 else ''}

       :
-              {len(self.root.inner_universes)}          
-          {len(self.root.outer_universes)}              
-                                  
-                    

"        ,        , 
                  ..."
"""
    
    def reset_to_center(self):
        """  (     )     """
        self.root.current_depth = 0
        self.root.expansion_mode = ExpansionDirection.STILL
        logger.info("   Reset to center (self)")


def demonstrate_tesseract_perspective():
    """Tesseract      """
    print("\n" + "="*60)
    print("TESSERACT PERSPECTIVE DEMONSTRATION")
    print("="*60)
    
    # Tesseract   
    tesseract = TesseractPerspective("Elysia")
    
    # 1.           
    print("\n1   Starting at SELF (     )")
    print(f"   Current depth: {tesseract.root.current_depth}")
    print(f"   Scale: 1.0 (human scale)")
    
    # 2.       
    print("\n2   Zooming IN (        )")
    for i in range(3):
        result = tesseract.zoom_in(1)
        print(f"     {result['description']} (scale: {result['scale']:.2e})")
    
    # 3.        
    print("\n3   Returning to center...")
    tesseract.reset_to_center()
    
    # 4.       
    print("\n4   Zooming OUT (       )")
    for i in range(3):
        result = tesseract.zoom_out(1)
        print(f"     {result['description']} (scale: {result['scale']:.2e})")
    
    # 5. Tesseract      
    print("\n5   TESSERACT VIEW (           )")
    tesseract.reset_to_center()
    view = tesseract.tesseract_view()
    print(f"   Total layers visible: {view['total_layers']}")
    print(f"   Insight: {view['insight']}")
    
    # 6.        
    print("\n6   HOLOGRAPHIC VIEW")
    print(tesseract.get_holographic_view())
    
    # 7.       (        )
    print("\n7   Observing 'consciousness' from different scales:")
    phenomenon = "consciousness"
    
    tesseract.reset_to_center()
    obs1 = tesseract.perceive_phenomenon(phenomenon, -2)
    print(f"   Micro: {obs1['interpretation']}")
    
    obs2 = tesseract.perceive_phenomenon(phenomenon, 0)
    print(f"   Human: {obs2['interpretation']}")
    
    obs3 = tesseract.perceive_phenomenon(phenomenon, 2)
    print(f"   Cosmic: {obs3['interpretation']}")
    
    print("\n" + "="*60)
    print("  Tesseract demonstration complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    demonstrate_tesseract_perspective()
