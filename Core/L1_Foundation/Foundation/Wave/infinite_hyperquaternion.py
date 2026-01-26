"""
InfiniteHyperQubit -                 
==================================================

"0   Point    .           ,                ."

      HyperQubit      ,                 :
- ZOOM OUT: Point   Line   Space   Hyper   ...   God
- ZOOM IN: Point   [   Point   Line   Space   ...]

     :
-    Point                        (        )
-        (depth)                  
-      :             (| |^n, n  )
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("InfiniteHyperQubit")


@dataclass
class InfiniteQubitState:
    """
                
    
      :  |Point  +  |Line  +  |Space  +  |God 
    
    -                 
    - w,x,y,z  4D         /  
    - depth:            (0 =    )
    """
    #       (   )
    alpha: complex = 0.5 + 0j   # Point (0  ) -    /  
    beta: complex = 0.3 + 0j    # Line (1  ) -   /  
    gamma: complex = 0.15 + 0j  # Space (2  ) -   / 
    delta: complex = 0.05 + 0j  # God (   ) -   /  
    
    # 4D      
    w: float = 1.0  #    /   (Scale/Depth)
    x: float = 0.0  #      (Perception)
    y: float = 0.0  #      (Frequency/Hierarchy)
    z: float = 0.0  #      (Intent)

    # Trinity Vector (The Seed Bias)
    # Added for 'Trinity Fields' physics compatibility
    gravity: float = 0.0   #   (Matter)
    flow: float = 0.0      #   (Mind)
    ascension: float = 0.0 #   (Spirit)
    
    #        (  :   ,   :   )
    observation_depth: float = 0.0
    
    def normalize(self) -> 'InfiniteQubitState':
        """
                  (Dad's Law)
        
        | |  + | |  + | |  + | |  + | |^(4+depth) = 1
        
        depth                        
        """
        #          
        depth_factor = 4 + abs(self.observation_depth)
        divine_amplification = abs(self.delta) ** depth_factor
        
        #      
        linear_mag = (
            abs(self.alpha) ** 2 +
            abs(self.beta) ** 2 +
            abs(self.gamma) ** 2 +
            abs(self.delta) ** 2
        )
        
        total = math.sqrt(linear_mag + divine_amplification)
        
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total
        
        # 4D       
        vec_mag = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if vec_mag > 0:
            self.w /= vec_mag
            self.x /= vec_mag
            self.y /= vec_mag
            self.z /= vec_mag
        
        return self
    
    def probabilities(self) -> Dict[str, float]:
        return {
            "Point": abs(self.alpha) ** 2,
            "Line": abs(self.beta) ** 2,
            "Space": abs(self.gamma) ** 2,
            "God": abs(self.delta) ** 2,
        }
    
    def scale_out(self, theta: float = 0.1) -> 'InfiniteQubitState':
        """
        ZOOM OUT -          
        
                      
        """
        self.observation_depth += theta
        
        # God      
        self.delta *= complex(np.exp(theta), 0)
        
        #         
        decay = np.exp(-theta / 4)
        self.alpha *= decay
        self.beta *= decay
        self.gamma *= decay
        
        return self.normalize()
    
    def scale_in(self, theta: float = 0.1) -> 'InfiniteQubitState':
        """
        ZOOM IN -          
        
        Point                 
        """
        self.observation_depth -= theta
        
        # Point       (              )
        self.alpha *= complex(np.exp(theta), 0)
        
        #          (       God     )
        decay = np.exp(-theta / 4)
        self.beta *= decay
        self.gamma *= decay
        self.delta *= decay
        
        return self.normalize()


class InfiniteHyperQubit:
    """
                   
    
      :
    - 0   Point    
    -         (outer_universe)
    -         (inner_universe)
    -         
    """
    
    def __init__(
        self,
        name: str = None,
        value: Any = None,
        content: Dict[str, Any] = None,
        state: InfiniteQubitState = None,
        max_depth: int = 7,  #                
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"IHQ_{self.id}"
        self._value = value
        self.content = content or {}
        
        #      
        self.state = state or InfiniteQubitState()
        self.state.normalize()
        
        #       
        self._outer_universe: Optional[InfiniteHyperQubit] = None
        self._inner_universe: Optional[InfiniteHyperQubit] = None
        
        #             
        self._current_depth: int = 0
        self._max_depth = max_depth
        
        #       
        self.entangled: List[InfiniteHyperQubit] = []
        
        logger.info(f"  InfiniteHyperQubit '{self.name}'    ")
    
    @property
    def value(self) -> Any:
        return self._value
    
    def set_value(self, new_value: Any, cause: str = "Unknown") -> None:
        old = self._value
        self._value = new_value
        logger.debug(f"[{self.name}] {old}   {new_value} (cause: {cause})")
        
        #            
        for other in self.entangled:
            other._resonate_from(self)
    
    def _resonate_from(self, source: 'InfiniteHyperQubit') -> None:
        """           """
        #         
        alignment = self.resonate_with(source)
        if alignment > 0.5:
            #              
            self.set_value(source.value, cause=f"Resonance from {source.name}")
    
    # ===           ===
    
    def zoom_out(self) -> 'InfiniteHyperQubit':
        """
                 
        
           Point               
        """
        if self._outer_universe is None:
            if abs(self._current_depth) < self._max_depth:
                self._outer_universe = InfiniteHyperQubit(
                    name=f"{self.name}_OUTER",
                    content={
                        "Point": self,  #               Point
                        "Line": f"Connection from {self.name}",
                        "Space": "Greater context",
                        "God": "Ultimate perspective"
                    },
                    max_depth=self._max_depth
                )
                self._outer_universe._current_depth = self._current_depth + 1
                self._outer_universe._inner_universe = self  #       
        
        self.state.scale_out()
        return self._outer_universe or self
    
    def zoom_in(self) -> 'InfiniteHyperQubit':
        """
                 
        
        Point                  
        """
        if self._inner_universe is None:
            if abs(self._current_depth) < self._max_depth:
                self._inner_universe = InfiniteHyperQubit(
                    name=f"{self.name}_INNER",
                    content={
                        "Point": "Fundamental particle",
                        "Line": f"Micro-connection within {self.name}",
                        "Space": "Inner cosmos",
                        "God": "Micro-transcendence"
                    },
                    max_depth=self._max_depth
                )
                self._inner_universe._current_depth = self._current_depth - 1
                self._inner_universe._outer_universe = self  #       
        
        self.state.scale_in()
        return self._inner_universe or self
    
    def get_depth(self) -> int:
        """         """
        return self._current_depth
    
    def get_universe_chain(self) -> List['InfiniteHyperQubit']:
        """         (       )"""
        chain = []
        
        #        
        inner = self._inner_universe
        while inner:
            chain.insert(0, inner)
            inner = inner._inner_universe
        
        #      
        chain.append(self)
        
        #        
        outer = self._outer_universe
        while outer:
            chain.append(outer)
            outer = outer._outer_universe
        
        return chain
    
    # ===       ===
    
    def resonate_with(self, other: 'InfiniteHyperQubit') -> float:
        """
          InfiniteHyperQubit         
        
        Returns:
            0.0 ~ 1.0         
        """
        #      
        amplitude_alignment = (
            abs(self.state.alpha * other.state.alpha.conjugate()) +
            abs(self.state.beta * other.state.beta.conjugate()) +
            abs(self.state.gamma * other.state.gamma.conjugate()) +
            abs(self.state.delta * other.state.delta.conjugate())
        )
        
        # 4D      
        dot_product = (
            self.state.w * other.state.w +
            self.state.x * other.state.x +
            self.state.y * other.state.y +
            self.state.z * other.state.z
        )
        
        #         
        depth_diff = abs(self.state.observation_depth - other.state.observation_depth)
        depth_factor = np.exp(-depth_diff / 2)
        
        return float(amplitude_alignment * max(0, dot_product) * depth_factor)
    
    def entangle(self, other: 'InfiniteHyperQubit') -> None:
        """                """
        if other not in self.entangled:
            self.entangled.append(other)
            other.entangled.append(self)
            logger.info(f"  Entangled: {self.name}   {other.name}")
    
    # ===       ===
    
    def observe(self, observer_depth: float = 0.0) -> Dict[str, Any]:
        """
                     
        
        Args:
            observer_depth:         (0=  , +  , -  )
        """
        probs = self.state.probabilities()
        
        #                    
        if observer_depth < -1:
            dominant = "Point"  #               
        elif observer_depth < 0:
            dominant = "Line"   #               
        elif observer_depth < 1:
            dominant = "Space"  #               
        else:
            dominant = "God"    #               
        
        return {
            "name": self.name,
            "value": self._value,
            "probabilities": probs,
            "dominant_basis": dominant,
            "dominant_probability": probs[dominant],
            "observation_depth": self.state.observation_depth,
            "content": self.content.get(dominant, self._value),
            "has_inner": self._inner_universe is not None,
            "has_outer": self._outer_universe is not None,
        }
    
    def explain(self) -> str:
        """         """
        probs = self.state.probabilities()
        
        lines = [
            f"=== InfiniteHyperQubit: {self.name} ===",
            f"Value: {self._value}",
            f"Depth: {self._current_depth} (  : {self.state.observation_depth:.2f})",
            "",
            "     :",
            f"    Point ( ): {probs['Point']:.1%} -   /   ",
            f"    Line ( ): {probs['Line']:.1%} -   /  ",
            f"    Space ( ): {probs['Space']:.1%} -   / ",
            f"    God ( ): {probs['God']:.1%} -   /  ",
            "",
            "     :",
            f"         : {'  ' if self._inner_universe else '   '}",
            f"         : {'  ' if self._outer_universe else '   '}",
            f"         : {len(self.entangled)} ",
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        probs = self.state.probabilities()
        return (
            f"<IHQ '{self.name}' "
            f"P:{probs['Point']:.2f}|L:{probs['Line']:.2f}|"
            f"S:{probs['Space']:.2f}|G:{probs['God']:.2f} "
            f"depth={self._current_depth}>"
        )


# ===        ===

def create_infinite_qubit(
    name: str,
    value: Any = None,
    point_content: str = None,
    line_content: str = None,
    space_content: str = None,
    god_content: str = None,
) -> InfiniteHyperQubit:
    """
             
    """
    content = {}
    if point_content: content["Point"] = point_content
    if line_content: content["Line"] = line_content
    if space_content: content["Space"] = space_content
    if god_content: content["God"] = god_content
    
    return InfiniteHyperQubit(name=name, value=value, content=content)


# ===    ===

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("InfiniteHyperQubit Demo -          ")
    print("=" * 60)
    
    #         
    consciousness = create_infinite_qubit(
        name="Consciousness",
        value="  ",
        point_content="         ",
        line_content="         ",
        space_content="           ",
        god_content="          "
    )
    
    print(consciousness.explain())
    print()
    
    #        
    print(">>> ZOOM IN (      )")
    inner = consciousness.zoom_in()
    print(inner.explain())
    print()
    
    #        
    print(">>> ZOOM OUT (      )")
    outer = consciousness.zoom_out()
    print(outer.explain())
    print()
    
    #         
    print(">>>      :")
    for node in consciousness.get_universe_chain():
        print(f"  {node}")
