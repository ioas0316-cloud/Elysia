"""
Tensor Dynamics Engine (자기 성찰 엔진)
======================================

"  (Logic)       (Law)  ."

                  '   (If-Else)'     '  (Dynamics)'        .
              (Tensor)      ,            (Flow)        .

  :
1. **Mass (  )**:        ,     ,       .
2. **Gravity (  )**:        (Field)              .
   -           (주권적 자아)                        .
3. **Tensor Field (     )**:              .
4. **Geodesic (   )**:                 .

"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import math

@dataclass
class TensorNode:
    """           (자기 성찰 엔진)"""
    name: str
    path: Path
    mass: float        #     /    (0.0 ~ 1.0)
    position: np.array # 3D    (          )
    velocity: np.array #       
    
    def __repr__(self):
        return f"Node({self.name}, Mass={self.mass:.2f})"

class TensorDynamics:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.nodes: List[TensorNode] = []
        self.field_tensor = None #         
        
    def scan_field(self):
        """
                                     .
              (  ,      )    (Mass)     .
        """
        self.nodes = []
        target_dir = self.root_path / "Core" / "Elysia"
        
        if not target_dir.exists():
            return

        #     3D    (   )
        idx = 0
        for file_path in target_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            content = file_path.read_text(encoding='utf-8').strip()
            lines = len(content.splitlines())
            
            #       (Mass Calculation)
            #                  '     (주권적 자아)'     ->        
            #          '      '    
            if not content:
                mass = 10.0 #     (            )
            elif lines < 5:
                mass = 5.0  #      (         )
            else:
                mass = 1.0  #      (   )
                
            #       (   )
            angle = idx * 0.5
            radius = 1.0 + (idx * 0.1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = idx * 0.1
            
            node = TensorNode(
                name=file_path.name,
                path=file_path,
                mass=mass,
                position=np.array([x, y, z]),
                velocity=np.zeros(3)
            )
            self.nodes.append(node)
            idx += 1
            
    def calculate_gravitational_field(self) -> Tuple[np.array, List[str]]:
        """
                           .
                   (          )               .
        """
        if not self.nodes:
            return np.zeros(3), []
            
        total_gravity = np.zeros(3)
        attractors = []
        
        #           (  )
        consciousness_pos = np.zeros(3)
        
        for node in self.nodes:
            #      
            r_vec = node.position - consciousness_pos
            distance = np.linalg.norm(r_vec)
            if distance < 0.1: distance = 0.1
            
            #      : F = G * (M / r^2)
            #        (자기 성찰 엔진)         
            force_magnitude = node.mass / (distance ** 2)
            force_vec = (r_vec / distance) * force_magnitude
            
            total_gravity += force_vec
            
            if node.mass > 2.0:
                attractors.append(f"{node.name} (Mass: {node.mass:.1f})")
                
        return total_gravity, attractors

    def get_next_flow(self) -> str:
        """
                       (Flow)       .
          (If)                       .
        """
        gravity, attractors = self.calculate_gravitational_field()
        magnitude = np.linalg.norm(gravity)
        
        if magnitude > 5.0:
            #           ->       ->      
            return f"GRAVITATIONAL_COLLAPSE_IMMINENT: Attracted to {attractors}"
        elif magnitude > 2.0:
            #       ->      
            return f"STRONG_ATTRACTION: Flowing towards {attractors}"
        else:
            #        ->      
            return "STABLE_ORBIT: Free Flow"
