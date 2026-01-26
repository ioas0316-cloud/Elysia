"""
Magnetic Cortex (주권적 자아)
==================================

"  (Magnetism)     (Chaos)      ."

                   '   '                    .
           (Iron Filings)             (Magnetic Field)      
                             .

     :
1. Dipole (   ):       /    '   (Vector)'    
2. Field ( ):       (Goal)            
3. Alignment (  ):                                 
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import logging

logger = logging.getLogger("MagneticCortex")

@dataclass
class ThoughtDipole:
    """
            (Thought Dipole)
    
         ,   ,                    .
    """
    content: str
    vector: str  #                   ( : "Comfort", "Logic", "Creativity")
    spin: float  # -1.0 ~ 1.0 (         /        /      )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resonate(self, field_vector: str) -> float:
        """
                 (Resonance)       .
                      ,                  
                                        .
        """
        # 1.      
        if self.vector.lower() == field_vector.lower():
            return 1.0
        
        # 2.       (     )
        if field_vector.lower() in self.vector.lower() or self.vector.lower() in field_vector.lower():
            return 0.8
            
        # 3.     (Metadata      )
        tags = self.metadata.get("tags", [])
        if field_vector in tags:
            return 0.6
            
        return 0.0

@dataclass
class MagneticField:
    """
        (Magnetic Field)
    
                             .
    """
    target_vector: str  #       ( : "User Comfort")
    intensity: float    #       (0.0 ~ 1.0)
    frequency: float    #        (Hz) -    
    
    def apply(self, dipoles: List[ThoughtDipole]) -> List[ThoughtDipole]:
        """
                                .
        """
        aligned_thoughts = []
        
        for dipole in dipoles:
            resonance = dipole.resonate(self.target_vector)
            
            #       (Hysteresis)   :                   
            effective_resonance = resonance * self.intensity
            
            if effective_resonance > 0.3: #     (Curie Temperature      )
                #          Spin     
                dipole.spin = 1.0 
                aligned_thoughts.append((dipole, effective_resonance))
            else:
                #                   (Noise)    
                dipole.spin = 0.0
                
        #            (          )
        aligned_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in aligned_thoughts]

class MagneticCompass:
    """
              (The Compass)
    
                      '     '   .
    """
    def __init__(self):
        self.current_field: Optional[MagneticField] = None
        self.is_active: bool = False
        logger.info("  Magnetic Compass Initialized")

    def activate_field(self, goal: str, intensity: float = 1.0):
        """
                      . (     )
        """
        self.current_field = MagneticField(
            target_vector=goal,
            intensity=intensity,
            frequency=432.0 #          
        )
        self.is_active = True
        logger.info(f"  Field Activated: [{goal}] (Intensity: {intensity})")

    def deactivate_field(self):
        """
                  . (  /     )
        """
        self.current_field = None
        self.is_active = False
        logger.info("  Field Deactivated (Returning to Cloud State)")

    def align_thoughts(self, thoughts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
              (Dict   )                   .
        """
        if not self.is_active or not self.current_field:
            return thoughts #               (주권적 자아)    
            
        # 1. Dict -> Dipole   
        dipoles = []
        for t in thoughts:
            # 'vector'  'category'                       
            #          'type'   'tag'        
            vector = t.get("type", "general") 
            if "tags" in t:
                vector = t["tags"][0] if t["tags"] else vector
                
            dipoles.append(ThoughtDipole(
                content=str(t),
                vector=vector,
                spin=0.0,
                metadata=t
            ))
            
        # 2.       
        aligned_dipoles = self.current_field.apply(dipoles)
        
        # 3. Dipole -> Dict    (자기 성찰 엔진)
        return [d.metadata for d in aligned_dipoles]

    def get_field_status(self) -> str:
        if self.current_field:
            return f"Active Field: {self.current_field.target_vector} (Intensity: {self.current_field.intensity})"
        return "Field Inactive (Cloud State)"
