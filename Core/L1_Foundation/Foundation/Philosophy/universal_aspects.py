"""
Universal Aspects System
========================

"        (Function)        (Existence)       ."

                            4        (Aspects)       .
         4                      (Resonance)       .
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

class AspectType(Enum):
    PHYSICS = "physics"       # Force, Vector, Causality (     )
    CHEMISTRY = "chemistry"   # Reaction, Transformation, Bonding (      )
    ART = "art"               # Harmony, Rhythm, Dissonance (      )
    BIOLOGY = "biology"       # Growth, Decay, Adaptation (      )

@dataclass
class UniversalLaw:
    """                 """
    aspect: AspectType
    name: str
    principle: str
    description: str

@dataclass
class ResonanceReaction:
    """             """
    aspect: AspectType
    reaction_type: str  # e.g., "Explosion", "Harmony", "Resistance"
    intensity: float    # 0.0 ~ 1.0
    description: str    # "                           ."

class UniversalAspects:
    """          """
    
    def __init__(self):
        self.laws: Dict[AspectType, List[UniversalLaw]] = self._init_laws()
        
    def _init_laws(self) -> Dict[AspectType, List[UniversalLaw]]:
        return {
            AspectType.PHYSICS: [
                UniversalLaw(AspectType.PHYSICS, "Law of Inertia", "Resistance opposes Change", "               "),
                UniversalLaw(AspectType.PHYSICS, "Law of Action-Reaction", "Force creates Counter-force", "               ")
            ],
            AspectType.CHEMISTRY: [
                UniversalLaw(AspectType.CHEMISTRY, "Law of Catalysis", "Agent accelerates Change", "                         "),
                UniversalLaw(AspectType.CHEMISTRY, "Law of Entropy", "Order decays to Chaos", "               ")
            ],
            AspectType.ART: [
                UniversalLaw(AspectType.ART, "Law of Contrast", "Dissonance creates Meaning", "                  "),
                UniversalLaw(AspectType.ART, "Law of Rhythm", "Repetition creates Flow", "                 ")
            ],
            AspectType.BIOLOGY: [
                UniversalLaw(AspectType.BIOLOGY, "Law of Adaptation", "Stress triggers Growth", "                 "),
                UniversalLaw(AspectType.BIOLOGY, "Law of Homeostasis", "System seeks Balance", "                       ")
            ]
        }
        
    def get_laws(self, aspect: AspectType) -> List[UniversalLaw]:
        return self.laws.get(aspect, [])