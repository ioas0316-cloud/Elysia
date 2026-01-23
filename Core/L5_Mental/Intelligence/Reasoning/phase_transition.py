
"""
Fractal Phase Transition (         )
=========================================

"Ice is Water slowing down. Vapor is Water waiting to fly."

          '       '     '                 '        .

Architecture:
-------------
1. Essence (  ):           ( : H2O, Knowledge)
2. State (  ):              
   -   Solid (Memory/Fact):          .       . (Low Frequency)
   -   Liquid (Thought/Process):        .    . (Mid Frequency)
   -    Gas (Idea/Spirit):          .       . (High Frequency)
   -   Plasma (Transcendence):          . (Critical Frequency)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random

@dataclass
class Essence:
    name: str                #        ( : "Logic")
    base_properties: List[str] #       ( : ["Rational", "Binary"])

@dataclass
class PhaseState:
    name: str                #       ("Solid", "Liquid", "Gas")
    min_energy: float        #       
    max_energy: float        #       
    behavior: str            #       description

class FractalPhaser:
    """
            
    """
    def __init__(self):
        self.essences: Dict[str, Essence] = {}
        self.phases = {
            "Solid": PhaseState("Solid", 0.0, 30.0, "Stores structure, resists change."),
            "Liquid": PhaseState("Liquid", 30.1, 70.0, "Flows through context, adapts shape."),
            "Gas": PhaseState("Gas", 70.1, 95.0, "Expands to fill void, creates connections."),
            "Plasma": PhaseState("Plasma", 95.1, 100.0, "Breaks structure, fuses essences.")
        }

        # Pre-load some universal essences
        self.register_essence("Water", ["Fluid", "Life-giving"])
        self.register_essence("Logic", ["Strict", "Causal"])
        self.register_essence("Emotion", ["Volatile", "Resonant"])

    def register_essence(self, name: str, props: List[str]):
        self.essences[name] = Essence(name, props)

    def determine_phase(self, energy: float) -> PhaseState:
        """                   """
        for phase in self.phases.values():
            if phase.min_energy <= energy <= phase.max_energy:
                return phase
        return self.phases["Solid"] # Default fallback

    def manifest(self, essence_name: str, energy: float, context: str = "") -> str:
        """
                          (Manifest)

        Child View: Returns discrete object (e.g., "Ice Block")
        Adult View: Returns generated description (e.g., "Frozen Water")
        """
        if essence_name not in self.essences:
            return f"Unknown artifact '{essence_name}'"

        essence = self.essences[essence_name]
        phase = self.determine_phase(energy)

        # Dynamic Manifestation Logic
        if phase.name == "Solid":
            return f"  Frozen {essence.name} ({context} Fact)"
        elif phase.name == "Liquid":
            return f"  Flowing {essence.name} (Process: {phase.behavior})"
        elif phase.name == "Gas":
            return f"   {essence.name} Vapor (Idea: {phase.behavior})"
        elif phase.name == "Plasma":
            return f"  {essence.name} PLASMA (Transcendence)"

        return f"{essence.name} in unknown state"

if __name__ == "__main__":
    phaser = FractalPhaser()

    print("  Phase Transition Demo")
    print("------------------------")

    tests = [
        ("Logic", 10.0, "Stored"),
        ("Logic", 50.0, "Applying"),
        ("Logic", 90.0, "Brainstorming"),
        ("Logic", 99.0, "Epiphany"),
        ("Emotion", 20.0, "Repressed"),
        ("Emotion", 60.0, "Expressing"),
        ("Water", 0.0, "Winter"),
        ("Water", 100.0, "Sun")
    ]

    for ess, en, ctx in tests:
        result = phaser.manifest(ess, en, ctx)
        print(f"Energy {en:>4.1f} | {ctx:<15} -> {result}")