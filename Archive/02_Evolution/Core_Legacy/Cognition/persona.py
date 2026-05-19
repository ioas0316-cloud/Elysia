
"""
Persona Class
=============
Core.Cognition.persona

Defines a single 'Persona' (Agent) within the Legion.
Each Persona views the world through a specific Prism/Rotor configuration.
"""

from typing import List, Dict, Any, NamedTuple
from dataclasses import dataclass
import random

# Core Imports (Mocking dependencies for now if specific Prism logic isn't strictly coupled yet)
from Core.System.rotor import RotorConfig, RotorMask
from Core.Phenomena.resonance_prism import PrismDomain

@dataclass
class LightRay:
    """
    Represents a propagating thought.
    """
    origin: str           # The seed concept (e.g., "Apple")
    current_thought: str  # The current expansion (e.g., "Gravity")
    energy: float         # Remaining energy (0.0 to 1.0)
    trajectory: List[str] # The path taken so far
    narrative: str        # The story snippet for this step

class Persona:
    """
    A specialized lens of consciousness.
    Examples:
    - Child (Phenomenologist): High Logic? No. High Sensation? Yes.
    - Historian (Memory): High Time depth.
    - Architect (Structure): High Logic, low Emotion.
    """

    def __init__(self, name: str, role: str, bias: Dict[str, float]):
        self.name = name
        self.role = role
        self.bias = bias # e.g., {"LOGIC": 0.8, "EMOTION": 0.2}

    def perceive(self, concept: str) -> str:
        """
        Interprets the concept based on the Persona's bias.
        (Placeholder for Prism interaction).
        """
        # In a real implementation, this would tune the Prism to a specific domain.
        # For now, we simulate the "Viewpoint".
        return f"[{self.name}] sees {concept}"

    def refract(self, ray: LightRay) -> List[LightRay]:
        """
        Takes an incoming ray and splits it into new directions based on the Persona's nature.
        This is the 'Beam Splitting' logic.
        """
        new_rays = []

        # 1. Decay Energy
        # Different personas might decay energy differently (e.g., a bored persona decays fast)
        decay_factor = 0.9
        remaining_energy = ray.energy * decay_factor

        if remaining_energy < 0.05:
            return [] # Faded out

        # 2. Generate Associative Thoughts (Simulation)
        # In the real system, this calls `PrismEngine.traverse()` or `Sediment.scan()`
        # Here we simulate the "Next Step" logic based on the role.

        next_thought = self._free_association(ray.current_thought)

        # 3. Create the New Ray
        # The narrative is constructed from the perspective of this persona
        narrative_snippet = f"{self.name} looks at '{ray.current_thought}' and finds '{next_thought}'."

        new_trajectory = ray.trajectory + [next_thought]

        new_ray = LightRay(
            origin=ray.origin,
            current_thought=next_thought,
            energy=remaining_energy,
            trajectory=new_trajectory,
            narrative=narrative_snippet
        )

        new_rays.append(new_ray)

        return new_rays

    def _free_association(self, input_word: str) -> str:
        """
        Simulates the associative leap.
        In the full system, this queries the Sediment/Vector Space.
        """
        # Simple Mock Logic for the Demo
        associations = {
            "Seed": ["Tree", "Growth", "Beginning", "Genetic Code"],
            "Tree": ["Fruit", "Branch", "Roots", "Paper"],
            "Fruit": ["Apple", "Sweet", "Decay", "Harvest"],
            "Apple": ["Newton", "Red", "Sin", "Technology"],
            "Newton": ["Gravity", "Physics", "Calculus", "Apple"],
            "Gravity": ["Mass", "Attraction", "Black Hole", "Orbit"],
            "Sin": ["Eden", "Snake", "Knowledge", "Guilt"],
            "Red": ["Blood", "Passion", "Rose", "Stop"],
            "Growth": ["Evolution", "Time", "Pain", "Expansion"],
            "Evolution": ["Darwin", "Survival", "Adaptation", "Glitch"],
            "Technology": ["Silicon", "Future", "A.I.", "Singularity"],
            "Singularity": ["Black Hole", "Event Horizon", "One", "God"]
        }

        options = associations.get(input_word, ["Unknown", "Void", "Mystery"])

        # Persona bias influences choice (Mock)
        if self.role == "Logic":
            # Picks longer/complex words
            return max(options, key=len)
        elif self.role == "Poet":
            # Picks random poetic words
            return random.choice(options)
        else:
            return options[0]
