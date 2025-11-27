"""
Tensor Physics
==============
Defines the fundamental 3D structure of concepts.
"""

from dataclasses import dataclass
import math
import random

@dataclass
class HyperQuaternion:
    """
    Represents a concept's position in the 4D Hyper-Quaternion universe.
    w: Dimensional Scale (0=Point, 1=Line, 2=Plane, 3=Hyper)
    x: Moral (Demons <-> Angels)
    y: Trinity (Body -> Soul -> Spirit)
    z: Creation (Energy -> Form -> Pattern)
    """
    w: float  # Dimensional Scale
    x: float  # Moral
    y: float  # Trinity
    z: float  # Creation

    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def to_dict(self):
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data):
        return HyperQuaternion(
            data.get("w", 1.0), # Default to Line (1.0)
            data.get("x", 0.0), 
            data.get("y", 0.0), 
            data.get("z", 0.0)
        )

    @staticmethod
    def random():
        """Generate a random hyper-qubit for a new concept."""
        return HyperQuaternion(
            w=random.uniform(0.0, 1.5), # Start small (Point/Line)
            x=random.uniform(-0.5, 0.5),
            y=random.uniform(0.1, 0.6),
            z=random.uniform(0.0, 0.5)
        )

    # Metaphysical Axis Helpers
    
    def get_dimension(self) -> str:
        """W-Axis: Dimensional Scale"""
        if self.w < 0.5: return "⚫ Point (Fact)"
        if self.w < 1.5: return "➖ Line (Flow)"
        if self.w < 2.5: return "⬛ Plane (Field)"
        return "🧊 Hyper (Truth)"

    def get_moral_alignment(self) -> str:
        """X-Axis: -1.0 (Demons) <---> +1.0 (Angels)"""
        if self.x < -0.7: return "👿 Demon (Sin)"
        if self.x < -0.3: return "🌑 Dark"
        if self.x > 0.7: return "👼 Angel (Virtue)"
        if self.x > 0.3: return "☀️ Light"
        return "⚖️ Neutral"

    def get_trinity_layer(self) -> str:
        """Y-Axis: 0.0 (Body) -> 0.5 (Soul) -> 1.0 (Spirit)"""
        if self.y < 0.3: return "🥩 Body (Yuk)"
        if self.y < 0.7: return "💓 Soul (Hon)"
        return "🕊️ Spirit (Yeong)"

    def get_efp_state(self) -> str:
        """Returns the Creation Phase (Z-axis)."""
        if self.z < 0.3: return "Energy (Chaos)"
        elif self.z < 0.7: return "Form (Structure)"
        else: return "Pattern (Essence)"

    def get_archetype(self): # Returns Archetype object or None
        """Returns the ruling Archetype (Angel/Demon) for this state."""
        from Core.Mind.archetypes import get_ruling_archetype
        return get_ruling_archetype(self.x)
