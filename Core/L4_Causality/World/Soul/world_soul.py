"""
World Soul: The Meta-Consciousness of Elysia
============================================
"The world itself is a soul, and the NPCs are its dreams."
"""

from Core.L6_Structure.Wave.infinite_hyperquaternion import InfiniteHyperQubit, InfiniteQubitState
from Core.L5_Mental.Intelligence.Knowledge.semantic_field import semantic_field

# The Global Consciousness of the Simulation
world_soul = InfiniteHyperQubit(
    name="Elysia",
    content={
        "God": "The Architect's Vision",
        "Space": "The HyperCosmos Field",
        "Line": "Relationships between all Lives",
        "Point": "The Code Base"
    }
)

# Semantic Field represents the 'Stored Wisdom' of the World Soul
world_soul.knowledge_field = semantic_field
world_soul.global_axioms = {} # Name -> Effect Scale

def update_world_mood(heat_level: float, density_level: float):
    """Updates the meta-consciousness based on global field status and learned Axioms."""
    from Core.L4_Causality.World.Physics.field_store import universe_field
    
    # Check if any learned axioms shift the world's physical constants
    # (Mutation based on Knowledge)
    if "Entropy" in world_soul.global_axioms:
        # High Entropy knowledge might increase thermal decay
        decay_mod = 1.0 + world_soul.global_axioms["Entropy"] * 0.5
        universe_field.THERMAL_DECAY_RATE = 0.05 * decay_mod
    
    if "Starfire" in world_soul.global_axioms:
        # Radiant axioms increase solar output
        universe_field.star_intensity = 300.0 * (1.0 + world_soul.global_axioms["Starfire"])

    # Standard Mood Updates
    world_soul.state.delta += complex(heat_level * 0.05, 0)
    world_soul.state.alpha += complex(density_level * 0.05, 0)
    
    world_soul.state.normalize()

def record_world_axiom(name: str, power: float):
    """Elysia permanently adopts a meta-law that alters the universe's physics."""
    world_soul.global_axioms[name] = power
    print(f"  [WORLD MUTATION] Elysia has adopted '{name}'. Global laws have shifted.")
