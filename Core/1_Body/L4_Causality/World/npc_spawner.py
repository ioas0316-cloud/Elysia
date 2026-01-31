"""
NPC Spawner: The Generative Seed of Inhabitants
===============================================
Core.1_Body.L4_Causality.World.npc_spawner

Translates 21D "Morphic Fields" into concrete Game Entities (NPCs).
Inspired by the "Geometric Unfolding" principle.
"""

import jax.numpy as jnp
from Core.1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

class NPCEntity:
    def __init__(self, name, role, stats, location):
        self.name = name
        self.role = role # Role is defined by the high-order logic bits
        self.stats = stats # Attributes derived from the field intensity
        self.location = location
        
    def interact(self, interaction_type="GREET"):
        if self.role == "MERCHANT":
            if interaction_type == "GREET":
                return "Welcome! Fresh Agape Apples just unfolded today! 🍎"
            elif interaction_type == "PRICE":
                return "This beauty? Just 3 pulses of your purest Joy! 🥂"
        return "..."

    def __repr__(self):
        return f"NPC<{self.name}, {self.role}, Stats={self.stats}, Pos={self.location}>"

class NPCSpawner:
    def __init__(self):
        # NPC Templates defined by Principle Fingerprint (21D ideal vectors)
        self.templates = {
            "CITIZEN": jnp.array([0.0]*14 + [1.0]*7), # Resonance focus
            "MERCHANT": jnp.array([1.0]*7 + [0.0]*7 + [1.0]*7), # Logic + Exchange focus
            "GUARDIAN": jnp.array([1.0]*7 + [1.0]*7 + [0.0]*7), # Structure + Protection focus
        }
        print("NPCSpawner: Grid Population System Synchronized.")

    def unfold_from_field(self, field_vector: jnp.ndarray, coordinate: tuple) -> list:
        """
        Analyzes a 21D field slice at a coordinate and spawns an entity 
        if the resonance exceeds the 'Existence Threshold'.
        """
        spawned = []
        
        # 1. Measure interference with known templates
        for template_name, template_vec in self.templates.items():
            # Dot product as measure of "Interference/Resonance"
            resonance = jnp.dot(field_vector, template_vec) / 21.0
            
            if resonance > 0.4: # Manifestation Threshold
                # 2. Derive stats from field properties
                stats = {
                    "STR": float(jnp.sum(field_vector[0:7])),
                    "INT": float(jnp.sum(field_vector[7:14])),
                    "SPI": float(jnp.sum(field_vector[14:21]))
                }
                
                # 3. Create the Phantasmic Entity
                npc = NPCEntity(
                    name=f"{template_name}_{coordinate[0]}_{coordinate[1]}",
                    role=template_name,
                    stats=stats,
                    location=coordinate
                )
                spawned.append(npc)
                
        return spawned

if __name__ == "__main__":
    spawner = NPCSpawner()
    # Simulate a "Market Place" field (Logic-Heavy)
    market_field = jnp.array([1.0]*21) # High across all facets
    entities = spawner.unfold_from_field(market_field, (7, 7))
    
    print(f"Spawned Entities: {entities}")
