"""
Semantic Nature (The Living Environment)
========================================
"The stage upon which the drama unfolds."

This module manages the population of SemanticObjects in the world.
It allows the environment to be populated by 'Concepts' rather than 'Meshes'.
"""

import uuid
import random
import math
from typing import List, Optional
from Core.1_Body.L4_Causality.World.Nature.semantic_object import SemanticObject
from Core.1_Body.L4_Causality.World.Nature.interaction_engine import InteractionEngine, InteractionResult
from Core.1_Body.L4_Causality.World.Physics.trinity_fields import TrinityVector

class SemanticNature:
    def __init__(self):
        self.objects: List[SemanticObject] = []
        self.interaction_engine = InteractionEngine()
        
    def manifest_concept(self, concept_id: str, name: str, position: List[float], properties: dict = {}) -> SemanticObject:
        """
        Spawns a concept into the physical world.
        """
        obj_id = str(uuid.uuid4())[:8]
        
        # In a real implementation, we would fetch Trinity Vector from the Concept
        # simple override for now
        trinity = None
        if concept_id == "Tree":
            trinity = TrinityVector(gravity=0.8, flow=0.2, ascension=0.4) # Deep roots, grows up
        elif concept_id == "Rock":
            trinity = TrinityVector(gravity=0.9, flow=0.0, ascension=0.1) # Solid
        elif concept_id == "BerryBush":
             trinity = TrinityVector(gravity=0.4, flow=0.5, ascension=0.3)
             
        obj = SemanticObject(
            id=obj_id,
            concept_id=concept_id,
            name=name,
            position=position,
            scale=[1.0, 1.0, 1.0],
            trinity_override=trinity,
            properties=properties
        )
        self.objects.append(obj)
        return obj

    def generate_wild_nature(self, count: int = 50, radius: float = 50.0):
        """
        Procedurally populates the area with random nature.
        """
        for _ in range(count):
            # Random Position
            angle = random.random() * 2 * math.pi
            r = math.sqrt(random.random()) * radius
            x = r * math.cos(angle)
            z = r * math.sin(angle)
            
            choice = random.random()
            if choice < 0.6: # 60% Trees
                self.manifest_concept("Tree", f"Wild Tree {int(x)}_{int(z)}", [x, 0, z])
            elif choice < 0.9: # 30% Rocks
                self.manifest_concept("Rock", f"Rock {int(x)}_{int(z)}", [x, 0, z])
            else: # 10% Bushes
                self.manifest_concept("BerryBush", f"Bush {int(x)}_{int(z)}", [x, 0, z], {"has_berries": True})

    def get_objects_in_range(self, position: List[float], radius: float) -> List[SemanticObject]:
        """
        Returns objects close to a location.
        """
        found = []
        for obj in self.objects:
            # Simple Euclidean distance
            dx = obj.position[0] - position[0]
            dy = obj.position[1] - position[1]
            dz = obj.position[2] - position[2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq <= radius * radius:
                found.append(obj)
        return found
        
    def interact(self, subject_name: str, tool_concept: str, target_id: str) -> InteractionResult:
        """
        Proxy for interactions.
        """
        target = next((o for o in self.objects if o.id == target_id), None)
        if not target:
            return InteractionResult(False, "Target not found.")
            
        result = self.interaction_engine.resolve(subject_name, tool_concept, target)
        
        if result.destroyed:
            self.objects.remove(target)
            
        return result

    def tick(self):
        """
        Environmental updates (Growth, Decay).
        """
        # Example: Trees slowly heal or grow
        for obj in self.objects:
            if obj.concept_id == "Tree" and obj.integrity < 100:
                obj.integrity += 0.1 # Regrowth
