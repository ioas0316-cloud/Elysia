"""
Core/Life/resource_system.py

A system for managing the passive change of resources for all living entities.
This encapsulates the logic for hunger, aging, and natural regeneration.
"""

from typing import List
from .entity import LivingEntity

class PassiveResourceSystem:
    """
    Manages the passive updates of resources for a collection of entities.
    """
    def __init__(self, entities: List[LivingEntity]):
        self.entities = entities

    def update(self):
        """
        Processes one tick of passive resource changes for all entities.
        This is designed to be called by the Kernel's main tick loop.
        """
        for entity in self.entities:
            if not entity.is_alive:
                continue

            # 1. Aging
            entity.age += 1
            if entity.age >= entity.max_age:
                entity.hp.subtract(entity.hp.max) # Die of old age

            # 2. Hunger and Hydration Depletion
            # Simple linear decay for now. Can be replaced with a more complex
            # model using Core/Math engines later.
            entity.hunger.subtract(0.15)
            entity.hydration.subtract(0.1)

            # 3. Starvation and Dehydration Damage
            if entity.hunger.current <= 0:
                entity.hp.subtract(2.0)

            if entity.hydration.current <= 0:
                entity.hp.subtract(0.5)

    def add_entity(self, entity: LivingEntity):
        """Adds a new entity to be managed by the system."""
        if entity not in self.entities:
            self.entities.append(entity)

    def remove_entity(self, entity: LivingEntity):
        """Removes an entity from the system."""
        self.entities = [e for e in self.entities if e.id != entity.id]
