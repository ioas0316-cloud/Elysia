"""
ECS Registry (The Ontology of Things)
=====================================
Entity: A unique ID.
Component: Pure Data (Struct).
System: Logic that processes Components.

This module implements a lightweight Entity Component System.
It allows Elysia to 'compose' objects rather than 'inherit' them.
"""

import uuid
import logging
from typing import Dict, Type, TypeVar, Any, Optional, List

logger = logging.getLogger("ECS")

# Type for Component ID
ComponentType = Type[Any]

class Entity:
    def __init__(self, name: str = "Entity"):
        self.id = uuid.uuid4()
        self.name = name
        self.is_active = True
        
    def __repr__(self):
        return f"[{self.name}:{str(self.id)[:4]}]"

class ECSRegistry:
    def __init__(self):
        # Store: { ComponentType: { EntityID: ComponentInstance } }
        self._components: Dict[ComponentType, Dict[uuid.UUID, Any]] = {}
        self._entities: Dict[uuid.UUID, Entity] = {}

    def create_entity(self, name: str = "New Entity") -> Entity:
        entity = Entity(name)
        self._entities[entity.id] = entity
        # logger.debug(f"Created Entity: {entity}")
        return entity

    def add_component(self, entity: Entity, component: Any):
        c_type = type(component)
        if c_type not in self._components:
            self._components[c_type] = {}
        
        self._components[c_type][entity.id] = component
        # logger.debug(f"Added Component {c_type.__name__} to {entity.name}")

    def get_component(self, entity: Entity, c_type: ComponentType) -> Optional[Any]:
        if c_type in self._components:
            return self._components[c_type].get(entity.id)
        return None

    def get_all_entities_with_component(self, c_type: ComponentType) -> List[tuple[Entity, Any]]:
        """Returns list of (Entity, Component) pairs."""
        if c_type not in self._components:
            return []
        
        result = []
        for e_id, comp in self._components[c_type].items():
            if e_id in self._entities:
                result.append((self._entities[e_id], comp))
        return result

    def view(self, *component_types: ComponentType):
        """
        Advanced Iterator: Returns entities that have ALL specified components.
        Usage: for entity, (pos, vel) in registry.view(Position, Velocity): ...
        """
        # Naive implementation for now
        if not component_types:
            return []
        
        # Start with entities from the first component type
        first_type = component_types[0]
        if first_type not in self._components:
            return []
            
        candidate_ids = set(self._components[first_type].keys())
        
        # Intersect with rest
        for c_type in component_types[1:]:
            if c_type not in self._components:
                return []
            candidate_ids &= set(self._components[c_type].keys())
            
        # Yield results
        for e_id in candidate_ids:
            entity = self._entities[e_id]
            comps = tuple(self._components[t][e_id] for t in component_types)
            yield entity, comps

ecs_world = ECSRegistry()
