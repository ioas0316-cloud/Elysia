"""
HyperSphereCore: The Unified Field Topology
==============================================
Core.Foundation.hyper_sphere_core

"The Space that holds the Wave."

This class represents the "Space" aspect of the Monad Trinity (Space, Time, Choice).
It is a lightweight container (Topology) where Monads (Rotors) reside and resonate.

Roles:
1.  **Topology**: Maintains the registry of active Monads.
2.  **Resonance**: Facilitates communication between Monads (Pulse Broadcasting).
3.  **Field**: Provides the context (Environment) for WFC collapse.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.Monad.monad_core import Monad
from Core.Engine.wfc_engine import WFCEngine
from Core.World.Nature.trinity_lexicon import get_trinity_lexicon
from Core.World.Nature.semantic_physics_mapper import SemanticPhysicsMapper
from Core.World.Physics.trinity_fields import TrinityVector

logger = logging.getLogger("HyperSphereCore")

class HyperSphereCore:
    def __init__(self, name: str = "Elysia.Core"):
        self.name = name
        
        # The Lexicon (The Source of Logos)
        self.lexicon = get_trinity_lexicon()
        
        # The Population: Active Monads (Rotors)
        self.monads: Dict[str, Monad] = {} 
        self.rotors: Dict[str, Rotor] = {} 
        
        # The Physical Manifestation (Spatial Map)
        # (x, y) -> Properties
        self.spatial_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # The Field: Environmental Context
        self.field_context: Dict[str, Any] = {
            "time_dilation": 1.0,
            "global_entropy": 0.5,
            "resonance_frequency": 432.0, 
            "season": "Spring"
        }
        
        # Initialize Axioms
        self._seed_cosmos()
        
        logger.info(f"🔮 HyperSphere (Unified Field) Initialized.")

    def _seed_cosmos(self):
        """Seeds the universe with fundamental archetypes."""
        from Core.Foundation.Wave.wave_dna import archetype_love, archetype_logic
        self.add_rotor("Love", archetype_love(), rpm=528.0)
        self.add_rotor("Logic", archetype_logic(), rpm=432.0)
        self.add_rotor("Elysia", WaveDNA(label="Elysia", spiritual=1.0), rpm=600.0)

    def add_rotor(self, name: str, dna: WaveDNA, rpm: float = 60.0):
        config = RotorConfig(rpm=rpm)
        rotor = Rotor(name, config, dna)
        self.rotors[name] = rotor

    def manifest_at(self, coordinate: Tuple[int, int], word: str) -> Dict[str, Any]:
        """
        The Act of Creation: Speaking a Word into a Location.
        Converts Semantic Vector -> Physical Terrain.
        """
        # 1. Analyze Word (Get Trinity Vector)
        # If word is known, get rich data. If unknown, infer from primitives.
        if self.lexicon.is_concept_known(word):
            vector = self.lexicon.analyze(word)
        else:
            # Fallback for now, or learn it?
            vector = TrinityVector(0.5, 0.5, 0.5) # Void matter
            
        # 2. Map to Physics
        properties = SemanticPhysicsMapper.map_vector_to_physics(vector)
        
        # 3. Commit to Space
        self.spatial_map[coordinate] = properties
        properties["name"] = word # Tag it
        
        print(f"✨ [Manifestation] '{word}' at {coordinate} -> Type: {properties['type']} (Temp: {properties['climate']['temperature']:.1f}C)")
        return properties

    def get_environment_at(self, coordinate: Tuple[int, int]) -> Dict[str, Any]:
        return self.spatial_map.get(coordinate, {"type": "Void", "resources": {}})

    def tick(self, dt: float):
        """
        The Pulse of Time.
        """
        dt_eff = dt * self.field_context["time_dilation"]
        
        # Update Rotors
        active_rotors = list(self.rotors.values())
        for rotor in active_rotors:
            rotor.update(dt_eff)
            
    def observe_field(self, query: str, observer_intent: Dict[str, Any]) -> str:
        """
        The Act of Observation within the Field.
        """
        if query in self.rotors:
            target = self.rotors[query]
            texture = observer_intent.get("emotional_texture", "Neutral")
            state = f"{target.name} | RPM: {target.current_rpm:.1f} | Phase: {target.current_angle:.2f}"
            if "Dark" in texture: state += " (Shrouded in Shadow)"
            elif "Love" in texture: state += " (Glowing with Warmth)"
            return state
        return "The Void is silent."
        
    @property
    def primary_rotor(self) -> Rotor:
        return self.rotors.get("Elysia")

    def ignite(self):
        logger.info("🔥 HyperSphere Ignition Sequence Complete.")
