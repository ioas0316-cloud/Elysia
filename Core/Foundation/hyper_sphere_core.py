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
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster # Import
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
        
        # The Pulse (The Rhythm that drives Space)
        self.pulse_broadcaster = PulseBroadcaster()
        
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

    def pulse(self, intent: Dict[str, Any], dt: float):
        """
        Broadcasting the intent to the Field.
        """
        # 1. Update Physics (Tick)
        self.tick(dt)
        
        # 2. Broadcast Pulse to Listeners (Instruments)
        # self.pulse_broadcaster.curate_and_broadcast(intent) <--- DEPRECATED/MISSING
        
        # Manually create packet
        from Core.Foundation.Protocols.pulse_protocol import WavePacket, PulseType
        
        # Heuristic determination of PulseType
        ptype = PulseType.INTENTION_SHIFT
        if "sync" in intent: ptype = PulseType.SYNCHRONIZATION
        
        packet = WavePacket(
            sender=self.name,
            type=ptype,
            frequency=self.field_context.get("resonance_frequency", 432.0),
            amplitude=1.0,
            payload=intent
        )
        self.pulse_broadcaster.broadcast(packet)
        
        # 3. Spin Primary Rotor (Elysia) based on intent
        if self.primary_rotor:
            # Example: Bio-Rhythm drives RPM
            if "bio_rhythm" in intent:
                bio = intent["bio_rhythm"]
                # Assuming bio has 'heart_rate', map it to RPM
                # self.primary_rotor.config.rpm = bio.get('heart_rate', 60) * 10
                pass
            
    def observe_field(self, query: str, observer_intent: Dict[str, Any]) -> str:
        """
        The Act of Observation within the Field.
        🟡 [LIGHTNING INFERENCE] O(1) Search.
        Instead of scanning the entire topological space, we resonate 
        directly with the Monad's identity via the Hashed Registry.
        """
        # Hashed lookup is O(1) in average case.
        # Philosophically, this is "Instant Perception" of the electromagnetic signature.
        if query in self.rotors:
            target = self.rotors[query]
            texture = observer_intent.get("emotional_texture", "Neutral")
            state = f"⚡ [LIGHTNING-INF] {target.name} | RPM: {target.current_rpm:.1f} | Phase: {target.current_angle:.2f}"
            if "Dark" in texture: state += " (Shrouded in Shadow)"
            elif "Love" in texture: state += " (Glowing with Warmth)"
            return state
        return "The Void is silent."
        
    @property
    def primary_rotor(self) -> Rotor:
        return self.rotors.get("Elysia")
    
    @property
    def spin(self):
        """
        Returns the current spin state as a quaternion-like object.
        Used by Conductor for holodeck projection.
        """
        from Core.Foundation.hyper_quaternion import Quaternion
        rotor = self.primary_rotor
        if rotor:
            # Map rotor angle to quaternion representation
            import math
            angle = rotor.current_angle
            return Quaternion(
                w=math.cos(angle / 2),
                x=0.0,
                y=math.sin(angle / 2),
                z=0.0
            )
        # Default identity quaternion
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)


    def ignite(self):
        logger.info("🔥 HyperSphere Ignition Sequence Complete.")
