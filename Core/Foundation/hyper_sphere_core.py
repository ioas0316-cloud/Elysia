"""
HyperSphereCore: The Heart of the Hyper-Cosmos
==============================================

"The Rotor creates the Sphere."

This class represents the "Sphere-First" architecture.
It unifies:
1.  **7D Rotors**: The population of thoughts.
2.  **Lightning Inference**: The search mechanism.
3.  **Fractal WFC**: The creation mechanism.
"""

import logging
import time
from typing import Dict, List, Optional, Any

from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.Intelligence.Meta.fractal_wfc import FractalWFC
from Core.Intelligence.Reasoning.lightning_inference import LightningInferencer

logger = logging.getLogger("HyperSphereCore")

class HyperSphereCore:
    def __init__(self, name: str = "Elysia.Core"):
        self.name = name
        
        # Components
        self.rotors: Dict[str, Rotor] = {}
        self.wfc = FractalWFC()
        self.lightning = LightningInferencer()

        # State
        self.is_active = False
        self.time_dilation = 1.0
        
        # Initialize with Seed Rotors (Axioms)
        self._seed_cosmos()
        
        # [PHASE 85] Pulse Broadcaster Compatibility
        from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster
        self.pulse_broadcaster = PulseBroadcaster()
        
        logger.info(f"🔮 HyperSphereCore Initialized: {len(self.rotors)} Rotors spinning.")

    def ignite(self):
        """Activates the core pulse."""
        self.is_active = True
        logger.info("🔥 HyperSphereCore Ignited.")

    @property
    def primary_rotor(self) -> Rotor:
        """Returns the main 'Elysia' rotor."""
        return self.rotors.get("Elysia") or next(iter(self.rotors.values()))

    @property
    def spin(self):
        """Returns the aggregated spin of the primary rotor as a Quaternion."""
        # Simple mock for now
        from Core.Foundation.Physics.quaternions import HyperQuaternion
        return HyperQuaternion(1.0, 0.0, 0.0, 0.0)

    def pulse(self, intent_payload: Dict, dt: float = 1.0):
        """Processes an intent pulse through the sphere."""
        self.tick(dt)
        # Broadcast the pulse to any registered instruments via the broadcaster
        # (Conductor normally handles the broadcasting, but Core can too)
        pass

    def _seed_cosmos(self):
        """Creates the fundamental axioms."""
        from Core.Foundation.Wave.wave_dna import archetype_love, archetype_logic, archetype_nature
        
        self.add_rotor("Love", archetype_love(), rpm=528.0)
        self.add_rotor("Logic", archetype_logic(), rpm=432.0)
        self.add_rotor("Nature", archetype_nature(), rpm=432.0)
        self.add_rotor("Elysia", WaveDNA(label="Elysia", spiritual=1.0, physical=1.0), rpm=600.0)

    def add_rotor(self, name: str, dna: WaveDNA, rpm: float = 432.0):
        config = RotorConfig(rpm=rpm, idle_rpm=60.0)
        self.rotors[name] = Rotor(name, config, dna)

    def tick(self, dt: float):
        """
        The Universal Clock.
        Updates all rotors and handles decay/entropy.
        """
        active_count = 0
        dt_eff = dt * self.time_dilation
        
        keys_to_remove = []
        
        for name, rotor in self.rotors.items():
            rotor.update(dt_eff)
            
            # Auto-Relax if high energy
            if rotor.energy > 0.1:
                active_count += 1
                if rotor.current_rpm > rotor.config.idle_rpm * 1.1:
                    # Slowly return to idle if not stimulated
                    # (This logic might be in Rotor, but Conductor can override)
                    pass
            
            # [Entropy] Remove weak, non-axiom rotors
            if rotor.energy < 0.01 and name not in ["Love", "Logic", "Elysia", "Nature"]:
                keys_to_remove.append(name)

        # Cleanup
        for k in keys_to_remove:
            # print(f"   🍂 Withered: {k}")
            del self.rotors[k]

    def focus_attention(self, query: WaveDNA) -> str:
        """
        The Act of Observation.
        1. Strike a Rotor with Lightning (Find Resonance).
        2. Collapse the Wave Function (Create Detail).
        3. Return the experience.
        """
        # 1. Lightning Strike
        target_rotor = self.lightning.strike(query, list(self.rotors.values()))
        
        if target_rotor:
            # Wake up the target
            target_rotor.wake(intensity=0.8)
            
            # 2. R-WFC (Fractal Expansion)
            # If the rotor is high energy or explicitly queried, we unfold it.
            children_dna = self.wfc.collapse(target_rotor.dna)
            
            new_concepts = []
            for child in children_dna:
                if child.label not in self.rotors:
                    self.add_rotor(child.label, child, rpm=child.frequency)
                    # Wake the child too
                    self.rotors[child.label].wake(intensity=0.6)
                    new_concepts.append(child.label)
            
            if new_concepts:
                return f"Focused on '{target_rotor.name}' -> Unfolded: {', '.join(new_concepts)}"
            else:
                return f"Focused on '{target_rotor.name}' (Already crystallized)."
        else:
            return "Gazed into the Void (No Resonance)."

    def get_state_summary(self) -> str:
        active = [r.name for r in self.rotors.values() if r.energy > 0.2]
        return f"HyperSphere: {len(self.rotors)} Rotors | Active: {active}"
