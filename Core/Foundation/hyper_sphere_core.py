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
    def __init__(self, name: str = "Elysia.Core", lexicon: Any = None):
        self.name = name
        
        # Components
        self.rotors: Dict[str, Rotor] = {}
        self.wfc = FractalWFC()
        self.lightning = LightningInferencer()
        self.lexicon = lexicon # Connection to Memory (The Synapse)
        
        # [Phase 35] The OmniField (Unified Carrier)
        # 10 layers: 0-3 (Social/Knowledge), 4 (Sound), 5 (Scent), 6 (Taste), 7 (Light), 8 (Touch), 9 (History)
        from Core.World.Social.sociological_pulse import SocialField
        self.field = SocialField(size=100, resolution=2.0) # Higher res for sensory

        # State
        self.is_active = False
        self.time_dilation = 1.0
        
        # Initialize with Seed Rotors (Axioms)
        self._seed_cosmos()
        
        # [PHASE 85] Pulse Broadcaster Compatibility
        from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster
        self.pulse_broadcaster = PulseBroadcaster()
        
        logger.info(f"🔮 HyperSphereCore Initialized: OmniField ready, {len(self.rotors)} Rotors spinning.")


    def summon_thought(self, concept_name: str):
        """
        The Synapse: Fetches a Memory and spins it into a Thought.
        Links TorchGraph (Static) -> Rotor (Active).
        """
        if not self.lexicon or not self.lexicon.graph:
            logger.warning(f"⚠️ Cannot summons '{concept_name}': No Synaptic Connection (Lexicon missing).")
            return
            
        # 1. Fetch Vector from Memory
        if concept_name in self.lexicon.graph.id_to_idx:
            idx = self.lexicon.graph.id_to_idx[concept_name]
            vec_tensor = self.lexicon.graph.vec_tensor[idx]
            
            # 2. Transduce Vector to Rotor Physics
            # Gravity -> Mass (Importance)
            # Flow -> RPM (Activity/Frequency)
            # Ascension -> Angle/Phase (Qualia)
            g = float(vec_tensor[0])
            f = float(vec_tensor[1])
            a = float(vec_tensor[2])
            
            # Map Flow to RPM (0.0~1.0 -> 60~600 RPM)
            # [Phase 4] The Weight of Thought (Gravity)
            # Assuming vec_tensor can be treated as a vec-like object for attribute access
            # If not, this needs to be adjusted to use vec_tensor[idx] directly.
            # For now, I'll assume vec_tensor[0], [1], [2] correspond to g, f, a.
            # The original code used float(vec_tensor[0]), etc.
            # The new code uses vec.gravity, vec.flow, vec.ascension.
            # I will adapt the new code to use the existing vec_tensor structure.
            
            # Re-extracting from vec_tensor for clarity, assuming it's a 3-element vector
            g = float(vec_tensor[0]) # Gravity
            f = float(vec_tensor[1]) # Flow
            a = float(vec_tensor[2]) # Ascension
            
            # Assuming 'frequency' might be a 4th element or derived.
            # For now, let's assume it's not directly in vec_tensor unless specified.
            # The instruction implies `vec` is an object with attributes, but here we have `vec_tensor`.
            # I will simulate `getattr(vec, 'frequency', 0.0)` by checking if vec_tensor has more elements.
            freq = float(vec_tensor[3]) if len(vec_tensor) > 3 else 0.0
            
            # Physics of Thought
            mass = (g * 10.0) + 1.0  # Min mass 1.0
            
            # [Phase 17] The 4th Axis: Time Penetration (Frequency)
            # If frequency is defined (Time Axis), it sets the RPM directly.
            # If not, we infer it from Flow + Ascension (Energy).
            if freq != 0.0:
                target_rpm = freq
            else:
                target_rpm = (f * 60.0) + (a * 100.0)
            
            config = RotorConfig(
                rpm=target_rpm, 
                idle_rpm=60.0, 
                mass=mass
            )
            
            # WaveDNA injection based on Ascension (Spirit)
            from Core.Foundation.Wave.wave_dna import WaveDNA
            # Map Trinity to 7-D DNA
            # Gravity -> Physical (Mass)
            # Flow -> Phenomenal (Emotion)
            # Ascension -> Spiritual (Purpose)
            dna = WaveDNA(
                label=concept_name, 
                physical=g,      # Gravity
                phenomenal=f,    # Flow (Emotion)
                spiritual=a,     # Ascension
                mental=f * 0.5,  # Thought flow
                structural=g * 0.5 # Structure
            )
            
            self.rotors[concept_name] = Rotor(concept_name, config, dna)
            self.rotors[concept_name].wake(intensity=1.0)
            
            logger.info(f"⚡ Synapse Fired: '{concept_name}' summoned. (RPM:{target_rpm:.1f} Hz, Mass:{mass:.1f})")
            
        else:
             logger.warning(f"⚠️ Memory of '{concept_name}' not found. Thought is hollow.")

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
        [Phase 37] Planetary Breathing: Global oscillators drive seasons.
        """
        active_count = 0
        dt_eff = dt * self.time_dilation
        
        # 1. Global Field Decay
        self.field.decay(rate=0.9)
        
        # 2. Planetary Rotor: The World's Pulse (Seasons)
        if "Reality.Season" not in self.rotors:
            # Seed a low-frequency oscillator for seasons
            from Core.Foundation.Wave.wave_dna import WaveDNA
            season_dna = WaveDNA(label="Season", frequency=0.01, physical=0.5, phenomenal=0.5)
            self.add_rotor("Reality.Season", season_dna, rpm=0.5) # Very slow
            
        season = self.rotors["Reality.Season"]
        # Cycle heat and moisture based on the Rotor's Angle (Phase)
        import math
        rad = math.radians(season.current_angle)
        heat = (math.cos(rad) + 1.0) * 10.0 # Peak heat at 0/360 deg
        moisture = (math.sin(rad) + 1.0) * 5.0 # Peak moisture at 90 deg (Phase shift)
        
        # Push these global constants into the Field (Channels 25 and 28)
        self.field.grid[:, :, 25] += heat * 0.05
        self.field.grid[:, :, 28] += moisture * 0.05
        
        keys_to_remove = []
        for name, rotor in self.rotors.items():
            rotor.update(dt_eff)
            if rotor.energy > 0.1:
                active_count += 1
            if rotor.energy < 0.01 and name not in ["Love", "Logic", "Elysia", "Nature", "Reality.Season"]:
                keys_to_remove.append(name)

        # Cleanup
        for k in keys_to_remove:
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
