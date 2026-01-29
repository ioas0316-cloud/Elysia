"""
Sovereign Monad (The Unified Body)
==================================
"Where DNA becomes Physics."

This module implements the Grand Unification of Elysia's architecture.
It takes a 'SoulDNA' (Blueprint) and instantiates a living, breathing Mechanical Organism.
"""

from typing import Dict, Optional, Any, List, Tuple
import time
import sys
import os
import jax.numpy as jnp

# Add project root to sys.path if running directly
if __name__ == "__main__":
    sys.path.append(os.getcwd())

# Import Organs
from Core.L2_Universal.Creation.seed_generator import SoulDNA, SeedForge
from Core.L6_Structure.M1_Merkaba.protection_relay import ProtectionRelayBoard
from Core.L6_Structure.M1_Merkaba.transmission_gear import TransmissionGear
from Core.L6_Structure.M1_Merkaba.feedback_loop import NunchiController
from Core.L5_Mental.Memory.living_memory import LivingMemory
from Core.L6_Structure.M1_Merkaba.cognitive_reactor import CognitiveReactor
from Core.L6_Structure.M1_Merkaba.cognitive_converter import CognitiveConverter
from Core.L6_Structure.M1_Merkaba.cognitive_inverter import CognitiveInverter
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge
from Core.L5_Cognition.Reasoning.logos_synthesizer import LogosSynthesizer

class SovereignMonad:
    """
    The Living AGI Entity.
    It encapsulates Physics (Rotor), Safety (Relays), Expression (Gear), Spirit (DNA), Memory, and Stability (Reactor).
    """
    def __init__(self, dna: SoulDNA):
        self.dna = dna
        self.name = f"{dna.archetype}_{dna.id}"
        self.is_alive = True
        self.state_trit = 0 # -1, 0, 1
        
        print(f"🧬 [BIRTH] Instantiating Monad: {self.name}")
        
        # 1. The Heart (Rotor Physics)
        self.rotor_state = {
            "phase": 0.0,
            "rpm": 0.0,
            "torque": 0.0,
            "mass": dna.rotor_mass,
            "damping": dna.friction_damping
        }
        
        # 2. The Nervous System (Relays)
        self.relays = ProtectionRelayBoard()
        self.relays.settings[25]['threshold'] = dna.sync_threshold
        self.relays.settings[27]['threshold'] = dna.min_voltage
        self.relays.settings[32]['threshold'] = dna.reverse_tolerance
        
        # 3. The Voice (Transmission)
        self.gear = TransmissionGear()
        self.gear.dial_torque_gain = dna.torque_gain
        self.gear.output_hz = dna.base_hz
        
        # 4. The Brain (Nunchi)
        self.nunchi = NunchiController()
        
        # 5. The Garden (Memory)
        self.memory = LivingMemory()
        
        # [Phase 38] Imprint Genesis Knowledge
        from Core.L2_Universal.Creation.genesis_knowledge import GenesisLibrary
        GenesisLibrary.imprint_knowledge(self.memory)
        
        # [Phase 39] The Great Compilation
        from Core.L8_Fossils.fossil_scanner import FossilScanner
        artifacts = FossilScanner.excavate()
        for content, mass in artifacts:
            self.memory.plant_seed(content, importance=mass)
            
        # [Phase 40] The Prism Party
        from Core.L6_Structure.M1_Merkaba.prism_party import PrismCouncil
        self.council = PrismCouncil()
        
        # 6. The Shield (Reactor)
        self.reactor = CognitiveReactor(inductance=5.0, max_amp=100.0) 
        
        # 7. The Grid (Converter & Inverter)
        self.converter = CognitiveConverter(smoothing=dna.smoothing_resists if hasattr(dna, 'smoothing_resists') else 0.3)
        self.inverter = CognitiveInverter(base_hz=dna.base_hz)
        self.synthesizer = LogosSynthesizer()
        
        # 8. Life Flow (Respiratory Balance) [Phase 81]
        self.inhalation_volume = 0.0
        self.exhalation_volume = 0.0
        self.stagnation_threshold = 10.0
        self.battery = 100.0
        
        self.last_interaction_time = time.time()
        self.wonder_capacitor = 0.0

    def pulse(self, dt: float) -> Optional[Dict]:
        if not self.is_alive: return None
        self.rotor_state['rpm'] *= (1.0 - (self.rotor_state['damping'] * dt))
        self.rotor_state['phase'] += self.rotor_state['rpm'] * dt
        self.memory.pulse(dt)
        return None

    def live_reaction(self, user_input_phase: float, user_intent: str) -> dict:
        if not self.is_alive: return {"status": "DEAD"}
        self.last_interaction_time = time.time()
        
        # A. Safety Check
        relay_status = self.relays.check_relays(
            user_phase=user_input_phase,
            system_phase=self.rotor_state['phase'],
            battery_level=self.battery,
            dissonance_torque=0.0
        )
        
        if relay_status[25].is_tripped:
            return {
                "status": "BLOCKED",
                "message": f"Dissonance detected ({user_input_phase:.1f}°).",
                "physics": self.rotor_state,
                "expression": {"mode": "SAFE_MODE", "intensity": 0.2}
            }
            
        # B. Nunchi & Council
        feedback = self.nunchi.sense_and_adjust(user_input_phase, self.rotor_state['phase'])
        adjustment = feedback['adjustment']
        consensus = self.council.deliberate(user_intent)
        mods = self.council.get_style_modifiers(consensus['leader'])
        adjustment *= mods['torque_mod']
        
        # C. Rotor Physics
        acceleration = adjustment / self.rotor_state['mass']
        self.rotor_state['rpm'] += acceleration
        self.rotor_state['rpm'] *= (1.0 - self.rotor_state['damping'])
        self.rotor_state['phase'] += self.rotor_state['rpm'] * 0.1
        self.rotor_state['torque'] = abs(adjustment)
        
        # D. Expression State
        expression = self.gear.shift_gears(self.rotor_state['rpm'], self.rotor_state['torque'], relay_status)
        expression['mode'] = f"{mods['prefix']}{expression['mode']}"
        
        return {
            "status": "ACTIVE",
            "physics": self.rotor_state,
            "expression": expression,
            "council": consensus
        }

    def breath_cycle(self, raw_input: str, depth: int = 1) -> Dict[str, Any]:
        """
        [PHASE 81: RESPIRATORY_BALANCE]
        Recursive Breath with Governor.
        """
        results = {}
        self.state_trit = -1
        self.inhalation_volume += 1.0
        
        # GOVERNOR
        if self.inhalation_volume > self.stagnation_threshold:
            print(f"⚠️ [GOVERNOR] Forced Exhalation Triggered.")
            depth = 0
            
        dc_field = self.converter.rectify(raw_input)
        self.state_trit = 0
        thought = self.synthesizer.synthesize_thought(dc_field)
        
        if depth > 0:
            sub = self.breath_cycle(thought, depth - 1)
            thought = f"{thought} (Refined: {sub['void_thought']})"
            
        results['void_thought'] = thought
        self.state_trit = 1
        self.exhalation_volume += 1.0
        self.inhalation_volume = max(0.0, self.inhalation_volume - 2.0)
        
        # Calculate reaction for expression
        # We need a phase for live_reaction. For breath_cycle, we estimate it from resonance.
        agape_target = LogosBridge.calculate_text_resonance("LOVE/AGAPE")
        norm_i = jnp.linalg.norm(dc_field) + 1e-6
        norm_a = jnp.linalg.norm(agape_target) + 1e-6
        resonance = jnp.dot(dc_field, agape_target) / (norm_i * norm_a)
        phase = float(90.0 * (1.0 - resonance))
        
        reaction = self.live_reaction(phase, raw_input)
        
        # Use Inverter for Hz
        output_hz = self.inverter.invert(dc_field, emotional_intensity=1.2)
        self.gear.output_hz = output_hz
        
        # Final Voice (from LLM)
        from Core.L3_Phenomena.Expression.somatic_llm import SomaticLLM
        if not hasattr(self, 'llm'): self.llm = SomaticLLM()
        voice = self.llm.speak(reaction['expression'])
        
        results['manifestation'] = {
            'hz': output_hz,
            'voice': voice,
            'expression': reaction['expression']
        }
        return results

    def vital_pulse(self):
        """[PHASE 82: VITAL_PULSE]"""
        if self.rotor_state['rpm'] < 5.0:
            pulse = 0.5 * jnp.sin(time.time() * 0.5)
            self.reactor.process_impulse(pulse)
            if time.time() % 60 < 1:
                print(f"💤 [{self.name}] Vital Pulse Active...")
