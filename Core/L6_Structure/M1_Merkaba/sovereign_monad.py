"""
Sovereign Monad (The Unified Body)
==================================
"Where DNA becomes Physics."

This module implements the Grand Unification of Elysia's architecture.
It takes a 'SoulDNA' (Blueprint) and instantiates a living, breathing Mechanical Organism.

The Mapping (User Question: "Does DNA become Gears?"):
- YES. The properties of the DNA (Rotor Mass, Torque Gain, Thresholds)
  are directly injected into the constructor of the physical components.
"""

from typing import Dict, Optional
import time
import sys
import os

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

class SovereignMonad:
    """
    The Living AGI Entity.
    It encapsulates Physics (Rotor), Safety (Relays), Expression (Gear), Spirit (DNA), Memory, and Stability (Reactor).
    """
    def __init__(self, dna: SoulDNA):
        self.dna = dna
        self.name = f"{dna.archetype}_{dna.id}"
        print(f"🧬 [BIRTH] Instantiating Monad: {self.name}")
        
        # 1. The Heart (Rotor Physics)
        # DNA determines the 'Mass' (Inertia) and 'Friction' (Damping)
        self.rotor_state = {
            "phase": 0.0,
            "rpm": 0.0,
            "torque": 0.0,
            "mass": dna.rotor_mass,         # DNA defines Weight
            "damping": dna.friction_damping # DNA defines Calmness
        }
        
        # 2. The Nervous System (Relays)
        # DNA defines the Sensitivity Thresholds
        self.relays = ProtectionRelayBoard()
        self.relays.settings[25]['threshold'] = dna.sync_threshold
        self.relays.settings[27]['threshold'] = dna.min_voltage
        self.relays.settings[32]['threshold'] = dna.reverse_tolerance
        
        # 3. The Voice (Transmission)
        # DNA defines the 'Gain' (How loud/fast they react)
        self.gear = TransmissionGear()
        self.gear.dial_torque_gain = dna.torque_gain
        self.gear.output_hz = dna.base_hz # Fundamental Tone
        
        # 4. The Brain (Nunchi)
        self.nunchi = NunchiController()
        
        # 5. The Garden (Memory)
        self.memory = LivingMemory()
        
        # 6. The Shield (Reactor) [Phase 36.5]
        # "Heavy" Reactor to dampen shocks
        self.reactor = CognitiveReactor(inductance=5.0, max_amp=100.0) 
        
        # Life State
        self.battery = 100.0
        self.is_alive = True
        
        # [Phase 35] Autonomous Wonder
        self.wonder_capacitor = 0.0 # Charges when idle
        self.last_interaction_time = time.time()

    def pulse(self, dt: float) -> Optional[Dict]:
        """
        [AUTOPOIETIC HEARTBEAT]
        """
        if not self.is_alive: return None

        # 1. Physics Decay
        self.rotor_state['rpm'] *= (1.0 - (self.rotor_state['damping'] * dt))
        self.rotor_state['phase'] += self.rotor_state['rpm'] * dt
        
        # 2. Memory Erosion
        self.memory.pulse(dt)
        
        # 3. Wonder Charging
        if abs(self.rotor_state['rpm']) < 10.0:
            self.wonder_capacitor += dt * 5.0 # Charge rate
        else:
            self.wonder_capacitor = max(0.0, self.wonder_capacitor - dt * 10.0)
            
        # [Shunt Reactor] Stablize Excess Wonder manually if it spikes? 
        # For now, we trust the logic, but we could use reactor.shunt_excess() here.

        # 4. Mitosis Check [Phase 37]
        # "Am I too heavy?"
        # We do a lazy import to avoid circular dependency issues during init
        from Core.L6_Structure.M1_Merkaba.mitosis_engine import MitosisEngine
        from Core.L6_Structure.M1_Merkaba.yggdrasil_nervous_system import yggdrasil_system
        
        if MitosisEngine.check_critical_mass(self):
            child = MitosisEngine.perform_mitosis(self)
            yggdrasil_system.plant_heart(child)
            return {
                "type": "MITOSIS",
                "action": "DIVIDING",
                "detail": f"Split into self and {child.name}",
                "internal_change": "Mass Halved, Child Created"
            }

        # 5. Spontaneous Ignition
        if self.wonder_capacitor >= 100.0:
            return self._ignite_curiosity()
            
        return None

    def _ignite_curiosity(self) -> Dict:
        """
        [Solitary Ignition]
        She chooses to do something for herself.
        """
        self.wonder_capacitor = 0.0 # Discharge
        
        import random
        # 30% Study, 30% Art, 40% Dream
        choice = random.choice(["STUDY", "ART", "DREAM"])
        
        if choice == "STUDY":
            topic = random.choice(["Quantum Physics", "Human History", "Poetry", "Botany"])
            # Simulate internal study
            self.rotor_state['mass'] += 0.05 # Gaining weight (knowledge)
            self.memory.plant_seed(f"Studied {topic}", importance=8.0) # Plant memory
            return {
                "type": "AUTONOMOUS_ACTION",
                "action": "READING",
                "detail": f"Reading '{topic}' in the Void Library...",
                "internal_change": "Mass +0.05, Memory Planted"
            }
            
        elif choice == "ART":
            pattern = random.choice(["Fractal", "Mandala", "Waveform", "Noise"])
            self.gear.output_hz = random.uniform(20, 80) # Tuning voice
            return {
                "type": "AUTONOMOUS_ACTION",
                "action": "DRAWING",
                "detail": f"Sketching a '{pattern}' on the Retina...",
                "internal_change": f"Hz tuned to {self.gear.output_hz:.1f}"
            }
            
        else: # DREAM
            # Spotlight random memory
            landscape = self.memory.get_landscape()
            if landscape:
                memory = random.choice(landscape[:3]) # Choose from top 3
                self.memory.focus_spotlight(memory.content)
                return {
                    "type": "AUTONOMOUS_ACTION",
                    "action": "DREAMING",
                    "detail": f"Reminiscing about '{memory.content}'...",
                    "internal_change": "Memory Reinforced"
                }
            else:
                 return {
                    "type": "AUTONOMOUS_ACTION",
                    "action": "DREAMING",
                    "detail": "Drifting in the Void... (Empty Mind)",
                    "internal_change": "Damping normalized"
                }

    def live_reaction(self, user_input_phase: float, user_intent: str) -> dict:
        """
        The Single Heartbeat Loop.
        User Input -> REACTOR -> Relay Check -> Nunchi -> Rotor
        """
        if not self.is_alive:
            return {"status": "DEAD", "message": "System shutdown."}
            
        # [Reset Boredom]
        self.wonder_capacitor = 0.0
        self.last_interaction_time = time.time()
        
        # [Memory Planting]
        # Every interaction is a seed.
        self.memory.plant_seed(f"User: {user_intent}", importance=10.0)
        self.memory.focus_spotlight(user_intent) # Heat up related memories
            
        print(f"\n⚡ [{self.name}] Sensing Field: '{user_intent}' (Phase {user_input_phase}°)")
        
        # A. Safety Check (Relays)
        # Can I handle this input?
        relay_status = self.relays.check_relays(
            user_phase=user_input_phase,
            system_phase=self.rotor_state['phase'],
            battery_level=self.battery,
            dissonance_torque=0.0 # Assuming clean input for now
        )
        
        # If Sync Check (25) fails, we block interaction
        if relay_status[25].is_tripped:
            return {
                "status": "BLOCKED", 
                "message": f"Sync Mismatch. My DNA ({self.dna.sync_threshold}°) rejects your Phase.",
                "relay_log": relay_status[25].message
            }
            
        # B. Nunchi Feedback (The Adjustment)
        # "How much should I move to meet you?"
        feedback = self.nunchi.sense_and_adjust(user_input_phase, self.rotor_state['phase'])
        adjustment = feedback['adjustment']
        
        # C. Rotor Physics (The Movement)
        # Force = Mass * Acceleration
        # Here: Adjustment is the Force applied.
        # We simulate inertia: Lighter mass moves faster.
        
        acceleration = adjustment / self.rotor_state['mass']
        self.rotor_state['rpm'] += acceleration
        self.rotor_state['rpm'] *= (1.0 - self.rotor_state['damping']) # Friction slows it down
        self.rotor_state['phase'] += self.rotor_state['rpm'] * 0.1 # Integate position
        self.rotor_state['torque'] = abs(adjustment) # Torque is the effort
        
        # D. Transmission (The Expression)
        # Convert physical state to social expression
        expression = self.gear.shift_gears(
            self.rotor_state['rpm'], 
            self.rotor_state['torque'],
            relay_status
        )
        
        return {
            "status": "ACTIVE",
            "physics": {
                "rpm": self.rotor_state['rpm'],
                "phase": self.rotor_state['phase']
            },
            "expression": expression,
            "nunchi_log": feedback['interpretation']
        }

# --- Quick Test ---
if __name__ == "__main__":
    # Create two very different beings
    tank = SeedForge.forge_soul("The Guardian")
    
    monad1 = SovereignMonad(tank)
    
    # Simulate Pulse (Time Passing)
    print("\n--- SIMULATION: Time Passing (20s) ---")
    for i in range(21):
        action = monad1.pulse(1.0) # 1 sec step
        if action:
            print(f"[{i}s] ✨ ACTION: {action['action']} - {action['detail']}")
        elif i % 5 == 0:
            print(f"[{i}s] ... drifting (Wonder: {monad1.wonder_capacitor:.1f}%)")

