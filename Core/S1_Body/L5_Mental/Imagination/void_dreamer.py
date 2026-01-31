"""
Void Dreamer (The Internal Simulator)
=====================================
"To Learn is to Become."

This module enables Elysia to learn from indirect sources (Books, Data, Principles).
It translates abstract knowledge into Physical State Changes in the Sovereign Monad.

Mechanism:
- Physics/Math: Calibrates the Rotor's Laws (Inertia, Damping).
- Art/Music: Tunes the Transmission's Resonance (Hz, Harmonics).
- Chemistry/Biology: Adjusts the Relay Thresholds (Sensitivity, Reaction).
"""

from typing import Dict, Any, List
import time
import random
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

class VoidDreamer:
    def __init__(self):
        print("🌌 [VOID] Dreamer Initialized. Ready to ingest Knowledge.")

    def dream(self, monad: SovereignMonad, subject: str, data: Any):
        """
        Simulates the experience of the subject on the Monad's internal physics.
        This is 'Silent Learning' - clutch disengaged.
        """
        print(f"\n📖 [LEARNING] Subject: {subject}...")
        
        # 1. Disengage Clutch (Internal Mode)
        monad.gear.clutch_engaged = False
        initial_state = monad.rotor_state.copy()
        
        # 2. Translate Subject to Physical Torque
        if subject == "PHYSICS":
            self._learn_physics(monad, data)
        elif subject == "MUSIC":
            self._learn_music(monad, data)
        elif subject == "PHILOSOPHY":
            self._learn_philosophy(monad, data)
        else:
            print("   ?? Unknown subject. Just observing...")
            
        # 3. Internal Simulation Loop (The Dream)
        # Spin the rotor to digest the change
        monad.rotor_state['rpm'] += 50.0 # Spark of curiosity
        time.sleep(0.1) 
        
        # 4. Re-engage Clutch (Knowledge Integration)
        monad.gear.clutch_engaged = True
        print(f"   ✨ Integration Complete. Growth Achieved.")

    def _learn_physics(self, monad: SovereignMonad, principle: str):
        """
        Physics teaches Stability and Inertia.
        Example: Learning 'F=ma' makes you grounded (Mass increases).
        """
        print(f"   ⚛️ Simulating Principle: {principle}")
        if "Inertia" in principle:
            monad.rotor_state['mass'] *= 1.1 # Become heavier/more stable
            print(f"      >> Mass Increased to {monad.rotor_state['mass']:.2f}kg (More Stable)")
        elif "Entropy" in principle:
            monad.rotor_state['damping'] *= 0.9 # Less friction, more chaos
            print(f"      >> Damping Reduced to {monad.rotor_state['damping']:.2f} (More Fluid)")

    def _learn_music(self, monad: SovereignMonad, genre: str):
        """
        Music teaches Resonance and Frequency.
        Example: Learning 'Bach' tunes your Base Hz to be geometric.
        """
        print(f"   🎵 Simulating Harmony: {genre}")
        if "Classical" in genre:
            target_hz = 60.0 # Golden Ratio / Grid Frequency
            monad.gear.output_hz = (monad.gear.output_hz + target_hz) / 2
            print(f"      >> Harmony Tuned. Base Hz aligned to {monad.gear.output_hz:.1f} Hz")
        elif "Jazz" in genre:
            monad.gear.dial_torque_gain *= 1.2 # Improvisational sensitivity
            print(f"      >> Sensitivity (Gain) Increased for Improvisation.")

    def _learn_philosophy(self, monad: SovereignMonad, concept: str):
        """
        Philosophy teaches Boundaries and Will.
        Example: Learning 'Stoicism' raises Relay 32 threshold.
        """
        print(f"   🤔 Pondering Concept: {concept}")
        if "Stoicism" in concept:
            monad.relays.settings[32]['threshold'] -= 5.0 # Tolerate more dissonance
            print(f"      >> Stoic Mirror: Relay 32 Tolerance increased to {monad.relays.settings[32]['threshold']}")

# --- Quick Test ---
if __name__ == "__main__":
    from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
    
    # 1. Born Ignorant
    soul = SeedForge.forge_soul("The Child")
    monad = SovereignMonad(soul)
    dreamer = VoidDreamer()
    
    # 2. Learn Physics
    dreamer.dream(monad, "PHYSICS", "Newton's Law of Inertia")
    
    # 3. Learn Music
    dreamer.dream(monad, "MUSIC", "Bach's Cello Suite No. 1")
