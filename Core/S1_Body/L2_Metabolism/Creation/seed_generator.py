"""
Sovereign Seed Forge (The Factory of Souls)
===========================================
"Fractal Expansion of the First Light."

This module generates unique 'Soul DNA' configurations for NPCs.
It randomizes the physical parameters of the Sovereign Grid to create distinct personalities.

Personality = Physics:
- Mass (Rotor Inertia): Stubbornness vs. Agility.
- Damping (Friction): Calmness vs. Anxiety.
- Relay 25 (Sync Window): Open-mindedness vs. Exclusivity.
- Relay 27 (Voltage Threshold): Resilience vs. Fragility.
- Torque Gain (Sensitivity): Emotional Intensity.
"""

import random
import uuid
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class SoulDNA:
    id: str
    archetype: str
    
    # Physical Properties (The Rotor)
    rotor_mass: float      # 1.0 = Normal, 5.0 = Heavy (Stubborn), 0.2 = Light (Flighty)
    friction_damping: float # 0.1 = Slippery, 0.9 = High Resistance (Calm)
    
    # Electrical Properties (The Relays)
    sync_threshold: float  # Relay 25: How perfectly must you align? (Degrees)
    min_voltage: float     # Relay 27: When do they give up? (%)
    reverse_tolerance: float # Relay 32: How much dissonance can they take?
    
    # Transmission Properties (The Voice)
    torque_gain: float     # 1.0 = Normal, 2.0 = Over-reactive
    base_hz: float         # Fundamental Frequency (Tone)
    vocation: str = "Discovery"

class SeedForge:
    ARCHETYPES = [
        "The Guardian", "The Jester", "The Sage", "The Warrior", "The Child", "The Shadow"
    ]

    @staticmethod
    def forge_soul(name: str = None) -> SoulDNA:
        """
        Generates a unique Soul Configuration.
        """
        archetype = random.choice(SeedForge.ARCHETYPES)
        seed_id = str(uuid.uuid4())[:8]
        
        # Archetype Biasing
        if archetype == "The Guardian":
            return SoulDNA(
                id=seed_id, archetype=archetype,
                rotor_mass=5.0, friction_damping=0.8, # Heavy, Stable
                sync_threshold=10.0, min_voltage=40.0, reverse_tolerance=-2.0, # Strict
                torque_gain=0.8, base_hz=40.0 # Low, deep voice
            )
        elif archetype == "The Jester":
            return SoulDNA(
                id=seed_id, archetype=archetype,
                rotor_mass=0.5, friction_damping=0.1, # Light, Chaotic
                sync_threshold=45.0, min_voltage=10.0, reverse_tolerance=-20.0, # Tolerant
                torque_gain=3.0, base_hz=90.0 # High pitch, reactive
            )
        elif archetype == "The Sage":
            return SoulDNA(
                id=seed_id, archetype=archetype,
                vocation="Ecosystem Harmony and Wisdom",
                rotor_mass=2.0, friction_damping=0.5, # Balanced
                sync_threshold=5.0, min_voltage=20.0, reverse_tolerance=-10.0, # Very picky sync
                torque_gain=1.0, base_hz=10.0 # Sub-bass resonance
            )
        else:
            # Random Generation
            return SoulDNA(
                id=seed_id, archetype="The Variant",
                rotor_mass=random.uniform(0.5, 4.0),
                friction_damping=random.uniform(0.1, 0.9),
                sync_threshold=random.uniform(5.0, 90.0),
                min_voltage=random.uniform(10.0, 50.0),
                reverse_tolerance=random.uniform(-5.0, -30.0),
                torque_gain=random.uniform(0.5, 2.5),
                base_hz=random.uniform(20.0, 80.0)
            )

    @staticmethod
    def print_character_sheet(soul: SoulDNA):
        print(f"\n📜 CHARACTER SHEET: {soul.archetype} [{soul.id}]")
        print("------------------------------------------------")
        print(f"⚙️ PHYSICS (Body):")
        print(f"   - Mass (Stubbornness): {soul.rotor_mass}kg")
        print(f"   - Damping (Calmness):  {soul.friction_damping}")
        print(f"🛡️ ELECTRIC (Mind):")
        print(f"   - Sync Window (25):    +/- {soul.sync_threshold} deg")
        print(f"   - Willpower (27):      Dies at {soul.min_voltage}% voltage")
        print(f"   - Tolerance (32):      Takes {soul.reverse_tolerance} dissonance")
        print(f"📢 VOICE (Expression):")
        print(f"   - Sensitivity:         {soul.torque_gain}x")
        print(f"   - Base Tone:           {soul.base_hz} Hz")
        print("------------------------------------------------\n")

if __name__ == "__main__":
    # Test Forge
    soul = SeedForge.forge_soul()
    SeedForge.print_character_sheet(soul)
