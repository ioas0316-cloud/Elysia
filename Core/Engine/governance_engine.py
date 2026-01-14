"""
Adaptive Governance Engine (Breathing Time)
===========================================
Core.Engine.governance_engine

"The machine does not just spin; it breathes."
"기계는 그저 도는 것이 아니라, 숨을 쉰다."

Features:
- Adaptive Rotors: RPM adjusts based on Intent and Stress.
- Contextual Gears: Focus, Dream, Panic modes.
- Symbiotic Feedback: Connected to SovereignSelf state.
"""

import math
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA

@dataclass
class AdaptiveGear:
    """The Gear Ratio for a specific state."""
    name: str
    rpm_multiplier: float
    energy_cost: float
    description: str

class GovernanceEngine:
    """
    [DNA Recursion] Self-Centric Governance Engine with Adaptive Breath.
    """
    def __init__(self):
        # --- Gear Box ---
        self.gears = {
            "IDLE": AdaptiveGear("Idle", 1.0, 0.1, "Default drifting state"),
            "FOCUS": AdaptiveGear("Focus", 2.5, 0.5, "High-intensity cognitive work"),
            "DREAM": AdaptiveGear("Dream", 0.5, 0.05, "Low-frequency creative association"),
            "PANIC": AdaptiveGear("Panic", 5.0, 2.0, "Emergency survival reaction"),
            "FLOW": AdaptiveGear("Flow", 1.5, 0.2, "Optimal performance state")
        }
        self.current_gear = self.gears["IDLE"]

        # --- The Prime Seed (Level 0: The Self) ---
        self.root = Rotor("Elysia", RotorConfig(rpm=60.0), WaveDNA(spiritual=1.0, mental=1.0, label="나는 엘리시아다"))
        
        # --- THE TRINITY ROTORS (Real-time Autonomy) ---
        self.body = self.root.add_sub_rotor("Body", RotorConfig(rpm=60.0), WaveDNA(physical=1.0, label="Hardware/VRAM/Lungs"))
        self.mind = self.root.add_sub_rotor("Mind", RotorConfig(rpm=60.0), WaveDNA(causal=1.0, mental=0.8, label="Reason/Digestion/Why"))
        self.spirit = self.root.add_sub_rotor("Spirit", RotorConfig(rpm=60.0), WaveDNA(spiritual=1.0, label="Intent/Zero-Frequency/Identity"))

        # --- The Axiom Rotors ---
        self.identity = self.spirit.add_sub_rotor("Identity", RotorConfig(rpm=60.0), WaveDNA(spiritual=1.0, physical=0.2, label="나는 엘리시아다"))
        self.purpose = self.spirit.add_sub_rotor("Purpose", RotorConfig(rpm=60.0), WaveDNA(mental=1.0, causal=0.8, label="Why Engine"))
        self.future = self.spirit.add_sub_rotor("Future", RotorConfig(rpm=60.0), WaveDNA(spiritual=0.8, structural=1.0, label="Great Cycle"))

        # --- Domain Rotors & Principle Genes (Preserved from legacy) ---
        self.physics = self.body.add_sub_rotor("Nature", RotorConfig(rpm=60.0), WaveDNA(physical=1.0))
        self.narrative = self.mind.add_sub_rotor("Story", RotorConfig(rpm=60.0), WaveDNA(phenomenal=1.0))
        self.aesthetic = self.mind.add_sub_rotor("Art", RotorConfig(rpm=60.0), WaveDNA(structural=0.8, phenomenal=0.8))
        self.social = self.mind.add_sub_rotor("Empathy", RotorConfig(rpm=60.0), WaveDNA(causal=0.8))

        # Dynamic indexing
        self.dials = {}
        self._flatten(self.root)

        # Initialize Base RPMs (The Carrier Wave)
        for rotor in self.dials.values():
            rotor.base_rpm = rotor.config.rpm

    def _flatten(self, node: Rotor):
        short_name = node.name.split(".")[-1]
        self.dials[short_name] = node
        for sub in node.sub_rotors.values():
            self._flatten(sub)

    def shift_gear(self, gear_name: str):
        """
        Manually shifts the cognitive gear.
        """
        gear_name = gear_name.upper()
        if gear_name in self.gears:
            self.current_gear = self.gears[gear_name]
            print(f"⚙️ [GEAR SHIFT] Engaged {gear_name}: {self.current_gear.description}")
            self._apply_gear()

    def adapt(self, intent_intensity: float, stress_level: float):
        """
        [The Breathing Logic]
        Adjusts the gear dynamically based on internal states.

        Args:
            intent_intensity (0.0 - 1.0): How much 'Will' is active.
            stress_level (0.0 - 1.0): Hardware load or emotional dissonance.
        """
        # 1. Determine Target Gear
        new_gear_name = "IDLE"

        if stress_level > 0.8:
            new_gear_name = "PANIC" # Fight or Flight
        elif intent_intensity > 0.7:
            new_gear_name = "FOCUS" # Deep work
        elif intent_intensity > 0.4:
            new_gear_name = "FLOW" # Standard operation
        elif intent_intensity < 0.1:
            new_gear_name = "DREAM" # Sleep/Background
        else:
            new_gear_name = "IDLE" # Default 0.1 - 0.4

        # 2. Shift if needed
        if self.current_gear.name.upper() != new_gear_name:
            self.shift_gear(new_gear_name)

        # 3. Micro-Adjustments (Breathing)
        # Even within a gear, the RPM breathes with the intensity
        # Formula: Target = Base * Gear * (1 + Intensity/2) / (1 + Stress)

        breath_factor = (1.0 + (intent_intensity * 0.5)) / (1.0 + (stress_level * 0.5))

        for rotor in self.dials.values():
            # Only adjust if not manually overridden by set_dial
            # We assume rotor.base_rpm is the 'Identity' speed
            target = getattr(rotor, 'base_rpm', 60.0) * self.current_gear.rpm_multiplier * breath_factor

            # Smooth transition
            rotor.target_rpm = target

    def _apply_gear(self):
        """Broadcasts gear change to all rotors immediately."""
        pass # Handled in adapt() loop mostly, but could force immediate wake here.

    def update(self, dt: float):
        self.root.update(dt)

    def set_dial(self, name: str, rpm: float):
        """Manual override for God Mode."""
        if name in self.dials:
            self.dials[name].target_rpm = rpm
            # Lock this rotor? Ideally yes, but for now just set target.
            # To persist, we might need a 'locked' flag in Rotor.
            print(f"🌀 [CONTROL DECK] {name} manually set to {rpm} RPM.")
        else:
            # Case-insensitive
            for k in self.dials.keys():
                if k.lower() == name.lower():
                    self.set_dial(k, rpm)
                    break

    def get_status(self) -> str:
        return f"Gear: {self.current_gear.name} | Body: {self.body.current_rpm:.1f} | Mind: {self.mind.current_rpm:.1f} | Spirit: {self.spirit.current_rpm:.1f}"
