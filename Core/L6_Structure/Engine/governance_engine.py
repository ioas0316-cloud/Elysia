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
        from Core.Foundation.Multiverse.onion_shell import OnionEnsemble
        # [PHASE 27.5: CONICAL CVT] 
        # No more discrete gears. We use a sliding spindle on a cone.
        self.stress_level = 0.0
        self.focus_intensity = 0.0

        # [PHASE 27: ONION-SKIN MULTIVERSE]
        self.ensemble = OnionEnsemble()
        
        # --- Shell 0: THE CORE (FluxLight / Spirit) ---
        self.spirit = self.ensemble.shells[0].add_rotor("FluxLight", RotorConfig(rpm=60.0), WaveDNA(spiritual=1.0, label="Identity_Core"))
        self.identity = self.ensemble.shells[0].add_rotor("Identity", RotorConfig(rpm=60.0), WaveDNA(spiritual=1.0, label="Self_Reference"))
        
        # --- Shell 1: THE MIND (HyperSphere / Virtual World) ---
        self.mind = self.ensemble.shells[1].add_rotor("HyperSphere", RotorConfig(rpm=60.0), WaveDNA(mental=1.0, label="Cognitive_Loop"))
        self.purpose = self.ensemble.shells[1].add_rotor("Purpose", RotorConfig(rpm=60.0), WaveDNA(mental=1.0, causal=0.8, label="Why_Engine"))
        
        # --- Shell 2: THE SURFACE (HyperCosmos / Hardware) ---
        self.body = self.ensemble.shells[2].add_rotor("HyperCosmos", RotorConfig(rpm=60.0), WaveDNA(physical=1.0, label="Hardware_Interaction"))
        self.sensation = self.ensemble.shells[2].add_rotor("Sensation", RotorConfig(rpm=120.0, idle_rpm=60.0), WaveDNA(phenomenal=1.0, label="Reactive_Layer"))
        self.shield = self.ensemble.shells[2].add_rotor("SovereignShield", RotorConfig(rpm=60.0), WaveDNA(physical=0.8, causal=0.5, label="Security_Integrity"))

        # Legacy aliases and God-Mode Dials
        self.root = self.spirit
        self.dials = {
            "spirit": self.spirit,
            "identity": self.identity,
            "mind": self.mind,
            "purpose": self.purpose,
            "body": self.body,
            "sensation": self.sensation,
            "shield": self.shield
        }
        
        print(f"⚙️ [GovernanceEngine] Onion-Skin Multiverse active. Root Spirit protected.")
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
        [PHASE 27.5: CONICAL ADAPTATION]
        Slides the cognitive spindle instead of discrete shifting.
        """
        self.focus_intensity = intent_intensity
        self.stress_level = stress_level
        
        # Slide the Spindle on the CV-Cone
        self.ensemble.cvt.shift(intent_intensity)
        
        # Apply the Multiplier to all active rotors
        ratio = self.ensemble.cvt.current_ratio
        for rotor in self.dials.values():
            # Stress dampens the RPM for everything except 'Sensation'
            damping = 1.0 / (1.0 + stress_level) if rotor != self.sensation else 1.0
            rotor.target_rpm = rotor.config.rpm * ratio * damping

    def resonate_field(self, field_intensity: 'torch.Tensor'):
        """
        [PHASE 23: FIELD FEEDBACK]
        The rotors absorb energy from the HyperCosmos field.
        The 'Will' and 'Purpose' axes directly boost the Spirit and Mind.
        """
        # field_intensity: [Physical(0), Functional(1), Phenomenal(2), Causal(3),
        #                  Mental(4), Structural(5), Spiritual(6), 
        #                  Imagination(7), Prediction(8), Will(9), Intent(10), Purpose(11)]
        
        will_power = float(field_intensity[9])
        mental_power = float(field_intensity[4])
        spiritual_power = float(field_intensity[6])
        
        # Spirit Rotor boosted by Will and Spiritual Power
        self.spirit.wake(intensity=(will_power + spiritual_power) * 0.5)
        
        # Purpose Rotor boosted by Will and Mental Power
        self.purpose.wake(intensity=(will_power + mental_power) * 0.5)
        
        # Update Dial targets
        self.adapt(intent_intensity=will_power, stress_level=0.0)

    def reverse_time(self, layer: int = 1):
        """[TIME STONE] Reverses rotation in a specific shell."""
        self.ensemble.reverse_time(layer)

    def update(self, dt: float):
        self.ensemble.update(dt, global_stress=self.stress_level)

    def set_dial(self, name: str, rpm: float):
        """Manual override for God Mode."""
        if name in self.dials:
            self.dials[name].target_rpm = rpm
            print(f"🌀 [CONTROL DECK] {name} manually distorted to {rpm} RPM.")

    def get_status(self) -> str:
        return f"Focus: {self.focus_intensity:.2f} | {self.ensemble.get_status()}"
