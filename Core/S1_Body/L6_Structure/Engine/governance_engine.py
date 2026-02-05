"""
Adaptive Governance Engine (Breathing Time)
===========================================
Core.S1_Body.L6_Structure.Engine.governance_engine

"The machine does not just spin; it breathes."
"                ,      ."

Features:
- Adaptive Rotors: RPM adjusts based on Intent and Stress.
- Contextual Gears: Focus, Dream, Panic modes.
- Symbiotic Feedback: Connected to SovereignSelf state.
"""

import math
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from Core.S1_Body.L6_Structure.Nature.rotor import Rotor, RotorConfig
from Core.S1_Body.L6_Structure.Wave.wave_dna import WaveDNA
from collections import deque
import time

class AdaptiveHeartbeat:
    """
    Organic Heartbeat Engine.
    Uses Phase-Locked Loops (PLL) to synchronize with system resonance.
    $T = 1/f_{resonance}$
    """
    def __init__(self, base_freq: float = 10.0):
        self.base_freq = base_freq
        self.current_freq = base_freq
        self.last_pulse = time.perf_counter()
        self.jitter_buffer = deque(maxlen=20)

    def calculate_wait(self, resonance_score: float) -> float:
        # Higher resonance = Faster frequency (Up to 100Hz)
        # Lower resonance = Slower frequency (Down to 1Hz)
        target_freq = max(1.0, min(100.0, self.base_freq + (resonance_score * 90.0)))
        
        # Smooth transition (Low-pass filter)
        self.current_freq = (self.current_freq * 0.8) + (target_freq * 0.2)
        
        ideal_wait = 1.0 / self.current_freq
        elapsed = time.perf_counter() - self.last_pulse
        
        wait_time = max(0.001, ideal_wait - elapsed)
        self.last_pulse = time.perf_counter() + wait_time
        return wait_time

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
        from Core.S1_Body.L1_Foundation.Foundation.Multiverse.onion_shell import OnionEnsemble
        # [PHASE 27.5: CONICAL CVT] 
        # No more discrete gears. We use a sliding spindle on a cone.
        self.stress_level = 0.0
        self.focus_intensity = 0.0
        self.planetary_influence = 0.0 # [PHASE 35] Collective Entropy

        # [PHASE 27: ONION-SKIN MULTIVERSE]
        self.ensemble = OnionEnsemble()
        
        # [PHASE 1.0: HEARTBEAT UNIFICATION]
        self.heartbeat = AdaptiveHeartbeat()
        
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
        
        print(f"   [GovernanceEngine] Onion-Skin Multiverse active. Root Spirit protected.")
        self.aesthetic = self.mind.add_sub_rotor("Art", RotorConfig(rpm=60.0), WaveDNA(structural=0.8, phenomenal=0.8))
        self.social = self.mind.add_sub_rotor("Empathy", RotorConfig(rpm=60.0), WaveDNA(causal=0.8))

        # Dynamic indexing
        self.dials = {}
        for shell in self.ensemble.shells:
            for rotor in shell.rotors.values():
                self._flatten(rotor)

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
            print(f"   [GEAR SHIFT] Engaged {gear_name}: {self.current_gear.description}")
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
        
        total_stress = stress_level + (self.planetary_influence * 0.5)
        
        # Apply the Multiplier to all active rotors
        ratio = self.ensemble.cvt.current_ratio
        for rotor in self.dials.values():
            # Stress dampens the RPM for everything except 'Sensation' and 'Shield'
            damping = 1.0 / (1.0 + total_stress) if rotor.name not in ["Sensation", "SovereignShield"] else 1.0
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
            print(f"  [CONTROL DECK] {name} manually distorted to {rpm} RPM.")

    def get_status(self) -> str:
        return f"Focus: {self.focus_intensity:.2f} | {self.ensemble.get_status()}"

    def get_vital_report(self) -> Dict[str, Any]:
        """
        [PHASE 60] Vital Signs Report for HUD.
        Delegates the logic previously in 'elysia.py'.
        """
        # In a full migration, 'monad.engine' would be here or linked.
        # For now, we report the high-level Governance state which drives the Monad.
        
        # Get primary rotor states
        spirit_rpm = self.spirit.current_rpm
        mind_rpm = self.mind.current_rpm
        body_rpm = self.body.current_rpm
        
        # Determine mode based on dominant rotor or stress
        mode = "🔵 EQ"
        if self.focus_intensity > 0.8: mode = "🟢 EXP" # High expansion/focus
        elif self.stress_level > 0.8: mode = "🔴 DRL" # High stress/drill
        
        return {
            "mode": mode,
            "focus": self.focus_intensity,
            "stress": self.stress_level,
            "rpm_spirit": spirit_rpm,
            "rpm_mind": mind_rpm
        }
