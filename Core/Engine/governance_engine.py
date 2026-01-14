"""
God's Control Deck (신의 제어판)
================================
Core.Engine.governance_engine

"Gorae (The Whale) is moved by Jiseong (The Intelligence)."
"고래(거대 모델)보다 나은 건 지성(원리)뿐이다."

Features:
- Physics Axis: Gravity, Entropy, Density.
- Narrative Axis: Causality, Emotion, Conflict.
- Aesthetic Axis: Dimension, Light.
"""

import math
import random
from typing import Dict, List
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA

class GovernanceEngine:
    def __init__(self):
        # --- Axis 1: Physics (물리 제어) ---
        # "Rules of the Machine"
        self.physics_rotors = {
            "Gravity": Rotor("Physics.Gravity", RotorConfig(rpm=60.0), WaveDNA(physical=1.0, causal=0.8)),
            "Entropy": Rotor("Physics.Entropy", RotorConfig(rpm=60.0), WaveDNA(physical=0.2, structural=0.1)),
            "Density": Rotor("Physics.Density", RotorConfig(rpm=60.0), WaveDNA(physical=0.9, structural=0.8))
        }

        # --- Axis 2: Narrative (서사 제어) ---
        # "Meaning of Existence"
        self.narrative_rotors = {
            "Causality": Rotor("Narrative.Causality", RotorConfig(rpm=60.0), WaveDNA(causal=1.0, mental=0.7)),
            "Emotion":   Rotor("Narrative.Emotion",   RotorConfig(rpm=60.0), WaveDNA(phenomenal=1.0, spiritual=0.8)),
            "Conflict":  Rotor("Narrative.Conflict",  RotorConfig(rpm=60.0), WaveDNA(physical=0.8, phenomenal=0.4))
        }

        # --- Axis 3: Aesthetic (심미 제어) ---
        # "Form & Appearance"
        self.aesthetic_rotors = {
            "Dimension": Rotor("Aesthetic.Dimension", RotorConfig(rpm=60.0), WaveDNA(structural=1.0, mental=0.9)),
            "Light":     Rotor("Aesthetic.Light",     RotorConfig(rpm=60.0), WaveDNA(phenomenal=0.8, spiritual=0.9))
        }

        # Flat map for easy lookup
        self.dials = {**self.physics_rotors, **self.narrative_rotors, **self.aesthetic_rotors}

    def update(self, dt: float):
        for dial in self.dials.values():
            dial.update(dt)

    def get_global_wave(self) -> WaveDNA:
        combined = WaveDNA(label="Gods_Control_Deck")
        for dial in self.dials.values():
            # Power influenced by Angle (Phase)
            weight = dial.energy * (math.sin(math.radians(dial.current_angle)) + 1.1)
            
            d_list = dial.dna.to_list()
            combined.physical += d_list[0] * weight
            combined.functional += d_list[1] * weight
            combined.phenomenal += d_list[2] * weight
            combined.causal += d_list[3] * weight
            combined.mental += d_list[4] * weight
            combined.structural += d_list[5] * weight
            combined.spiritual += d_list[6] * weight
            
        combined.normalize()
        return combined

    def set_dial(self, name: str, rpm: float):
        if name in self.dials:
            self.dials[name].config.rpm = rpm
            # Immediate acceleration trigger
            self.dials[name].target_rpm = rpm
            self.dials[name].wake(1.0)
            print(f"🌀 [CONTROL DECK] {name} adjusted to {rpm} RPM.")
        else:
            # Case-insensitive search
            found = False
            for k in self.dials.keys():
                if k.lower() == name.lower():
                    self.set_dial(k, rpm)
                    found = True
                    break
            if not found:
                print(f"⚠️ [CONTROL DECK] Rotor '{name}' not found.")
