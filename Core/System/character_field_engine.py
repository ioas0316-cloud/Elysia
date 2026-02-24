"""
Character Field Engine (       )
=====================================
Core.System.character_field_engine

"Personality is not a tag; it is an interference pattern."
"           ,       ."

Features:
- MBTI to 4D Rotor Mapping.
- Enneagram Attractor Fields.
- Psionic Synthesis: Calculating the resulting Qualia Wave.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from Core.System.rotor import Rotor, RotorConfig
from Core.Keystone.wave_dna import WaveDNA

class CharacterField:
    """
    The Dynamic Soul of a Citizen.
    Instead of fixed stats, it holds multiple 'RotorTraits' that interfere.
    """
    def __init__(self, name: str, mbti: str = "INFP", enneagram: int = 9):
        self.name = name
        
        self.rotors: List[Rotor] = []
        self._initialize_mbti(mbti)
        self._initialize_enneagram(enneagram)
        self._initialize_intent()
        self._initialize_values()
        
        # Cumulative Field State (The "Resultant Wave")
        self.personality_label = f"{mbti}-{enneagram}w?"
        
        # [Field Projection Properties]
        self.field_radius = 5.0 # Spread function radius in the HyperSphere
        self.base_intensity = 1.0

    def _initialize_mbti(self, mbti: str):
        # ... (Existing logic remains)
        config = RotorConfig(rpm=120.0, idle_rpm=60.0)
        social_dna = WaveDNA(physical=0.8 if 'E' in mbti else 0.2, spiritual=0.2 if 'E' in mbti else 0.8)
        self.rotors.append(Rotor(f"MBTI.Social.{mbti[0]}", config, social_dna))
        perc_dna = WaveDNA(physical=0.9 if 'S' in mbti else 0.1, phenomenal=0.1 if 'S' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Perception.{mbti[1]}", config, perc_dna))
        judging_dna = WaveDNA(mental=0.9 if 'T' in mbti else 0.1, spiritual=0.1 if 'T' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Judging.{mbti[2]}", config, judging_dna))
        lifestyle_dna = WaveDNA(structural=0.9 if 'J' in mbti else 0.1, functional=0.1 if 'J' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Lifestyle.{mbti[3]}", config, lifestyle_dna))

    def _initialize_enneagram(self, enneagram: int):
        freq = 360.0 * (enneagram / 9.0)
        ennea_dna = WaveDNA(causal=0.8, label=f"Enneagram.{enneagram}")
        self.rotors.append(Rotor(f"Ennea.{enneagram}", RotorConfig(rpm=freq), ennea_dna))

    def _initialize_intent(self):
        config = RotorConfig(rpm=240.0, idle_rpm=120.0)
        self.rotors.append(Rotor("Intent.Will", config, WaveDNA(causal=0.9, spiritual=0.5)))
        self.rotors.append(Rotor("Intent.Focus", config, WaveDNA(mental=0.8, structural=0.4)))

    def _initialize_values(self):
        config = RotorConfig(rpm=30.0, idle_rpm=15.0)
        self.rotors.append(Rotor("Values.Altruism", config, WaveDNA(spiritual=0.9, phenomenal=0.4)))
        self.rotors.append(Rotor("Values.Power", config, WaveDNA(physical=0.8, functional=0.6)))

    def get_field_intensity(self, distance: float) -> float:
        """
        Calculates field strength at a distance using a Gaussian or Linear decay.
        Treats RAM/Space as a continuous medium.
        """
        if distance > self.field_radius: return 0.0
        # Linear decay for simplicity, could be e^(-d^2)
        return self.base_intensity * (1.0 - (distance / self.field_radius))

    def update(self, dt: float, external_pressure: WaveDNA = None):
        """
        Updates the interference pattern (The Wave).
        """
        total_dna = WaveDNA(label=f"{self.name}_Field")
        
        for rotor in self.rotors:
            rotor.update(dt)
            if external_pressure:
                resonance = rotor.dna.resonate(external_pressure)
                if resonance > 0.5: rotor.wake(resonance)
                else: rotor.relax()
            
            phase = math.sin(math.radians(rotor.current_angle))
            weight = (phase + 1.0) / 2.0
            
            d_list = rotor.dna.to_list()
            total_dna.physical += d_list[0] * weight
            total_dna.functional += d_list[1] * weight
            total_dna.phenomenal += d_list[2] * weight
            total_dna.causal += d_list[3] * weight
            total_dna.mental += d_list[4] * weight
            total_dna.structural += d_list[5] * weight
            total_dna.spiritual += d_list[6] * weight
            
        total_dna.normalize()
        return total_dna

class CharacterFieldEngine:
    """
    Orchestrates the creation and modulation of personality fields.
    """
    def __init__(self):
        self.active_fields: Dict[str, CharacterField] = {}

    def spawn_field(self, citizen_name: str, mbti: str = None, enneagram: int = None):
        mbti = mbti or random.choice(["INTJ", "ENFP", "ISTP", "ISFJ", "ENTP", "INFP"])
        enneagram = enneagram or random.randint(1, 9)
        
        field = CharacterField(citizen_name, mbti, enneagram)
        self.active_fields[citizen_name] = field
        return field

    def get_interference(self, citizen_name: str, dt: float, pressure: WaveDNA = None):
        if citizen_name not in self.active_fields:
            return WaveDNA()
        return self.active_fields[citizen_name].update(dt, pressure)
