"""
Character Field Engine (인격 장 엔진)
=====================================
Core.Engine.character_field_engine

"Personality is not a tag; it is an interference pattern."
"인격은 딱지가 아니라, 간섭 무늬다."

Features:
- MBTI to 4D Rotor Mapping.
- Enneagram Attractor Fields.
- Psionic Synthesis: Calculating the resulting Qualia Wave.
"""

import math
import random
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA

@dataclass
class RotorTrait:
    name: str
    frequency: float  # Base RPM
    axis_weights: torch.Tensor # (4,) - Alignment in X,Y,Z,W
    phase_offset: float = 0.0

class CharacterField:
    """
    The Dynamic Soul of a Citizen.
    Instead of fixed stats, it holds multiple 'RotorTraits' that interfere.
    """
    def __init__(self, name: str, mbti: str = "INFP", enneagram: int = 9):
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.rotors: List[Rotor] = []
        self._initialize_mbti(mbti)
        self._initialize_enneagram(enneagram)
        
        # Cumulative Field State (The "Resultant Wave")
        self.current_state = torch.zeros(4, device=self.device)
        self.personality_label = f"{mbti}-{enneagram}w?"

    def _initialize_mbti(self, mbti: str):
        """
        Maps MBTI to 4D Rotor Physics.
        E/I: Amplitude (Social Energy)
        S/N: Resonance Axis (Physical vs Phenomenal)
        T/F: Resonance Axis (Mental vs Spiritual)
        J/P: Harmonic Stability
        """
        config = RotorConfig(rpm=120.0, idle_rpm=60.0)
        
        # Dimensional Mappings
        # E/I -> (1, 0, 0, 0) vs (-1, 0, 0, 0)
        social_dna = WaveDNA(physical=0.8 if 'E' in mbti else 0.2, spiritual=0.2 if 'E' in mbti else 0.8)
        self.rotors.append(Rotor(f"MBTI.Social.{mbti[0]}", config, social_dna))
        
        # S/N -> (0, 1, 0, 0) vs (0, -1, 0, 0)
        perc_dna = WaveDNA(physical=0.9 if 'S' in mbti else 0.1, phenomenal=0.1 if 'S' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Perception.{mbti[1]}", config, perc_dna))
        
        # T/F -> (0, 0, 1, 0) vs (0, 0, -1, 0)
        judging_dna = WaveDNA(mental=0.9 if 'T' in mbti else 0.1, spiritual=0.1 if 'T' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Judging.{mbti[2]}", config, judging_dna))
        
        # J/P -> Stability weight
        lifestyle_dna = WaveDNA(structural=0.9 if 'J' in mbti else 0.1, functional=0.1 if 'J' in mbti else 0.9)
        self.rotors.append(Rotor(f"MBTI.Lifestyle.{mbti[3]}", config, lifestyle_dna))

    def _initialize_enneagram(self, enneagram: int):
        """
        Maps Enneagram types to Causal Attractors.
        """
        # Enneagram as a circular frequency bias (Type 1-9)
        freq = 360.0 * (enneagram / 9.0)
        ennea_dna = WaveDNA(causal=0.8, label=f"Enneagram.{enneagram}")
        self.rotors.append(Rotor(f"Ennea.{enneagram}", RotorConfig(rpm=freq), ennea_dna))

    def update(self, dt: float, external_pressure: WaveDNA = None):
        """
        Updates the interference pattern.
        """
        total_dna = WaveDNA(label=f"{self.name}_Field")
        
        for rotor in self.rotors:
            # 1. Update Physics
            rotor.update(dt)
            
            # 2. Modulate by Pressure (Excitation)
            if external_pressure:
                resonance = rotor.dna.resonate(external_pressure)
                if resonance > 0.5:
                    rotor.wake(resonance)
                else:
                    rotor.relax()
            
            # 3. Summing interference
            # Each rotor contributes its DNA scaled by its current Angle (Wave position)
            # sin(theta) determines the current phase of the trait
            phase = math.sin(math.radians(rotor.current_angle))
            weight = (phase + 1.0) / 2.0 # 0.0 ~ 1.0
            
            # Add to the field
            total_dna.physical += rotor.dna.physical * weight
            total_dna.functional += rotor.dna.functional * weight
            total_dna.phenomenal += rotor.dna.phenomenal * weight
            total_dna.causal += rotor.dna.causal * weight
            total_dna.mental += rotor.dna.mental * weight
            total_dna.structural += rotor.dna.structural * weight
            total_dna.spiritual += rotor.dna.spiritual * weight
            
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
