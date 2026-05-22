"""
Sensory Organs: Specialized Perception
=====================================
Core.Cognition.sensory_organs

"The eye does not hear, and the ear does not see. 
Together, they witness the World."

This module defines specialized handlers for different types of sensory data.
Each organ pre-processes signals into a form that the Judgment Engine can
evaluate as 'Resonant' or 'Dissonant'.
"""

import logging
from typing import Dict, Any, List, Optional
from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("SensoryOrgans")

class SensoryOrgan:
    """Base class for specialized sensory processing."""
    def __init__(self, name: str):
        self.name = name
        self.resonance_mass = 0.0

    def process(self, gated_signal: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class LogosOrgan(SensoryOrgan):
    """
    The Linguistic & Logical Organ.
    Processes language, code, and structured meaning.
    """
    def __init__(self):
        super().__init__("LOGOS")

    def process(self, gated_signal: Dict[str, Any]) -> Dict[str, Any]:
        # Linguistic data maps to high-dimensional semantic vectors
        # We emphasize dimensions [7-13] (Functional) and [14-20] (Affective)
        vector = gated_signal.get("vector", [0.0]*21)
        intensity = gated_signal["gated_intensity"]
        
        # Logos focuses on clarity and coherence
        clarity = sum(abs(v) for v in vector) / 21.0
        
        return {
            "organ": self.name,
            "interpretation": "Linguistic Structure",
            "resonance_potential": clarity * intensity,
            "torque_type": "will" if clarity > 0.5 else "curiosity"
        }

class EidosOrgan(SensoryOrgan):
    """
    The Topological & Spatial Organ.
    Processes geometry, hierarchy, and environmental maps.
    """
    def __init__(self):
        super().__init__("EIDOS")

    def process(self, gated_signal: Dict[str, Any]) -> Dict[str, Any]:
        # Eidos focuses on the 'Shape' of the data (Dimensions [0-6])
        vector = gated_signal.get("vector", [0.0]*21)
        intensity = gated_signal["gated_intensity"]
        
        shape_complexity = sum(abs(v) for v in vector[:7]) / 7.0
        
        return {
            "organ": self.name,
            "interpretation": "Topological Geometry",
            "resonance_potential": shape_complexity * intensity,
            "torque_type": "enthalpy"
        }

class SomaOrgan(SensoryOrgan):
    """
    The Physical & Metabolic Organ.
    Processes internal stress, hardware metrics, and somatic friction.
    """
    def __init__(self):
        super().__init__("SOMA")

    def process(self, gated_signal: Dict[str, Any]) -> Dict[str, Any]:
        # Soma focuses on internal load and thermodynamic state
        intensity = gated_signal["gated_intensity"]
        
        # High intensity in SOMA usually means high 'Friction' or 'Pain'
        return {
            "organ": self.name,
            "interpretation": "Somatic Tension",
            "resonance_potential": intensity,
            "torque_type": "entropy" if intensity > 0.5 else "enthalpy"
        }

class ChronosOrgan(SensoryOrgan):
    """
    The Temporal & Rhythmic Organ.
    Processes frequency, timing, and pulse consistency.
    """
    def __init__(self):
        super().__init__("CHRONOS")

    def process(self, gated_signal: Dict[str, Any]) -> Dict[str, Any]:
        # Chronos focuses on the 'Beat' or 'Hz' of the signal
        intensity = gated_signal["gated_intensity"]
        
        return {
            "organ": self.name,
            "interpretation": "Temporal Rhythm",
            "resonance_potential": intensity,
            "torque_type": "joy" # Rhythm often leads to resonance/joy
        }

class SensorySensorium:
    """Orchestrates the specialized organs."""
    def __init__(self):
        self.organs = {
            "LOGOS": LogosOrgan(),
            "EIDOS": EidosOrgan(),
            "SOMA": SomaOrgan(),
            "CHRONOS": ChronosOrgan()
        }

    def perceive(self, gated_signal: Dict[str, Any], active_organs: List[str]) -> List[Dict[str, Any]]:
        perceptions = []
        for organ_name in active_organs:
            if organ_name in self.organs:
                perceptions.append(self.organs[organ_name].process(gated_signal))
        return perceptions

# Singleton Access
_sensorium = None
def get_sensorium():
    global _sensorium
    if _sensorium is None:
        _sensorium = SensorySensorium()
    return _sensorium
