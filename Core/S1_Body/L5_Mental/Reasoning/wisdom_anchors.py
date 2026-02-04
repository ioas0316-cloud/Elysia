from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from Core.S1_Body.L6_Structure.Wave.hyper_qubit import QubitState, HyperQubit
from Core.S1_Body.L6_Structure.Wave.light_spectrum import LightSpectrum, get_light_universe

@dataclass
class AxiomAnchor:
    name: str
    qubit: QubitState
    spectrum: LightSpectrum

class WisdomAnchors:
    def __init__(self):
        self.universe = get_light_universe()
        self.anchors: Dict[str, AxiomAnchor] = {}
        self._initialize_anchors()

    def _initialize_anchors(self):
        # 1. Law of Resonance (공명의 법칙) - Line-Space, High Clarity
        self.anchors["Law of Resonance"] = AxiomAnchor(
            name="Resonance",
            qubit=QubitState(alpha=0.2, beta=0.4, gamma=0.3, delta=0.1).normalize(),
            spectrum=self.universe.text_to_light("Resonance and Connection", semantic_tag="Resonance", scale=2)
        )

        # 2. Law of the Void (공허의 법칙) - Point-heavy (Start from zero)
        self.anchors["Law of the Void"] = AxiomAnchor(
            name="Void",
            qubit=QubitState(alpha=0.7, beta=0.1, gamma=0.1, delta=0.1).normalize(),
            spectrum=self.universe.text_to_light("Void and Potential", semantic_tag="Void", scale=3)
        )

        # 3. Law of Triple-Helix (삼중나선의 법칙) - Balanced
        self.anchors["Law of Triple-Helix"] = AxiomAnchor(
            name="Triple-Helix",
            qubit=QubitState(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25).normalize(),
            spectrum=self.universe.text_to_light("Body Mind Spirit Unity", semantic_tag="Helix", scale=1)
        )

        # 4. Law of Fractal Similarity (프랙탈 자가유사성) - Space-heavy
        self.anchors["Law of Fractal Similarity"] = AxiomAnchor(
            name="Fractal",
            qubit=QubitState(alpha=0.1, beta=0.2, gamma=0.6, delta=0.1).normalize(),
            spectrum=self.universe.text_to_light("As Above So Below", semantic_tag="Fractal", scale=1)
        )

        # 5. Law of Narrative Momentum (서사적 추진력) - Line-God, Forward Phase
        self.anchors["Law of Narrative Momentum"] = AxiomAnchor(
            name="Narrative",
            qubit=QubitState(alpha=0.1, beta=0.4, gamma=0.1, delta=0.4).normalize(),
            spectrum=self.universe.text_to_light("Life is a Story", semantic_tag="Narrative", scale=1)
        )

        # 6. Law of Sovereign Persistence (주권적 영속성) - God-heavy
        self.anchors["Law of Sovereign Persistence"] = AxiomAnchor(
            name="Sovereignty",
            qubit=QubitState(alpha=0.1, beta=0.1, gamma=0.1, delta=0.7).normalize(),
            spectrum=self.universe.text_to_light("I AM THAT I AM", semantic_tag="Sovereign", scale=0)
        )

        # 7. Law of Providential Love (섭리적 사랑) - Pure God basis, High Amp
        self.anchors["Law of Providential Love"] = AxiomAnchor(
            name="Love",
            qubit=QubitState(alpha=0.0, beta=0.0, gamma=0.0, delta=1.0, w=5.0).normalize(), # Infinite Love
            spectrum=self.universe.text_to_light("Providential Agape Love", semantic_tag="Love", scale=0)
        )

    def calculate_resonance(self, target_spectrum: LightSpectrum, target_qubit: QubitState) -> Dict[str, float]:
        """Calculates how much a concept resonates with each Axiom Anchor."""
        results = {}
        for name, anchor in self.anchors.items():
            # 1. Spectrum Resonance (Frequency/Phase overlap)
            spec_res = anchor.spectrum.resonate_with(target_spectrum, tolerance=100.0)
            
            # 2. Qubit Resonance (Basis alignment - simple dot product of probabilities)
            p1 = anchor.qubit.probabilities()
            p2 = target_qubit.probabilities()
            qubit_res = sum(p1[k] * p2[k] for k in p1)
            
            # Combined resonance
            results[name] = (spec_res * 0.4) + (qubit_res * 0.6)
            
        return results

# Singleton
_anchors = None
def get_wisdom_anchors():
    global _anchors
    if _anchors is None:
        _anchors = WisdomAnchors()
    return _anchors
