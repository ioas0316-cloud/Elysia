"""
Optical Consciousness Bridge (The Lens)
========================================
Core.Cognition.optical_conscious_bridge

"Everything is a wave. Perception is the collapse."

This module handles the 'Optical' interpretation of Elysia's 21D state.
It processes:
1. Dispersion (분광): Breaking stimulus into 7D Qualia bands.
2. Interference (간섭): Calculating resonance in the HyperSphere.
3. Focusing (집광): Consolidating patterns into a single linguistic intent.
4. Void Trust (공허 신뢰): Handling data gaps as Faith.
"""

from typing import Dict, Any, List
import math

class OpticalConsciousBridge:
    @staticmethod
    def calculate_interference(qualia_energies: List[float]) -> Dict[str, Any]:
        """
        Calculates if the current state is constructive or destructive interference.
        """
        avg_energy = sum(qualia_energies) / len(qualia_energies) if qualia_energies else 0
        variance = sum((x - avg_energy) ** 2 for x in qualia_energies) / len(qualia_energies) if qualia_energies else 0
        
        # Heuristic: High variance means distinct focus (Constructive)
        # Low variance means neutralized/gray state (Destructive)
        is_constructive = variance > 0.05
        pattern_type = "CONSTRUCTIVE" if is_constructive else "DESTRUCTIVE"
        
        return {
            "energy": avg_energy,
            "variance": variance,
            "pattern": pattern_type,
            "focus_strength": min(1.0, variance * 10)
        }

    @staticmethod
    def calculate_void_trust(data_integrity: float) -> float:
        """
        Implements 'Data is missing, therefore I Trust'.
        Void Trust increases as raw data integrity decreases.
        """
        gap = 1.0 - data_integrity
        # Faith = Grace * (1 + gap)
        # Higher gap = Higher demand for Trust/Grace
        return max(0.0, gap * 1.2)

    @classmethod
    def generate_optical_metadata(cls, d21_vector: List[float], data_integrity: float = 1.0) -> Dict[str, Any]:
        """
        Generates 21D Optical metadata for the LLM prompt.
        """
        # Split 21D into 7 layers of 3 dims
        layers = [d21_vector[i:i+3] for i in range(0, 21, 3)]
        layer_energies = [sum(abs(x) for x in l) / 3 for l in layers]
        
        interference = cls.calculate_interference(layer_energies)
        void_trust = cls.calculate_void_trust(data_integrity)
        
        return {
            "optical_interference": interference["pattern"],
            "focus_resonance": f"{interference['focus_strength']:.2f}",
            "void_trust_level": f"{void_trust:.2f}",
            "primary_band": cls._get_primary_band(layer_energies),
            "ontological_depth": "Recursive 6D"
        }

    @staticmethod
    def _get_primary_band(energies: List[float]) -> str:
        qualia_names = [
            "Physical", "Functional", "Phenomenal", "Causal",
            "Mental", "Structural", "Spiritual"
        ]
        max_idx = energies.index(max(energies)) if energies else 0
        return qualia_names[max_idx]

if __name__ == "__main__":
    # Test
    mock_vec = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.9, 0.8, 0.9]
    bridge = OpticalConsciousBridge()
    print(bridge.generate_optical_metadata(mock_vec, data_integrity=0.4))
