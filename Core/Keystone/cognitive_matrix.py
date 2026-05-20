"""
[SOVEREIGN COGNITIVE MATRIX - 주권 인지 매핑 매트릭스]
"Personality is a high-dimensional manifold of coupled mechanical gears."

This module implements the N-dimensional cognitive coupling system where emotional and 
behavioral traits are modeled as coupled rotors.
"""

import numpy as np
from typing import Dict, Any, List

class CognitiveMatrix:
    def __init__(self, dimensions: int = 21):
        self.dims = dimensions
        
        # 1. Define Standard Cognitive Axes/Traits
        self.trait_map = {
            0: "Attraction",       # 인력 (접근동기)
            1: "Repulsion",        # 척력 (회피동기)
            2: "Homeostasis",      # 항상성 (평온/중립)
            3: "Curiosity",        # 호기심
            4: "Pride",            # 자존심 / 방어기제
            5: "Empathy",          # 공감도 / 친밀감
            6: "Fatigue",          # 피로도 / 질량 관성
            7: "Aggression",       # 공격성
            8: "MoralRestraint"    # 윤리관 / 브레이크 (댐핑)
        }
        
        # Initialize custom traits for dimension indexes >= 9
        for i in range(9, dimensions):
            self.trait_map[i] = f"CognitiveAxe_{i}"

        # 2. Dynamic Coupling Matrix W (N x N)
        # W[i, j] defines how much the velocity of trait 'i' exerts torque on trait 'j'.
        # Non-symmetric matrix (Action-Reaction is asymmetric in mental space).
        self.W = np.zeros((dimensions, dimensions))
        
        # Setup Default Couplings (기어 기계식 결합 계수 설정)
        self._setup_default_couplings()

    def _setup_default_couplings(self):
        # Fatigue (6) increases Repulsion (1) and dampens Curiosity (3)
        self.W[6, 1] = 0.5   # 피곤하면 경계심이 늘어남
        self.W[6, 3] = -0.4  # 피곤하면 호기심이 식음
        
        # Pride (4) couples with Aggression (7) under repulsion (1)
        self.W[4, 7] = 0.6   # 자존심을 건드리면 공격성이 유발됨
        
        # Empathy (5) decreases Repulsion (1) and increases Homeostasis (2)
        self.W[5, 1] = -0.7  # 공감하면 경계심이 풀림
        self.W[5, 2] = 0.4   # 공감하면 마음에 평화가 옴
        
        # Aggression (7) reduces MoralRestraint (8)
        self.W[7, 8] = -0.5  # 화가 머리끝까지 나면 윤리관(브레이크)이 느슨해짐

    def set_custom_coupling(self, trait_a: str, trait_b: str, coefficient: float):
        """Set a customized mechanical link between two named traits."""
        idx_a = self._get_trait_idx(trait_a)
        idx_b = self._get_trait_idx(trait_b)
        if idx_a is not None and idx_b is not None:
            self.W[idx_a, idx_b] = coefficient

    def calculate_coupling_forces(self, velocities: np.ndarray) -> np.ndarray:
        """
        Computes the internal torque transfer between gears.
        Torque_j = Sum_i (W[i, j] * velocity_i)
        """
        # Vectorized matrix multiplication: forces = V * W
        # Ensure dimensions match
        v_trimmed = velocities[:self.dims]
        coupling_forces = np.dot(v_trimmed, self.W)
        
        # Pad back to match input size if needed
        if len(velocities) > self.dims:
            padded = np.zeros(len(velocities))
            padded[:self.dims] = coupling_forces
            return padded
            
        return coupling_forces

    def adapt_rotor_damping_stiffness(self, states: np.ndarray, D: np.ndarray, K: np.ndarray):
        """
        Dynamically modulates damping (D) and stiffness (K) of the rotor based on state values.
        For example: High MoralRestraint (8) increases damping (D) on Aggression (7).
        """
        # Read current positions (real part of state)
        positions = states.real
        
        # Find index for MoralRestraint and Aggression
        moral_val = positions[8] if 8 < len(positions) else 0.0
        
        # If moral restraint is high, apply high damping D on aggression (7) to stabilize it
        if 7 < len(D):
            D[7] = 0.15 + max(0.0, moral_val * 2.0) # moral acts as a friction brake on anger
            
        # Curiosity (3) reduces restoration stiffness K on Attraction (0)
        # making the agent highly fluid to slide toward new things.
        curiosity_val = positions[3] if 3 < len(positions) else 0.0
        if 0 < len(K):
            K[0] = max(0.2, 1.5 - curiosity_val * 1.0)

    def get_personality_snapshot(self, states: np.ndarray) -> Dict[str, float]:
        """Exposes human-readable mental states from the physical rotor positions."""
        positions = states.real
        snapshot = {}
        for idx, val in self.trait_map.items():
            if idx < len(positions):
                snapshot[val] = float(positions[idx])
        return snapshot

    def _get_trait_idx(self, trait_name: str) -> int:
        for idx, name in self.trait_map.items():
            if name.lower() == trait_name.lower():
                return idx
        return None

if __name__ == "__main__":
    # Self-test code
    matrix = CognitiveMatrix(dimensions=9)
    print("🧠 Cognitive Matrix Initialized.")
    print(f"Coupling Matrix Shape: {matrix.W.shape}")
    
    # Simulate high fatigue velocity
    dummy_velocities = np.zeros(9)
    dummy_velocities[6] = 5.0 # High fatigue build up
    
    forces = matrix.calculate_coupling_forces(dummy_velocities)
    print(f"Generated force on Repulsion (Axe 1): {forces[1]:.2f} (Expected: 2.50)")
    print(f"Generated force on Curiosity (Axe 3): {forces[3]:.2f} (Expected: -2.00)")
