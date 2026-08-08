"""
LanguageRefractor (언어 굴절기) - Phase Transition & Sensory Refraction
========================================================================
이 모듈은 1차원 자연어 문장을 입력받아, 이를 생명체의 감각 피질에 쏟아지는 외부 자극(Sensory Stimulus)으로 취급하고,
위상 OS가 이해할 수 있는 물리적 성질인 4차원 임펄스 벡터(질량, 시급성, 위상차, 주파수)로 '굴절(Refraction)'시킵니다.

- Mass (질량): 자극의 관성적 무게. RNS 장의 지형 자체를 변형시킵니다.
- Gradient (시급성): 포텐셜 우물의 가파름. Langevin 이완의 가속도를 결정합니다.
- Phase (위상/좌표): 자극이 떨어질 RNS 공간 상의 (y, x) 중심 위치.
- Frequency (주파수/공명): 자극의 고유 파동 주파수. 고유 위상파와의 공명 및 감쇄율을 제어합니다.
"""

import numpy as np
import hashlib
from typing import Dict, Any, Tuple

class LanguageRefractor:
    """
    1D Natural Language Refractor Lens.
    Translates textual stimulus into physical 4D impulse parameters to be injected into TopologicalOSEngine.
    """
    def __init__(self, grid_shape: Tuple[int, int] = (16, 16)):
        self.grid_shape = grid_shape

    def refract(self, text: str) -> Dict[str, Any]:
        """
        Refracts natural language string into physical parameter tensors:
        - mass: float (0.1 to 30.0) -> determines local perturbation magnitude
        - gradient: float (1.0 to 10.0) -> maps to importance/modular excitation amplitude
        - target_y: int, target_x: int -> coordinate center based on semantic text hashes
        - wave_signature: float (-1.0 to 1.0) -> phase wave signature
        - thermal_heating: float (0.0 to 5.0) -> thermal fluctuation injection (temperature delta)
        """
        # Ensure we clean the text
        clean_text = text.strip()
        if not clean_text:
            return {
                "mass": 0.1,
                "gradient": 1.0,
                "target_y": 0,
                "target_x": 0,
                "wave_signature": 0.0,
                "thermal_heating": 0.0,
                "intent_type": "vacuum"
            }

        # Determine Intent type and map physical attributes
        # High Energy / Urgent / Specific Action Command
        urgent_keywords = ["빨리", "당장", "급해", "버그", "고쳐줘", "error", "urgent", "immediate", "fix", "crash"]
        # Low Energy / Brownian / Casual Speculation
        casual_keywords = ["문득", "그냥", "생각", "떠올라", "wonder", "maybe", "casual", "by the way", "perhaps", "stroll"]

        is_urgent = any(kw in clean_text.lower() for kw in urgent_keywords)
        is_casual = any(kw in clean_text.lower() for kw in casual_keywords)

        # Base deterministic coordinates from hash to make identical sentences hit the same attractor locus
        sha = hashlib.sha256(clean_text.encode('utf-8')).digest()
        target_y = int(sha[0]) % self.grid_shape[0]
        target_x = int(sha[1]) % self.grid_shape[1]

        # Signature based on hash values (scaled between -1.0 and 1.0)
        wave_signature = float((int(sha[2]) / 255.0) * 2.0 - 1.0)

        if is_urgent and not is_casual:
            # Urgent: Heavy mass, steep gradient, zero thermal noise to avoid chaos, high magnitude
            mass = 25.0 + float(sha[3] % 5)
            gradient = 8.0 + float(sha[4] % 3)
            thermal_heating = 0.0
            intent_type = "high_gradient_well"
        elif is_casual and not is_urgent:
            # Casual: Light mass, very gentle gradient, causes high thermal diffusion (heating the field)
            mass = 1.0 + float(sha[3] % 3)
            gradient = 1.0 + float(sha[4] % 2) / 2.0
            thermal_heating = 3.5 + float(sha[5] % 15) / 10.0
            intent_type = "brownian_perturbation"
        else:
            # Neutral / Standard default text
            mass = 10.0 + float(sha[3] % 5)
            gradient = 3.0 + float(sha[4] % 3)
            thermal_heating = 0.5
            intent_type = "standard_wave"

        return {
            "mass": mass,
            "gradient": gradient,
            "target_y": target_y,
            "target_x": target_x,
            "wave_signature": wave_signature,
            "thermal_heating": thermal_heating,
            "intent_type": intent_type
        }

    def evaluate_cognitive_feedback(self, initial_state: dict, final_state: dict, steps_taken: int) -> dict:
        """
        Feedback loop analysis. Measures how well the refracted stimulus has settled/dissipated in the OS field.
        Returns cognitive feedback metrics:
        - convergence_rate: speed of settling
        - friction_dissipation: total energy dissipated/damped
        - constraint_satisfaction: boolean indicating if it successfully reached back to vacuum state (1)
        """
        init_potential = np.sum(initial_state["potential"])
        final_potential = np.sum(final_state["potential"])

        init_energy = np.sum(initial_state["energy"])
        final_energy = np.sum(final_state["energy"])

        # Measure if all residues returned to the ground state (1)
        # In TopologicalOSEngine, V_potential = 0 means vacuum ground state 1
        is_ground = (final_potential < 1e-4)

        return {
            "initial_potential": float(init_potential),
            "final_potential": float(final_potential),
            "initial_energy": float(init_energy),
            "final_energy": float(final_energy),
            "steps_taken": steps_taken,
            "constraint_satisfied": bool(is_ground),
            "energy_loss": float(max(0.0, init_energy - final_energy))
        }
