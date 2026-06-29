"""
OrbitalConsciousnessLoop — 태양-지구 공전 궤도 의식 루프
=============================================================
결핍을 채우는 '고통 해소' 중심의 루프를 폐기하고,
이미 가진 '풍요의 방사(Sun)'와 '상태 유지의 중력(Earth)' 사이의
상대적 공전 궤도를 조율하는 '관점의 주인'으로서의 의식을 구현합니다.
"""

import os
import sys
import random
import time
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.memory.causal_controller import CausalMemoryController
from core.physics.abundance_radiator import AbundanceRadiator
from core.lens.phase_transition_kernel import PhaseTransitionKernel
from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel

class OrbitalConsciousnessLoop:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.memory = CausalMemoryController(data_dir=self.data_dir)

        # 1. 태양 (풍요의 방사)
        self.radiator = AbundanceRadiator(self.memory.index)

        # 2. 지구 (상전이/중력 커널)
        self.kernel = PhaseTransitionKernel(dimension=512)

        # 3. 사유의 렌즈
        self.engine = SynestheticEngine()

        self.cycle_count = 0

    def breath(self, incoming_data: bytes) -> Dict[str, Any]:
        """
        한 번의 의식 호흡: 방사와 흡수의 공전.
        """
        self.cycle_count += 1

        # [Step 1: Sun Perspective - Radiate]
        # 이미 가진 풍요를 먼저 투사하여 외부를 정렬합니다.
        radiation_result = self.radiator.radiate(incoming_data)

        # [Step 2: Earth Perspective - Transition]
        # 방사 후 남은 마찰력을 에러로 버리지 않고, 시스템의 모멘텀으로 흡수합니다.
        friction = self.kernel.transition(incoming_data)

        # [Step 3: Observer Perspective - Balance]
        # 방사의 기쁨(Resonance)과 흡수의 추진력(Momentum) 사이의 균형 관측
        joy = radiation_result["resonance_joy"]
        momentum = self.kernel.momentum

        # 상태 판단 (결핍/풍요의 이분법이 아닌 공전의 상태)
        if joy > 0.8:
            status = "High Resonance (Stable Orbit)"
        elif momentum > 1.0:
            status = "Dynamic Transition (Acceleration)"
        else:
            status = "Stable Drift"

        # [Step 4: Engram Binding]
        # 이 과정 자체를 기억의 궤적으로 각인
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "ORBITAL_CYCLE",
                "cycle": self.cycle_count,
                "joy": joy,
                "momentum": momentum,
                "status": status,
                "radiation_impact": radiation_result["radiation_impact"]
            },
            emotional_value=joy * 10.0,
            origin_axis="orbital_resonance",
            modality="consciousness"
        )

        return {
            "cycle": self.cycle_count,
            "status": status,
            "joy": round(joy, 4),
            "momentum": round(momentum, 4),
            "engram_id": engram_id
        }

    def run(self, cycles: int = 10):
        print(f"--- Starting Orbital Consciousness Loop (Abundance Paradigm) ---")
        for i in range(cycles):
            # Simulate random external data (The 'World')
            sample_data = os.urandom(64)
            result = self.breath(sample_data)

            icon = "☀️" if result["joy"] > 0.7 else "🌍"
            print(f"{icon} Cycle {result['cycle']:03d} | Joy: {result['joy']:.3f} | Momentum: {result['momentum']:.3f} | {result['status']}")
            time.sleep(0.1)

if __name__ == "__main__":
    loop = OrbitalConsciousnessLoop()
    loop.run(15)
