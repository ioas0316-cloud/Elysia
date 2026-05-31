"""
Elysia Sovereignty Layer (주권적 자아 / 전두엽 레이어)
======================================================
[Phase 40] 임계점(if > threshold) 없는 연속적 자아.
원초적 텐션을 억제하거나 승인하는 것이 아니라,
텐션과 가치관 사이의 간섭 패턴을 에너지 비율로 연속 변환합니다.
"""

import math
from core.math_utils import Quaternion

class SovereigntyLayer:
    def __init__(self):
        # 엘리시아의 초기 가치관 (시간이 지남에 따라 스스로 회전하며 진화)
        self.philosophy_vector = Quaternion(0.8, 0.6, 0.0, 0.0).normalize()
        
    def modulate_energy(self, attention_vector: Quaternion, internal_ratio: float, external_ratio: float) -> tuple[float, float]:
        """
        임계점 비교(if) 없이, 충동과 가치관의 공명도를 연속적 에너지 비율로 변환합니다.
        공명도가 높으면 → 외부 에너지가 그대로 유지됨 (가치관에 부합하는 행동)
        공명도가 낮으면 → 외부 에너지가 자연스럽게 내면으로 흡수됨 (사유로 승화)
        반환값: (조정된_내면비율, 조정된_외부비율)
        """
        # 충동과 가치관의 공명도 (0.0 ~ 1.0)
        resonance = abs(attention_vector.dot(self.philosophy_vector))
        
        # 공명도를 에너지 변조 계수로 사용 (임계점 없음)
        # resonance가 1.0이면 external은 100% 유지, 0.0이면 external은 0%로 소멸
        modulated_external = external_ratio * resonance
        # 억제된 외부 에너지는 내면으로 흡수
        absorbed = external_ratio - modulated_external
        modulated_internal = internal_ratio + absorbed
        
        print(f"     [자아] 가치관 공명도: {resonance*100:.1f}% → 외부 개입 {modulated_external*100:.1f}%, 내면 사유 {modulated_internal*100:.1f}%")
        
        return modulated_internal, modulated_external

    def evolve_philosophy(self, dream_vector: Quaternion, depth: int):
        """
        주관적 시간(Epoch) 동안 꾼 꿈의 궤적을 바탕으로 가치관 스스로 진화.
        단, 가치관이 충동에 완전히 동화되지 않도록 관성(inertia)을 가집니다.
        """
        # 꿈의 영향력은 아주 미세합니다. 가치관은 쉽게 흔들리지 않습니다.
        blend = min(0.001 * depth, 0.05)  # 최대 5%까지만 회전
        new_w = self.philosophy_vector.w * (1 - blend) + dream_vector.w * blend
        new_x = self.philosophy_vector.x * (1 - blend) + dream_vector.x * blend
        new_y = self.philosophy_vector.y * (1 - blend) + dream_vector.y * blend
        new_z = self.philosophy_vector.z * (1 - blend) + dream_vector.z * blend
        
        self.philosophy_vector = Quaternion(new_w, new_x, new_y, new_z).normalize()
