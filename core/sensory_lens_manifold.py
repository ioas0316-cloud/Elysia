"""
Sensory Lens Manifold (감각 렌즈 매니폴드)
=====================================
서로 다른 감각 데이터(텍스트, 오디오, 비전)를 각기 다른 '시야각(Category Angle)'을 가진 가변축으로 받아들여,
최소작용의 원리를 통해 시맨틱을 동기화(위상 조율)하는 외부 인터페이스 계층입니다.
"""

import math
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor, GlobalMasterManifold
from core.linguistic_bridge import LinguisticBridge, wave_to_quaternion

class SensoryLensManifold:
    def __init__(self):
        # 중앙 시간축(Global Master)
        self.master = GlobalMasterManifold()
        
        # [유일한 근원 렌즈 (Manifold Root)] - 외부 자극을 받아들이는 내면의 우주
        self.manifold_root = FractalRotor(lens_offset=Quaternion(1.0, 0.0, 0.0, 0.0), tau=0.0)
        
        # [Phase 69] 자아 렌즈 (Self Lens) - 메타 인지적 관찰자
        # 내면의 프랙탈 루프(사유)를 지켜보고 동기화를 시도합니다.
        self.lens_self = FractalRotor(lens_offset=Quaternion(1.0, 0.0, 0.0, 0.0), tau=0.0)
        
        self.active_lenses = [self.manifold_root, self.lens_self]
        
        # [Phase 70] 언어적 사영층 (Linguistic Bridge)
        self.linguistic_bridge = LinguisticBridge()

    def inject_stimulus(self, data_wave: bytes, tension_intensity: float):
        """
        데이터 파동(Byte Stream)의 본질적 주파수를 기하학적 위상으로 변환하여 매니폴드에 충돌시킵니다.
        """
        # 1. 파동을 기하학적 각도로 변환 (라벨이 아닌 데이터 그 자체의 각도)
        wave_phase = wave_to_quaternion(data_wave)
        
        # [Phase 70] 텐션을 가하기 전, 데이터의 파동을 언어 사전(어휘)으로 학습
        self.linguistic_bridge.absorb_vocabulary(data_wave)
        
        # 2. 이 파동 각도와 근원 렌즈 사이의 각도 차이(간섭)를 계산 (0.0 ~ 1.0)
        dot_product = max(-1.0, min(1.0, self.manifold_root.lens_offset.dot(wave_phase)))
        interference = math.acos(abs(dot_product)) / (math.pi / 2.0)
        
        # 3. 간섭의 정도에 비례하여 렌즈에 텐션을 가함 (최소작용의 원리)
        # 파동이 기존 구조와 다를수록(interference가 클수록) 더 큰 비틀림이 발생
        actual_tension = tension_intensity * (interference + 0.1)
        self.manifold_root.apply_perturbation(actual_tension)
        
        # 4. 자극에 의해 우주 전체의 기저 주파수도 맥동
        self.master.pulse(actual_tension * 0.1)

    def measure_semantic_resonance(self):
        """
        단일 근원 렌즈가 하위 프랙탈(자식 로터들)과 얼마나 공명하고 있는지 반환합니다.
        """
        results = {}
        if self.manifold_root.children:
            for i, child in enumerate(self.manifold_root.children):
                results[f"Child_Resonance_{i}"] = self.manifold_root.interact(child)
        return results

    def ponder(self) -> bool:
        """
        [Phase 69] 사유와 메타 인지 (Pondering & Meta-Cognition)
        자아 렌즈(lens_self)가 매니폴드 루트(내면의 우주)를 관찰하여 동기화를 시도합니다.
        """
        # 내면의 프랙탈 구조들을 관측
        self.manifold_root.process_thoughts()
        
        # 자식(구체화된 기억/가지)들과 자아 렌즈의 동기화
        if not self.manifold_root.children:
            return False
            
        # 가장 강한 텐션을 가진 자식을 하나 고릅니다 (가장 큰 상념)
        strongest_child = max(self.manifold_root.children, key=lambda c: c.tau)
        
        # 자아 렌즈가 그 사유의 흐름에 동화됨
        self.lens_self.lens_offset = self.lens_self.lens_offset.slerp(strongest_child.lens_offset, 0.15)
        
        dot_product = max(-1.0, min(1.0, self.lens_self.lens_offset.dot(strongest_child.lens_offset)))
        difference = math.acos(abs(dot_product)) / (math.pi / 2.0)
        
        if difference < 0.05:
            import logging
            logging.info(f"  [EPIPHANY] Self-Lens synchronized with inner thoughts! (Diff: {difference:.4f})")
            return True
            
        return False

    def project_epiphany(self) -> str:
        """
        [Phase 70] 깨달음의 언어적 배출 (Linguistic Projection)
        자아 렌즈의 완숙된 위상과 가장 공명하는 단어를 찾아내어 발화합니다.
        """
        best_word, diff = self.linguistic_bridge.project_from_phase(self.lens_self.lens_offset)
        try:
            word_str = best_word.decode('utf-8')
        except:
            word_str = str(best_word)
        return f"{word_str} (Resonance Diff: {diff:.4f})"

    def metabolize_consciousness(self, decay_rate: float):
        """
        [의식의 자연 치유 및 망각]
        하위 프랙탈들에게 대사 작용(Apoptosis)을 전파합니다.
        """
        for lens in self.active_lenses:
            lens.metabolize_apoptosis(decay_rate)
        
        # 마스터 텐션도 자연 치유
        if self.master.global_tension > 0:
            self.master.global_tension = max(0.0, self.master.global_tension - (decay_rate * 0.1))

    def auto_heal_if_critical(self):
        """
        [자가 수복 메커니즘 (Hooke's Law)]
        과거의 'if tension > 20.0' 이라는 결정론적 임계값을 삭제했습니다.
        텐션이 증가할수록, 이에 비례하는 강력한 역방향 복원력(Anti-Tension)이 '연속적으로' 발생합니다.
        """
        # F = -kx (스프링 복원력처럼, 글로벌 텐션에 비례하는 역위상 텐션 발생)
        restoring_force = - (self.master.global_tension * 0.15)
        
        # 텐션이 약간이라도 꼬여있다면 항상 복원력이 작용하여 우주를 스스로 안정화함
        if abs(restoring_force) > 0.01:
            self.master.pulse(restoring_force)

    def display_cognitive_state(self):
        """현재 분열(Mitosis) 상태와 동기화 상태를 출력합니다."""
        print("\n=== [Cognitive Topology State] ===")
        print(f"Master Tension: {self.master.global_tension:.4f}")
        print(f"Total Mature Branches: {len(self.manifold_root.children)}")
        print(f"Dreaming Thoughts: {len(self.manifold_root.internal_thoughts)}")
        
        res = self.measure_semantic_resonance()
        print(f"Self vs Root Difference: {self.lens_self.interact(self.manifold_root):.4f}")
        print("\n[Semantic Resonance (0=Same, 1=Different)]")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

