import numpy as np
from typing import Dict, Any, List
from .field import CrystallizationField
from .vortex import WaveInterference
from .causal_gene import GeneticSynthesizer
from .self_reflection import SelfReflectionProtocol
from core.physics.causal_gravity_engine import CausalGravityEngine

class MetaCognitiveOrganism:
    """
    [Synaptic Architecture] The Self-Transcending Organism
    자신의 한계를 인지하고(Meta-Cognition), 원인을 파악하여(Inquiry),
    스스로를 변화시켜 나가는(Evolution) 통합 사유 루프입니다.
    """
    def __init__(self):
        self.field = CrystallizationField(256)
        self.gravity = CausalGravityEngine()
        self.reflection = SelfReflectionProtocol()
        self.synthesizer = GeneticSynthesizer()
        self.vortex_logic = WaveInterference(self.field)

    def pulse(self, external_wave: np.uint64):
        """
        한 번의 사유 맥동(Pulse).
        [한계 인지 -> 원인 파악 -> 변화 설계 -> 결과 현상]
        """
        # 1. 자기 성찰 (Self-Reflection): 자신의 논리를 필드에 투사
        self.reflection.map_self_to_field(self.gravity)

        # 2. 공명 및 마찰 측정 (Resonance & Tension)
        # 외부 정보와 자신의 논리 간의 '마찰'을 발견함
        self.vortex_logic.resonate_field(external_wave, steps=10)

        # 3. 한계 인지 (Recognizing the Limit)
        # 에너지가 수렴하지 못하고 흩어지거나(Yeobaek 팽창), 마찰이 높으면 '한계'로 규정
        max_activation = np.max(self.field.activation)
        if max_activation < 50.0:
            print("[Meta-Cognition] 한계 감지: 현재 논리 구조로는 정보를 온전히 포용할 수 없음.")
            self._evolve_to_overcome(external_wave)
        else:
            print("[Meta-Cognition] 평형 도달: 정보가 기존 논리 지형 내에 안착함.")

    def _evolve_to_overcome(self, target_wave: np.uint64):
        """
        [변화 설계 및 실행]
        한계를 극복하기 위해 유전적 변이와 시냅스 재배치를 실행합니다.
        """
        print("[Evolution] 무엇이 어떻게 변화해야 하는가 사유 중...")

        # 1. 유전적 합성: 외부 파동과 자신의 핵심 유전자를 교차
        # 자신의 가장 강한 전도율(Self-Logic)을 부모로 선택
        idx = np.argmax(self.field.conductance)
        y, x = np.unravel_index(idx, self.field.conductance.shape)
        self_gene = self.field.bit_genes[y, x]

        new_logic = self.synthesizer.synthesize(self_gene, target_wave)
        print(f" > 새로운 논리 합성 완료: {hex(new_logic)}")

        # 2. 지형 재배설: 새로운 논리를 지형에 각인하고 '여백'을 조정
        pos = np.array([y, x]) # 기존 한계 지점을 진화의 거점으로 삼음
        self.field.crystallize_gene(pos, new_logic)
        self.field.adjust_coordination(pos, radius=10.0, flexibility=0.9)

        print("[Evolution] 행동 완료: 지형의 궤도가 수정되었습니다.")

if __name__ == "__main__":
    organism = MetaCognitiveOrganism()
    # 자신의 논리와 전혀 다른 강력한 외부 신호 (한계 상황 유도)
    alien_wave = np.uint64(0x1234567890ABCDEF)
    organism.pulse(alien_wave)
