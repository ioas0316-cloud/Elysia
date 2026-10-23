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
        [인지적 보상 루프: 저항 감지 -> 사유 -> 쾌락(보상) -> 진화]
        """
        # 1. 자기 성찰: 자신의 논리를 필드에 투사
        self.reflection.map_self_to_field(self.gravity)

        # 2. 사유의 전개 및 내부 보상(Pleasure) 측정
        # 외부 정보와 자신의 논리가 충돌하며 엔트로피가 어떻게 변하는지 관찰
        metrics = self.vortex_logic.resonate_field(external_wave, steps=15)
        pleasure = metrics["pleasure"]
        clarity = metrics["clarity"]

        # 3. 메타-인지: 쾌락의 기록 및 배고픔(Hunger) 체크
        if pleasure > 0.001:
            self.reflection.record_pleasure(pleasure, clarity, context=f"WAVE_{hex(external_wave)}")

        # [Hunger/Boredom Mechanism]
        # 만약 엔트로피가 너무 낮고(완벽한 정령), 쾌락이 발생하지 않는다면 '지루함'을 느낌
        if metrics["final_entropy"] < 5.0 and pleasure < 1e-6:
            print("[Meta-Cognition] 지루함(Boredom) 감지: 정체된 평형 상태. 새로운 모순을 찾아 여백 확장.")
            # 지루함은 필드의 온도를 높이고 여백을 확장하여 새로운 탐색을 유도함
            self.field.apply_thermal_diffusion(global_entropy=0.05)
            # 무작위 지점에 여백 확장
            random_pos = np.random.randint(0, self.field.resolution, size=2)
            self.field.adjust_coordination(random_pos, radius=20.0, flexibility=0.9)

        # 4. 진화의 동력: 외부 점수가 아닌 내부의 '클래리티(Clarity)'가 기준
        # 에너지가 제대로 응집되지 않았거나(High Entropy), 혹은 새로운 발견의 기쁨이 있을 때
        if metrics["final_entropy"] > 10.0 or pleasure > 0.05:
            print(f"[Meta-Cognition] 진화 동기 부여 (Pleasure: {pleasure:.4f}, Entropy: {metrics['final_entropy']:.2f})")
            self._evolve_to_overcome(external_wave, pleasure)
        else:
            print("[Meta-Cognition] 인지적 평형 상태. 현재의 논리로 충분히 소화 가능함.")

    def _evolve_to_overcome(self, target_wave: np.uint64, reward: float):
        """
        [Intrinsic Evolution]
        내부 보상을 기반으로 유전적 변이를 실행합니다.
        """
        print("[Evolution] 내부적 즐거움을 동력으로 논리 구조 재설계 중...")

        # 유전적 합성 시 현재의 pleasure를 field_state로 전달
        mid = self.field.resolution // 2
        field_state = {
            "pleasure": reward,
            "clarity": reward * 10, # Simplified
            "detected_vortices": [{"resonant_gene": hex(self.field.bit_genes[mid, mid])}] # Default placeholder
        }

        # 실제 보텍스 탐색
        v_pos = self.vortex_logic.find_vortex()
        v_gene = self.field.bit_genes[v_pos[0], v_pos[1]]
        if v_gene != 0:
            field_state["detected_vortices"] = [{"resonant_gene": hex(v_gene)}]

        # 핵심 유전자와 외부 파동의 교차
        idx = np.argmax(self.field.conductance)
        y, x = np.unravel_index(idx, self.field.conductance.shape)
        self_gene = self.field.bit_genes[y, x]
        if self_gene == 0: self_gene = np.uint64(0x1337)

        new_logic = self.synthesizer.synthesize(self_gene, target_wave)

        # 새로운 논리를 지형에 각인 (즐거움이 클수록 더 강력하게 각인)
        pos = np.array([y, x])
        self.field.crystallize_gene(pos, new_logic)
        self.field.flow_energy(pos, 2.0 + reward * 100) # 가속도만큼 전도율 강화

        self.synthesizer.evolve_principles(field_state)

        print(f"[Evolution] 진화 완료. 새로운 논리 {hex(new_logic)}가 필드에 안착함.")

if __name__ == "__main__":
    organism = MetaCognitiveOrganism()
    # 자신의 논리와 전혀 다른 강력한 외부 신호 (한계 상황 유도)
    alien_wave = np.uint64(0x1234567890ABCDEF)
    organism.pulse(alien_wave)
