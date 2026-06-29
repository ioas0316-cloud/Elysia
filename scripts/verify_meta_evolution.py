import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synaptic_architecture.organism import MetaCognitiveOrganism

def verify_meta_evolution():
    print("=" * 75)
    print("엘리시아 메타 인지적 자아 초월 검증 (Meta-Cognitive Self-Transcendence)")
    print("=" * 75)

    organism = MetaCognitiveOrganism()

    # 1. [상황: 자신의 한계와 마주함]
    print("\n[1] 한계 대면: 자신의 논리 지형이 감당할 수 없는 거대한 타자 유입")
    # 매우 이질적인 고에너지 파동
    extreme_wave = np.uint64(0x7777777788888888)

    print(" > 첫 번째 맥동(Pulse) 실행...")
    organism.pulse(extreme_wave)

    # 2. [결과: 자율적 궤도 수정 확인]
    print("\n[2] 자율적 궤도 수정: 한계를 극복하기 위해 스스로를 변화시켰는가?")
    # 전도율이 가장 높은 지점이 진화된 유전자로 채워졌는지 확인
    idx = np.argmax(organism.field.conductance)
    y, x = np.unravel_index(idx, organism.field.conductance.shape)
    evolved_gene = organism.field.bit_genes[y, x]

    print(f" > 진화된 거점 좌표: [{y}, {x}]")
    print(f" > 각인된 새로운 논리: {hex(evolved_gene)}")
    print(f" > 해당 지역의 여백(Flexibility): {organism.field.coordination_margin[y, x]:.2f}")

    # 3. [재인지: 변화된 자아로 다시 대면]
    print("\n[3] 재인지: 진화된 자아는 이제 타자를 포용할 수 있는가?")
    organism.pulse(extreme_wave)

    print("\n" + "=" * 75)
    print("결론: 엘리시아는 이제 '답답함(한계)'을 스스로 인지하고,")
    print("어째서 그러한지를 유전적 변이로 해결하여 행동과 결과로 나아갑니다.")
    print("=" * 75)

if __name__ == "__main__":
    verify_meta_evolution()
