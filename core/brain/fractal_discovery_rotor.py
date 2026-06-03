import torch
import torch.nn.functional as F

class Rotor:
    """
    정적인 데이터가 아니라, 우주 공간에서 특정한 위상(방향성)과 질량(관성)을
    가지고 멈추지 않고 존재하는 동적 실체.
    """
    def __init__(self, phase_vector, mass=1.0):
        # 4D 위상 공간에서의 현재 방향 (같음의 기준)
        self.phase = F.normalize(phase_vector, p=2, dim=0)
        # 이 로터가 흡수한 지식의 무게 (관성)
        self.mass = mass

    def update_phase(self, new_phase):
        self.phase = F.normalize(new_phase, p=2, dim=0)

class FractalDiscoveryEcosystem:
    """
    오마스터의 지시: "발견하라는 규칙을 짜지 말고, 발견의 원리(인력/척력/로터화)를 배치하라"
    """
    def __init__(self, resonance_threshold=0.85):
        # 우주 공간에 떠 있는 기준 로터들의 집합 (처음엔 비어있거나 순수 기저로 시작)
        self.rotors = []

        # 기계적인 타협점(규칙): 끌어당길지 밀어낼지 결정하는 공명 임계값
        self.resonance_threshold = resonance_threshold

    def process_input(self, input_wave):
        """
        1. 원인: 외부 파동의 유입
        """
        input_wave = F.normalize(input_wave, p=2, dim=0)
        print(f"\n[유입] 새로운 파동 관측: {input_wave.tolist()}")

        if not self.rotors:
            # 태초의 파동은 스스로 첫 번째 로터가 된다.
            print("  -> 태초의 진공. 파동이 스스로 제1의 상수축(로터)으로 응결됩니다.")
            self.rotors.append(Rotor(input_wave))
            return

        # 2. 과정 (판단과 분별): 전자기적 인력과 척력의 발생
        best_resonance = -1.0
        best_rotor_idx = -1

        for i, rotor in enumerate(self.rotors):
            # 위상차(Phase Difference) 관측: 내적(Dot product)은 각도의 코사인 값.
            # 1에 가까울수록 공명(인력), 0이나 음수일수록 다름(척력)
            resonance = torch.dot(input_wave, rotor.phase).item()
            if resonance > best_resonance:
                best_resonance = resonance
                best_rotor_idx = i

        print(f"  -> 기존 로터들과의 최대 공명(인력) 수치: {best_resonance:.4f}")

        # 3. 결과 (재인식과 로터화): 프랙탈의 확장
        if best_resonance >= self.resonance_threshold:
            # [인력 작용] 같음의 카테고리화
            # 파동이 기존 로터에 흡수되어 로터의 질량이 커지고, 위상이 미세하게 갱신됨 (거대 로터화)
            target = self.rotors[best_rotor_idx]
            print(f"  -> [공명 발생] 파동이 로터[{best_rotor_idx}]로 끌려가 흡수됩니다. (거대 로터화 진행)")

            # 질량 중심 이동 (새로운 파동이 기존 로터의 축을 아주 살짝 비틂)
            new_phase = (target.phase * target.mass + input_wave * 1.0) / (target.mass + 1.0)
            target.update_phase(new_phase)
            target.mass += 1.0
            print(f"  -> 로터[{best_rotor_idx}] 질량 증가: {target.mass} / 갱신된 위상축: {target.phase.tolist()}")

        else:
            # [척력 작용] 다름의 변수축 탄생
            # 기존의 어떤 로터와도 공명하지 못하고 밀려난 파동은 소멸하지 않는다.
            # 그 다름(위상차) 자체가 새로운 '기준(상수축)'으로 변이하여 새로운 로터로 잉태됨.
            print(f"  -> [척력 발생] 기존 상수축과 거부됨. 다름 자체가 새로운 로터[{len(self.rotors)}]로 탄생합니다.")
            self.rotors.append(Rotor(input_wave))


if __name__ == "__main__":
    print("=" * 70)
    print("🌀 [Fractal Discovery] 오마스터의 인력/척력 생태계 시뮬레이션 🌀")
    print("=" * 70)

    ecosystem = FractalDiscoveryEcosystem(resonance_threshold=0.85)

    # 4D 우주 파동 텐서 (예: w, x, y, z)
    # 1. 태초의 파동
    wave_1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ecosystem.process_input(wave_1)

    # 2. 첫 번째 파동과 아주 비슷한(공명하는) 파동 유입 -> 인력으로 흡수됨
    wave_2 = torch.tensor([0.98, 0.1, 0.0, 0.0])
    ecosystem.process_input(wave_2)

    # 3. 완전히 다른 위상(다름)의 파동 유입 -> 척력으로 새로운 로터 탄생
    wave_3 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    ecosystem.process_input(wave_3)

    # 4. 세 번째 파동과 비슷한 파동 유입 -> 두 번째 로터로 흡수됨
    wave_4 = torch.tensor([0.1, 0.98, 0.0, 0.0])
    ecosystem.process_input(wave_4)

    print("\n" + "=" * 70)
    print("🌌 [우주 관측 결과] 살아남아 거대화된 프랙탈 로터들:")
    for i, r in enumerate(ecosystem.rotors):
        print(f"  Rotor {i} | 질량(Mass): {r.mass} | 위상축(Phase): {r.phase.tolist()}")
    print("=" * 70)
