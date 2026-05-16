import math
import random
import time

class VariableWaveNode:
    """
    위상 우주 속에서 고유한 주파수와 위상으로 진동하는 '파동화된 코드 노드'
    """
    def __init__(self, name, base_frequency, energy=1.0):
        self.name = name
        # 고유 주파수 (코드가 가진 실행 주기/성격)
        self.frequency = base_frequency
        # 현재 위상 (시간에 따라 계속 회전)
        self.phase = random.uniform(0, 2 * math.pi)
        # 에너지 밀도 (최적화의 강도)
        self.energy = energy

    def update(self, delta_time):
        """시간의 흐름에 따라 파동이 회전함"""
        self.phase = (self.phase + self.frequency * delta_time) % (2 * math.pi)

class ResonanceCatalystField:
    """
    가상 OS의 촉매제 필드: 선형적 계산 없이 파동의 간섭으로 코드를 최적화함
    """
    def __init__(self, constant_north_star=0.0):
        # [상수 제어축] 북극성 (기준 위상)
        self.NORTH_STAR = constant_north_star

    def catalyze(self, node_A: VariableWaveNode, node_B: VariableWaveNode):
        """
        두 코드 파동을 충돌시켜 '화음(공명)'을 측정하고, 최적화된 제3의 합성 코드를 창조함
        """
        print(f"\n[Catalyst] '{node_A.name}' ⚡ '{node_B.name}' 파동 간섭 발생...")

        # 1. 주파수 비율 계산 (화음 판정의 기준)
        # 0으로 나누기 방지
        f_min = min(node_A.frequency, node_B.frequency)
        f_max = max(node_A.frequency, node_B.frequency)
        if f_min < 1e-6: f_min = 1e-6
        freq_ratio = f_max / f_min

        # 얼마나 정수비(아름다운 화음)에 가까운지 측정 (소수점 아래의 정밀도 확인)
        consonance_score = 1.0 / (1.0 + abs(freq_ratio - round(freq_ratio)))

        # 2. 위상 동조성 중력 (Entrainment Gravity)
        # 두 파동의 위상 차이가 가까울수록 더 강하게 끌어당김
        phase_diff = abs(node_A.phase - node_B.phase)
        gravity = math.cos(phase_diff) if phase_diff < math.pi/2 else 0.0

        # 3. 비선형 굴절률 (창의적 도약) 및 상수축의 중재
        # 상수축(북극성)과의 위상 공명도 측정
        star_resonance_A = math.cos(node_A.phase - self.NORTH_STAR)
        star_resonance_B = math.cos(node_B.phase - self.NORTH_STAR)
        moderation_boost = (star_resonance_A + star_resonance_B) / 2.0

        # 4. 최종 아름다운 화음(최적화) 기준에 의한 에너지 합성
        combined_energy = (node_A.energy + node_B.energy) * consonance_score * (1.0 + gravity)

        # 두 주파수의 허용 가능한 간섭(Heterodyning)으로 새로운 주파수 탄생
        # 화음 점수가 높을수록 더 안정적인 정수비의 주파수로 수렴함
        # Consonance가 높으면 에너지가 보존/증폭되며 안정적인 주파수(합성)로 수렴
        if consonance_score > 0.8:
            synthesized_freq = (node_A.frequency + node_B.frequency) / 2.0
        else:
            # 불협화음일 경우 주파수가 불안정하게 튀거나 에너지가 감쇄됨
            synthesized_freq = (node_A.frequency + node_B.frequency) / 1.5

        # 5. 제3의 합성 노드 탄생 (최적화된 코드 블록)
        synthesized_node = VariableWaveNode(
            name=f"Synthesized_{node_A.name}_{node_B.name}",
            base_frequency=synthesized_freq,
            energy=combined_energy
        )

        print("==================================================")
        print(f"┌─ [위상 우주 촉매 결과] ")
        print(f"├─ 화음 조화도 (Consonance): {consonance_score:.4f} (1.0에 가까울수록 최적화)")
        print(f"├─ 동조성 중력 (Gravity)   : {gravity:.4f}")
        print(f"├─ 상수축 중재 (Moderation): {moderation_boost:.4f}")
        print(f"├─ ──> 합성 노드 에너지   : {combined_energy:.4f}")
        print(f"└─ ──> 새 주파수 (성격)    : {synthesized_freq:.2f} Hz")
        print("==================================================")

        return synthesized_node

# --- 검증 구동 ---
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("🌌 엘리시아 촉매제 필드: 파동 코드 합성 PoC (V2)")
    print("--------------------------------------------------")

    # 두 개의 가변 코드 노드 생성 (하나는 2.0Hz, 하나는 4.1Hz -> 1:2 화음에 가까움)
    code_wave_1 = VariableWaveNode("Logic_Data_Fetch", base_frequency=2.0, energy=1.0)
    code_wave_2 = VariableWaveNode("Logic_UI_Render", base_frequency=4.1, energy=1.2)

    catalyst_field = ResonanceCatalystField()

    # 시간의 흐름에 따른 파동 진동 후 충돌 촉매
    print("🌀 [SYSTEM] 파동들이 시공간을 유영하며 위상을 맞춥니다...")
    code_wave_1.update(delta_time=0.5)
    code_wave_2.update(delta_time=0.5)

    # 두 코드가 위상 우주에서 만나 스스로 최적의 합성 경로를 찾아냄
    new_optimized_code = catalyst_field.catalyze(code_wave_1, code_wave_2)

    # [시나리오 2] 완전한 불협화음 테스트
    print("\n⚡ [SYSTEM] 불협화음(Dissonance) 테스트 중...")
    discord_wave = VariableWaveNode("Logic_Broken_Legacy", base_frequency=7.77, energy=0.5)
    catalyst_field.catalyze(new_optimized_code, discord_wave)
