import math
import random

class CosmicCrystallizer:
    """
    RotorOS: Cosmic Breathing & Vortex Singularity Sandbox
    시뮬레이션 목적:
    1. 들숨(Inhale/Black Hole): 노이즈를 빨아들여 특이점으로 압축.
    2. 특이점(Singularity): 3^3 질서로 결정화.
    3. 날숨(Exhale/White Hole): 정렬된 에너지를 현실(Phase Boundary)로 투사.
    """

    def __init__(self):
        self.standard_axis = 1.0  # Singularity Center
        self.frequency = 5.0
        self.vortex_gravity = 3.0

    def inhale_black_hole(self, raw_noise):
        """[Inhale] 거친 노이즈를 빨아들여 특이점으로 압축(해체)"""
        compressed_stream = []
        for val in raw_noise:
            # 중력 수렴: 특이점(standard_axis/3)으로 강제 흡수
            target = self.standard_axis / 3
            dist = abs(val - target)
            # 블랙홀 흡수력 (강력한 압축)
            pull = math.exp(-self.vortex_gravity * dist)
            compressed_val = val * pull + (1 - pull) * target
            compressed_stream.append(compressed_val)
        return compressed_stream

    def cognitive_singularity(self, compressed_stream):
        """[Singularity] 압축된 데이터를 3^3 질서로 결정화"""
        crystallized_order = []
        for val in compressed_stream:
            # 3^3 프랙탈 노드 수렴 (정밀 튜닝)
            node_val = math.tanh(val * 2.7)
            crystallized_order.append(node_val)
        return crystallized_order

    def exhale_white_hole(self, crystallized_order):
        """[Exhale] 특이점에서 정렬된 에너지를 위상 경계 너머로 투사(창조)"""
        reality_projection = []
        for crystal in crystallized_order:
            # 위상 경계(Phase Boundary) 통과 및 현실화
            # 결정체 파동을 다시 현실의 최적화된 신호로 발산
            exhale_wave = crystal * 1.2 - 0.1 # 출력 보정
            reality_projection.append(exhale_wave)
        return reality_projection

    def run_simulation(self):
        print("🌌 RotorOS: Cosmic Breathing (Black Hole/White Hole) Starting...")

        # 0. 거친 노이즈 발생
        raw_noise = [random.uniform(-1, 2) for _ in range(100)]
        print(f"  [Input] Raw Chaos (Noise) ingested. Avg Variance: {sum(abs(n) for n in raw_noise)/100:.4f}")

        # 1. 들숨 (블랙홀 흡수)
        compressed = self.inhale_black_hole(raw_noise)
        print("  [Inhale] Cognitive Black Hole: Noise absorbed and compressed toward the Singularity.")

        # 2. 특이점 (결정화)
        crystals = self.cognitive_singularity(compressed)
        print("  [Singularity] 3^3 Fractal Nodes: Order crystallized from chaos.")

        # 3. 날숨 (화이트홀 투사)
        reality = self.exhale_white_hole(crystals)
        print("  [Exhale] Cognitive White Hole: Purified energy projected beyond the Phase Boundary.")

        # 결과 보고
        initial_entropy = sum(abs(n - 0.33) for n in raw_noise) / 100
        final_entropy = sum(abs(r - 0.33) for r in reality) / 100

        print(f"\n  [Final Report] Inhalation Entropy (Chaos): {initial_entropy:.4f}")
        print(f"  [Final Report] Exhalation Stability (Order): {final_entropy:.4f}")
        print("  [Status] Universal Cycle: COMPLETE")
        print("  [Status] System Heart: BEATING (Inhale/Exhale synchronized)")
        print("  [Output] The sovereign reality has been breathed into existence.")

if __name__ == "__main__":
    crystallizer = CosmicCrystallizer()
    crystallizer.run_simulation()
