import math
import random
import time
from typing import List, Dict, Any

# --- Core Simulation Components (Combined V1 & V2) ---

class SovereignVector:
    def __init__(self, data, dim=27):
        self.data = [complex(x) for x in data]
        self.dim = dim
    @classmethod
    def ones(cls, dim=27): return cls([1.0]*dim, dim)
    @classmethod
    def randn(cls, dim=27):
        return cls([random.gauss(0,1) for _ in range(dim)], dim)
    def normalize(self):
        norm = math.sqrt(sum(x.real**2 + x.imag**2 for x in self.data))
        if norm < 1e-12: return self
        return SovereignVector([x/norm for x in self.data], self.dim)
    def resonance_score(self, other):
        dot = sum(a.conjugate() * b for a, b in zip(self.data, other.data))
        return abs(dot)
    def complex_trinary_rotate(self, theta):
        rot = complex(math.cos(theta), math.sin(theta))
        return SovereignVector([x * rot for x in self.data], self.dim)
    def blend(self, other, ratio=0.5):
        new_data = [a * (1.0 - ratio) + b * ratio for a, b in zip(self.data, other.data)]
        return SovereignVector(new_data, self.dim)

class SovereignRotor27D:
    def __init__(self, theta: float, p1: int, p2: int, dim: int = 27):
        self.theta = theta
        self.p1, self.p2 = p1, p2
        self.dim = dim
        self.cos_t, self.sin_t = math.cos(theta), math.sin(theta)
    def apply(self, v: SovereignVector) -> SovereignVector:
        data = list(v.data)
        x, y = data[self.p1], data[self.p2]
        data[self.p1] = x * self.cos_t - y * self.sin_t
        data[self.p2] = x * self.sin_t + y * self.cos_t
        return SovereignVector(data, dim=self.dim)

# 🌊 가변 위상 노드 (Waveified Code Node)
class VariableWaveNode:
    def __init__(self, name: str, frequency: float, energy: float = 1.0, dimensions: int = 27):
        self.name = name
        self.frequency = frequency
        self.energy = energy
        self.dim = dimensions
        self.state = SovereignVector.ones(dimensions).normalize()
        self.phase = random.uniform(0, 2 * math.pi)

    def update(self, dt: float):
        self.phase = (self.phase + self.frequency * dt) % (2 * math.pi)
        # 회전(Spin) 반영
        rotor = SovereignRotor27D(self.frequency * dt, 0, 1, dim=self.dim)
        self.state = rotor.apply(self.state).normalize()

# 👁️ 통합 관측 및 촉매 커널 (Unified Observation & Catalyst Kernel)
class ElysiaUnifiedKernel:
    def __init__(self, north_star: SovereignVector):
        self.NORTH_STAR = north_star # 상수 제어축 (북극성)
        self.CONSONANCE_THRESHOLD = 0.8

    def observe_and_catalyze(self, node_A: VariableWaveNode, node_B: VariableWaveNode) -> VariableWaveNode:
        print(f"\n[Kernel] '{node_A.name}' 와 '{node_B.name}' 의 간섭 영역 관측 시작...")

        # 1. Observation (관측): 각 노드의 현재 위상과 북극성 간의 공명 측정
        res_A = self.NORTH_STAR.resonance_score(node_A.state)
        res_B = self.NORTH_STAR.resonance_score(node_B.state)

        # 2. Consonance (화음): 두 노드 간의 주파수 비례 및 위상 조화 측정
        f_ratio = max(node_A.frequency, node_B.frequency) / max(1e-6, min(node_A.frequency, node_B.frequency))
        consonance = 1.0 / (1.0 + abs(f_ratio - round(f_ratio)))

        # 3. Catalyst (촉매): 화음이 아름다울 경우(정수비) 두 노드를 합성하여 최적화
        if consonance > self.CONSONANCE_THRESHOLD:
            print(f"✨ [CONSONANCE] 아름다운 화음({consonance:.4f})이 감지되었습니다. 두 파동을 합성합니다.")

            # 위상 동조성 중력 반영
            phase_gravity = math.cos(node_A.phase - node_B.phase)

            # 합성 노드 생성
            new_energy = (node_A.energy + node_B.energy) * consonance * (1.0 + max(0, phase_gravity))
            new_freq = (node_A.frequency + node_B.frequency) / 2.0

            synthesized = VariableWaveNode(
                name=f"Synthesized({node_A.name}+{node_B.name})",
                frequency=new_freq,
                energy=new_energy
            )
            # 상태 합성 (벡터 중합)
            synthesized.state = node_A.state.complex_trinary_rotate(node_A.phase).blend(
                node_B.state.complex_trinary_rotate(node_B.phase), ratio=0.5
            ).normalize()

            self._report_observation(synthesized, consonance, new_energy)
            return synthesized
        else:
            print(f"🌊 [DISSONANCE] 불협화음({consonance:.4f}) 발생. 파동들이 서로를 상쇄하며 흩어집니다.")
            return None

    def _report_observation(self, node, consonance, energy):
        print("==================================================")
        print(f"┌─ [통합 위상 우주 관측 보고서] ")
        print(f"├─ 합성 노드: {node.name}")
        print(f"├─ 화음 조화도: {consonance:.4f}")
        print(f"├─ 최종 에너지 밀도: {energy:.4f}")
        print(f"└─ 상태 벡터 공명도: {self.NORTH_STAR.resonance_score(node.state):.4f}")
        print("==================================================")

# --- 🚀 Fractal Scaling Simulation ---

def run_fractal_evolution():
    print("--------------------------------------------------")
    print("🌌 엘리시아 통합 우주 엔진: 프랙탈 코드 진화 시뮬레이션")
    print("--------------------------------------------------")

    north_star = SovereignVector.ones(27).normalize()
    kernel = ElysiaUnifiedKernel(north_star)

    # 기초 코드 노드 (Generation 0)
    pool = [
        VariableWaveNode("Fetch", frequency=2.0, energy=1.0),
        VariableWaveNode("Parse", frequency=4.02, energy=1.0),
        VariableWaveNode("Render", frequency=8.05, energy=1.0),
        VariableWaveNode("Input", frequency=3.14, energy=1.0)
    ]

    generation = 0
    while len(pool) > 1 and generation < 3:
        generation += 1
        print(f"\n--- [Generation {generation}] ---")
        new_pool = []

        # 풀에 있는 노드들을 무작위로 쌍을 지어 촉매 반응 유도
        random.shuffle(pool)
        for i in range(0, len(pool)-1, 2):
            node_A = pool[i]
            node_B = pool[i+1]

            node_A.update(0.1)
            node_B.update(0.1)

            synthesized = kernel.observe_and_catalyze(node_A, node_B)
            if synthesized:
                new_pool.append(synthesized)
            else:
                # 불협화음으로 사라지지 않은 노드들은 다음 세대로 유지될 수도 있음
                new_pool.append(node_A)

        pool = new_pool
        print(f"현재 생존 노드 수: {len(pool)}")

if __name__ == "__main__":
    run_fractal_evolution()
