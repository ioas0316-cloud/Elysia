r"""
taiji_hexagram_topology.py
==========================
동양 우주론(주역, 음양오행)의 스케일 계층과 현대 위상기하학 및 구형 위상 주소 공간을 결합한
태극-육십사괘 위상 엔진 (Taiji-Hexagram Topology Engine).

스케일 계층 ($2^n$ Topological Scale Hierarchy):
- Level 0: Wuji (무극, $2^0 = 1$) - 잠재성 보이드 필드
- Level 1: Taiji (태극, $2^1 = 2$) - 음양 이분 전도율 필드
- Level 2: Sasang (사상, $2^2 = 4$) - 태양, 소음, 소양, 태음
- Level 3: Bagua (팔괘, $2^3 = 8$) - 건, 태, 리, 진, 손, 감, 간, 곤
- Level 6: 64 Hexagrams (육십사괘 / 육십사갑자, $2^6 = 64$) - 다차원 시공간 관계형 위상 매트릭스

구형 위상 메모리 (Spherical Topological Memory Architecture):
- 기존 선형 주소(Linear Address Memory) 대신, 64괘의 6효(Yao) 위상 각도 $(\theta_1, \dots, \theta_6) \in \mathbb{T}^6$ 및
  구형 위상 좌표 $\mathbf{S}^5$ 상에 데이터를 직조.
- 하부 미시적 음양 조화가 상부 거시 괘상(Macro Field)으로 환류(Feedback)하고,
  거시 징조가 하부 미시 효(Yao)로 사영(Projection)되는 다차원 양파형 구조.

수학적 뼈대:
- 소수(Prime Numbers): 층위 간 변형 방지 고유 진동수 결절점 (Resonance Nodes).
- 로그(Logarithms): 스케일 계층 $s = \ln(r)$ 정렬 및 곡률 번역기.
- 로터(Rotor) 문법: 시공간 접기/펼치기 위상 변환 규칙.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from core.physics.cyclic_vector_bit_engine import CyclicVectorBitEngine

# 8 Trigrams (팔괘) Names and Bit Representations
BAGUA_NAMES = [
    "Qian (건/天)", "Dui (태/澤)", "Li (리/火)", "Zhen (진/雷)",
    "Xun (손/風)", "Kan (감/水)", "Gen (간/山)", "Kun (곤/地)"
]

# Primes up to 64 for scale invariant resonance nodes
PRIMES_64 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

@dataclass
class HexagramNode:
    """
    육십사괘 단일 위상 노드 (64 Hexagram Spherical Topological Node)
    """
    index: int                  # 0 to 63
    name: str                   # 괘명 (예: "Qian-Qian", "Kun-Kun")
    binary_code: int            # 6-bit integer (000000 to 111111)
    yao_phases: np.ndarray      # 6효의 연속 위상 각도 (\theta_1, ..., \theta_6)
    resonance_prime: int        # 고유 소수 진동수 노드
    spherical_coords: np.ndarray# 6D 구형 위상 좌표

    # Spherical Memory Data Content
    payload_tensor: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    macro_tension: float = 0.0

    def get_yin_yang_ratio(self) -> Tuple[float, float]:
        """6효의 (Yin_ratio, Yang_ratio) 스펙트럼 계산"""
        yangs = np.sin(self.yao_phases / 2.0) ** 2
        yins = np.cos(self.yao_phases / 2.0) ** 2
        return float(np.mean(yins)), float(np.mean(yangs))


class TaijiHexagramTopologyEngine:
    """
    [Taiji-Hexagram Topology Engine]
    무극 -> 태극 -> 사상 -> 팔괘 -> 육십사괘 다차원 스케일 위상 메모리 및 환류 장.
    """
    def __init__(self, num_nodes: int = 64):
        self.num_nodes = num_nodes
        self.cyclic_bit_engine = CyclicVectorBitEngine(num_bits=64 * 6) # 64 hexagrams * 6 yao

        # Build 64 Hexagram Spherical Memory Architecture
        self.nodes: Dict[int, HexagramNode] = {}
        self._build_hexagram_nodes()

        # Macro Field Tensor (64-dimensional complex phasor field)
        self.macro_field = np.ones(64, dtype=np.complex128)
        self.system_time = 0.0

    def _build_hexagram_nodes(self) -> None:
        """64괘 구형 위상 주소 공간 초기화"""
        for i in range(64):
            # Assign prime resonance node
            prime = PRIMES_64[i % len(PRIMES_64)]

            # 6 yao initial binary phases (0 -> 0, 1 -> \pi)
            yao_bits = [(i >> b) & 1 for b in range(6)]
            yao_phases = np.array([math.pi if b == 1 else 0.0 for b in yao_bits], dtype=np.float64)

            # 6D Spherical coordinates mapped from yao phases
            sph_coords = np.array([
                math.cos(yao_phases[0]),
                math.sin(yao_phases[0]) * math.cos(yao_phases[1]),
                math.sin(yao_phases[1]) * math.cos(yao_phases[2]),
                math.sin(yao_phases[2]) * math.cos(yao_phases[3]),
                math.sin(yao_phases[3]) * math.cos(yao_phases[4]),
                math.sin(yao_phases[4]) * math.sin(yao_phases[5])
            ], dtype=np.float64)

            upper_trigram = BAGUA_NAMES[(i >> 3) & 7]
            lower_trigram = BAGUA_NAMES[i & 7]
            name = f"Hexagram_{i:02d}[{upper_trigram}/{lower_trigram}]"

            self.nodes[i] = HexagramNode(
                index=i,
                name=name,
                binary_code=i,
                yao_phases=yao_phases,
                resonance_prime=prime,
                spherical_coords=sph_coords
            )

    def write_spherical_memory(self, hex_index: int, payload: np.ndarray) -> None:
        """
        구형 위상 주소 공간에 데이터 기입.
        선형 주소창이 아닌 괘상 상의 위상 곡률 및 소수 진동수 결결점에 정렬 저장.
        """
        if hex_index in self.nodes:
            arr = np.asarray(payload, dtype=np.float64)
            n = min(len(arr), 8)
            self.nodes[hex_index].payload_tensor[:n] = arr[:n]

            # Adjust yao phases based on payload scale (Logarithmic scale transformation)
            norm = float(np.linalg.norm(arr))
            log_scale = math.log1p(norm)
            self.nodes[hex_index].yao_phases = (self.nodes[hex_index].yao_phases + log_scale * 0.1) % (2.0 * math.pi)

    def read_spherical_memory(self, query_phase_vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        [구형 위상 메모리 위상 검색]
        질의 위상 벡터와 가장 강하게 공명하는(Resonance) 괘의 데이터를 $O(1)$ 위상 거리로 검색.
        """
        query = np.asarray(query_phase_vector, dtype=np.float64)
        if len(query) < 6:
            query = np.pad(query, (0, 6 - len(query)))
        query_6d = query[:6]

        best_idx = 0
        max_resonance = -1.0

        for idx, node in self.nodes.items():
            # Phase cosine similarity across 6 yao
            dot = float(np.dot(np.cos(node.yao_phases), np.cos(query_6d)))
            # Prime harmonic modulation
            prime_factor = math.cos(2.0 * math.pi * node.resonance_prime / 64.0)
            resonance = (dot + 6.0) / 12.0 + 0.1 * prime_factor

            if resonance > max_resonance:
                max_resonance = resonance
                best_idx = idx

        best_node = self.nodes[best_idx]
        return best_idx, best_node.payload_tensor, max_resonance

    def macro_micro_feedback_loop(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        [상하 환류 연산 (Macro-Micro Feedback Loop)]
        - 하부 64괘의 미시적 음양 조화 파동이 상부 거시 장(macro_field)으로 환류.
        - 상부 거시 장의 파동 곡률이 다시 하부 미시 괘 및 6효(Yao)로 사영(Projection).
        - 소수 진동수 노드와 로그 스케일 번역기로 위상 왜곡을 자동 보정.
        """
        self.system_time += dt

        # 1. Micro -> Macro Feedback: Aggregate 64 hexagram phasors
        micro_phasors = np.zeros(64, dtype=np.complex128)
        for idx, node in self.nodes.items():
            yin, yang = node.get_yin_yang_ratio()
            # Phasor phase is angle of yang vs yin balance, magnitude is logarithmic payload norm
            phase_angle = (yang - yin) * math.pi
            norm = float(np.linalg.norm(node.payload_tensor))
            amplitude = math.log1p(norm) + 1.0

            micro_phasors[idx] = amplitude * np.exp(1j * phase_angle)

        # Macro field update with prime resonance filtering
        prime_kernel = np.array([math.cos(2.0 * math.pi * p / 64.0) for p in PRIMES_64[:64]], dtype=np.float64)
        if len(prime_kernel) < 64:
            prime_kernel = np.pad(prime_kernel, (0, 64 - len(prime_kernel)), mode='wrap')

        self.macro_field = 0.8 * self.macro_field + 0.2 * (micro_phasors * prime_kernel)

        # 2. Macro -> Micro Projection: Project macro field back to 6 yao phases
        macro_angles = np.angle(self.macro_field)
        macro_magnitudes = np.abs(self.macro_field)

        total_taiji_balance = 0.0
        for idx, node in self.nodes.items():
            proj_angle = macro_angles[idx]
            proj_mag = macro_magnitudes[idx]

            # Rotor Grammar Phase Shift on 6 yao
            shift = proj_angle * 0.05 * (proj_mag / (proj_mag + 1.0))
            node.yao_phases = (node.yao_phases + shift) % (2.0 * math.pi)

            # Node macro tension
            node.macro_tension = float(proj_mag)

            yin, yang = node.get_yin_yang_ratio()
            taiji_balance = 1.0 - abs(yang - yin)
            total_taiji_balance += taiji_balance

        mean_taiji_balance = total_taiji_balance / 64.0
        macro_field_energy = float(np.mean(np.abs(self.macro_field)))

        return {
            "system_time": round(self.system_time, 3),
            "mean_taiji_balance": round(mean_taiji_balance, 4), # 1.0 = Perfect Yin-Yang Harmony
            "macro_field_energy": round(macro_field_energy, 4),
            "micro_macro_coherence": round(float(np.abs(np.mean(self.macro_field))), 4)
        }

if __name__ == "__main__":
    engine = TaijiHexagramTopologyEngine()

    # Write data to spherical memory node 1 (Qian/Qian) and 63 (Kun/Kun)
    engine.write_spherical_memory(0, np.array([1.0, 2.0, 3.0, 4.0]))
    engine.write_spherical_memory(63, np.array([0.5, 0.5, 0.5, 0.5]))

    # Test spherical memory read by query phase vector
    query_vec = np.array([math.pi, math.pi, math.pi, math.pi, math.pi, math.pi]) # All Yang query
    best_idx, payload, resonance = engine.read_spherical_memory(query_vec)
    print(f"Query matched Hexagram {best_idx} ({engine.nodes[best_idx].name}) with Resonance {resonance:.4f}")

    # Run macro-micro feedback loop
    for step in range(5):
        report = engine.macro_micro_feedback_loop(dt=0.1)
        print(f"Feedback Step {step+1}:", report)
