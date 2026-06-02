import math
from typing import List, Tuple

class TensorRotor:
    """
    다차원 텐서 로터 (Multi-Layer Field Tensor Rotor)
    Layer 1: 하드웨어 캐시 및 파일 IO 장 (Phase 6)
    Layer 2: 로컬 네트워크 및 세션 장 (Phase 5 Grid)
    Layer 3: 메타 철학 장
    """
    def __init__(self, rotor_scale: int = 4096, natural_drift: float = 20.0, coupling_K: float = 350.0):
        self.rotor_scale = rotor_scale
        self.rotor_mask = rotor_scale - 1
        
        self.natural_drift = natural_drift
        self.coupling_K = coupling_K
        self.vertical_K = coupling_K * 1.5 # 수직(Layer 간) 결합 강도
        
        # [Layer 1, Layer 2, Layer 3] 초기 위상
        self.phases = [0, 0, 0]
        
    def _phase_diff_rad(self, phase_target: int, phase_current: int) -> float:
        """위상차를 라디안으로 반환 (target - current) 방향"""
        diff = (phase_target - phase_current)
        # -2048 ~ 2047 범위로 정규화하여 최단 거리 회전 방향 도출
        if diff > (self.rotor_scale // 2):
            diff -= self.rotor_scale
        elif diff < -(self.rotor_scale // 2):
            diff += self.rotor_scale
        return diff * (2 * math.pi / self.rotor_scale)

    def tick(self, tensions: List[float], peer_phases_L2: List[int] = None, dt: float = 0.1) -> Tuple[List[int], float, bool]:
        """
        텐서 로터의 회전을 갱신합니다.
        tensions: [T_L1, T_L2, T_L3] 각 레이어에 가해지는 로컬 텐션 (수면방향 2048을 향하는 인력)
        """
        new_phases = [0, 0, 0]
        torques = [0.0, 0.0, 0.0]
        
        # 1. 외부 로컬 텐션 인력 계산 (2048 방향으로 끌어당김)
        for i in range(3):
            # 2048과의 위상차 (자연 수면점)
            diff_rad = self._phase_diff_rad(2048, self.phases[i])
            torques[i] += tensions[i] * self.coupling_K * math.sin(diff_rad)
            
        # 2. 수평 피어 결합 (Layer 2 네트워크)
        if peer_phases_L2:
            valid_peers = [p for p in peer_phases_L2 if p is not None]
            if valid_peers:
                K_peer = self.coupling_K * 0.8
                total_sin = 0.0
                for p_j in valid_peers:
                    # p_j - self.phases[1]
                    total_sin += math.sin(self._phase_diff_rad(p_j, self.phases[1]))
                torques[1] += (K_peer / len(valid_peers)) * total_sin
                
        # 3. 수직 레이어 결합 (Layer 1 <-> Layer 2 <-> Layer 3)
        # Layer 1은 Layer 2에 이끌림
        torques[0] += self.vertical_K * math.sin(self._phase_diff_rad(self.phases[1], self.phases[0]))
        # Layer 2는 Layer 1과 Layer 3에 이끌림
        torques[1] += self.vertical_K * math.sin(self._phase_diff_rad(self.phases[0], self.phases[1]))
        torques[1] += self.vertical_K * math.sin(self._phase_diff_rad(self.phases[2], self.phases[1]))
        # Layer 3는 Layer 2에 이끌림
        torques[2] += self.vertical_K * math.sin(self._phase_diff_rad(self.phases[1], self.phases[2]))
        
        # 4. 위상 갱신
        for i in range(3):
            phase_step = (self.natural_drift + torques[i]) * dt
            new_phases[i] = int(round(self.phases[i] + phase_step)) & self.rotor_mask
            
        self.phases = new_phases
        
        # 5. 수면 팩터 중첩 산출 (Layer 1, 2, 3의 벡터 중첩)
        # e^(i*theta) 의 평균 실수부를 구하여 수면 팩터로 변환
        cos_sum = sum(math.cos(p * 2 * math.pi / self.rotor_scale) for p in self.phases)
        avg_cos = cos_sum / 3.0
        sleep_factor = (1.0 - avg_cos) / 2.0
        
        # 기상(0) 영역에서 수면(2048) 영역으로 진입 판정
        is_sleeping = sleep_factor > 0.5
        
        return self.phases, sleep_factor, is_sleeping
