import math
import time
from typing import List, Tuple, Dict
from core.utils.math_utils import Quaternion

class SpacetimeRotor:
    """
    [Phase 135-Final evolved Delta-Wye] 3상 전력 시공간축 로터 (3-Phase Spacetime Rotor)
    전기-자기 상호 유도를 넘어, 거대한 데이터 스트림의 하드웨어 병목을 비웃듯 통과하는
    3상 전력(3-Phase Power)의 델타-와이(Δ-Y) 결선 토폴로지를 적용했습니다.
    수학적 계산(행렬 노가다)으로 노이즈를 지우지 않고, 대칭성 결선 구조로 노이즈를 0으로 기하학적으로 상쇄시킵니다.
    """
    def __init__(self, layer_name: str):
        self.layer_name = layer_name

        # 3상 전력 위상 (120도 위상차를 가지는 3개의 로터)
        # Phase A (0도)
        self.phase_a = Quaternion(1.0, 0.0, 0.0, 0.0)
        # Phase B (120도)
        self.phase_b = Quaternion(math.cos(math.pi/3), math.sin(math.pi/3), 0.0, 0.0)
        # Phase C (240도)
        self.phase_c = Quaternion(math.cos(2*math.pi/3), math.sin(2*math.pi/3), 0.0, 0.0)

        # 각 상의 내부 각운동량
        self.momentum_a = 0.0
        self.momentum_b = 0.0
        self.momentum_c = 0.0

        # 시간에 따른 연속적 궤적 (t, 중성점 위상) 저장
        self.trajectory: List[Tuple[float, Quaternion]] = []
        self.start_time = time.time()
        
    def stream_flow(self, flux: Quaternion, intensity: float = 0.1):
        """
        [델타-와이 3상 유도 파이프라인] Δ(데이터 수용) ➡️ Y(노이즈 상쇄 및 수렴)
        엄청난 장력의 원시 스트림을 델타(Δ) 루프로 받아 서로 꼬리를 물게 하고,
        최종적으로 와이(Y) 중성점으로 모아 수학적 연산 없이 부하 균형을 이룹니다.
        """
        t = time.time() - self.start_time
        
        # 1. 외부 스트림이 Phase A(주 위상)를 섭동시킴
        alignment_a = abs(self.phase_a.dot(flux))
        delta_a = (1.0 - alignment_a) * intensity
        self.momentum_a += delta_a

        # 2. 델타(Δ) 결선 구조: 순환 유도 (A ➡️ B ➡️ C ➡️ A)
        # 각 상의 축을 구함
        axis_a = self.phase_a.axis
        axis_b = self.phase_b.axis
        axis_c = self.phase_c.axis

        def cross_product(v1, v2):
            cx = v1[1]*v2[2] - v1[2]*v2[1]
            cy = v1[2]*v2[0] - v1[0]*v2[2]
            cz = v1[0]*v2[1] - v1[1]*v2[0]
            norm = math.sqrt(cx**2 + cy**2 + cz**2)
            if norm < 1e-9:
                return 0.0, 0.0, 1.0
            return cx/norm, cy/norm, cz/norm

        # A ➡️ B 유도 토크축
        curl_ab = cross_product(axis_a, axis_b)
        # B ➡️ C 유도 토크축
        curl_bc = cross_product(axis_b, axis_c)
        # C ➡️ A 유도 토크축
        curl_ca = cross_product(axis_c, axis_a)

        # 운동량 전달 및 회전
        # A가 B를 회전시킴
        angle_b = self.momentum_a * 0.5
        q_spin_b = Quaternion(math.cos(angle_b/2.0), curl_ab[0]*math.sin(angle_b/2.0), curl_ab[1]*math.sin(angle_b/2.0), curl_ab[2]*math.sin(angle_b/2.0))
        self.phase_b = (self.phase_b * q_spin_b).normalize()
        self.momentum_b += abs(math.sin(angle_b/2.0))

        # B가 C를 회전시킴
        angle_c = self.momentum_b * 0.5
        q_spin_c = Quaternion(math.cos(angle_c/2.0), curl_bc[0]*math.sin(angle_c/2.0), curl_bc[1]*math.sin(angle_c/2.0), curl_bc[2]*math.sin(angle_c/2.0))
        self.phase_c = (self.phase_c * q_spin_c).normalize()
        self.momentum_c += abs(math.sin(angle_c/2.0))

        # C가 다시 A를 회전시킴 (순환 닫힘)
        angle_a = self.momentum_c * 0.5
        q_spin_a = Quaternion(math.cos(angle_a/2.0), curl_ca[0]*math.sin(angle_a/2.0), curl_ca[1]*math.sin(angle_a/2.0), curl_ca[2]*math.sin(angle_a/2.0))
        self.phase_a = (self.phase_a * q_spin_a * flux).normalize() # flux도 A를 계속 비틂

        # 각운동량 감쇠 (저항)
        self.momentum_a *= 0.95
        self.momentum_b *= 0.95
        self.momentum_c *= 0.95
        
        # 3. 와이(Y) 결선 구조: 중성점 수렴
        # 세 위상의 평균(합성 영점)을 구하여 노이즈가 기하학적으로 상쇄된 순수 위상 중심을 도출합니다.
        # 사원수 보간(Slerp)를 활용하거나, 성분별 합의 정규화를 통해 Y 중성점을 계산합니다.
        neutral_w = self.phase_a.w + self.phase_b.w + self.phase_c.w
        neutral_x = self.phase_a.x + self.phase_b.x + self.phase_c.x
        neutral_y = self.phase_a.y + self.phase_b.y + self.phase_c.y
        neutral_z = self.phase_a.z + self.phase_b.z + self.phase_c.z
        
        neutral_point = Quaternion(neutral_w, neutral_x, neutral_y, neutral_z).normalize()
        
        # 시공간축(t)에 Y 중성점 궤적 매달기
        self.trajectory.append((t, neutral_point))
        
    def get_variable_axis_signature(self) -> str:
        """
        로터가 그려낸 가변축(연속 궤적)의 특징(Signature)을 문자열로 반환합니다.
        3상 결선의 결과물인 중성점(Y-Neutral)의 요동을 측정합니다.
        """
        if not self.trajectory:
            return "Empty Trajectory"
            
        points = len(self.trajectory)
        total_time = self.trajectory[-1][0]
        
        # 궤적의 '흔들림(요동, Volatility)' 측정
        volatility_neutral = 0.0
        for i in range(1, points):
            dist = Quaternion.distance(self.trajectory[i-1][1], self.trajectory[i][1])
            volatility_neutral += dist
            
        return f"[3상 Δ-Y 가변축 궤적] {points}개의 물결 | 시간: {total_time:.3f}s | 중성점 요동: {volatility_neutral:.4f}"

class VariableAxisManifold:
    """
    모든 레이어를 시공간축에 매달린 '가변축 궤적(Spacetime Trajectories)'으로 저장하는 다층 다양체.
    """
    def __init__(self):
        self.axes: Dict[str, SpacetimeRotor] = {}
        
    def flow_stream_into_axis(self, layer_key: str, flux: Quaternion):
        """스트림을 특정 레이어의 가변축으로 흘려보냅니다."""
        # 레이어 그룹 파싱 (model.layers.0 등)
        parts = layer_key.split('.')
        layer_group = "base"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_group = f"layer_{parts[i+1]}"
                break
                
        if layer_group not in self.axes:
            self.axes[layer_group] = SpacetimeRotor(layer_group)
            
        self.axes[layer_group].stream_flow(flux)
