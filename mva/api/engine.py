import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import math

def calculate_projection_variance(points: np.ndarray, quaternion: List[float]) -> float:
    """주어진 쿼터니언으로 공간을 회전한 뒤, 2D 평면(X-Y)에 투영된 점들의 분산을 계산합니다."""
    # 쿼터니언을 회전 행렬로 변환
    rot = R.from_quat(quaternion)
    rotated_points = rot.apply(points)

    # Z축을 관측축(시선)으로 가정하고 X-Y 평면(투영면)의 데이터 분산을 측정
    # 점들이 일렬로 정렬(줄삭제)되었다면 특정 축(예: Y축) 방향의 분산이 최소가 됨
    # 여기서는 선형 정렬을 찾기 위해 PCA와 유사하게 2D 공분산 행렬의 최소 고유값을 구함
    xy_points = rotated_points[:, :2]

    if len(xy_points) < 2:
        return 0.0

    cov_matrix = np.cov(xy_points.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)

    # 가장 작은 고유값이 작을수록 점들이 한 선상에 가까이 모여있다는 의미 (공명/일렬 정렬)
    min_eig = float(np.min(eigenvalues))
    # 부동 소수점 오차로 인한 매우 작은 음수 방지
    return max(0.0, min_eig)

def find_resonance_angle(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float]:
    """현재 시간(t)에서의 점들의 궤적을 기반으로 최적의 공명 각도(쿼터니언)를 찾습니다."""

    # 1. 현재 시간 t에서의 3D 좌표 계산
    current_points = []
    for p in points_data:
        base_pos = p['position']
        vel = p['velocity']
        phase = p['phase']

        # app.js 의 궤적 로직과 동일하게 시뮬레이션
        x = base_pos[0] + math.sin(time_t + phase) * 0.5 + vel[0] * time_t * 0.1
        y = base_pos[1] + math.cos(time_t + phase) * 0.5 + vel[1] * time_t * 0.1
        z = base_pos[2] + vel[2] * time_t

        current_points.append([x, y, z])

    points_array = np.array(current_points)

    # 2. 쿼터니언 공간 랜덤/그리드 탐색 (MVA 수준의 간단한 탐색)
    best_quaternion = [0, 0, 0, 1]
    min_variance = float('inf')

    # X, Y, Z 축에 대해 대략적인 각도를 탐색하여 '테트리스 줄삭제' 지점 찾기
    for angle_x in np.linspace(0, np.pi, 10):
        for angle_y in np.linspace(0, np.pi, 10):
            for angle_z in np.linspace(0, np.pi, 10):
                rot = R.from_euler('xyz', [angle_x, angle_y, angle_z])
                quat = rot.as_quat() # [x, y, z, w]

                variance = calculate_projection_variance(points_array, quat)

                if variance < min_variance:
                    min_variance = variance
                    best_quaternion = quat.tolist()

    return best_quaternion, min_variance

def generate_symbolic_regression(points_data: List[Dict[str, Any]], best_quaternion: List[float], time_t: float) -> str:
    """공명 상태에서의 2D 투영 궤적을 기반으로 기하학적 수식을 생성(역기록)합니다."""
    # 실제로는 궤적을 피팅해야 하지만 MVA에서는 현재 점들의 관계를 수식화하여 보여줌
    # 예: y = a * sin(b*x + c)
    rot = R.from_quat(best_quaternion)

    current_points = []
    for p in points_data:
        base_pos = p['position']
        vel = p['velocity']
        phase = p['phase']
        x = base_pos[0] + math.sin(time_t + phase) * 0.5 + vel[0] * time_t * 0.1
        y = base_pos[1] + math.cos(time_t + phase) * 0.5 + vel[1] * time_t * 0.1
        z = base_pos[2] + vel[2] * time_t
        current_points.append([x, y, z])

    rotated = rot.apply(np.array(current_points))
    xy_points = rotated[:, :2]

    # 단순 1차 함수 피팅 (y = ax + b)
    if len(xy_points) > 1:
        x_coords = xy_points[:, 0]
        y_coords = xy_points[:, 1]

        # 선형 회귀
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]

        # 공명이 일어났으므로 점들은 거의 이 선분 위에 존재함
        formula = f"y = {m:.2f} * x + {c:.2f} (Variance: minimized)"
        return formula

    return "Insufficient data for regression"
