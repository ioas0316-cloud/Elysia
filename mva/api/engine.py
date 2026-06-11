import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import math
import json
import os
from scipy.optimize import curve_fit


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

def complex_wave_func(x, A, w, p, C):
    # Re( A * exp(i(w*x + p)) ) + C  == A * cos(w*x + p) + C
    # We fit a real cosine wave to represent the projected Euler wave
    return A * np.cos(w * x + p) + C

def save_formula_to_archive(formula: str, variance: float, quaternion: List[float]):
    archive_path = 'mva/api/archives.json'
    archives = []
    if os.path.exists(archive_path):
        with open(archive_path, 'r') as f:
            try:
                archives = json.load(f)
            except:
                pass
    archives.append({
        "formula": formula,
        "variance": variance,
        "quaternion": quaternion
    })
    with open(archive_path, 'w') as f:
        json.dump(archives, f, indent=4, ensure_ascii=False)

def generate_symbolic_regression(points_data: List[Dict[str, Any]], best_quaternion: List[float], time_t: float) -> str:
    """공명 상태에서의 2D 투영 궤적을 기반으로 기하학적 수식을 생성(역기록)합니다."""
    rot = R.from_quat(best_quaternion)

    current_points = []
    for p in points_data:
        base_pos = p['position']
        vel = p['velocity']
        phase = p['phase']

        # 적용된 Zeta Factor (Tension)을 함께 반영하여 현재 좌표 계산
        tension = p.get('zeta_factor', 1.0)

        # fractal.py 의 로직에 맞게 가속도와 위상 적용
        x = base_pos[0] + math.sin(time_t + phase) * 0.5 * tension + vel[0] * time_t * 0.1
        y = base_pos[1] + math.cos(time_t + phase) * 0.5 * tension + vel[1] * time_t * 0.1
        z = base_pos[2] + vel[2] * time_t
        current_points.append([x, y, z])

    rotated = rot.apply(np.array(current_points))
    xy_points = rotated[:, :2]

    if len(xy_points) > 2:
        x_coords = xy_points[:, 0]
        y_coords = xy_points[:, 1]

        try:
            # 파동 함수(Cos)로 피팅 시도 (오일러 공식의 실수부)
            # 초기 추정값: 진폭 A=1.0, 주파수 w=1.0, 위상 p=0.0, 상수 C=평균y
            popt, _ = curve_fit(complex_wave_func, x_coords, y_coords, p0=[1.0, 1.0, 0.0, np.mean(y_coords)], maxfev=2000)
            A, w, p, C = popt
            formula = f"y = {A:.2f} * e^(i({w:.2f}*x + {p:.2f})) + {C:.2f}"
        except Exception as e:
            # 피팅 실패 시 선형 회귀로 Fallback
            A_mat = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A_mat, y_coords, rcond=None)[0]
            formula = f"y = {m:.2f} * x + {c:.2f} (Linear Fallback)"

        return formula

    return "Insufficient data for regression"

def get_current_points(points_data: List[Dict[str, Any]], time_t: float) -> np.ndarray:
    current_points = []
    for p in points_data:
        base_pos = p['position']
        vel = p['velocity']
        phase = p['phase']
        tension = p.get('zeta_factor', 1.0)

        x = base_pos[0] + math.sin(time_t + phase) * 0.5 * tension + vel[0] * time_t * 0.1
        y = base_pos[1] + math.cos(time_t + phase) * 0.5 * tension + vel[1] * time_t * 0.1
        z = base_pos[2] + vel[2] * time_t
        current_points.append([x, y, z])
    return np.array(current_points)

def evaluate_current_state(points_data: List[Dict[str, Any]], quaternion: List[float], time_t: float) -> Tuple[float, bool, str]:
    points_array = get_current_points(points_data, time_t)

    norm = math.sqrt(sum([q**2 for q in quaternion]))
    if norm < 1e-9:
        quaternion = [0, 0, 0, 1]
    else:
        quaternion = [q/norm for q in quaternion]

    variance = calculate_projection_variance(points_array, quaternion)

    # Calculate global variance (random sampling over angles)
    sampled_variances = []
    for _ in range(5):
        random_quat = np.random.rand(4)
        random_quat /= np.linalg.norm(random_quat)
        sampled_variances.append(calculate_projection_variance(points_array, random_quat))

    global_avg_variance = np.mean(sampled_variances)
    if global_avg_variance < 1e-9: global_avg_variance = 1e-9

    relative_drop_rate = (global_avg_variance - variance) / global_avg_variance

    # 줄삭제(공명) 판단 기준: 분산이 전역 평균 대비 80% 이상 감소했거나, 절대 분산이 매우 작을 때
    is_resonant = relative_drop_rate > 0.8 or variance < 0.1
    formula = ""

    if is_resonant:
        formula = generate_symbolic_regression(points_data, quaternion, time_t)
        save_formula_to_archive(formula, variance, quaternion)

    return variance, is_resonant, formula


# 엘리시아 자율 관측 궤적 저장소 (관성을 위한 메모리)
auto_observe_memory = {
    "current_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    "velocity_quat": np.array([0.0, 0.0, 0.0, 0.0])
}

def elysia_auto_observe_step(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float, bool, str]:
    """
    엘리시아가 스스로 공간의 분산(Tension Gradient)을 느끼고 쿼터니언을 회전시킵니다.
    경사 하강법과 운동량(Momentum/Damping)을 결합하여 시선을 움직입니다.
    """
    points_array = get_current_points(points_data, time_t)

    curr_q = auto_observe_memory["current_quat"]
    vel_q = auto_observe_memory["velocity_quat"]

    curr_var = calculate_projection_variance(points_array, curr_q)

    # 쿼터니언 미소 변화량에 대한 분산 기울기(Gradient) 계산
    delta = 0.05
    grad_q = np.zeros(4)
    for i in range(4):
        q_plus = curr_q.copy()
        q_plus[i] += delta
        q_plus /= np.linalg.norm(q_plus)
        var_plus = calculate_projection_variance(points_array, q_plus)
        grad_q[i] = (var_plus - curr_var) / delta

    # 물리적 제동 계수 (Damping) 및 관성 (Momentum)
    learning_rate = 0.02
    momentum = 0.8

    vel_q = momentum * vel_q - learning_rate * grad_q

    # Update Quat
    next_q = curr_q + vel_q
    norm = np.linalg.norm(next_q)
    if norm > 1e-9:
        next_q /= norm
    else:
        next_q = np.array([0.0, 0.0, 0.0, 1.0])

    auto_observe_memory["current_quat"] = next_q
    auto_observe_memory["velocity_quat"] = vel_q

    # 평가
    variance, is_resonant, formula = evaluate_current_state(points_data, next_q.tolist(), time_t)

    return next_q.tolist(), variance, is_resonant, formula
