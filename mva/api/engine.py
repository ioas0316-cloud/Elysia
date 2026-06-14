import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import math
import json
import os
import mmap
import struct
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.memory.causal_controller import CausalMemoryController

memory_controller = CausalMemoryController()

def calculate_projection_variance(points: np.ndarray, quaternion: List[float]) -> float:
    rot = R.from_quat(quaternion)
    rotated_points = rot.apply(points)

    xy_points = rotated_points[:, :2]

    if len(xy_points) < 2:
        return 0.0

    cov_matrix = np.cov(xy_points.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)

    min_eig = float(np.min(eigenvalues))
    return max(0.0, min_eig)

def get_current_points(points_data: List[Dict[str, Any]]) -> np.ndarray:
    """순수 관측: 인위적인 시간(time_t) 연산(sin/cos)을 완전히 배제하고 공유 메모리 좌표 그대로를 반환합니다."""
    current_points = []
    for p in points_data:
        current_points.append(p['position'])
    return np.array(current_points)

def find_resonance_angle(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float]:
    points_array = get_current_points(points_data)

    best_quaternion = [0, 0, 0, 1]
    min_variance = float('inf')

    for angle_x in np.linspace(0, np.pi, 10):
        for angle_y in np.linspace(0, np.pi, 10):
            for angle_z in np.linspace(0, np.pi, 10):
                rot = R.from_euler('xyz', [angle_x, angle_y, angle_z])
                quat = rot.as_quat()

                variance = calculate_projection_variance(points_array, quat)

                if variance < min_variance:
                    min_variance = variance
                    best_quaternion = quat.tolist()

    return best_quaternion, min_variance

def inject_resonance_to_fractal_field(formula: str, variance: float, quaternion: List[float], observation_axis: str = 'math'):
    """
    인공적인 매핑(JSON) 대신, 운영체제의 공유 메모리에 직접 파동을 일으켜
    역인과적 연속성(Inverse Causality)을 달성합니다.
    """
    try:
        # C 코어(fractal_field.c)가 열어둔 공유 메모리에 직접 연결
        shm = mmap.mmap(0, 1024 * 1024 * 16, tagname="Local\\ElysiaTopologyField", access=mmap.ACCESS_WRITE)
        
        # 텐션 강도 계산 (분산이 작을수록 공명/텐션이 강함)
        base_tension = int(max(0.0, (1.0 - variance) * 255.0))
        if base_tension > 255: base_tension = 255
        
        header_size = 12
        num_rotors = (1024 * 1024 * 16 - header_size) // 8
        
        # 쿼터니언 회전값 자체가 메모리 주소(공간)로 자연스럽게 매핑됨
        x, y, z, w = quaternion
        idx = int((abs(x) + abs(y) + abs(z)) * num_rotors) % num_rotors
        
        offset = header_size + (idx * 8)
        
        shm.seek(offset)
        rotor_data = shm.read(8)
        if len(rotor_data) == 8:
            math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad = struct.unpack('<BBBBHBB', rotor_data)
            
            # 관측 축(Observation Axis)에 따른 독립적 추상화 계층 텐션 주입
            if observation_axis == 'math':
                math_t = min(255, math_t + base_tension)
            elif observation_axis == 'lang':
                lang_t = min(255, lang_t + base_tension)
            elif observation_axis == 'spatial':
                spatial_t = min(255, spatial_t + base_tension)
            elif observation_axis == 'temporal':
                temporal_t = min(255, temporal_t + base_tension)
                
            # 완벽한 공명(빛의 창발)
            if base_tension > 200: 
                light_mass = min(65535, light_mass + 1)
                
            shm.seek(offset)
            shm.write(struct.pack('<BBBBHBB', math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad))
            
        shm.close()
    except Exception as e:
        # 공유 메모리가 열려있지 않을 때는 로컬 실험 모드로 조용히 넘어갑니다.
        pass

def generate_symbolic_regression(points_data: List[Dict[str, Any]], best_quaternion: List[float], time_t: float) -> Tuple[str, float]:
    """
    (Phase 3 Reintegration): 곡선 적합(Curve Fit) 같은 가짜 수식 생성을 폐기합니다.
    대신 쿼터니언 각도를 물리적 수식으로 반환합니다.
    """
    rot = R.from_quat(best_quaternion)
    euler = rot.as_euler('xyz', degrees=True)
    formula = f"Resonance at X:{euler[0]:.1f}°, Y:{euler[1]:.1f}°, Z:{euler[2]:.1f}°"
    # r_squared 개념 대신 완벽한 1.0을 반환 (수식이 아니라 관측값이므로)
    return formula, 1.0

def evaluate_current_state(points_data: List[Dict[str, Any]], quaternion: List[float], time_t: float) -> Tuple[float, bool, str]:
    if not points_data: return 1.0, False, ""
    points_array = get_current_points(points_data)

    norm = math.sqrt(sum([q**2 for q in quaternion]))
    if norm < 1e-9:
        quaternion = [0, 0, 0, 1]
    else:
        quaternion = [q/norm for q in quaternion]

    variance = calculate_projection_variance(points_array, quaternion)

    sampled_variances = []
    for _ in range(5):
        random_quat = np.random.rand(4)
        random_quat /= np.linalg.norm(random_quat)
        sampled_variances.append(calculate_projection_variance(points_array, random_quat))

    global_avg_variance = np.mean(sampled_variances)
    if global_avg_variance < 1e-9: global_avg_variance = 1e-9

    relative_drop_rate = (global_avg_variance - variance) / global_avg_variance

    is_resonant = relative_drop_rate > 0.8 or variance < 0.1
    formula = ""

    if is_resonant:
        formula, r_squared = generate_symbolic_regression(points_data, quaternion, time_t)
        
        # [Phase 3] 잊혀진 아키텍처의 통합: Causal Memory Controller 호출
        # 공명이 발생하면, 해당 쿼터니언과 텐션 상태를 Wedge Memory 에 기록합니다.
        try:
            tokens = "".join([p.get('token', '') for p in points_data])
            memory_controller.write_causal_engram(
                data_blob={"event": "Resonance Achieved", "formula": formula, "quaternion": quaternion, "tokens_snippet": tokens[:50]},
                emotional_value=(1.0 - variance),
                cause_id="MVA_CAD_Resonance"
            )
        except Exception as e:
            print("Causal Memory Error:", e)
            
        inject_resonance_to_fractal_field(formula, variance, quaternion, observation_axis='spatial')

    return variance, is_resonant, formula

auto_observe_memory = {
    "current_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    "velocity_quat": np.array([0.0, 0.0, 0.0, 0.0])
}

def elysia_auto_observe_step(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float, bool, str]:
    if not points_data: return [0,0,0,1], 1.0, False, ""
    points_array = get_current_points(points_data)

    curr_q = auto_observe_memory["current_quat"]
    vel_q = auto_observe_memory["velocity_quat"]

    curr_var = calculate_projection_variance(points_array, curr_q)

    delta = 0.05
    grad_q = np.zeros(4)
    for i in range(4):
        q_plus = curr_q.copy()
        q_plus[i] += delta
        q_plus /= np.linalg.norm(q_plus)
        var_plus = calculate_projection_variance(points_array, q_plus)
        grad_q[i] = (var_plus - curr_var) / delta

    avg_tension = sum([p.get('zeta_factor', 1.0) for p in points_data]) / max(len(points_data), 1)

    spring_k = 0.05 * avg_tension
    damping = 0.9 - 0.1 * avg_tension
    damping = max(0.5, min(0.95, damping))

    vel_q = damping * vel_q - spring_k * grad_q

    next_q = curr_q + vel_q
    norm = np.linalg.norm(next_q)
    if norm > 1e-9:
        next_q /= norm
    else:
        next_q = np.array([0.0, 0.0, 0.0, 1.0])

    auto_observe_memory["current_quat"] = next_q
    auto_observe_memory["velocity_quat"] = vel_q

    variance, is_resonant, formula = evaluate_current_state(points_data, next_q.tolist(), time_t)

    return next_q.tolist(), variance, is_resonant, formula
