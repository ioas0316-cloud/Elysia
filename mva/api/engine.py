import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import math
import json
import os
from scipy.optimize import curve_fit

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

def find_resonance_angle(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float]:
    current_points = []
    for p in points_data:
        base_pos = p['position']
        vel = p['velocity']
        phase = p['phase']

        x = base_pos[0] + math.sin(time_t + phase) * 0.5 + vel[0] * time_t * 0.1
        y = base_pos[1] + math.cos(time_t + phase) * 0.5 + vel[1] * time_t * 0.1
        z = base_pos[2] + vel[2] * time_t

        current_points.append([x, y, z])

    points_array = np.array(current_points)

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

def complex_wave_func(x, A, w, p, C):
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

def generate_symbolic_regression(points_data: List[Dict[str, Any]], best_quaternion: List[float], time_t: float) -> Tuple[str, float]:
    rot = R.from_quat(best_quaternion)

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

    rotated = rot.apply(np.array(current_points))
    xy_points = rotated[:, :2]

    if len(xy_points) > 2:
        x_coords = xy_points[:, 0]
        y_coords = xy_points[:, 1]

        try:
            popt, _ = curve_fit(complex_wave_func, x_coords, y_coords, p0=[1.0, 1.0, 0.0, np.mean(y_coords)], maxfev=2000)
            A, w, p, C = popt
            formula = f"y = {A:.2f} * e^(i({w:.2f}*x + {p:.2f})) + {C:.2f}"

            y_pred = complex_wave_func(x_coords, *popt)
            ss_res = np.sum((y_coords - y_pred)**2)
            ss_tot = np.sum((y_coords - np.mean(y_coords))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception as e:
            A_mat = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A_mat, y_coords, rcond=None)[0]
            formula = f"y = {m:.2f} * x + {c:.2f} (Linear Fallback)"

            y_pred = m * x_coords + c
            ss_res = np.sum((y_coords - y_pred)**2)
            ss_tot = np.sum((y_coords - np.mean(y_coords))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return formula, r_squared

    return "Insufficient data for regression", 0.0

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
        if r_squared >= 0.90:
            save_formula_to_archive(formula, variance, quaternion)
        else:
            is_resonant = False
            formula = f"[검증 실패] R² = {r_squared:.2f} (< 0.90) - 노이즈"

    return variance, is_resonant, formula

auto_observe_memory = {
    "current_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    "velocity_quat": np.array([0.0, 0.0, 0.0, 0.0])
}

def elysia_auto_observe_step(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float, bool, str]:
    points_array = get_current_points(points_data, time_t)

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
