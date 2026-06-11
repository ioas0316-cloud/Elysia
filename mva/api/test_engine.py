from engine import calculate_projection_variance, find_resonance_angle, generate_symbolic_regression, evaluate_current_state, elysia_auto_observe_step
from fractal import map_to_movement_field
import numpy as np

def test_variance():
    # Test variance calculation with dummy data
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    quat = [0.0, 0.0, 0.0, 1.0] # No rotation
    var = calculate_projection_variance(points, quat)
    # Variance along Y should be 0 since all points are on X axis
    assert var < 1e-5

def test_find_resonance():
    res = map_to_movement_field('테스트')
    best_quat, min_var = find_resonance_angle(res, 0.0)
    assert len(best_quat) == 4
    assert min_var >= 0.0

def test_auto_observe():
    res = map_to_movement_field('우주')
    # First step
    next_q, var, is_res, form = elysia_auto_observe_step(res, 0.0)
    assert len(next_q) == 4

def test_symbolic_regression():
    res = map_to_movement_field('하늘고래')
    best_quat, variance = find_resonance_angle(res, 0.0)
    formula, r_squared = generate_symbolic_regression(res, best_quat, 0.0)
    assert type(formula) == str
    assert type(r_squared) == np.float64 or type(r_squared) == float

def test_evaluate_state():
    res = map_to_movement_field('프랙탈')
    quat = [0, 0, 0, 1]
    variance, is_resonant, formula = evaluate_current_state(res, quat, 0.0)
    assert type(variance) == float
    assert type(is_resonant) == bool
