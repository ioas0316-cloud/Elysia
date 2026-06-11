import pytest
from fractal import decompose_hangul, map_to_movement_field
from engine import calculate_projection_variance, find_resonance_angle

def test_decompose_hangul():
    cho, jung, jong = decompose_hangul('가')
    assert cho == 0  # ㄱ
    assert jung == 0 # ㅏ
    assert jong == 0 # (없음)

    cho, jung, jong = decompose_hangul('힣')
    assert cho == 18 # ㅎ
    assert jung == 20 # ㅣ
    assert jong == 27 # ㅎ

def test_map_to_movement_field():
    res = map_to_movement_field('하늘')
    assert len(res) == 2
    assert res[0]['token'] == '하'
    assert res[1]['token'] == '늘'
    assert 'position' in res[0]
    assert 'velocity' in res[0]

def test_engine_basic():
    # 간단한 테스트
    res = map_to_movement_field('하늘')
    best_quat, variance = find_resonance_angle(res, 0.0)
    assert len(best_quat) == 4
    assert variance >= 0.0

def test_symbolic_regression():
    from engine import generate_symbolic_regression
    res = map_to_movement_field('하늘고래')
    best_quat, variance = find_resonance_angle(res, 0.0)
    formula = generate_symbolic_regression(res, best_quat, 0.0)
    assert type(formula) == str
    assert 'y =' in formula or 'Insufficient' in formula

def test_evaluate_current_state():
    from engine import evaluate_current_state
    res = map_to_movement_field('하늘고래')
    variance, is_res, form = evaluate_current_state(res, [0,0,0,1], 0.0)
    assert type(variance) == float
    assert type(is_res) == bool
    assert type(form) == str
