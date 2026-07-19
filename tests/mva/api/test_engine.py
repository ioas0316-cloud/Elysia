from mva.api.engine import evaluate_current_state, elysia_auto_observe_step
from mva.api.fractal import map_to_movement_field
import numpy as np

def test_auto_observe():
    res = map_to_movement_field('우주')
    # First step
    next_q, tension, is_res, form = elysia_auto_observe_step(res, 0.0)
    assert len(next_q) == 4
    assert type(tension) == float
    assert type(is_res) == bool

def test_evaluate_state():
    res = map_to_movement_field('프랙탈')
    quat = [0, 0, 0, 1]
    tension, is_resonant, formula = evaluate_current_state(res, quat, 0.0)
    assert type(tension) == float
    assert type(is_resonant) == bool
    assert type(formula) == str
