from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.pc_sensory_motor_body import (  # noqa: E402
    ScreenRetinaSensor,
    WindowsForegroundSensor,
)
from core.embodiment.self_modifying_organism import (  # noqa: E402
    BASE_AXIS_NAMES,
    SelfModifyingCausalOrganism,
)


def _make_stationary_frame_provider():
    """Real high-res frame where center salient patch is warm/bright and corners are dark cool."""
    img = np.full((240, 320, 3), [0.10, 0.12, 0.28], dtype=np.float64)
    for y in range(100, 140):
        for x in range(140, 180):
            if ((x // 2) + (y // 2)) % 2 == 0:
                img[y, x] = [0.95, 0.35, 0.12]
            else:
                img[y, x] = [0.15, 0.10, 0.10]
    return lambda: img


def test_uncontrolled_world_prediction_error_drops_through_genuine_self_modification() -> None:
    org = SelfModifyingCausalOrganism()
    retina = ScreenRetinaSensor(frame_provider=_make_stationary_frame_provider())
    fg = WindowsForegroundSensor(title_provider=lambda: "FixedWorld")

    errors: list[float] = []
    for _ in range(5):
        tr = org.run_autonomous_cycle(retina, fg, forced_action="zoom_salient_focus")
        errors.append(tr.contrast.how_much_l2)

    assert tr.contrast.why_attribution != "unseen_action_effect"
    assert errors[0] > 0.08
    assert errors[-1] < errors[0] * 0.45
    assert tr.modification.past_error_after <= tr.modification.past_error_before


def test_persistent_unexplained_residual_spawns_new_perceptual_axis() -> None:
    org = SelfModifyingCausalOrganism()
    s0 = np.full(10, 0.2, dtype=np.float64)
    s_next = s0.copy()
    s_next[1] += 0.45  # persistent diagonal residual in r_mean & edge_sharpness
    s_next[7] += 0.45

    spawned = []
    for _ in range(5):
        s_b = org.project_state(s0)
        pred = org.predict_next(s_b, "probe_workspace_file")
        act = org.project_state(s_next)
        c = org.compare_prediction_and_reality(s_b, "probe_workspace_file", pred, act)
        m = org.self_modify(s0, "probe_workspace_file", pred, act, c)
        org.action_counts["probe_workspace_file"] += 1
        org.episode_buffer.append(
            {"cycle": m.cycle, "action": "probe_workspace_file", "raw_before": s0.tolist(), "raw_after": s_next.tolist()}
        )
        if m.new_axis_name:
            spawned.append(m.new_axis_name)

    assert len(spawned) >= 1
    assert len(org.axis_names) > len(BASE_AXIS_NAMES)
    assert org.projection_matrix.shape[0] == len(org.axis_names)
    assert org.transition_weights["probe_workspace_file"].shape == (len(org.axis_names), len(org.axis_names) + 1)


def test_harmful_self_modification_triggers_automatic_rollback() -> None:
    org = SelfModifyingCausalOrganism()
    s0 = np.full(10, 0.3, dtype=np.float64)
    s_mild = s0.copy()
    s_mild[4] += 0.12

    # Cycle 1: small residual -> weight update kept & pending verification stored
    s0_d = org.project_state(s0)
    pred1 = org.predict_next(s0_d, "zoom_salient_focus")
    act1 = org.project_state(s_mild)
    c1 = org.compare_prediction_and_reality(s0_d, "zoom_salient_focus", pred1, act1)
    org.self_modify(s0, "zoom_salient_focus", pred1, act1, c1)
    org.action_counts["zoom_salient_focus"] += 1

    # Cycle 2: reality suddenly reverses much larger -> triggers automatic rollback!
    s_reversed = s0.copy()
    s_reversed[4] -= 0.45
    pred2 = org.predict_next(s0_d, "zoom_salient_focus")
    act2 = org.project_state(s_reversed)
    c2 = org.compare_prediction_and_reality(s0_d, "zoom_salient_focus", pred2, act2)
    m2 = org.self_modify(s0, "zoom_salient_focus", pred2, act2, c2)

    assert m2.mutation_kind == "rollback"
    assert m2.kept is False
    assert org.mutations_reverted == 1
    assert np.allclose(org.transition_weights["zoom_salient_focus"], 0.0)


def test_external_world_change_while_still_attributed_to_external_autonomy() -> None:
    org = SelfModifyingCausalOrganism()
    step = {"i": 0}

    def _shifting_world() -> np.ndarray:
        step["i"] += 1
        if step["i"] == 1:
            return np.full((240, 320, 3), 0.15, dtype=np.float64)
        return np.full((240, 320, 3), 0.85, dtype=np.float64)

    retina = ScreenRetinaSensor(frame_provider=_shifting_world)
    fg = WindowsForegroundSensor(title_provider=lambda: "DynamicScene")
    tr = org.run_autonomous_cycle(retina, fg, forced_action="observe_still")

    assert tr.contrast.why_attribution == "external_world_autonomy"
    assert tr.contrast.how_much_l2 > 0.3


def test_organism_persistence_roundtrip_and_real_screen_cycle(tmp_path: Path) -> None:
    org = SelfModifyingCausalOrganism()
    retina = ScreenRetinaSensor(frame_provider=_make_stationary_frame_provider())
    fg = WindowsForegroundSensor(title_provider=lambda: "WorkspaceWindow")

    for _ in range(4):
        org.run_autonomous_cycle(retina, fg, workspace_root=ROOT)

    save_p = tmp_path / "org.json"
    org.save(save_p)
    reloaded = SelfModifyingCausalOrganism.load(save_p)

    assert reloaded.cycle_count == 4
    assert len(reloaded.episode_buffer) == 4
    assert sum(reloaded.action_counts.values()) == 4
    assert reloaded.projection_matrix.shape[1] == 10
