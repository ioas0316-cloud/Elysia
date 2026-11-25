from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in (_REPO_ROOT / "L2_cognitive_engine", _REPO_ROOT / "L3_interface"):
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

from core.embodiment import (
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)
from core.topology.continuous_rotor_spectrum_field import (
    ContinuousBlindCADRotor,
    ContinuousRotorSpectrumField,
    apply_continuous_scale_vision,
    build_combat_interaction_rotor,
    build_gravity_metabolism_rotor,
)
from simulators.opaque_external_tools import OpaqueParametricCADKernel


def test_gravity_metabolism_rotor_emerges_standing_sitting_lying_sleeping_without_if_else() -> None:
    rotor = build_gravity_metabolism_rotor()

    energetic = rotor.converge(
        {"gravity_load": -0.2, "metabolic_energy": 0.9, "fatigue_pressure": -0.7, "support_comfort": -0.3}
    )
    moderate = rotor.converge(
        {"gravity_load": 0.3, "metabolic_energy": 0.0, "fatigue_pressure": 0.25, "support_comfort": 0.35}
    )
    exhausted = rotor.converge(
        {"gravity_load": 0.9, "metabolic_energy": -0.85, "fatigue_pressure": 0.95, "support_comfort": 0.9}
    )

    assert energetic.emergent_outputs["posture_height"] > moderate.emergent_outputs["posture_height"] > exhausted.emergent_outputs["posture_height"]
    assert energetic.emergent_outputs["posture_height"] > 0.5
    assert exhausted.emergent_outputs["posture_height"] < -0.5
    assert exhausted.emergent_outputs["metabolic_expenditure"] < -0.5
    assert exhausted.emergent_outputs["support_seeking"] > 0.6


def test_combat_interaction_rotor_rotates_attack_guard_evade_flee_continuously() -> None:
    rotor = build_combat_interaction_rotor()

    close_strong = rotor.converge(
        {"target_proximity": 0.9, "incoming_threat": -0.2, "self_reserve": 0.9, "evasion_space": 0.1}
    )
    close_attacked = rotor.converge(
        {"target_proximity": 0.85, "incoming_threat": 0.95, "self_reserve": 0.2, "evasion_space": -0.7}
    )
    open_evade = rotor.converge(
        {"target_proximity": 0.2, "incoming_threat": 0.9, "self_reserve": 0.6, "evasion_space": 0.95}
    )
    weak_retreat = rotor.converge(
        {"target_proximity": -0.7, "incoming_threat": 0.9, "self_reserve": -0.9, "evasion_space": 0.5}
    )

    assert close_strong.emergent_outputs["engage_vs_retreat"] > 0.6
    assert close_attacked.emergent_outputs["guard_compression"] > 0.65
    assert open_evade.emergent_outputs["lateral_evade"] > 0.7
    assert weak_retreat.emergent_outputs["engage_vs_retreat"] < -0.6


def test_variable_resistor_dial_modulates_sensitivity_and_inertia() -> None:
    r_fluid = build_combat_interaction_rotor()
    r_rigid = build_combat_interaction_rotor()
    r_fluid.set_resistance("incoming_threat", 0.25)
    r_rigid.set_resistance("incoming_threat", 6.0)

    stim = {"target_proximity": 0.5, "incoming_threat": 0.95, "self_reserve": 0.0, "evasion_space": -0.3}
    s_fluid = r_fluid.converge(stim, max_iters=6)
    s_rigid = r_rigid.converge(stim, max_iters=6)

    assert s_fluid.dial_values["incoming_threat"] > s_rigid.dial_values["incoming_threat"]
    assert s_fluid.emergent_outputs["guard_compression"] > s_rigid.emergent_outputs["guard_compression"]


def test_continuous_visual_scale_dial_transitions_macro_to_micro_monotonically() -> None:
    img = np.full((200, 280, 3), 20, dtype=np.uint8)
    for y in range(12, 188, 16):
        for x in range(12, 268, 18):
            img[y : y + 6, x : x + 8, :] = [235, 225, 70]

    macro = apply_continuous_scale_vision(img, scale_dial=0.05)
    meso = apply_continuous_scale_vision(img, scale_dial=0.50)
    micro = apply_continuous_scale_vision(img, scale_dial=0.95)

    assert macro["blur_radius"] > meso["blur_radius"] > micro["blur_radius"]
    assert macro["pooling_stride"] >= meso["pooling_stride"] >= micro["pooling_stride"]
    assert macro["perceived_units"] < meso["perceived_units"] <= micro["perceived_units"]
    assert micro["perceived_units"] >= 40


def test_continuous_rotor_plasticity_reduces_prediction_error_and_deforms_W_M_R() -> None:
    rotor = build_combat_interaction_rotor()
    stim = {"target_proximity": 0.75, "incoming_threat": 0.85, "self_reserve": 0.3, "evasion_space": -0.4}
    target = {"engage_vs_retreat": -0.65, "guard_compression": 0.90, "lateral_evade": -0.20}

    first = rotor.adapt_from_friction(stim, target)
    last = first
    for _ in range(5):
        last = rotor.adapt_from_friction(stim, target)

    assert last["error_after"] < first["error_before"]
    summary = rotor.plasticity_summary()
    assert summary["plasticity_updates"] == 6
    assert summary["drift_W_norm"] > 0.0
    assert summary["drift_M_norm"] > 0.0
    assert summary["drift_R_norm"] > 0.0

    # Full JSON round-trip preserves deformed (W, M, R)
    restored = ContinuousRotorSpectrumField.from_dict(rotor.to_dict())
    assert restored.plasticity_summary()["drift_M_norm"] == summary["drift_M_norm"]
    assert np.allclose(restored.coupling, rotor.coupling)
    assert np.allclose(restored.resistances, rotor.resistances)


def test_structural_axis_emergence_grows_new_coupled_axis_without_resetting_learned_weights() -> None:
    rotor = build_gravity_metabolism_rotor()
    prev_M = rotor.readout.copy()
    grown = rotor.grow_coupled_axis(
        "ambient_cold_stress",
        initial_resistance=0.60,
        residual_outputs={"metabolic_expenditure": 0.75, "support_seeking": 0.50},
    )
    assert grown["grown"] is True
    assert rotor.dim == 5
    assert "ambient_cold_stress" in rotor.emergent_axes
    assert np.allclose(rotor.readout[:, :4], prev_M)

    cold_state = rotor.converge(
        {"gravity_load": 0.0, "metabolic_energy": 0.1, "fatigue_pressure": 0.1, "support_comfort": 0.0, "ambient_cold_stress": 0.95}
    )
    warm_state = rotor.converge(
        {"gravity_load": 0.0, "metabolic_energy": 0.1, "fatigue_pressure": 0.1, "support_comfort": 0.0, "ambient_cold_stress": -0.95}
    )
    assert cold_state.emergent_outputs["metabolic_expenditure"] > warm_state.emergent_outputs["metabolic_expenditure"]


def test_continuous_blind_cad_rotor_discovers_cavity_with_shuffled_opaque_ids() -> None:
    # Shuffle/rename opaque operator IDs so NO hardcoded `"op_k0"` / `"op_k3"` could ever work!
    kernel = OpaqueParametricCADKernel(blind_mode=True, grid_size=24)
    kernel._op_map = {
        "mystery_gamma": "boolean_union",
        "mystery_delta": "axial_extrude",
        "mystery_alpha": "box_solid",
        "mystery_omega": "boolean_diff",
        "mystery_beta": "uniform_scale",
    }
    cad_rotor = ContinuousBlindCADRotor()

    r1 = cad_rotor.step_blind_probe(kernel)
    assert r1["op_used"] == "mystery_alpha"  # discovered 0-ary creator purely via arity friction
    assert r1["volume"] > 0.1

    r2 = cad_rotor.step_blind_probe(kernel)
    assert r2["op_used"] == "mystery_omega"  # discovered cavity-carving op purely by surprise!
    assert r2["hollow_voxels"] > 0
    assert r2["emerged_axis"] == "emergent_cavity_topology"
    assert "emergent_cavity_topology" in cad_rotor.param_rotor.axis_names

    # Repeat probing reduces prediction error as (W, M, R) adapt
    r3 = cad_rotor.step_blind_probe(kernel)
    assert r3["error_after"] <= r3["error_before"]
    p_sum = cad_rotor.plasticity_summary()
    assert p_sum["total_plasticity_updates"] >= 6
    assert p_sum["drift_M_norm"] > 0.0


def test_vertical_spine_persists_plasticity_across_restart_and_runs_self_vs_human_audit(tmp_path: Path) -> None:
    def _pattern() -> np.ndarray:
        arr = np.full((240, 320, 3), 22, dtype=np.uint8)
        for y in range(12, 228, 16):
            for x in range(12, 308, 18):
                arr[y : y + 6, x : x + 8, :] = [235, 220, 75]
        return arr

    state_dir = tmp_path / "persisted_spine"
    mind1 = ScaleRotorEmbodiedMind(state_dir)
    spine1 = mind1.get_vertical_spine(max_lock_ticks=3)
    retina1 = ScreenRetinaSensor(frame_provider=_pattern)
    actuator = UniversalMotorActuator(dry_run=True)

    for _ in range(5):
        spine1.step_vertical_cycle(retina=retina1, actuator=actuator, fg_title="audit_test")

    audit1 = spine1.observe_self_vs_human_boundary()
    m1 = audit1["measured"]
    assert m1["cycles_lived"] == 5
    assert m1["total_plasticity_updates"] >= 10
    assert m1["total_drift_M_norm"] > 0.0
    assert m1["current_axis_count"] > m1["birth_axis_count"]
    assert "emergent_cavity_topology" in m1["emergent_axes"]
    assert len(audit1["self_perceived_gaps_ko"]) >= 3
    assert len(audit1["unperceivable_blind_spots_ko"]) >= 3

    # Reload a brand-new mind from disk -> learned (W, M, R), emergent axes, and scale_dial MUST survive!
    mind2 = ScaleRotorEmbodiedMind(state_dir)
    spine2 = mind2.get_vertical_spine(max_lock_ticks=3)
    audit2 = spine2.observe_self_vs_human_boundary()
    assert audit2["measured"]["cycles_lived"] == 5
    assert audit2["measured"]["persisted_scale_dial"] == m1["persisted_scale_dial"]
    assert audit2["measured"]["emergent_axes"] == m1["emergent_axes"]
    assert audit2["measured"]["total_drift_M_norm"] == m1["total_drift_M_norm"]
