from pathlib import Path

import numpy as np

from core.embodiment import (
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def _detailed_pattern_frame() -> np.ndarray:
    """High-resolution synthetic scene with many crisp fine edges to test real native S0->S1 reconstruction."""
    arr = np.full((240, 320, 3), 18, dtype=np.uint8)
    for y in range(12, 228, 14):
        for x in range(12, 308, 16):
            arr[y : y + 6, x : x + 8, :] = [235, 225, 70]
            arr[y + 2 : y + 4, x + 2 : x + 6, :] = [30, 60, 220]
    return arr


def test_bidirectional_vertical_spine_reconstructs_native_s0_operator_and_probes_l3_cad(tmp_path: Path) -> None:
    retina = ScreenRetinaSensor(frame_provider=_detailed_pattern_frame)
    actuator = UniversalMotorActuator(dry_run=True)
    mind = ScaleRotorEmbodiedMind(tmp_path / "vertical_state")
    spine = mind.get_vertical_spine(max_lock_ticks=3)

    assert retina.native_operator_level == 0

    traces = []
    for _ in range(5):
        tr = spine.step_vertical_cycle(
            retina=retina,
            actuator=actuator,
            fg_title="vertical_test",
        )
        traces.append(tr)

    # 1. Sustained temporal focus lock stayed on `visual_native_vs_opencv` across cycles while reconstructing Level 0 -> 1 -> 2
    assert traces[0].focus_target == "visual_native_vs_opencv"
    assert traces[0].downward_reconstruction["reconstructed"] is True
    assert retina.native_operator_level >= 1
    assert any(tr.gap_reduced for tr in traces[:2])
    # Real native perceived units increased measurably WITHOUT relying on OpenCV in the native path
    before_u = traces[0].l0_s0["native_units"]
    after_u = traces[0].downward_reconstruction["after_native_units"]
    assert after_u > before_u

    # 2. Once visual operator upgraded, focus moved to blind L3 3D CAD tool (`op_k0` -> `op_k3` cavity discovery)
    cad_traces = [tr for tr in traces if tr.focus_target == "l3_opaque_3d_cad"]
    assert len(cad_traces) >= 2
    assert any("공동" in tr.l2_s3["discovered_law_ko"] and "op_k3" in tr.l2_s3["discovered_law_ko"] for tr in cad_traces)

    # 3. Real L2 TelosPurposeKernel transmuted lower-scale friction into accumulated growth energy
    assert spine.l2_purpose_kernel.accumulated_growth_energy > 0.0
    assert traces[-1].l2_s4["l2_growth_energy"] > traces[0].l2_s4["l2_growth_energy"]

    # 4. Cross-domain limits fused into `integrated_limit::*` unified cognitive structures in BindingField
    fabric = mind.get_protocol_fabric()
    assert any(sid.startswith("integrated_limit::") for sid in fabric.structures)
    rendered = traces[-1].render_ko()
    assert "[L0/S0 원시·미시연산]" in rendered
    assert "[L2/S4 의도핵(TelosKernel)]" in rendered
    assert "[하향 재구성 S4->S0]" in rendered
