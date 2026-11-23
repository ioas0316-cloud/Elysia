from __future__ import annotations

import json
from pathlib import Path

from core.sensory.multimodal_dial_balance_inference_engine import (
    DEFAULT_DIAL_DASHBOARD_PATH,
    DEFAULT_DIAL_STATE_PATH,
    MultimodalDialBalanceEngine,
    render_multimodal_dial_dashboard,
)


def test_multimodal_variable_resistor_dial_cross_inference_and_scaling() -> None:
    """
    Verifies ELYSIA's coupled 6-Dial Potentiometer Equilibrium Network inside ONE `CausalField`:
      1. Blind Vision (`m0_vision` masked) reconstructs accurately across 4 noisy unseen
         phenomena solely from `(Touch, Hearing, Thermal, Language, Tool)` (`MAE <= 0.16`, 100% sign).
      2. Combinatorial error collapse as coupled dials grow `k = 1 -> 5`: 5 coupled dials crush
         ambiguity & reconstruction error sharply compared to 1 isolated dial.
      3. Double-Blind Unseen Live Copper Wire (`Vision` AND `Touch` BOTH masked) reconstructs both
         hidden modalities accurately from `(Hearing, Thermal, Language, Tool)` (`MAE <= 0.18`, 100% sign).
      4. Persists state JSON and renders clean 4-panel dashboard PNG.
    """
    engine = MultimodalDialBalanceEngine(state_path=DEFAULT_DIAL_STATE_PATH)
    report = engine.run_multimodal_dial_curriculum()

    print("\n=== [ELYSIA Multimodal Variable-Resistor Dial Verification Log] ===")
    print(report.summary_ko)

    # Experiment 1: Blind Vision reconstruction across all 4 noisy unseen variants
    assert len(report.blind_vision_results) == 4
    for r in report.blind_vision_results:
        assert r.sign_accuracy == 1.0
        assert r.mean_abs_error <= 0.16

    # Experiment 2: Steep combinatorial error & uncertainty reduction from k=1 -> k=5
    curve = report.scaling_curve
    assert len(curve) == 5
    k1 = curve[0]
    k5 = curve[-1]
    assert k1.mean_abs_error >= 0.28
    assert k5.mean_abs_error <= 0.10
    assert k5.mean_abs_error < 0.20 * k1.mean_abs_error
    assert k5.mean_uncertainty < k1.mean_uncertainty
    assert k5.combinatorial_pairs == 10

    # Experiment 3: Double-Blind hidden live wire reconstruction
    db = report.double_blind_result
    assert db.blind_modalities == ["m0_vision", "m1_touch"]
    assert db.sign_accuracy == 1.0
    assert db.mean_abs_error <= 0.18
    assert db.inferred_blind_values["m0_vision"] < -0.50
    assert db.inferred_blind_values["m1_touch"] < -0.35

    # Shared 6D CausalField & persistent JSON check
    assert report.causal_field_voxels == 12  # 6 modality cells + 6 lived anchors
    assert report.causal_field_beams >= 40
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(saved["scaling_curve"]) == 5
    assert len(saved["blind_vision_results"]) == 4

    # Visual Dashboard verification
    dash_path = render_multimodal_dial_dashboard(
        engine=engine,
        report=report,
        output_path=DEFAULT_DIAL_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
