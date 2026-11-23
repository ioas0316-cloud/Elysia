from __future__ import annotations

import json
from pathlib import Path

from core.sensory.reentrant_self_listening_loop_engine import (
    DEFAULT_REENTRANT_DASHBOARD_PATH,
    DEFAULT_REENTRANT_STATE_PATH,
    ReentrantSelfListeningOrganism,
    render_reentrant_self_listening_dashboard,
)


def test_reentrant_self_listening_and_invariant_contrast_ecosystem() -> None:
    """
    Verifies ELYSIA's unbroken Re-entrant Self-Listening & Invariant/Change Contrast Loop
    operating across Speech, Motor Grip, and an Unseen Polarity-Inverting Medium:
      1. Cycle 0 detects mismatch between intent ("why") and own perceived echo ("how it landed").
      2. Cycle 1 holds aligned invariants fixed and nudges ONLY mismatched axes, explicitly
         logging `[Fixed Invariants | Changed Variables -> Observed Echo Delta]`.
      3. Cycle 2 predicts the exact echo BEFORE emitting (prediction_error <= 0.03) and solves
         inverse compensation cleanly (final_gap <= 0.05), including sign-inversion discovery
         on the unseen medium!
      4. Synchronizes inside shared 6D CausalField, persists JSON, and renders 4-panel PNG.
    """
    organism = ReentrantSelfListeningOrganism(state_path=DEFAULT_REENTRANT_STATE_PATH)
    report = organism.run_ecosystem_reentrant_curriculum()

    print("\n=== [ELYSIA Re-entrant Self-Listening Verification Log] ===")
    print(report.summary_ko)
    for st in report.streams:
        print(f"  {st.summary_ko}")

    assert len(report.streams) == 3
    by_id = {s.stream_id: s for s in report.streams}

    for sid in (
        "Stream_A_Speech_SelfListening",
        "Stream_B_Motor_Control",
        "Stream_C_Unseen_Channel",
    ):
        st = by_id[sid]
        assert st.initial_gap >= 0.18
        assert st.final_gap <= 0.05
        assert st.final_prediction_error <= 0.03
        assert "합의 성공" in st.dialogue_summary_ko
        assert len(st.cycles) == 3

        audit_c1 = st.cycles[1].audit
        assert audit_c1 is not None
        assert len(audit_c1.fixed_invariants) >= 2
        assert len(audit_c1.changed_variables) >= 2

    # Specific audit verification on Stream A Speech & Stream C Sign-Inversion
    st_a = by_id["Stream_A_Speech_SelfListening"]
    assert set(st_a.cycles[1].audit.changed_variables) == {"clarity", "warmth"}
    assert set(st_a.cycles[1].audit.fixed_invariants) == {"precision", "firmness", "novelty", "safety"}

    st_c = by_id["Stream_C_Unseen_Channel"]
    # Polarity-inverted precision (+0.60 intent requires negative emission ~ -0.533)
    assert st_c.cycles[-1].emitted_vec[2] < -0.45
    assert abs(st_c.cycles[-1].perceived_echo_vec[2] - 0.60) < 0.02

    # Shared 6D CausalField populated cleanly
    assert report.causal_field_voxels >= 12
    assert report.causal_field_beams >= 12

    # Persistent JSON validation
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(saved["streams"]) == 3

    # Visual Dashboard validation
    dash_path = render_reentrant_self_listening_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_REENTRANT_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
