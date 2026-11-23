from __future__ import annotations

import json
from pathlib import Path

from core.sensory.experiential_adaptation_and_play_engine import (
    DEFAULT_EXPERIENTIAL_DASHBOARD_PATH,
    DEFAULT_EXPERIENTIAL_STATE_PATH,
    ExperientialAdaptationOrganism,
    render_experiential_adaptation_dashboard,
)


def test_experiential_transfer_mismatch_branching_and_play_creation() -> None:
    """
    Verifies ELYSIA's living experiential growth ring across Problem-Solving, Failure-Adaptation,
    Transfer Mastery, and Creative Play inside ONE shared `CausalField`:
      1. Enc 1 (Raw Overload) learns first schema `schema_00` (SIMPLE_SUMMARIZE_DIGEST).
      2. Enc 2 (Duplicate Storm) tries `schema_00` first, hits real mismatch, explores and
         branches `schema_01` (DEDUPLICATE_AND_COMPACT) linked backward to `schema_00`.
      3. Enc 3 (Cyclic Lock) fails prior habit, explores and branches `schema_02`
         (SEVER_CYCLIC_REFERENCE_LOCK).
      4. Enc 4 (Unseen Duplicate Storm) solves cleanly on Attempt 1 (no mismatch).
      5. Calm Play recombines `schema_01` + `schema_02` into emergent `schema_03`
         (HYBRID_HYGIENE_PLAY_SYNTHESIS) BEFORE seeing a mixed crisis.
      6. Enc 5 (Unseen Mixed Duplicate + Cyclic Lock Crisis) recalls `schema_03` and
         solves on Attempt 1!
      7. Persists state JSON and renders clean 4-panel dashboard PNG.
    """
    organism = ExperientialAdaptationOrganism(state_path=DEFAULT_EXPERIENTIAL_STATE_PATH)
    report = organism.run_full_experiential_growth_curriculum()

    print("\n=== [ELYSIA Experiential Adaptation & Play Verification Log] ===")
    print(report.summary_ko)
    for e in report.encounters:
        print(
            f"  {e.encounter_id:26s} | attempts={len(e.attempts)} | mismatch={str(e.mismatch_triggered):5s} | "
            f"bloat {e.initial_bloat:.2f}->{e.final_bloat:.2f} | resolved={e.resolved_action_code}"
        )

    encs = {e.encounter_id: e for e in report.encounters}

    # Enc 1: First experience creates schema_00
    e1 = encs["Enc_1_Raw_Overload"]
    assert not e1.mismatch_triggered
    assert len(e1.attempts) == 1
    assert e1.resolved_action_code == "SIMPLE_SUMMARIZE_DIGEST"
    assert e1.final_bloat <= 0.10
    assert e1.new_schema_created_id == "schema_00"

    # Enc 2: Prior schema_00 fails -> mismatch -> explores & branches schema_01
    e2 = encs["Enc_2_Duplicate_Storm"]
    assert e2.mismatch_triggered is True
    assert e2.attempts[0].action_code == "SIMPLE_SUMMARIZE_DIGEST"
    assert e2.attempts[0].mismatch_detected is True
    assert len(e2.attempts) >= 2
    assert e2.resolved_action_code == "DEDUPLICATE_AND_COMPACT"
    assert e2.final_bloat <= 0.15
    assert e2.new_schema_created_id == "schema_01"

    # Enc 3: Cyclic lock triggers mismatch -> explores & branches schema_02
    e3 = encs["Enc_3_Cyclic_Lock"]
    assert e3.mismatch_triggered is True
    assert e3.attempts[0].mismatch_detected is True
    assert e3.resolved_action_code == "SEVER_CYCLIC_REFERENCE_LOCK"
    assert e3.final_bloat <= 0.10
    assert e3.new_schema_created_id == "schema_02"

    # Enc 4: Unseen duplicate flood transfers schema_01 immediately (1st try!)
    e4 = encs["Enc_4_Unseen_Dup_Transfer"]
    assert e4.mismatch_triggered is False
    assert len(e4.attempts) == 1
    assert e4.attempts[0].source_schema_id == "schema_01"
    assert e4.final_bloat <= 0.15

    # Playful Synthesis schema_03 & Enc 5 mixed crisis 1-shot mastery!
    sch_by_id = {s.schema_id: s for s in report.schemas}
    assert len(report.schemas) == 4
    s3 = sch_by_id["schema_03"]
    assert s3.origin_kind == "PLAYFUL_SYNTHESIS"
    assert s3.action_code == "HYBRID_HYGIENE_PLAY_SYNTHESIS"
    assert set(s3.parent_schema_ids) == {"schema_01", "schema_02"}

    e5 = encs["Enc_5_Unseen_Mixed_Crisis"]
    assert e5.mismatch_triggered is False
    assert len(e5.attempts) == 1
    assert e5.attempts[0].source_schema_id == "schema_03"
    assert e5.resolved_action_code == "HYBRID_HYGIENE_PLAY_SYNTHESIS"
    assert e5.final_bloat <= 0.10

    # CausalField materialization check
    for sid in ("schema_00", "schema_01", "schema_02", "schema_03"):
        assert sid in organism.field_sys.causal_field.voxels
    assert report.causal_field_beams >= 9

    # Persistent JSON validation
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(saved["schemas"]) == 4
    assert len(saved["encounters"]) == 5

    # Visual Dashboard validation
    dash_path = render_experiential_adaptation_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_EXPERIENTIAL_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
