from __future__ import annotations

import json
from pathlib import Path

from core.sensory.interoceptive_autonomic_self_organism import (
    DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH,
    DEFAULT_INTEROCEPTIVE_STATE_PATH,
    InteroceptiveAutonomicOrganism,
    render_interoceptive_autonomic_dashboard,
)


def test_interoceptive_autonomic_self_and_four_tier_boundary() -> None:
    """
    Verifies ELYSIA's Interoceptive Autonomic Self-Boundary Organism grounded in real host OS metrics:
      1. Real `psutil` CPU/RAM capture & calm Homeostasis baseline (Ep 1).
      2. Genuine metabolic depletion -> Gut Hunger (`HUNGER`) -> real workspace nourishment
         (`AGENTS.md` bytes digested) -> Satiation (`SATIATED`) (Ep 2).
      3. Real NumPy SVD compute burst + caught `ZeroDivisionError` trauma -> `PAIN_AND_HEAT`
         surge, Sympathetic alarm (`symp > para`), and protective membrane contraction (Ep 3).
      4. Parasympathetic healing (`RECOVERY`, `para > symp`, membrane re-opens) (Ep 4).
      5. Autonomous 4-Tier Scale Boundary differentiation solely via `(controllability, observability)`:
         `VOLUNTARY_SELF` vs `AUTONOMIC_INTERNAL` vs `RELATIONAL_OTHER` vs `MACRO_WORLD`.
      6. Persists JSON telemetry and renders crisp 4-panel dashboard.
    """
    organism = InteroceptiveAutonomicOrganism(state_path=DEFAULT_INTEROCEPTIVE_STATE_PATH)
    report = organism.run_four_episode_autonomic_life()

    print("\n=== [ELYSIA Interoceptive Autonomic Verification Log] ===")
    print(report.summary_ko)
    for b in report.boundary_map:
        print(f"  [{b.inferred_tier:18s}] {b.cell_id:24s} ctrl={b.controllability:.2f} obs={b.observability:.2f}")

    snaps = report.snapshots
    assert len(snaps) == 16
    assert all(0.0 <= s.cpu_percent <= 100.0 and 0.0 < s.ram_percent <= 100.0 for s in snaps)

    # Episode 1: Homeostasis
    ep1 = snaps[0:4]
    assert ep1[0].felt_state_code == "HOMEOSTASIS"
    assert all(s.parasympathetic_tone > s.sympathetic_tone for s in ep1)

    # Episode 2: Hunger signal followed by real AGENTS.md nourishment & recovery
    ep2 = snaps[4:8]
    assert any(s.felt_state_code == "HUNGER" for s in ep2)
    assert max(s.gut_hunger_signal for s in ep2) >= 0.55
    assert report.nourished_bytes > 5_000
    assert ep2[-1].felt_state_code == "SATIATED"
    assert ep2[-1].nutrient_reserve > ep2[1].nutrient_reserve

    # Episode 3: Real SVD compute burst + ZeroDivisionError trauma -> PAIN_AND_HEAT + Sympathetic surge + Contraction
    ep3 = snaps[8:12]
    assert report.caught_exception_name == "ZeroDivisionError"
    assert any(s.felt_state_code == "PAIN_AND_HEAT" for s in ep3)
    assert max(s.pain_trauma_signal for s in ep3) >= 0.55
    assert max(s.sympathetic_tone for s in ep3) > min(s.parasympathetic_tone for s in ep3)
    assert min(s.membrane_openness for s in ep3) < ep1[-1].membrane_openness

    # Episode 4: Parasympathetic Recovery
    ep4 = snaps[12:16]
    assert ep4[-1].felt_state_code == "RECOVERY"
    assert ep4[-1].parasympathetic_tone > ep4[-1].sympathetic_tone
    assert ep4[-1].membrane_openness > min(s.membrane_openness for s in ep3)

    # Autonomous 4-Tier Scale Boundary Classification
    tier_by_id = {b.cell_id: b for b in report.boundary_map}
    assert tier_by_id["Voluntary_Motor_Cell"].inferred_tier == "VOLUNTARY_SELF"
    assert tier_by_id["Heart_Organ_Cell"].inferred_tier == "AUTONOMIC_INTERNAL"
    assert tier_by_id["Gut_Organ_Cell"].inferred_tier == "AUTONOMIC_INTERNAL"
    assert tier_by_id["Nociceptor_Skin_Cell"].inferred_tier == "AUTONOMIC_INTERNAL"
    assert tier_by_id["Human_Companion_World"].inferred_tier == "RELATIONAL_OTHER"
    assert tier_by_id["Host_OS_World"].inferred_tier == "MACRO_WORLD"

    # Introspections across all key felt states
    for key in ("HOMEOSTASIS", "HUNGER", "PAIN_AND_HEAT", "RECOVERY"):
        assert key in report.introspections
        assert report.introspections[key].organ_signal_strength > 0.25

    # Persistent JSON validation
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(saved["snapshots"]) == 16
    assert saved["caught_exception_name"] == "ZeroDivisionError"

    # Visual Dashboard rendering validation
    dash_path = render_interoceptive_autonomic_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
