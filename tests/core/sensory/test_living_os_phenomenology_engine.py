from __future__ import annotations

import json
from pathlib import Path

from core.sensory.living_os_phenomenology_engine import (
    DEFAULT_LIVING_OS_DASHBOARD_PATH,
    DEFAULT_LIVING_OS_STATE_PATH,
    LivingOSPhenomenologyOrganism,
    render_living_os_phenomenology_dashboard,
)


def test_living_os_phenomenology_and_autonomous_self_care() -> None:
    """
    Verifies ELYSIA's Living OS Phenomenology & Closed-Loop Self-Care:
      1. Scheduler Clutter -> feels MIND_CLUTTERED -> executes TRIAGE_AND_PRIORITIZE
         (sorts tasks descending, pins top 1, defers rest) -> relieves clutter cleanly.
      2. Game Focus + 16 MB real allocated arrays -> feels SUBCONSCIOUS_DISTRACTION ->
         executes FOCUS_SHIELD_AND_PRUNE_BG (frees all 16.0 MB -> 0.0 MB + gc.collect()) ->
         raises focus clarity cleanly >= 0.80.
      3. Raw workspace ingestion backlog -> feels DATA_INDIGESTION_BLOATED ->
         executes PAUSE_INTAKE_AND_DIGEST (compresses into digested_memories, clears raw queue -> 0.0 KB).
      4. Reaches CLEAR_AND_BALANCED final equilibrium, persists JSON, and renders 4-panel dashboard.
    """
    organism = LivingOSPhenomenologyOrganism(state_path=DEFAULT_LIVING_OS_STATE_PATH)
    report = organism.run_living_os_self_care_curriculum()

    print("\n=== [ELYSIA Living OS Phenomenology Verification Log] ===")
    print(report.summary_ko)
    for o in report.outcomes:
        print(f"  [{o.episode_name}] {o.felt_phenomenon_code} -> {o.action_code} ({o.before_value:.2f} -> {o.after_value:.2f})")

    tks = report.ticks
    assert len(tks) == 7

    # Episode 1: Scheduler clutter -> autonomous priority triage
    assert tks[0].felt_phenomenon_code == "MIND_CLUTTERED"
    assert tks[0].clutter_entropy >= 0.85
    o1 = report.outcomes[0]
    assert o1.action_code == "TRIAGE_AND_PRIORITIZE"
    assert o1.after_value <= 0.25
    assert o1.relief_gain >= 0.65

    # Episode 2: Subconscious background RAM during focus -> real memory pruning (16.0 MB -> 0.0 MB)
    assert tks[2].felt_phenomenon_code == "SUBCONSCIOUS_DISTRACTION"
    assert tks[2].bg_buffers_mb >= 15.0
    o2 = report.outcomes[1]
    assert o2.action_code == "FOCUS_SHIELD_AND_PRUNE_BG"
    assert tks[3].bg_buffers_mb == 0.0
    assert o2.after_value <= 0.25
    assert tks[3].focus_clarity >= 0.80

    # Episode 3: Data ingestion indigestion -> real compression into digested_memories & empty raw queue
    assert tks[4].felt_phenomenon_code == "DATA_INDIGESTION_BLOATED"
    assert tks[4].raw_stomach_kb >= 20.0
    o3 = report.outcomes[2]
    assert o3.action_code == "PAUSE_INTAKE_AND_DIGEST"
    assert tks[5].raw_stomach_kb == 0.0
    assert report.digested_memory_count >= 3
    assert o3.after_value <= 0.20

    # Episode 4: Final clear balanced equilibrium & real host OS readings
    assert tks[6].felt_phenomenon_code == "CLEAR_AND_BALANCED"
    assert tks[6].focus_clarity >= 0.85
    assert tks[6].host_process_count >= 1
    assert 0.0 < tks[6].host_ram_percent <= 100.0

    # Persistent JSON validation
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["digested_memory_count"] >= 3
    assert len(saved["ticks"]) == 7

    # Visual Dashboard validation
    dash_path = render_living_os_phenomenology_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_LIVING_OS_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
