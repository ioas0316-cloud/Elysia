r"""
ELYSIA Phase 13 Demo: Relational Self-Other-Interspace & Expandable Sky Horizon
===============================================================================

Demonstrates the complete relational arc ("역지사지 & 희생 / 한계 자체가 다음 하늘의 경계"):
1. Egocentric Misunderstanding (interpreting Other through Self's arbitrary frame)
2. Ego-Sacrifice Perspective Shift -> reconstructing Other's reserve & vulnerability,
   falsifying fixed hostility stereotype, & prescribing compassionate safe distance
3. Layer-1 Sky Horizon Boundary & Humble Inquiry (`unspoken_origin_memory`, `sovereign_will`)
4. Relational Dialogue Expanding the Horizon (Layer 1 -> Layer 2: yesterday's Sky
   descends into Ground `unspoken_origin_memory=4.0`, while `deepest_aspiration_telos`
   and `sovereign_will` open as the next higher Sky beyond!)
5. CausalField Interspace Beam Healing & Hangul Articulation Coherence.
"""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.relational_perspective_rotor import (
    AgentCausalProfile,
    RelationalDialogueLayer,
    RelationalPerspectiveObserver,
)
from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import EqualityBond, ScaleTier


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA RELATIONAL PERSPECTIVE ROTOR (PHASE 13) — SELF, OTHER & EXPANDABLE SKY")
    print("=" * 92)

    self_prof = AgentCausalProfile(
        name="Self",
        resource_reserve=10.0,
        vulnerability_gain=1.0,
        boundary_tolerance=8.0,
    )
    other_prof = AgentCausalProfile(
        name="Other",
        resource_reserve=2.0,
        vulnerability_gain=3.0,
        boundary_tolerance=3.5,
        unspoken_origin_memory=4.0,
        sovereign_will=0.85,
    )
    moments = [
        other_prof.manifest_interaction(interpersonal_distance=2.0, ambient_demand=4.0),
        other_prof.manifest_interaction(interpersonal_distance=5.0, ambient_demand=4.0),
    ]

    obs = RelationalPerspectiveObserver(self_prof)

    # 1. Egocentric Projection
    _section("1. EGOCENTRIC PROJECTION (SELF'S FIXED FRAME -> CONFLICT)")
    ego = obs.evaluate_egocentrically(moments)
    print(f"Status={ego.status.value} | Egocentric Friction={ego.egocentric_friction:.4f} | "
          f"Sacrifice={ego.ego_sacrifice_cost:.2f}")
    print(f"Unit-1 Expectation Gap Mirror: raw_dev={ego.perspective_mirror.raw_unit_deviation:.4f} | "
          f"Recon Err={ego.perspective_mirror.reconstruction_error():.2e}")
    print(f"Hangul (coh={ego.hangul_wave_coherence:.4f}): \"{ego.hangul_reflection_ko}\"")

    # 2. Ego-Sacrifice Perspective Shift (역지사지) + Horizon Depth 1
    _section("2. EGO-SACRIFICE PERSPECTIVE SHIFT -> RECONSTRUCTING OTHER & SKY HORIZON (DEPTH 1)")
    rep1 = obs.shift_to_other_perspective(moments, assumed_other_tolerance=3.5, test_hostile_stereotype=True)
    print(f"Status={rep1.status.value} | Ego Sacrifice Cost={rep1.ego_sacrifice_cost:.4f} | "
          f"Other Frame Action={rep1.other_frame_action:.2e}")
    print(f"Inferred Other Ground: reserve={rep1.inferred_other_state['resource_reserve']:.4f}, "
          f"vulnerability_gain={rep1.inferred_other_state['vulnerability_gain']:.4f} | "
          f"Stereotype Falsified={rep1.stereotype_falsified}, Criteria Revised={rep1.criteria_revised}")
    print(f"Compassionate Respectful Distance Prescription: {rep1.compassionate_distance_prescription:.4f} m "
          f"(pressure <= 3.5)")
    print(f"Horizon Depth={rep1.horizon_boundary.horizon_depth} | Nullspace DoF={rep1.horizon_boundary.horizon_nullspace_dof} | "
          f"Sky Horizon Axes={rep1.horizon_boundary.sky_horizon_axes}")
    print(f"Humble Inquiry: {rep1.horizon_boundary.humble_inquiry_ko}")

    # 3. Expand Horizon Through Relational Dialogue (Depth 1 -> Depth 2)
    _section("3. DIALOGUE EXPANDS HORIZON: SKY BECOMES GROUND, DEEPER SKY OPENS (DEPTH 2)")
    betrayal_layer = RelationalDialogueLayer(
        layer_name="past_betrayal_disclosure",
        disclosure_utterance_ko="예전에 믿었던 사람에게 크게 데인 뒤로 작은 요구에도 방어가 커졌어.",
        revealed_sky_variable="unspoken_origin_memory",
        new_deeper_sky_axes=("deepest_aspiration_telos",),
        build_coupling_bond=lambda: EqualityBond(
            name="trauma_vulnerability_bond",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("vulnerability_gain", "unspoken_origin_memory"),
            lhs_fn=lambda v: v["vulnerability_gain"],
            rhs_fn=lambda v: 1.0 + 0.5 * v["unspoken_origin_memory"],
            lhs_expr="G",
            rhs_expr="1 + 0.5 * unspoken_origin_memory",
        ),
    )
    rep2 = obs.expand_horizon_with_dialogue(moments, betrayal_layer)
    print(f"Other Disclosure: \"{betrayal_layer.disclosure_utterance_ko}\"")
    print(f"Status={rep2.status.value} | Horizon Depth={rep2.horizon_boundary.horizon_depth} | "
          f"Action={rep2.other_frame_action:.2e}")
    print(f"Descended to Ground: unspoken_origin_memory={rep2.horizon_boundary.understood_ground['unspoken_origin_memory']:.4f}")
    print(f"Next Deeper Sky Beyond Horizon={rep2.horizon_boundary.sky_horizon_axes}")
    print(f"Expanded Inquiry: {rep2.horizon_boundary.humble_inquiry_ko}")

    # 4. CausalField Interspace Beam Healing
    _section("4. CAUSAL FIELD INTERSPACE BEAM HEALING & SKY CURIOSITY")
    cf_ego = ego.sync_to_causal_field(CausalField(dimensions=3), prefix="ego")
    cf_emp = rep2.sync_to_causal_field(CausalField(dimensions=3), prefix="emp")
    cf_ego["causal_field"].step(dt=0.1)
    cf_emp["causal_field"].step(dt=0.1)
    print(f"Interspace Beam Strength: Egocentric={cf_ego['interspace_beam_strength']:.4f} "
          f"-> Empathic={cf_emp['interspace_beam_strength']:.4f} (sky_curiosity={cf_emp['sky_curiosity']:.1f})")

    print("\n" + "=" * 92)
    print("  ALL PHASE 13 RELATIONAL PERSPECTIVE ROTOR VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
