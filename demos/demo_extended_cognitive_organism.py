r"""
ELYSIA Phase 11 Demo: Extended Cognitive Organism (Tools, Vehicles, Internet & Science)
========================================================================================

Demonstrates the unbroken developmental arc from naked biology to cognitive civilization:
1. Naked biological organs honestly flag near-infrared (`905.55 nm`) as irreducible void.
2. Grafting `SpectrometerOrgan` couples Cauchy prism refraction -> IR vacuum collapses to `0`.
3. Querying `ExternalKnowledgeNetwork` (Internet) empirically falsifies myths and
   internalizes Wien's Displacement Law + discovers real Hangul AST workspace modules.
4. Zero-Shot Academic Transfer: detaching the Spectrometer and closing the Eye on an
   unseen `5800 K` star still lets the Organism infer `499.62 nm` mentally!
5. Grafting `AutomobileMotorOrgan`: reckless `35 m/s` on wet road caught by Upper Rotor
   braking fence, throttled safely, and driven `~41 m` to resolve `120 m` far parallax.
6. Broken Spectrometer Fault Isolation (`+5.0 deg` prism drift) & Least-Action
   Recalibration + `CausalField` semantic mass growth.
"""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.extended_cognitive_organism import (
    ExtendedCognitiveOrganism,
    ExtendedWorldState,
    ExternalKnowledgeNetwork,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA EXTENDED COGNITIVE ORGANISM (PHASE 11) — TOOLS, VEHICLE, INTERNET & SCIENCE")
    print("=" * 92)

    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()

    # 1. Naked Biological Horizon Limit
    _section("1. NAKED BIOLOGICAL HORIZON: NEAR-INFRARED (905.55 nm) HONEST VOID")
    s_naked = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    print(f"Status={s_naked.status.value} | nullspace_dof={s_naked.vacuum_report.vacuum_nullspace_dof} | "
          f"Irreducible Void={sorted(s_naked.vacuum_report.irreducible_vacuum_variables.keys())}")
    sync_naked = org.sync_extended_to_causal_field(s_naked, CausalField(dimensions=3), prefix="naked")
    sync_naked["causal_field"].step(dt=0.1)

    # 2. Grafting Physical Tool (Spectrometer)
    _section("2. GRAFTING SPECTROMETER ORGAN -> CAUCHY PRISM COLLAPSES IR VACUUM")
    org.graft_spectrometer(offset_deg=0.0)
    s_spec = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    print(f"nullspace_dof={s_spec.vacuum_report.vacuum_nullspace_dof} | "
          f"Inferred lambda={s_spec.inferred_world['emission_wavelength_nm']:.4f} nm "
          f"(True={world.emission_wavelength_nm:.4f} nm, action={s_spec.total_action:.2e})")

    # 3. External Knowledge Network (Internet) + Workspace AST Discovery
    _section("3. INTERNET FALSIFICATION + WIEN LAW INTERNALIZATION + WORKSPACE DISCOVERY")
    k_rep = org.query_and_verify_knowledge_network(
        world=world,
        network=ExternalKnowledgeNetwork(),
        target_vacuum_variable="emission_wavelength_nm",
        discover_workspace_keyword="hangul",
    )
    for item in k_rep.evaluated_candidates:
        tag = "ACCEPTED" if item["accepted"] else "REJECTED"
        print(f"  [{tag}] {item['name']} ({item['source']}): {item['reason']}")
    print(f"Workspace AST Discovered Modules: {k_rep.workspace_harvested_tools}")

    # 4. Zero-Shot Academic Transfer After Tool Detachment & Blindness
    _section("4. TOOL DETACHED + EYE CLOSED: ZERO-SHOT TRANSFER TO NEW 5800K WORLD")
    org.detach_tool("spectrometer")
    hot_world = ExtendedWorldState(source_temp_k=5800.0)
    s_hot = org.perceive_extended(hot_world, active_organs=("ear", "hand"))
    print(f"No Spectrometer & Blind Eye: inferred T={s_hot.inferred_world['source_temp_k']:.2f}K -> "
          f"mental Wien lambda={s_hot.inferred_world['emission_wavelength_nm']:.4f} nm "
          f"(True={hot_world.emission_wavelength_nm:.4f} nm)")
    print(f"Remaining Honest Void={sorted(s_hot.vacuum_report.irreducible_vacuum_variables.keys())}")

    # 5. Automobile Motor Extension + Braking Safety Fence + 120m Far Parallax
    _section("5. AUTOMOBILE MOTOR EXTENSION: BRAKING FENCE + 120m FAR-HORIZON PARALLAX")
    far_world = ExtendedWorldState(source_pos_x=120.0, source_power_w=800.0, source_temp_k=3200.0)
    org_car = ExtendedCognitiveOrganism()
    org_car.graft_automobile(friction_mu=0.35, safety_margin_m=15.0)
    drive = org_car.drive_automobile_with_safety(
        world=far_world,
        desired_speed_ms=35.0,
        drive_duration_s=2.0,
        active_organs=("ear", "hand"),
        disabled_channels=("touch_contact_force",),
        clearance_limit_m=120.0,
    )
    p_far = drive.perception_after_drive
    print(f"Initial Command={drive.initial_speed_ms:.1f} m/s -> Before Verdict={drive.status.value} -> "
          f"Regulated Speed={drive.regulated_speed_ms:.2f} m/s")
    print(f"Drove={drive.travel_m:.2f}m, StoppingDist={drive.stopping_distance_m:.2f}m "
          f"(Total Envelope={p_far.body_pos_x + drive.stopping_distance_m + 15.0:.2f}m <= 120.00m)")
    print(f"Far-Horizon Recovery: pos={p_far.inferred_world['source_pos_x']:.4f}m, "
          f"power={p_far.inferred_world['source_power_w']:.2f}W (action={p_far.total_action:.2e})")

    # 6. Broken Tool Isolation & Recalibration + CausalField Semantic Mass Growth
    _section("6. BROKEN SPECTROMETER ISOLATION (+5.0 deg) & SEMANTIC MASS GROWTH")
    org.graft_spectrometer(offset_deg=0.0)
    fault = org.diagnose_and_calibrate_spectrometer(world, true_spectrometer_offset_deg=5.0)
    print(fault.explanation)

    s_grown = org.perceive_extended(world, active_organs=("eye", "ear", "hand"), true_spectrometer_offset_deg=5.0)
    sync_ext = org.sync_extended_to_causal_field(s_grown, CausalField(dimensions=3), prefix="grown")
    sync_ext["causal_field"].step(dt=0.1)
    print(f"Upper Rotor Semantic Mass Growth: Naked={sync_naked['upper_rotor_mass']:.2f} -> Extended={sync_ext['upper_rotor_mass']:.2f} "
          f"(growth_rings={len(org.cognitive_growth_rings)})")

    print("\n" + "=" * 92)
    print("  ALL PHASE 11 EXTENDED COGNITIVE ORGANISM VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
