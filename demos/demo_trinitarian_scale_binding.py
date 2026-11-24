r"""
ELYSIA Phase 8 Demo: Trinitarian Scale-Binding Protocol & Equality-Manifold Engine
================================================================================

Demonstrates live bidirectional equality/fence coupling across:
1. Variable-Resistor Electrical -> Thermal Circuit (dial flow, IFT sensitivities,
   inverse-causality dial prescription, overheat safety fence regulation, tamper healing)
2. Cellular Mitosis vs Organism Homeostasis ("Digital Cancer": 100% locally valid
   Layer-1 mitosis rules violating Layer-3 vascular/tissue homeostasis fences)
3. Continuous Narrative-Causal Free-Will Worldline (authentic continuity vs
   locally valid contextless mastery-jump exploit)
4. Physical CausalField Beam Tear & Heal projection.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.physics.trinitarian_scale_binding_protocol import (
    build_cellular_homeostasis_system,
    build_narrative_causal_worldline_system,
    build_variable_resistor_circuit_system,
    sync_to_causal_field,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA TRINITARIAN SCALE-BINDING PROTOCOL (PHASE 8) — LEAST-ACTION CONVERGENCE DEMO")
    print("=" * 92)

    # 1. Variable-Resistor Circuit
    _section("1. VARIABLE-RESISTOR CIRCUIT: DIAL FLOW, INVERSE PROOF, OVERHEAT & TAMPER")
    circuit = build_variable_resistor_circuit_system()
    v0 = circuit.observe_and_discern()
    print(f"[Nominal R=4.0] Status={v0.status.value} | I={circuit.quantities['I'].value:.4f}A "
          f"| P={circuit.quantities['P'].value:.4f}W | T_final={circuit.quantities['T_final'].value:.2f}C")

    rep_8 = circuit.turn_dial("R", 8.0)
    v8 = circuit.observe_and_discern()
    analytic_dp_dr = -144.0 / 64.0
    solver_dp_dr = v8.sensitivities["P"]["R"]
    print(f"\n[Dial R -> 8.0] Converged={rep_8.converged} (iters={rep_8.iterations}, "
          f"max_res={rep_8.max_residual:.2e})")
    print(f"  Values: I={circuit.quantities['I'].value:.4f}A, P={circuit.quantities['P'].value:.4f}W, "
          f"T_final={circuit.quantities['T_final'].value:.2f}C")
    print(f"  Sensitivity dP/dR: IFT={solver_dp_dr:.6f} vs Analytic={analytic_dp_dr:.6f}")
    print(v8.explanation)

    _, after_reg = circuit.regulate_to_equilibrium(adjustable_dials=["R"])
    print(f"\n[After Intent Regulation] Status={after_reg.status.value} | "
          f"R={circuit.quantities['R'].value:.4f} Ohm | P={circuit.quantities['P'].value:.4f}W")

    rep_hot = circuit.turn_dial("R", 1.0)
    v_hot = circuit.observe_and_discern()
    print(f"\n[Dial R -> 1.0 Overheat] max_res(L1/L2)={rep_hot.max_residual:.2e} | "
          f"T_final={circuit.quantities['T_final'].value:.2f}C > T_max=85C")
    print(v_hot.explanation)
    _, after_cool = circuit.regulate_to_equilibrium(adjustable_dials=["R"])
    print(f"[After Thermal Regulation] Status={after_cool.status.value} | "
          f"R={circuit.quantities['R'].value:.4f} Ohm | T_final={circuit.quantities['T_final'].value:.2f}C")

    circuit.inject_tamper("I", 10.0)
    v_tamper = circuit.observe_and_discern()
    print(f"\n[Tamper I := 10.0] Status={v_tamper.status.value}")
    print(v_tamper.explanation)

    torn_sync = sync_to_causal_field(circuit, verdict=v_tamper, causal_field=CausalField(dimensions=3), prefix="c_torn")
    torn_sync["causal_field"].step(dt=0.1)

    _, after_heal = circuit.regulate_to_equilibrium()
    print(f"\n[Self-Healed] Status={after_heal.status.value} | I={circuit.quantities['I'].value:.4f}A "
          f"(mode={circuit.quantities['I'].pin_mode.value})")

    healed_sync = sync_to_causal_field(circuit, verdict=after_heal, causal_field=CausalField(dimensions=3), prefix="c_heal")
    healed_sync["causal_field"].step(dt=0.1)
    print(f"CausalField Min Beam Strength: Torn={torn_sync['min_beam_strength']:.4f} "
          f"-> Healed={healed_sync['min_beam_strength']:.4f}")

    # 2. Cellular Mitosis vs Organism Homeostasis ("Digital Cancer")
    _section("2. CELLULAR MITOSIS vs ORGANISM HOMEOSTASIS ('DIGITAL CANCER')")
    bio = build_cellular_homeostasis_system()
    bio.turn_dial("r_div", 0.85)
    lower_bio = bio.observe_lower_perspective()
    v_bio = bio.observe_and_discern()
    print(f"[Oncogenic r_div=0.85] Isolated Lower Subsystem Valid={lower_bio['lower_valid']} "
          f"(max_residual={lower_bio['max_residual']:.2e})")
    print(f"  Cell Count N_next={bio.quantities['N_next'].value:.2f} (cap=125), "
          f"O2_demand={bio.quantities['O2_demand'].value:.2f} (supply=140)")
    print(v_bio.explanation)
    _, after_bio = bio.regulate_to_equilibrium(adjustable_dials=["r_div"])
    print(f"\n[After Homeostasis Regulation] Status={after_bio.status.value} | "
          f"r_div={bio.quantities['r_div'].value:.6f} | "
          f"N_next={bio.quantities['N_next'].value:.2f} | "
          f"O2_demand={bio.quantities['O2_demand'].value:.2f} | "
          f"total_action={after_bio.total_action:.2e}")

    # 3. Narrative-Causal Free-Will Worldline
    _section("3. NARRATIVE-CAUSAL FREE-WILL WORLDLINE vs CONTEXTLESS EXPLOIT")
    story = build_narrative_causal_worldline_system()
    v_story0 = story.observe_and_discern()
    print(f"[Authentic Free-Will Arc] Status={v_story0.status.value} | "
          f"causal_gain={story.quantities['causal_mastery_gain'].value:.2f} | "
          f"S_future={story.quantities['S_future'].value:.2f}")

    story.pin_value("prep_past", 0.5, relax=False)
    story.pin_value("trust_past", 0.5, relax=False)
    story.turn_dial("effort_cost", 4.0, relax=False)
    story.turn_dial("claimed_mastery_jump", 50.0, relax=True)
    lower_story = story.observe_lower_perspective()
    v_story_exploit = story.observe_and_discern(adjustable_dials=["claimed_mastery_jump"])
    print(f"\n[Contextless Claim=+50 Jump] Naive Tick Lower Valid={lower_story['lower_valid']} "
          f"| Causal Readiness Gain={story.quantities['causal_mastery_gain'].value:.2f}")
    print(v_story_exploit.explanation)
    _, after_story = story.regulate_to_equilibrium(adjustable_dials=["claimed_mastery_jump"])
    print(f"[After Narrative Regulation] claimed_mastery_jump={story.quantities['claimed_mastery_jump'].value:.4f} "
          f"| upper_fence_ok={after_story.upper_fence_verdict} "
          f"(gap vs S_future intent stays honestly flagged={not after_story.intent_satisfied})")

    print("\n" + "=" * 92)
    print("  ALL TRINITARIAN SCALE-BINDING VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
