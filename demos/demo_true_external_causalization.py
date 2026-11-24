r"""
ELYSIA Phase 17 Demo: Template-Free External Causalization & Self-Conviction Rewiring
=====================================================================================

Demonstrates Elysia expanding her empty open `CausalMindGraph` with ZERO
developer code templates and ZERO hardcoded domain vocabularies:
1. Honest refusal of a false external law & noisy distractor
2. Blind symbolic structure discovery (Pendulum, Ecology, Economics)
3. Real AST extraction from workspace `sensory_orchestra_os.py` & unseen external `.py`
4. Cross-modal 3-hop cognitive-sensory expansion (`code -> data -> academic`)
   + end-to-end sensitivity + physical `CausalField` growth + SSD/SHA-256 & 17-organ audit
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.sensory_orchestra_os import cauchy_refractive_index
from core.orchestration.true_external_causalization import (
    ASTCodeCausalizer,
    ExternalCausalLearner,
    SymbolicDataCausalizer,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA PHASE 17 — TRUE TEMPLATE-FREE EXTERNAL CAUSALIZATION & SELF-CONVICTION")
    print("=" * 92)

    rng = np.random.default_rng(77)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        slab = tmp / "phase17_mind.ice"
        sensor_py = tmp / "unseen_sensor_driver.py"
        sensor_py.write_text(
            "import math\n"
            "def calibrate_flux(raw_sensor_mv: float) -> float:\n"
            "    normalized_flux = 0.5 * math.sqrt(raw_sensor_mv) + 0.2\n"
            "    return normalized_flux\n",
            encoding="utf-8",
        )

        # 1. Blind structure-agnostic discovery across 3 unseen domains
        _section("1. BLIND STRUCTURE-AGNOSTIC DISCOVERY (ZERO TEMPLATES)")
        th = rng.uniform(-math.pi, math.pi, 80)
        om = rng.uniform(-3.0, 3.0, 80)
        al = -9.81 * np.sin(th) - 0.25 * om
        p1 = SymbolicDataCausalizer.discover({"theta": th, "omega": om, "alpha": al}, "alpha")
        print(f"  Pendulum  : {p1.formula_str} (val R2={p1.validation_r2:.6f}, k={int(p1.mdl_complexity)})")

        prey = rng.uniform(0.5, 5.0, 80)
        pred = rng.uniform(0.5, 4.0, 80)
        dp = 1.2 * prey - 0.4 * prey * pred
        p2 = SymbolicDataCausalizer.discover({"prey": prey, "predator": pred, "dprey_dt": dp}, "dprey_dt")
        print(f"  Ecology   : {p2.formula_str} (val R2={p2.validation_r2:.6f}, k={int(p2.mdl_complexity)})")

        pr = rng.uniform(2.0, 20.0, 90)
        inc = rng.uniform(10.0, 100.0, 90)
        junk = rng.normal(0.0, 5.0, 90)
        dem = 120.0 / pr + 0.5 * inc
        p3 = SymbolicDataCausalizer.discover({"price": pr, "income": inc, "distractor": junk, "demand": dem}, "demand")
        print(f"  Economics : {p3.formula_str} (val R2={p3.validation_r2:.6f}, kept_vars={p3.input_vars})")

        # 2. Direct Pure AST Extraction from Real Workspace Module
        _section("2. DIRECT AST HARVESTING OF REAL WORKSPACE SOURCE")
        orch_path = Path(r"c:\Elysia\L1_causal_field\core\orchestration\sensory_orchestra_os.py")
        real_ops = ASTCodeCausalizer.extract_operators_from_file(orch_path)
        cauchy_op = next(o for o in real_ops if o.output_var == "cauchy_refractive_index")
        ast_out = cauchy_op.evaluator({"dispersion_strength": 1.35})
        live_out = cauchy_refractive_index(1.35)
        print(f"  AST Match : {cauchy_op.formula_str}")
        print(f"  Max Error : {float(np.max(np.abs(ast_out - live_out))):.2e} against live implementation")

        # 3. Deficit -> Counterfactual Lie Refusal -> Cross-Modal 3-Hop Conviction & Rewiring
        _section("3. DEFICIT -> HONEST REFUSAL -> CROSS-MODAL 3-HOP SELF-CONVICTION")
        with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
            cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
            learner = ExternalCausalLearner(living_cycle=cycle)

            # Reject a lie over unexplainable noise
            bad_p = learner.learn_from_external_sources(
                "z",
                {"x": rng.uniform(1, 10, 40), "z": rng.normal(0, 5, 40)},
                academic_texts=[("rumor", "`z = 10.0 * x + 999.0`")],
            )
            print(f"  False Law : convinced={bad_p.convinced}, rewired={bad_p.rewired}")
            print(f"              {bad_p.conviction_statement_ko}")

            raw_mv = rng.uniform(1.0, 25.0, 60)
            flux = 0.5 * np.sqrt(raw_mv) + 0.2
            load = rng.uniform(0.5, 4.0, 60)
            stress = 3.0 * flux * load + 1.5 * (flux ** 2)
            integ = 100.0 / (1.0 + 0.25 * stress)

            acad_doc = (
                "열응력(thermal_stress)이 커질수록 구조 건전성(structural_integrity)은 감소한다.\n"
                "`구조 건전성(structural_integrity) = 100.0 / (1.0 + 0.25 * thermal_stress)`"
            )
            rep = learner.bridge_multi_hop_goal(
                stage_specs=[
                    {"target_var": "normalized_flux", "observations": {"raw_sensor_mv": raw_mv, "normalized_flux": flux}, "code_files": [sensor_py], "allow_data_discovery": False},
                    {"target_var": "thermal_stress", "observations": {"normalized_flux": flux, "ambient_load": load, "thermal_stress": stress}},
                    {"target_var": "구조 건전성", "observations": {"thermal_stress": stress, "구조 건전성": integ}, "academic_texts": [("mechanics", acad_doc)], "allow_data_discovery": False},
                ],
                end_to_end_inputs={"raw_sensor_mv": 9.0, "ambient_load": 2.0},
                final_target="구조 건전성",
                sensitivity_input="raw_sensor_mv",
            )
            print(f"  3-Hop Summary : {rep.summary_ko}")
            print(f"  Prediction    : structural_integrity(9mV, load=2) = {float(rep.inference_trace.target_value):.4f}")
            print(f"  Sensitivity   : d(integrity)/d(raw_sensor_mv) = {rep.end_to_end_sensitivity:.4f}")
            for i, sp in enumerate(rep.stage_proofs, start=1):
                print(f"  Stage {i} Proof : {sp.conviction_statement_ko}")

            # 4. Dynamic CausalField Growth + Persistence + 17-Organ Self-Audit
            _section("4. CAUSAL FIELD EXPANSION & 17-ORGAN SELF-AUDIT")
            field = CausalField(dimensions=3)
            sync = learner.sync_mind_to_causal_field(causal_field=field)
            field.step(dt=0.1)
            audit = cycle.introspect_own_architecture()
            print(
                f"  CausalField   : {sync['concept_count']} Concepts, {sync['operator_count']} Operators, "
                f"{sync['beams_linked']} Beams, dt=0.1 OK"
            )
            print(
                f"  Persistence   : SSD Crystals={mem_rt.ice.num_crystals}, Provenance Valid={cycle.verify_autobiographical_integrity()}"
            )
            print(
                f"  Self-Audit    : {audit.total_organ_files} Organ Files | {audit.total_organ_classes} Classes | {audit.total_organ_lines} Lines"
            )


if __name__ == "__main__":
    run_demo()
