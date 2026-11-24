"""
Verification suite for ELYSIA Phase 17:
True Template-Free External Causalization & Self-Conviction Rewiring.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.sensory_orchestra_os import cauchy_refractive_index
from core.orchestration.true_external_causalization import (
    ASTCodeCausalizer,
    ExternalCausalLearner,
    OpenAcademicCausalizer,
    SymbolicDataCausalizer,
    UnsafeASTError,
)
from core.physics.causal_field import CausalField


def test_symbolic_data_causalizer_discovers_three_unrelated_unknown_worlds_without_hardcoding():
    rng = np.random.default_rng(7)

    # 1. Nonlinear damped pendulum angular acceleration
    n = 80
    theta = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-3.0, 3.0, size=n)
    alpha = -9.81 * np.sin(theta) - 0.25 * omega
    op_pend = SymbolicDataCausalizer.discover(
        {"theta": theta, "omega": omega, "alpha": alpha}, target_var="alpha"
    )
    assert op_pend is not None
    assert op_pend.validation_r2 > 0.999
    assert op_pend.mdl_complexity <= 3
    assert set(op_pend.input_vars) == {"theta", "omega"}
    assert abs(float(op_pend.evaluator({"theta": 0.5, "omega": -0.8})) - (-9.81 * math.sin(0.5) - 0.25 * (-0.8))) < 1e-4

    # 2. Lotka-Volterra biological predator-prey dynamics
    prey = rng.uniform(0.5, 5.0, size=n)
    predator = rng.uniform(0.5, 4.0, size=n)
    dprey_dt = 1.2 * prey - 0.4 * prey * predator
    op_eco = SymbolicDataCausalizer.discover(
        {"prey": prey, "predator": predator, "dprey_dt": dprey_dt}, target_var="dprey_dt"
    )
    assert op_eco is not None
    assert op_eco.validation_r2 > 0.999
    assert op_eco.mdl_complexity <= 3
    assert set(op_eco.input_vars) == {"prey", "predator"}

    # 3. Microeconomic inverse demand with irrelevant distractor noise pruned by MDL
    n_eco = 90
    price = rng.uniform(2.0, 20.0, size=n_eco)
    income = rng.uniform(10.0, 100.0, size=n_eco)
    noise_col = rng.normal(0.0, 5.0, size=n_eco)
    demand = 120.0 / price + 0.5 * income
    op_econ = SymbolicDataCausalizer.discover(
        {"price": price, "income": income, "distraction_noise": noise_col, "demand": demand},
        target_var="demand",
    )
    assert op_econ is not None
    assert op_econ.validation_r2 > 0.999
    assert op_econ.mdl_complexity <= 3
    assert "distraction_noise" not in op_econ.input_vars
    assert set(op_econ.input_vars) == {"price", "income"}


def test_ast_code_causalizer_harvests_real_unseen_python_files_and_workspace_module(tmp_path):
    rc_path = tmp_path / "external_circuit.py"
    rc_path.write_text(
        '''import math

def damped_response(time_s: float, tau_s: float, gain: float) -> float:
    rc_voltage = gain * (1.0 - math.exp(-time_s / max(1e-6, tau_s)))
    return rc_voltage
''',
        encoding="utf-8",
    )
    ops = ASTCodeCausalizer.extract_operators_from_file(rc_path)
    assert len(ops) == 1
    op_rc = ops[0]
    assert op_rc.output_var == "rc_voltage"
    got = op_rc.evaluator({"time_s": 2.0, "tau_s": 4.0, "gain": 5.0})
    expected = 5.0 * (1.0 - math.exp(-2.0 / 4.0))
    assert math.isclose(float(got), expected, abs_tol=1e-12)

    # Directly parse real workspace file sensory_orchestra_os.py via AST (zero import execution)
    real_orch_path = Path(r"c:\Elysia\L1_causal_field\core\orchestration\sensory_orchestra_os.py")
    real_ops = ASTCodeCausalizer.extract_operators_from_file(real_orch_path)
    cauchy_op = next((o for o in real_ops if o.output_var == "cauchy_refractive_index"), None)
    assert cauchy_op is not None
    ast_val = cauchy_op.evaluator({"dispersion_strength": 1.35})
    live_val = cauchy_refractive_index(1.35)
    assert float(np.max(np.abs(ast_val - live_val))) < 1e-12

    # Safety gate rejects malicious code strictly
    bad_path = tmp_path / "unsafe_probe.py"
    bad_path.write_text("import os\ndef bad(x):\n    return os.system('echo no')\n", encoding="utf-8")
    ok, _ = ASTCodeCausalizer.validate_file_safety(bad_path)
    assert ok is False
    try:
        ASTCodeCausalizer.extract_operators_from_file(bad_path, strict=True)
        raised = False
    except UnsafeASTError:
        raised = True
    assert raised is True


def test_open_academic_causalizer_learns_brand_new_vocabulary_and_falsifies_wrong_claims():
    text = """
    세포막 전위(membrane_potential)가 임계값(threshold)을 넘으면 활동전위(action_potential)가 치솟는다.
    `action_potential = 0.5 * (1.0 + tanh(4.0 * (membrane_potential - threshold)))`

    가격(price)이 높아질수록 수요량(demand)은 감소한다.
    가격(price)이 높아질수록 수요량(demand)은 증가한다.
    """
    parsed = OpenAcademicCausalizer.extract_from_text(text, source_name="neuro_econ_notes")
    aliases = parsed["aliases"]
    assert aliases["세포막 전위"] == "membrane_potential"
    assert aliases["임계값"] == "threshold"
    assert aliases["활동전위"] == "action_potential"
    assert aliases["가격"] == "price"
    assert aliases["수요량"] == "demand"

    eq_ops = [op for op in parsed["operators"] if op.source_kind == "academic_equation"]
    assert len(eq_ops) == 1
    ap_op = eq_ops[0]
    assert ap_op.output_var == "action_potential"
    assert set(ap_op.input_vars) == {"membrane_potential", "threshold"}
    assert float(ap_op.evaluator({"membrane_potential": 1.5, "threshold": 0.5})) > 0.99

    qual_ops = [op for op in parsed["operators"] if op.source_kind == "academic_qualitative"]
    assert len(qual_ops) == 2

    rng = np.random.default_rng(11)
    p = rng.uniform(2.0, 20.0, size=60)
    inc = rng.uniform(10.0, 100.0, size=60)
    d = 120.0 / p + 0.5 * inc
    data = {"price": p, "income": inc, "demand": d}

    v_true = OpenAcademicCausalizer.verify_qualitative_claim(qual_ops[0], data)
    v_false = OpenAcademicCausalizer.verify_qualitative_claim(qual_ops[1], data)
    assert v_true["verified"] is True and v_true["falsified"] is False
    assert v_false["verified"] is False and v_false["falsified"] is True


def test_self_conviction_engine_commits_only_when_held_out_proof_succeeds_and_rejects_lies(tmp_path):
    learner = ExternalCausalLearner()
    rng = np.random.default_rng(21)
    x = rng.uniform(1.0, 10.0, size=40)
    y = rng.uniform(1.0, 5.0, size=40)
    z_noise = rng.normal(0.0, 5.0, size=40)

    # Lie + pure noise must NOT convince or modify the mind graph
    bad_proof = learner.learn_from_external_sources(
        target_var="z",
        observations={"x": x, "y": y, "z": z_noise},
        academic_texts=[("bad_blog", "`z = 10.0 * x + 999.0`")],
        allow_data_discovery=True,
    )
    assert bad_proof.convinced is False
    assert bad_proof.rewired is False
    assert len(learner.mind.operators) == 0
    assert len(bad_proof.rejected_candidates) >= 1

    # Honest physical signal + matching principle convinces Elysia & permanently rewires mind
    z_true = 2.5 * x - 0.8 * y
    good_proof = learner.learn_from_external_sources(
        target_var="z",
        observations={"x": x, "y": y, "z": z_true},
        academic_texts=[("honest_ref", "`z = 2.5 * x - 0.8 * y`")],
    )
    assert good_proof.convinced is True
    assert good_proof.rewired is True
    assert good_proof.held_out_r2 > 0.999
    assert good_proof.after_rmse < 1e-6
    assert len(learner.mind.operators) == 1


def test_cross_modal_3hop_causal_chaining_code_to_data_to_academic_and_sensitivity(tmp_path):
    rng = np.random.default_rng(33)
    sensor_file = tmp_path / "external_sensor_driver.py"
    sensor_file.write_text(
        '''import math

def calibrate_flux(raw_sensor_mv: float) -> float:
    normalized_flux = 0.5 * math.sqrt(raw_sensor_mv) + 0.2
    return normalized_flux
''',
        encoding="utf-8",
    )

    n = 60
    raw_mv = rng.uniform(1.0, 25.0, size=n)
    flux = 0.5 * np.sqrt(raw_mv) + 0.2
    load = rng.uniform(0.5, 4.0, size=n)
    stress = 3.0 * flux * load + 1.5 * (flux ** 2)
    integrity = 100.0 / (1.0 + 0.25 * stress)

    acad_text = (
        "열응력(thermal_stress)이 커질수록 구조 건전성(structural_integrity)은 감소한다.\n"
        "`구조 건전성(structural_integrity) = 100.0 / (1.0 + 0.25 * thermal_stress)`"
    )

    learner = ExternalCausalLearner()
    stage_specs = [
        {
            "target_var": "normalized_flux",
            "observations": {"raw_sensor_mv": raw_mv, "normalized_flux": flux},
            "code_files": [sensor_file],
            "allow_data_discovery": False,
        },
        {
            "target_var": "thermal_stress",
            "observations": {"normalized_flux": flux, "ambient_load": load, "thermal_stress": stress},
            "allow_data_discovery": True,
        },
        {
            "target_var": "구조 건전성",
            "observations": {"thermal_stress": stress, "구조 건전성": integrity},
            "academic_texts": [("structural_mechanics", acad_text)],
            "allow_data_discovery": False,
        },
    ]

    rep = learner.bridge_multi_hop_goal(
        stage_specs=stage_specs,
        end_to_end_inputs={"raw_sensor_mv": 9.0, "ambient_load": 2.0},
        final_target="구조 건전성",
        sensitivity_input="raw_sensor_mv",
    )

    assert rep.reached_before is False
    assert rep.reached_after is True
    assert all(pr.convinced and pr.rewired for pr in rep.stage_proofs)
    assert len(rep.inference_trace.fired_chain) == 3

    exp_flux = 0.5 * math.sqrt(9.0) + 0.2
    exp_stress = 3.0 * exp_flux * 2.0 + 1.5 * (exp_flux ** 2)
    exp_integrity = 100.0 / (1.0 + 0.25 * exp_stress)
    assert math.isclose(float(rep.inference_trace.target_value), exp_integrity, abs_tol=1e-3)
    assert rep.end_to_end_sensitivity is not None and rep.end_to_end_sensitivity < -1.0


def test_persistent_open_mind_graph_saves_reloads_and_seals_ssd_provenance(tmp_path):
    slab = tmp_path / "open_mind.ice"
    mind_json = tmp_path / "learned_mind.json"
    rng = np.random.default_rng(44)
    sensor_file = tmp_path / "sensor_conv.py"
    sensor_file.write_text(
        "import math\ndef to_flux(raw_sensor_mv):\n    normalized_flux = 0.5 * math.sqrt(raw_sensor_mv) + 0.2\n    return normalized_flux\n",
        encoding="utf-8",
    )

    raw_mv = rng.uniform(1.0, 25.0, size=50)
    flux = 0.5 * np.sqrt(raw_mv) + 0.2
    load = rng.uniform(0.5, 4.0, size=50)
    stress = 3.0 * flux * load + 1.5 * (flux ** 2)
    integrity = 100.0 / (1.0 + 0.25 * stress)

    with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
        cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
        learner = ExternalCausalLearner(living_cycle=cycle)
        learner.bridge_multi_hop_goal(
            stage_specs=[
                {"target_var": "normalized_flux", "observations": {"raw_sensor_mv": raw_mv, "normalized_flux": flux}, "code_files": [sensor_file], "allow_data_discovery": False},
                {"target_var": "thermal_stress", "observations": {"normalized_flux": flux, "ambient_load": load, "thermal_stress": stress}},
                {"target_var": "structural_integrity", "observations": {"thermal_stress": stress, "structural_integrity": integrity}, "academic_texts": [("mechanics", "`구조 건전성(structural_integrity) = 100.0 / (1.0 + 0.25 * thermal_stress)`")], "allow_data_discovery": False},
            ],
            end_to_end_inputs={"raw_sensor_mv": 9.0, "ambient_load": 2.0},
            final_target="구조 건전성",
        )
        assert mem_rt.ice.num_crystals >= 3
        assert cycle.verify_autobiographical_integrity() is True
        learner.save_mind(mind_json)

    # Reboot into fresh learner and run multi-hop inference directly from reloaded ASTs
    reborn = ExternalCausalLearner()
    reloaded_info = reborn.load_mind(mind_json)
    assert reloaded_info["operators"] == 3
    tr = reborn.mind.infer({"raw_sensor_mv": 9.0, "ambient_load": 2.0}, "구조 건전성")
    assert tr.reached is True
    assert math.isclose(float(tr.target_value), 100.0 / (1.0 + 0.25 * (3.0 * 1.7 * 2.0 + 1.5 * 1.7 * 1.7)), abs_tol=1e-3)


def test_dynamic_causal_field_grows_voxels_and_beams_from_learned_external_knowledge(tmp_path):
    learner = ExternalCausalLearner()
    sync_before = learner.sync_mind_to_causal_field(causal_field=CausalField(dimensions=3))
    assert sync_before["concept_count"] == 0
    assert sync_before["operator_count"] == 0
    assert sync_before["beams_linked"] == 0

    rng = np.random.default_rng(55)
    a = rng.uniform(1.0, 10.0, size=40)
    b = 2.0 * a + 1.0
    c = b ** 2 - 3.0 * b
    learner.learn_from_external_sources("b", {"a": a, "b": b}, academic_texts=[("l1", "`b = 2.0 * a + 1.0`")])
    learner.learn_from_external_sources("c", {"b": b, "c": c})

    field = CausalField(dimensions=3)
    sync_after = learner.sync_mind_to_causal_field(causal_field=field)
    assert sync_after["concept_count"] >= 3
    assert sync_after["operator_count"] == 2
    assert sync_after["beams_linked"] >= 4
    for vox in field.voxels.values():
        assert tuple(vox.tensor.shape) == (4,)
    field.step(dt=0.1)


def test_17_organ_architectural_self_audit_recognizes_phase_17():
    cycle = ElysiaUnifiedLivingCycle()
    audit = cycle.introspect_own_architecture()
    assert audit.total_organ_files >= 17
    rel_paths = [m["relative_path"] for m in audit.verified_organ_modules]
    assert "orchestration/true_external_causalization.py" in rel_paths
