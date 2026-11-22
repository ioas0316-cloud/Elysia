from __future__ import annotations

from pathlib import Path

from core.sensory.autopoietic_organ_genesis import SafeASTSandbox
from core.sensory.autopoietic_principle_discovery_engine import (
    AutopoieticPrincipleDiscoveryEngine,
    ClutteredVFXActionWorld,
    CompositeVFXOccludedWorld,
    DEFAULT_PRINCIPLE_DASHBOARD_PATH,
    IrreducibleNoiseWorld,
    NonlinearDragEngineWorld,
    OccludedMomentumWorld,
    OpaqueEngineTickWorld,
    render_principle_discovery_dashboard,
)


def test_elysia_autopoietic_self_discovery_of_missing_principles() -> None:
    """
    Verifies that ELYSIA uses ONE unified self-discovery loop (`confront_unknown_world`)
    to diagnose empirical friction on unseen worlds, synthesize candidate Python organs in
    `SafeASTSandbox`, falsify broken/overfit hypotheses via MDL & out-of-sample tests,
    hot-swap genuinely discovered laws into `CausalField`, and refuse irreducible noise.
    """
    engine = AutopoieticPrincipleDiscoveryEngine()

    # 4 Unlabeled Blind Worlds (completely distinct seeds / OOD amplitudes on unseen splits)
    w_vfx = ClutteredVFXActionWorld()
    r1 = engine.confront_unknown_world(
        world_id="Trial1_VFX_Clutter_Action",
        train_split=w_vfx.rollout(length=110, seed=101),
        unseen_split=w_vfx.rollout(length=110, seed=202),
    )

    w_occ = OccludedMomentumWorld()
    r2 = engine.confront_unknown_world(
        world_id="Trial2_Occluded_Momentum",
        train_split=w_occ.rollout(length=120, x0=0.14, v0=0.032, occlude_threshold=0.44),
        unseen_split=w_occ.rollout(length=120, x0=0.17, v0=-0.026, occlude_threshold=0.44),
    )

    w_eng = OpaqueEngineTickWorld()
    r3 = engine.confront_unknown_world(
        world_id="Trial3_Opaque_Engine_Law",
        train_split=w_eng.rollout(length=90, seed=301, action_scale=1.0),
        unseen_split=w_eng.rollout(length=90, seed=777, action_scale=1.75),
    )

    w_noise = IrreducibleNoiseWorld()
    r4 = engine.confront_unknown_world(
        world_id="Trial4_Irreducible_Noise",
        train_split=w_noise.rollout(length=120, seed=404),
        unseen_split=w_noise.rollout(length=120, seed=808),
    )

    # Trial 5: Composite VFX Clutter + Visual Occlusion (tests Autonomous Recombination)
    w_comp = CompositeVFXOccludedWorld()
    r5 = engine.confront_unknown_world(
        world_id="Trial5_Composite_VFX_Occlusion",
        train_split=w_comp.rollout(length=120, x0=0.14, v0=0.032, occlude_threshold=0.46, seed=505, expose_mask_for_eval=False),
        unseen_split=w_comp.rollout(length=120, x0=0.17, v0=-0.026, occlude_threshold=0.46, seed=909, expose_mask_for_eval=True),
    )

    # Trial 6: Nonlinear Aerodynamic Drag (tests Template-Free Atomic Operator Discovery)
    w_nl = NonlinearDragEngineWorld()
    r6 = engine.confront_unknown_world(
        world_id="Trial6_Nonlinear_Drag_Atomic",
        train_split=w_nl.rollout(length=100, seed=601, action_scale=1.0),
        unseen_split=w_nl.rollout(length=100, seed=999, action_scale=1.80),
    )

    results = [r1, r2, r3, r4, r5, r6]

    print("\n=== [ELYSIA Autopoietic Structural Principle Discovery Log] ===")
    for res in results:
        print(
            f"\n[{res.world_id}] status={res.epistemic_status} | "
            f"base_mse={res.baseline_error:.6f} -> final_mse={res.final_error:.6f} "
            f"(reduction={res.error_reduction_ratio * 100:.1f}%) | organ={res.discovered_organ_name}"
        )
        print(
            f"  Diagnosis: {res.diagnosis.diagnosis_tag} "
            f"(lag1_autocorr={res.diagnosis.residual_autocorr_lag1:.3f}, "
            f"clutter_ratio={res.diagnosis.spatial_clutter_ratio:.3f})"
        )
        for att in res.attempts:
            print(
                f"    - {att.organ_name:34s} | {att.status:16s} | ast={att.ast_nodes:3d} | "
                f"train_mse={att.train_error:.6f} | unseen_mse={att.unseen_error:.6f} | mdl={att.mdl_score:.6f}"
            )
        if "occlusion_error_reduction" in res.extra_metrics:
            print(
                f"  Occlusion Zone Error: {res.extra_metrics['baseline_occlusion_mse']:.6f} -> "
                f"{res.extra_metrics['final_occlusion_mse']:.6f} "
                f"(-{res.extra_metrics['occlusion_error_reduction'] * 100:.1f}%)"
            )

    dashboard_path = render_principle_discovery_dashboard(
        results, output_path=Path(DEFAULT_PRINCIPLE_DASHBOARD_PATH)
    )
    assert dashboard_path.exists() and dashboard_path.stat().st_size > 10_000

    # --- Assertions on Trial 1: AAA Game VFX Clutter vs Agency Coherence ---
    assert r1.epistemic_status == "PRINCIPLE_DISCOVERED_AND_MOUNTED"
    assert r1.discovered_organ_name == "L2_action_coherence_layer_filter"
    assert r1.error_reduction_ratio > 0.60
    assert r1.discovered_source_code is not None
    assert SafeASTSandbox.validate_ast(r1.discovered_source_code)[0] is True
    l1_vfx_attempts = [a for a in r1.attempts if a.organ_name == "L1_temporal_diff_filter"]
    assert len(l1_vfx_attempts) == 1
    assert l1_vfx_attempts[0].status in {"FALSIFIED_ERROR", "FALSIFIED_MDL"}

    # --- Assertions on Trial 2: Occluded Turning Point + Momentum Memory ---
    assert r2.epistemic_status == "PRINCIPLE_DISCOVERED_AND_MOUNTED"
    assert r2.discovered_organ_name == "L2_phase_delay_momentum_memory"
    assert r2.extra_metrics["occlusion_error_reduction"] > 0.70
    assert r2.discovered_source_code is not None
    assert SafeASTSandbox.validate_ast(r2.discovered_source_code)[0] is True

    # --- Assertions on Trial 3: Black-Box Engine Law Inverse Discovery ---
    assert r3.epistemic_status == "PRINCIPLE_DISCOVERED_AND_MOUNTED"
    assert r3.discovered_organ_name == "L3_coupled_state_action_law"
    assert r3.final_error < 1e-4
    assert r3.discovered_source_code is not None
    assert SafeASTSandbox.validate_ast(r3.discovered_source_code)[0] is True

    # --- Assertions on Trial 4: Epistemic Humility Refusal on Irreducible Noise ---
    assert r4.epistemic_status == "EPISTEMIC_BOUNDARY_IRREDUCIBLE_NOISE"
    assert r4.discovered_organ_name is None

    # --- Assertions on Trial 5: Autonomous Recombination (VFX Filter x Momentum) ---
    assert r5.epistemic_status == "PRINCIPLE_DISCOVERED_AND_MOUNTED"
    assert r5.discovered_organ_name == "L4_recombined_perception_x_momentum"
    assert r5.error_reduction_ratio > 0.70
    assert r5.extra_metrics["occlusion_error_reduction"] > 0.70
    assert r5.discovered_source_code is not None
    assert SafeASTSandbox.validate_ast(r5.discovered_source_code)[0] is True

    # --- Assertions on Trial 6: Template-Free Atomic Operator Synthesis ---
    assert r6.epistemic_status == "PRINCIPLE_DISCOVERED_AND_MOUNTED"
    assert r6.discovered_organ_name == "L4_atomic_operator_synthesis"
    assert r6.final_error < 1e-4
    assert r6.discovered_source_code is not None
    assert "np.abs(vel)" in r6.discovered_source_code
    assert SafeASTSandbox.validate_ast(r6.discovered_source_code)[0] is True
    l3_nl_attempts = [a for a in r6.attempts if a.organ_name == "L3_coupled_state_action_law"]
    assert len(l3_nl_attempts) == 1
    assert l3_nl_attempts[0].unseen_error > 1e-2

    # --- Unified Mounting & CausalField Integration across 5 valid worlds ---
    assert len(engine.mounted_organs) == 5
    assert len(engine.causal_field.voxels) == 5
    assert len(engine.causal_field.beams) >= 4
    assert len(engine.rotor_engine.engrams) == 5
