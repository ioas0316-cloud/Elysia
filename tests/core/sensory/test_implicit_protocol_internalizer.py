from __future__ import annotations

from pathlib import Path

from core.sensory.implicit_protocol_internalizer import (
    BoundaryClosureSyntaxWorld,
    DEFAULT_PROTOCOL_INTERNALIZER_DASHBOARD_PATH,
    ImplicitProtocolInternalizer,
    RelationalKeyGateWorld,
    ThermalPhaseMatterWorld,
    render_implicit_protocol_dashboard,
)


def test_elysia_fathoms_implicit_protocols_across_three_worlds() -> None:
    """
    Verifies that ONE `ImplicitProtocolInternalizer` resting on `TrinitarianScaleRotorProtocol`
    autonomously discovers WHY ambiguous surface phenomena behave differently across 3 unrelated
    blind worlds, rejects irrelevant distractors, crystallizes `(A ⊗ B) ⊗ C` s=2 super-molecules,
    and achieves 100% unseen prediction purely by traversing its own inclusion tree.
    """
    internalizer = ImplicitProtocolInternalizer()

    w1 = ThermalPhaseMatterWorld()
    r1 = internalizer.internalize_world(
        world_id="World1_ThermalPhase",
        train_episodes=w1.rollout(n_episodes=90, seed=101),
        unseen_episodes=w1.rollout(n_episodes=60, seed=701),
    )

    w2 = RelationalKeyGateWorld()
    r2 = internalizer.internalize_world(
        world_id="World2_RelationalGate",
        train_episodes=w2.rollout(n_episodes=80, seed=202),
        unseen_episodes=w2.rollout(n_episodes=60, seed=702),
    )

    w3 = BoundaryClosureSyntaxWorld()
    r3 = internalizer.internalize_world(
        world_id="World3_BoundaryClosure",
        train_episodes=w3.rollout(n_episodes=80, seed=303),
        unseen_episodes=w3.rollout(n_episodes=60, seed=703),
    )

    reports = [r1, r2, r3]

    print("\n=== [ELYSIA Implicit Protocol Internalization Log] ===")
    for rep in reports:
        print(
            f"\n[{rep.world_id}] Surface Ambiguity={rep.surface_only_ambiguity:.3f} -> "
            f"Residual={rep.protocol_residual_ambiguity:.3f} | "
            f"Discovered Cause='{rep.discovered_cause_key}' | "
            f"Rejected Distractor={rep.rejected_distractor_keys} | "
            f"Unseen Acc={rep.unseen_prediction_accuracy * 100:.1f}%"
        )
        for rule in rep.discovered_rules:
            print(
                f"  - {rule.super_molecule_id}: "
                f"({rule.surface_atom.split('::')[-1]} ⊗ {rule.cause_atom.split('::')[-1]}) "
                f"⊗ {rule.outcome_atom.split('::')[-1]} | tree_size={len(rule.constituent_tree_ids)}"
            )

    dash_path = render_implicit_protocol_dashboard(
        internalizer=internalizer,
        reports=reports,
        output_path=Path(DEFAULT_PROTOCOL_INTERNALIZER_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- World 1 Assertions ---
    assert r1.discovered_cause_key == "ambient_thermal"
    assert "bg_lighting" in r1.rejected_distractor_keys
    assert r1.surface_only_ambiguity > 0.50
    assert r1.protocol_residual_ambiguity < 0.05
    assert r1.unseen_prediction_accuracy == 1.0
    assert len(r1.discovered_rules) == 3

    # --- World 2 Assertions ---
    assert r2.discovered_cause_key == "self_coupling"
    assert "sky_tint" in r2.rejected_distractor_keys
    assert r2.surface_only_ambiguity > 0.35
    assert r2.protocol_residual_ambiguity < 0.05
    assert r2.unseen_prediction_accuracy == 1.0
    assert len(r2.discovered_rules) == 2

    # --- World 3 Assertions ---
    assert r3.discovered_cause_key == "outer_boundary"
    assert "whitespace_pad" in r3.rejected_distractor_keys
    assert r3.surface_only_ambiguity > 0.35
    assert r3.protocol_residual_ambiguity < 0.05
    assert r3.unseen_prediction_accuracy == 1.0
    assert len(r3.discovered_rules) == 3

    # --- Trinitarian Inclusion Integrity (every s=2 super-molecule keeps 5-node hierarchy) ---
    for rep in reports:
        for rule in rep.discovered_rules:
            node = internalizer.protocol.nodes[rule.super_molecule_id]
            assert node.scale_level == 2
            tree = internalizer.protocol.zoom_in_inspect(rule.super_molecule_id)
            assert len(tree) == 5
