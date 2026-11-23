from __future__ import annotations

from pathlib import Path

from core.sensory.analogical_boundary_reflection_engine import (
    AnalogicalBoundaryReflectionEngine,
    DEFAULT_ANALOGICAL_DASHBOARD_PATH,
    build_counterfactual_tests_for_domain,
    build_physical_domain_profiles,
    render_analogical_reflection_dashboard,
)


def test_analogical_reflection_on_invisible_electricity_and_heat() -> None:
    """
    Verifies ELYSIA's Analogical Boundary Reflection:
      1. Anchors visible lived `Water_Flow` as the sensory foothold.
      2. Confronts invisible `Invisible_Electricity` & `Invisible_Heat_Conduction`.
      3. Preserves isomorphic structure (`drive_gradient`, `constriction_resistance`),
         explicitly pinpoints where the Water analogy breaks (`open_cut`, `immediacy`),
         tears invalidated analogy beams in `CausalField`, and binds discovered boundaries
         into `reinterpreted::<target>` at `scale_level == 2`.
      4. Beats Naive Water Analogy cleanly (100% vs 50%) on unseen counterfactual tests!
    """
    profiles = build_physical_domain_profiles()
    engine = AnalogicalBoundaryReflectionEngine()
    engine.anchor_lived_foothold(profiles["Water_Flow"])

    rep_elec = engine.reflect_on_unknown_via_foothold(
        source=profiles["Water_Flow"],
        target=profiles["Invisible_Electricity"],
        counterfactual_tests=build_counterfactual_tests_for_domain("Invisible_Electricity"),
    )
    rep_heat = engine.reflect_on_unknown_via_foothold(
        source=profiles["Water_Flow"],
        target=profiles["Invisible_Heat_Conduction"],
        counterfactual_tests=build_counterfactual_tests_for_domain("Invisible_Heat_Conduction"),
    )

    reports = [rep_elec, rep_heat]

    print("\n=== [ELYSIA Analogical Boundary Reflection Log] ===")
    for rep in reports:
        print(f"\n{rep.reflection_summary_ko}")
        for item in rep.probe_contrasts:
            print(
                f"  - {item.probe_name:24s} | Water={item.source_val:+.2f} vs "
                f"{rep.target_domain_id}={item.target_val:+.2f} | diff={item.divergence:.2f} | {item.status}"
            )
        print(f"  Torn Analogy Beams: {rep.torn_analogical_beams}")
        print(
            f"  Unseen Accuracy: Naive Water={rep.naive_analogy_accuracy * 100:.0f}% -> "
            f"Reinterpreted s=3={rep.reinterpreted_accuracy * 100:.0f}%"
        )

    dash_path = render_analogical_reflection_dashboard(
        engine=engine,
        profiles=profiles,
        reports=reports,
        output_path=Path(DEFAULT_ANALOGICAL_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- Electricity Assertions ---
    assert rep_elec.target_is_invisible_fog is True
    assert set(rep_elec.aligned_probes) == {"drive_gradient_response", "constriction_resistance"}
    assert set(rep_elec.breakdown_probes) == {"open_cut_persistence", "propagation_immediacy"}
    assert len(rep_elec.torn_analogical_beams) == 2
    assert rep_elec.naive_analogy_accuracy < 0.70
    assert rep_elec.reinterpreted_accuracy == 1.0
    assert engine.protocol.nodes[rep_elec.reinterpreted_molecule_id].scale_level == 3

    # --- Heat Conduction Assertions ---
    assert rep_heat.target_is_invisible_fog is True
    assert set(rep_heat.aligned_probes) == {"drive_gradient_response", "constriction_resistance"}
    assert set(rep_heat.breakdown_probes) == {"open_cut_persistence", "propagation_immediacy"}
    assert len(rep_heat.torn_analogical_beams) == 2
    assert rep_heat.naive_analogy_accuracy < 0.70
    assert rep_heat.reinterpreted_accuracy == 1.0
    assert engine.protocol.nodes[rep_heat.reinterpreted_molecule_id].scale_level == 3

    # --- Verify Electricity vs Heat reinterpreted DIFFERENT outcomes on the SAME counterfactual probe ---
    assert (
        engine.reinterpreted_probe_values["Invisible_Electricity"]["open_cut_persistence"]
        != engine.reinterpreted_probe_values["Invisible_Heat_Conduction"]["open_cut_persistence"]
    )
    assert any(b.strength < 0.05 for b in engine.protocol.causal_field.beams)
