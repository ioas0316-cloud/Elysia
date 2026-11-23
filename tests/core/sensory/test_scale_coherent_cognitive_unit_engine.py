from __future__ import annotations

from pathlib import Path

from core.sensory.scale_coherent_cognitive_unit_engine import (
    DEFAULT_COGNITIVE_UNIT_DASHBOARD_PATH,
    PhysicalGeometricObject,
    ScaleCoherentCognitiveUnitEngine,
    render_cognitive_unit_genesis_dashboard,
)


def test_scale_coherent_cognitive_unit_genesis_and_deliberation() -> None:
    """
    Verifies ELYSIA's scale-coherent cognitive unit architecture across 4 phases:
      1. Raw unlabeled sensory encounters differentiate Size/Area vs Perimeter vs Inertial Mass.
      2. Deliberation refutes false definitions ("same size => same mass",
         "same area => same perimeter") with concrete lived counterexamples, while
         convincing valid causal definitions.
      3. Synthesizes novel `s=2` unit `UNIT_DENSITY = UNIT_AREA ⊗ UNIT_INERTIAL_MASS`
         retaining all lower scales inside its Trinitarian inclusion tree.
      4. Correctly judges an unseen counter-intuitive pair (tiny heavy iron vs giant light foam).
    """
    engine = ScaleCoherentCognitiveUnitEngine()

    training_objects = [
        PhysicalGeometricObject("Cube_Light", width=20.0, height=20.0, interior_cells=400.0, boundary_walk_steps=80.0, push_force=10.0, observed_accel=2.0),
        PhysicalGeometricObject("Cube_Heavy", width=20.0, height=20.0, interior_cells=400.0, boundary_walk_steps=80.0, push_force=10.0, observed_accel=0.25),
        PhysicalGeometricObject("Square_40x40", width=40.0, height=40.0, interior_cells=1600.0, boundary_walk_steps=160.0, push_force=10.0, observed_accel=1.0),
        PhysicalGeometricObject("Bar_80x20", width=80.0, height=20.0, interior_cells=1600.0, boundary_walk_steps=200.0, push_force=10.0, observed_accel=1.0),
    ]
    engine.encounter_objects(training_objects)
    units = engine.differentiate_cognitive_units()

    v1 = engine.deliberate_claim(
        claim_id="same_size_implies_same_mass",
        claim_text_ko="겉보기 크기가 같으면 무게(관성 질량)도 같다",
        premise_unit="UNIT_AREA",
        implied_unit="UNIT_INERTIAL_MASS",
    )
    v2 = engine.deliberate_claim(
        claim_id="same_area_implies_same_perimeter",
        claim_text_ko="넓이가 같으면 둘레도 같다",
        premise_unit="UNIT_AREA",
        implied_unit="UNIT_PERIMETER",
    )
    v3 = engine.deliberate_claim(
        claim_id="mass_tracks_push_resistance",
        claim_text_ko="관성 질량은 밀었을 때의 가속 저항과 일관되게 결합된다",
        premise_unit="UNIT_PUSH_RESISTANCE",
        implied_unit="UNIT_INERTIAL_MASS",
    )

    density_unit = engine.synthesize_higher_unit_density()

    unseen_iron = PhysicalGeometricObject(
        "Unseen_Iron_Pellet", width=10.0, height=10.0, interior_cells=100.0, boundary_walk_steps=40.0, push_force=10.0, observed_accel=0.10
    )
    unseen_foam = PhysicalGeometricObject(
        "Unseen_Foam_Block", width=60.0, height=60.0, interior_cells=3600.0, boundary_walk_steps=240.0, push_force=10.0, observed_accel=0.90
    )
    winner, dens_iron, dens_foam = engine.judge_unseen_pair_by_unit(
        "UNIT_DENSITY", unseen_iron, unseen_foam
    )

    print("\n=== [ELYSIA Scale-Coherent Cognitive Unit Genesis & Deliberation Log] ===")
    for uid in ("UNIT_AREA", "UNIT_PERIMETER", "UNIT_INERTIAL_MASS", "UNIT_DENSITY"):
        u = units[uid]
        print(f"  [s={u.scale_level}] {uid:20s} <- {u.constituent_units} | {u.linguistic_basis}")
    print("\n[Deliberation Verdicts]")
    for v in (v1, v2, v3):
        print(f"  - {v.claim_id:30s} -> {v.verdict} | {v.reasoning_ko}")
    print(
        f"\n[Unseen Counter-Intuitive Judgment] Winner={winner} | "
        f"Iron_Density={dens_iron:.4f} vs Foam_Density={dens_foam:.5f} ({dens_iron / dens_foam:.1f}x)"
    )

    dash_path = render_cognitive_unit_genesis_dashboard(
        engine=engine,
        unseen_a=unseen_iron,
        unseen_b=unseen_foam,
        dens_a=dens_iron,
        dens_b=dens_foam,
        output_path=Path(DEFAULT_COGNITIVE_UNIT_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- Cognitive Unit Differentiation Assertions ---
    for expected_unit in ("UNIT_AREA", "UNIT_PERIMETER", "UNIT_INERTIAL_MASS"):
        assert expected_unit in units
        assert engine.protocol.nodes[expected_unit].scale_level == 1

    # --- Refutation & Conviction Assertions ---
    assert v1.verdict == "REFUTED_BY_COUNTEREXAMPLE"
    assert v1.counterexample_pair == ("Cube_Light", "Cube_Heavy")

    assert v2.verdict == "REFUTED_BY_COUNTEREXAMPLE"
    assert v2.counterexample_pair == ("Square_40x40", "Bar_80x20")

    assert v3.verdict == "CONVINCED_AND_BOUND"
    assert v3.counterexample_pair is None

    # --- Higher-Order s=2 Density Synthesis & Inclusion Integrity ---
    assert density_unit.scale_level == 2
    assert density_unit.constituent_units == ("UNIT_AREA", "UNIT_INERTIAL_MASS")
    dens_tree = engine.protocol.zoom_in_inspect("UNIT_DENSITY")
    assert len(dens_tree) >= 5
    dens_ids = {n.node_id for n in dens_tree}
    assert {"UNIT_DENSITY", "UNIT_AREA", "UNIT_INERTIAL_MASS", "atom_push_force", "atom_interior_cells"}.issubset(dens_ids)

    # --- Unseen Counter-Intuitive Pair ---
    assert winner == "Unseen_Iron_Pellet"
    assert dens_iron / dens_foam > 50.0
