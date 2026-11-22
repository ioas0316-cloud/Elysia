from __future__ import annotations

from pathlib import Path

from core.sensory.trinitarian_scale_rotor_protocol import (
    DEFAULT_TRINITARIAN_DASHBOARD_PATH,
    TrinitarianScaleRotorProtocol,
    quat_geodesic_angle,
    render_trinitarian_protocol_dashboard,
)


def test_trinitarian_atom_to_molecule_inclusion_and_zoom_retuning() -> None:
    """
    Verifies the pure Trinitarian Scale-Rotor Inclusion Protocol:
      1. Non-commutative Hamilton rotor binding (`A ⊗ B != B ⊗ A`).
      2. Emergent molecular Chromatic character (`Red + Blue -> Yellow` synthesis).
      3. Multi-scale recursive inclusion (`s=0 atoms -> s=1 molecule -> s=2 super-molecule`).
      4. Top-down `zoom_in_inspect` & `re_tune_constituent` propagating phase/chromatic
         reconfiguration upward to parent scales while tearing invalidated beams.
    """
    proto = TrinitarianScaleRotorProtocol()

    # 1. Register elemental s=0 atomic rotors
    a_vis = proto.register_atom("atom_visual_red", (1.0, 0.0, 0.0), 0.90, (1.0, 0.0, 0.0))
    a_res = proto.register_atom("atom_tactile_resist", (0.0, 1.0, 0.0), 1.10, (0.0, 1.0, 0.0))
    a_pass = proto.register_atom("atom_pass_through", (0.0, 0.0, 1.0), 1.40, (0.20, 0.10, 0.70))
    a_word = proto.register_atom("atom_word_byeok", (0.5, 0.5, 1.0), 0.85, (0.15, 0.70, 0.15))

    # 2. Verify order sensitivity (non-commutative action -> consequence binding)
    ab = proto.resonate_and_bind("atom_visual_red", "atom_tactile_resist", molecule_id="tmp_AB")
    ba = proto.resonate_and_bind("atom_tactile_resist", "atom_visual_red", molecule_id="tmp_BA")
    order_angle = quat_geodesic_angle(ab.quaternion, ba.quaternion)
    assert order_angle > 0.20

    # Clean real hierarchy: s=0 -> s=1 -> s=2
    proto2 = TrinitarianScaleRotorProtocol()
    proto2.register_atom("atom_visual_red", (1.0, 0.0, 0.0), 0.90, (1.0, 0.0, 0.0))
    proto2.register_atom("atom_tactile_resist", (0.0, 1.0, 0.0), 1.10, (0.0, 1.0, 0.0))
    proto2.register_atom("atom_pass_through", (0.0, 0.0, 1.0), 1.40, (0.20, 0.10, 0.70))
    proto2.register_atom("atom_word_byeok", (0.5, 0.5, 1.0), 0.85, (0.15, 0.70, 0.15))

    mol_wall = proto2.resonate_and_bind(
        "atom_visual_red", "atom_tactile_resist", bond_coherence=0.90, molecule_id="mol_wall"
    )
    assert mol_wall.scale_level == 1
    assert mol_wall.constituents == ("atom_visual_red", "atom_tactile_resist")
    # Pure Red (1,0,0) + Pure Blue (0,1,0) must birth emergent Yellow character > 0.35!
    assert mol_wall.chromatic_triad[2] > 0.35

    super_byeok = proto2.resonate_and_bind(
        "mol_wall", "atom_word_byeok", bond_coherence=0.92, molecule_id="super_byeok"
    )
    assert super_byeok.scale_level == 2
    assert super_byeok.constituents == ("mol_wall", "atom_word_byeok")

    # Zoom-in traversal must expose every contained constituent down to s=0 atoms
    inspected = proto2.zoom_in_inspect("super_byeok")
    inspected_ids = {n.node_id for n in inspected}
    assert inspected_ids == {
        "super_byeok",
        "mol_wall",
        "atom_word_byeok",
        "atom_visual_red",
        "atom_tactile_resist",
    }

    before_triads = {
        "atom_visual_red": proto2.nodes["atom_visual_red"].chromatic_triad.copy(),
        "atom_tactile_resist": proto2.nodes["atom_tactile_resist"].chromatic_triad.copy(),
        "mol_wall": mol_wall.chromatic_triad.copy(),
        "super_byeok": super_byeok.chromatic_triad.copy(),
    }

    # 3. Friction occurs: user/ELYSIA discovers the red surface actually passes right through!
    # Zoom IN to `mol_wall` and swap `atom_tactile_resist` -> `atom_pass_through`, checking upward propagation!
    shifts = proto2.re_tune_constituent(
        target_molecule_id="mol_wall",
        old_child_id="atom_tactile_resist",
        new_child_id="atom_pass_through",
        new_coherence=0.94,
    )

    torn_beams = [
        b.strength
        for b in proto2.causal_field.beams
        if {b.source_id, b.target_id} == {"atom_tactile_resist", "mol_wall"}
    ]
    assert len(torn_beams) == 1 and torn_beams[0] < 0.05
    assert shifts["mol_wall"] > 0.15
    assert shifts["super_byeok"] > 0.15
    assert "atom_pass_through" in {n.node_id for n in proto2.zoom_in_inspect("super_byeok")}

    after_triads = {
        "mol_wall": proto2.nodes["mol_wall"].chromatic_triad.copy(),
        "super_byeok": proto2.nodes["super_byeok"].chromatic_triad.copy(),
    }

    print("\n=== [Trinitarian Scale-Rotor Inclusion Protocol Verification] ===")
    print(f"Non-commutativity geodesic angle (A*B vs B*A): {order_angle:.4f} rad")
    print(
        f"Emergent Chromatic Triad (Red + Blue -> Mol s=1): "
        f"R={before_triads['mol_wall'][0]:.3f}, B={before_triads['mol_wall'][1]:.3f}, "
        f"Y={before_triads['mol_wall'][2]:.3f}"
    )
    print(
        f"Zoom-In Re-Tuning Upward Propagation Shifts: "
        f"mol_wall={shifts['mol_wall']:.4f} rad, super_byeok={shifts['super_byeok']:.4f} rad | "
        f"Old Beam={torn_beams[0]:.2f}"
    )

    dash_path = render_trinitarian_protocol_dashboard(
        protocol=proto2,
        before_triads=before_triads,
        after_triads=after_triads,
        shifts=shifts,
        output_path=Path(DEFAULT_TRINITARIAN_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000
