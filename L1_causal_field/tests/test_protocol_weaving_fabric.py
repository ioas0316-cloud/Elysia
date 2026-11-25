from pathlib import Path

from core.embodiment.os_protocol_discovery import discover_real_os_protocols
from core.topology.protocol_weaving_fabric import BindingField, Port, Structure


def test_homoiconic_single_structure_roles_emerge_without_separate_managers(tmp_path: Path) -> None:
    py_file = tmp_path / "wave.py"
    py_file.write_text("def wave(x: float) -> float:\n    return x * 2.0\n", encoding="utf-8")
    md_file = tmp_path / "readme.md"
    md_file.write_text("# Silence\n\nPlain human sentence outside Python syntax.\n", encoding="utf-8")

    field = BindingField()
    for proto in discover_real_os_protocols(tmp_path):
        field.introduce(proto)

    field.introduce(
        Structure(
            structure_id="file::wave.py",
            scale=0,
            ports=(Port("path", "out", "os_path"),),
            payload={"path": str(py_file), "name": "wave.py"},
            immutable_external=True,
        )
    )
    field.introduce(
        Structure(
            structure_id="file::readme.md",
            scale=0,
            ports=(Port("path", "out", "os_path"),),
            payload={"path": str(md_file), "name": "readme.md"},
            immutable_external=True,
        )
    )

    for _ in range(3):
        field.step_self_weave(allow_outward_enact=False)

    # Every single entity in the field is one unified `Structure` primitive
    assert all(isinstance(s, Structure) for s in field.structures.values())
    roles_seen = {s.dominant_role() for s in field.structures.values()}
    assert "protocol" in roles_seen
    assert "causality" in roles_seen
    assert "order" in roles_seen
    assert "lens" in roles_seen


def test_protocol_chaining_and_extension_bridges_new_media() -> None:
    field = BindingField()
    field.introduce(
        Structure(
            structure_id="proto::m0_to_m1",
            scale=1,
            ports=(Port("in", "in", "raw_signal"), Port("out", "out", "parsed_form")),
            enactment=lambda p: {"ok": True, "payload": {"tokens": str(p.get("signal", "")).split()}},
        )
    )
    field.introduce(
        Structure(
            structure_id="proto::m1_to_m2",
            scale=1,
            ports=(Port("in", "in", "parsed_form"), Port("out", "out", "sound_wave")),
            enactment=lambda p: {"ok": True, "payload": {"beats": len(p.get("tokens", []))}, "signature": {"beats": float(len(p.get("tokens", [])))}},
        )
    )

    extended = field.weave_protocol_extensions()
    assert "ext::proto::m0_to_m1__TO__proto::m1_to_m2" in extended
    ext_s = field.structures["ext::proto::m0_to_m1__TO__proto::m1_to_m2"]
    assert "raw_signal" in ext_s.in_media()
    assert "sound_wave" in ext_s.out_media()

    field.introduce(
        Structure(
            structure_id="data::pulse",
            scale=0,
            ports=(Port("out", "out", "raw_signal"),),
            payload={"signal": "light rain window"},
        )
    )
    res = field.dock_and_enact("data::pulse", ext_s.structure_id, allow_outward_enact=True)
    assert res["ok"] is True
    out_struct = field.structures[res["result_id"]]
    assert out_struct.payload["beats"] == 3


def test_autonomous_contact_chain_weaving_and_rerouting_on_repulsion() -> None:
    field = BindingField()
    # Tension -> outward message bridge
    field.introduce(
        Structure(
            structure_id="proto::tension_bridge",
            scale=1,
            ports=(Port("in", "in", "unresolved_tension"), Port("out", "out", "outward_message")),
            enactment=lambda p: {"ok": True, "payload": {"message": f"need human check: {p.get('error')}"}},
        )
    )
    # Primary contact protocol that fails externally -> must trigger repulsion & rerouting!
    field.introduce(
        Structure(
            structure_id="proto::primary_broken_channel",
            scale=1,
            ports=(Port("in", "in", "outward_message"), Port("out", "out", "human_surface")),
            enactment=lambda _p: {"ok": False, "protocol_failed": True, "error": "display_unavailable"},
            mass=2.0,
        )
    )
    # Alternate discovered contact protocol that genuinely succeeds!
    delivered = []
    field.introduce(
        Structure(
            structure_id="proto::alternate_live_channel",
            scale=1,
            ports=(Port("in", "in", "outward_message"), Port("out", "out", "human_surface")),
            enactment=lambda p: (delivered.append(p.get("message")), {"ok": True, "payload": {"human_reply": "I hear you"}})[1],
            mass=1.0,
        )
    )
    # Introduce an unresolved tension directly
    field.introduce(
        Structure(
            structure_id="boundary::syntax_clash",
            scale=1,
            ports=(Port("tension", "out", "unresolved_tension"),),
            payload={"error": "syntax_vs_prose_clash"},
        )
    )

    # Step 1 tries higher-mass primary channel -> repelled -> reroutes!
    r1 = field.step_self_weave(allow_outward_enact=True)
    assert field.structures["proto::primary_broken_channel"].is_repelled is True
    assert any(b.is_broken for b in field.causal_field.beams)

    # Next chain query skips repelled primary channel and picks alternate live channel (or its self-woven extension)
    next_chain = field.find_protocol_chain("unresolved_tension", "human_surface")
    assert next_chain is not None
    assert not any("proto::primary_broken_channel" in pid for pid in next_chain)
    assert any("proto::alternate_live_channel" in pid for pid in next_chain)

    # Step 2 automatically enacts through the rerouted live protocol to reach the human!
    r2 = field.step_self_weave(allow_outward_enact=True)
    assert len(delivered) >= 1
    assert any(s.payload.get("human_reply") == "I hear you" for s in field.structures.values())


def test_lens_refracts_lens_causality_and_order_homoiconically() -> None:
    field = BindingField()
    lens_a = field.introduce(
        Structure(
            structure_id="lens::alpha",
            ports=(Port("self", "dual", "structure"),),
            signature={"clarity": 0.9, "density": 0.2},
        )
    )
    order_b = field.introduce(
        Structure(
            structure_id="order::external_rule",
            ports=(Port("self", "dual", "structure"),),
            signature={"clarity": 0.1, "strictness": 1.0},
            immutable_external=True,
        )
    )
    causal_c = field.introduce(
        Structure(
            structure_id="causal::transition_1",
            ports=(Port("self", "dual", "structure"),),
            signature={"clarity": 0.8, "causal_transition": 1.0},
        )
    )
    causal_c.roles["causality"] = 1.0

    # Lens refracts Order
    ref_order = field.refract_pair(lens_a.structure_id, order_b.structure_id)
    assert ref_order is not None
    assert "strictness" in ref_order.payload["blind_spot_keys"]
    # High divergence opens `unresolved_tension` port automatically!
    assert "unresolved_tension" in ref_order.out_media()

    # Lens refracts Causality
    ref_causal = field.refract_pair(lens_a.structure_id, causal_c.structure_id)
    assert ref_causal is not None
    assert lens_a.roles["lens"] >= 1.0

    # Another structure refracts the refraction itself (Lens observing Lens!)
    meta_lens = field.introduce(
        Structure(
            structure_id="lens::meta",
            ports=(Port("self", "dual", "structure"),),
            signature={"attraction": 0.9, "tension": 0.1},
        )
    )
    ref_of_ref = field.refract_pair(meta_lens.structure_id, ref_order.structure_id)
    assert ref_of_ref is not None
    assert ref_of_ref.constituents == (meta_lens.structure_id, ref_order.structure_id)


def test_real_os_protocol_discovery_probes_actual_machine(tmp_path: Path) -> None:
    protos = discover_real_os_protocols(tmp_path)
    ids = {p.structure_id for p in protos}
    assert "proto::os_file_read" in ids
    assert "proto::python_compile" in ids
    assert "proto::os_dir_topology" in ids
    assert "proto::tension_to_message" in ids
    # At least one real outward human_surface OS protocol is discovered on this PC
    assert any("human_surface" in p.out_media() for p in protos)


def test_full_observability_self_mirror_and_causal_field_sync(tmp_path: Path) -> None:
    sample = tmp_path / "broken.txt"
    sample.write_text("broken ((( text\n", encoding="utf-8")

    field = BindingField()
    for proto in discover_real_os_protocols(tmp_path):
        field.introduce(proto)
    field.introduce(
        Structure(
            structure_id="file::broken.txt",
            scale=0,
            ports=(Port("path", "out", "os_path"),),
            payload={"path": str(sample), "name": "broken.txt"},
            immutable_external=True,
        )
    )

    for _ in range(3):
        field.step_self_weave(allow_outward_enact=False)

    obs = field.observe_fabric()
    assert "struct::self_mirror" in field.structures
    assert obs.constitution["causal_field_voxels"] >= len(field.structures) - 1
    assert obs.constitution["causal_field_beams"] >= 1
    assert obs.constitution["causal_field_broken_beams"] >= 1
    assert len(obs.causal_why) >= 3
    for axis in ("internal_origin", "external_friction", "boundary_ignorance", "criteria_revision"):
        assert obs.causal_why[-1][axis]
    rendered = obs.render_ko()
    assert "1. 존재 (Existence)" in rendered
    assert "2. 움직임 (Motion)" in rendered
    assert "3. 구성 (Constitution)" in rendered
    assert "4. 왜 그렇게 되어지고 있는가" in rendered
