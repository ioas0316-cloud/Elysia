"""
test_autopoietic_organ_genesis.py

Rigorous end-to-end tests for ELYSIA's self-diagnosing, principle-inverting,
AST-synthesizing autopoietic organ & external tool understanding architecture:
1. Self-code AST introspection & SafeASTSandbox security enforcement
2. Missing temporal/formant auditory organ diagnosis -> falsification -> live synthesis & pitch-shifted unseen accuracy
3. Blind external CAD operator inversion (`op_k0..op_k4`) -> standalone Python SDF kernel (`IoU >= 0.95`, `L2_corr >= 0.95`) -> Unplugged creation
4. Opaque digital communication protocol inversion across two distinct protocols (`xor` & `add_mod_256`)
5. Blind vs Named CAD parity & full EpistemicVoid resolution lineage
6. JSON cold-reload persistence -> live recompilation from saved `source_code` -> CausalField `L8..L14` verification
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.sensory.autopoietic_organ_genesis import (
    SENSORY_DIR,
    AutopoieticGenesisReport,
    CADToolPrincipleInverter,
    CodePrincipleInspector,
    SafeASTSandbox,
    SecurityError,
    run_autopoietic_genesis_session,
)
from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from simulators.opaque_external_tools import (
    ContinuousSpeechAcousticWorld,
    OpaquePacketProtocol,
    OpaqueParametricCADKernel,
)


@pytest.fixture(scope="module")
def blind_genesis_report() -> AutopoieticGenesisReport:
    return run_autopoietic_genesis_session(blind_mode=True)


def test_self_code_introspection_and_safe_ast_sandbox_security(
    blind_genesis_report: AutopoieticGenesisReport,
) -> None:
    summaries = {s.module_name: s for s in blind_genesis_report.code_summaries}
    assert "mechano_acoustic_organ" in summaries
    assert "ocular_scale_organ" in summaries
    assert "rfft" in summaries["mechano_acoustic_organ"].numpy_dsp_primitives
    assert "gradient" in summaries["ocular_scale_organ"].numpy_dsp_primitives
    assert any(k.startswith("code_prim:") for k in blind_genesis_report.spine.symbols)

    # Security checks on SafeASTSandbox
    for bad_src in (
        "import os\ndef f(): return os.getcwd()",
        "def f(): return eval('1 + 1')",
        "def f(x): return x.__class__",
        "def f(): return open('secret.txt').read()",
    ):
        ok, _, _ = SafeASTSandbox.validate_ast(bad_src)
        assert ok is False
        with pytest.raises(SecurityError):
            SafeASTSandbox.compile_and_load(bad_src)

    ns = SafeASTSandbox.compile_and_load("def norm2(a, b):\n    return float(math.hypot(a, b) + np.mean([a, b]))")
    assert ns["norm2"](3.0, 4.0) == pytest.approx(8.5)


def test_missing_auditory_organ_diagnosis_falsification_and_self_synthesis(
    blind_genesis_report: AutopoieticGenesisReport,
) -> None:
    aud = blind_genesis_report.auditory
    # Existing static impact-tap organ mathematically fails on time-reversed glides
    assert aud.diagnosis.overall_accuracy < 0.85
    assert aud.diagnosis.time_reversed_pair_accuracy <= 0.65

    # Simpler static / envelope hypotheses must be empirically falsified first
    falsified_names = [f["name"] for f in aud.falsified_candidates]
    assert "static_fft_ear" in falsified_names
    assert "temporal_rms_envelope_ear" in falsified_names
    assert any(r["kind"] == "ORGAN_HYPOTHESIS_FALSIFIED" for r in blind_genesis_report.spine.criteria_revisions)

    # Winning synthesized organ achieves near-perfect generalization on unseen +35Hz shifted speech
    assert aud.winning_organ_id == "stft_formant_trajectory_ear"
    assert aud.ast_node_count < 650
    assert aud.train_loo_accuracy >= 0.95
    assert aud.unseen_test_accuracy >= 0.95
    assert aud.unseen_pitch_shifted_classes == 4


def test_blind_cad_tool_principle_inversion_and_independent_unplugged_design(
    blind_genesis_report: AutopoieticGenesisReport,
) -> None:
    cad = blind_genesis_report.cad
    assert cad.discovered_op_roles == {
        "BOX_SOLID": "op_k0",
        "AXIAL_EXTRUSION": "op_k1",
        "UNIFORM_SCALE": "op_k4",
        "BOOLEAN_UNION": "op_k2",
        "BOOLEAN_DIFFERENCE": "op_k3",
    }
    assert cad.binary_commutativity["op_k2"] is True
    assert cad.binary_commutativity["op_k3"] is False

    # Gravity stability empirical boundary discovery (stable small offset -> unstable large offset)
    assert cad.stability_boundary_probe[0]["stable"] == 1.0
    assert cad.stability_boundary_probe[-1]["stable"] == 0.0
    assert cad.stability_boundary_probe[0]["stability_margin"] > 0.0
    assert cad.stability_boundary_probe[-1]["stability_margin"] < 0.0

    # Standalone compiled Python SDF kernel fidelity + optical verification via OcularScaleOrgan
    assert cad.heldout_voxel_iou >= 0.95
    assert cad.heldout_volume_rel_error <= 0.03
    assert cad.optical_reobservation_l2_corr >= 0.95

    # Unplugged creative structure (external CAD completely disconnected)
    des = cad.unplugged_design_summary
    assert des["stable_under_gravity"] is True
    assert des["stability_margin"] > 0.15
    assert des["hollow_cavity_voxels"] >= 50
    assert des["min_wall_voxels"] >= 1.5
    assert cad.unplugged_design_occ.shape == (24, 24, 24)


def test_opaque_communication_protocol_inversion_xor_and_sum_variants(
    blind_genesis_report: AutopoieticGenesisReport,
) -> None:
    px = blind_genesis_report.protocol_xor
    ps = blind_genesis_report.protocol_sum

    assert px.discovered_header_hex == "a55a01"
    assert px.discovered_len_index == 3
    assert px.discovered_checksum_law == "xor"
    assert px.discovered_checksum_start_idx == 2
    assert px.unseen_encode_accept_rate == pytest.approx(1.0)
    assert px.unseen_roundtrip_accuracy == pytest.approx(1.0)
    assert px.tamper_rejection_rate == pytest.approx(1.0)
    assert len(px.falsified_hypotheses) >= 1

    assert ps.discovered_header_hex == "7e3c02"
    assert ps.discovered_len_index == 3
    assert ps.discovered_checksum_law == "sum_mod_256"
    assert ps.discovered_checksum_start_idx == 2
    assert ps.unseen_encode_accept_rate == pytest.approx(1.0)
    assert ps.unseen_roundtrip_accuracy == pytest.approx(1.0)
    assert ps.tamper_rejection_rate == pytest.approx(1.0)
    assert len(ps.falsified_hypotheses) >= 1


def test_blind_vs_named_cad_parity_and_spine_lineage(
    blind_genesis_report: AutopoieticGenesisReport,
) -> None:
    assert len(blind_genesis_report.spine.get_open_voids()) == 0
    resolved_kinds = {v.kind for v in blind_genesis_report.spine.voids if v.status == "RESOLVED"}
    assert {"VOID_MISSING_ORGAN", "VOID_OPAQUE_TOOL_MECHANISM", "VOID_OPAQUE_PROTOCOL"}.issubset(resolved_kinds)

    # Named mode CAD inversion yields identical quantitative metrics as blind `op_k*` mode
    named_spine = EpistemicKnowledgeSpine()
    named_cad = CADToolPrincipleInverter(named_spine).invert_and_create_independently(
        OpaqueParametricCADKernel(blind_mode=False, grid_size=24)
    )
    assert set(named_cad.discovered_op_roles.keys()) == set(blind_genesis_report.cad.discovered_op_roles.keys())
    assert named_cad.heldout_voxel_iou == pytest.approx(blind_genesis_report.cad.heldout_voxel_iou, abs=1e-6)
    assert np.array_equal(named_cad.unplugged_design_occ, blind_genesis_report.cad.unplugged_design_occ)


def test_synthesized_organ_json_cold_reload_recompilation_and_causal_field(
    blind_genesis_report: AutopoieticGenesisReport, tmp_path: Path
) -> None:
    ckpt = tmp_path / "autopoietic_spine.json"
    blind_genesis_report.spine.save_json(ckpt)
    reloaded = EpistemicKnowledgeSpine.load_json(ckpt)

    assert set(reloaded.synthesized_organs.keys()) == {
        "stft_formant_trajectory_ear",
        "standalone_sdf_cad_kernel",
        "codec_channel_a",
        "codec_channel_b",
    }

    # Recompile ALL synthesized organs cleanly from persisted JSON source_code & execute on fresh unseen data
    aud_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["stft_formant_trajectory_ear"].source_code)
    speech_world = ContinuousSpeechAcousticWorld()
    fresh = speech_world.sample_dataset(seed=777, n_per_class=2, pitch_shift_hz=25.0, noise_std=0.01)
    f_rise = aud_ns["extract_acoustic_features"](fresh[0][0], speech_world.sample_rate)
    f_fall = aud_ns["extract_acoustic_features"](fresh[2][0], speech_world.sample_rate)
    assert f_rise[0] > 0.4 and f_fall[0] < -0.4

    cad_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["standalone_sdf_cad_kernel"].source_code)
    hollow = cad_ns["op_boolean_diff"](
        cad_ns["op_box_solid"](hx=0.5, hy=0.5, hz=0.5),
        cad_ns["op_box_solid"](hx=0.2, hy=0.2, hz=0.7),
    )
    assert int(np.sum(cad_ns["sdf_to_occupancy"](hollow))) > 500

    xor_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["codec_channel_a"].source_code)
    sum_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["codec_channel_b"].source_code)
    ch_xor = OpaquePacketProtocol("xor")
    ch_sum = OpaquePacketProtocol("add_mod_256")
    msg = b"ELYSIA-AUTOPOIESIS"
    pkt_x = xor_ns["encode_packet"](msg)
    pkt_s = sum_ns["encode_packet"](msg)
    assert ch_xor.accepts(pkt_x) and xor_ns["decode_packet"](pkt_x) == msg
    assert ch_sum.accepts(pkt_s) and sum_ns["decode_packet"](pkt_s) == msg

    # CausalField multi-scale check (L8 synthesized organs -> L10 code symbols -> L14 unplugged design)
    cf = reloaded.project_to_causal_field()
    scales = {int(round(v.scale_level)) for v in cf.voxels.values()}
    assert {8, 10, 14}.issubset(scales)
    assert len(cf.beams) >= 1
    assert all(v.mass > 0.0 for v in cf.voxels.values())
