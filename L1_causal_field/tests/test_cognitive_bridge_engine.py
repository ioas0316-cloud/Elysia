from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in (_REPO_ROOT / "L2_cognitive_engine", _REPO_ROOT / "L3_interface"):
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

from core.embodiment import (
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)
from core.topology.cognitive_bridge_engine import (
    AirplaneBridgeFabricator,
    CognitiveInformationFabric,
)
from core.topology.continuous_rotor_spectrum_field import (
    ContinuousBlindCADRotor,
    build_cognitive_attention_rotor,
)
from simulators.opaque_external_tools import OpaqueParametricCADKernel


def test_airplane_bridges_solve_missing_organs_without_reducing_to_floats() -> None:
    fabric = CognitiveInformationFabric()
    fab = AirplaneBridgeFabricator()

    # 1. Bridge A: Somatic Homeostasis -> Real PC Telemetry + Non-reductive causal laws
    s_spec, s_unit = fab.execute_somatic_homeostasis_bridge(
        fabric,
        unresolved_error_load=0.55,
        posture_spectrum=-0.45,
        cycle_count=3,
    )
    assert s_spec.last_verified_ok is True
    assert len(s_unit.causal_mechanisms) >= 2
    assert len(s_unit.constraints_and_stakes) >= 2
    assert "os::tracemalloc" in s_unit.evidence_sources

    # 2. Bridge B: Tactile 3D Materiality -> Real Blind 3D CAD Physics
    kernel = OpaqueParametricCADKernel(blind_mode=True, grid_size=24)
    cad_rotor = ContinuousBlindCADRotor()
    cad_rotor.step_blind_probe(kernel)
    cad_res = cad_rotor.step_blind_probe(kernel)
    t_spec, t_unit = fab.execute_tactile_3d_bridge(fabric, cad_res)
    assert t_spec.last_verified_ok is True
    assert cad_res["hollow_voxels"] > 0
    assert "내부공동" in t_unit.concepts

    # 3. Bridge C: Temporal Acoustic Irreversibility (16kHz PCM where static FFT fails!)
    a_spec, a_unit = fab.execute_temporal_acoustic_bridge(fabric)
    assert a_spec.last_verified_ok is True
    assert "static_fft_blind_diff=0.0000" in a_spec.last_verification_metric
    assert "시간비가역성" in a_unit.concepts

    # 4. Bridge D: Black-Box Wire Protocol Inversion (both xor and add_mod_256 without oracle!)
    p1_spec, p1_unit = fab.execute_opaque_protocol_bridge(fabric, variant="xor")
    p2_spec, p2_unit = fab.execute_opaque_protocol_bridge(fabric, variant="add_mod_256")
    assert p1_spec.last_verified_ok is True
    assert p2_spec.last_verified_ok is True
    assert "mode=xor" in p1_spec.last_verification_metric
    assert "mode=add_mod_256" in p2_spec.last_verification_metric

    # 5. Bridge E: Offline-Blocked Web -> Real Local Human Corpus Reader
    c_spec, c_unit = fab.execute_local_human_corpus_bridge(fabric, query="감정")
    assert c_spec.last_verified_ok is True
    assert len(c_unit.concepts) >= 3
    assert len(c_unit.evidence_sources) >= 1 and Path(c_unit.evidence_sources[0]).is_file()

    # 6. Cross-Domain Isomorphisms & Knowledge -> Rotor Transfer!
    isos = fabric.discover_cross_domain_isomorphisms(min_similarity=0.80)
    assert len(isos) >= 4
    domains_linked = {tuple(sorted([iso.domain_a, iso.domain_b])) for iso in isos}
    assert len(domains_linked) >= 4

    rotor = build_cognitive_attention_rotor()
    prev_dim = rotor.dim
    xfer = fabric.transfer_isomorphism_to_rotor(rotor, isos[0])
    assert xfer["transferred"] is True
    assert rotor.dim == prev_dim + 1


def test_vertical_spine_bridges_offline_wall_and_persists_cognitive_fabric(tmp_path: Path) -> None:
    def _pattern() -> np.ndarray:
        arr = np.full((240, 320, 3), 24, dtype=np.uint8)
        for y in range(12, 228, 16):
            for x in range(12, 308, 18):
                arr[y : y + 6, x : x + 8, :] = [235, 220, 75]
        return arr

    state_dir = tmp_path / "airplane_spine_state"
    mind1 = ScaleRotorEmbodiedMind(state_dir)
    spine1 = mind1.get_vertical_spine(max_lock_ticks=3)
    retina = ScreenRetinaSensor(frame_provider=_pattern)
    actuator = UniversalMotorActuator(dry_run=True)

    traces = [
        spine1.step_vertical_cycle(retina=retina, actuator=actuator, fg_title="bridge_test")
        for _ in range(5)
    ]

    # Even in an offline sandbox, the Telos world cycle must bridge to real local human corpora & reduce gap!
    world_traces = [t for t in traces if t.focus_target == "telos_human_world"]
    assert len(world_traces) >= 1
    assert world_traces[0].gap_reduced is True
    assert "bridge::local_human_corpus_airplane" in spine1.cognitive_fabric.bridges

    audit = spine1.observe_self_vs_human_boundary()
    m = audit["measured"]
    assert m["verified_bridges_count"] >= 5
    assert len(m["domains_bridged"]) >= 5
    assert m["cross_domain_isomorphisms_count"] >= 4
    assert m["rotor_transfers_count"] >= 1

    # Reload from disk and verify Cognitive Information Fabric + Bridges + Isomorphisms survive!
    mind2 = ScaleRotorEmbodiedMind(state_dir)
    spine2 = mind2.get_vertical_spine(max_lock_ticks=3)
    m2 = spine2.observe_self_vs_human_boundary()["measured"]
    assert m2["verified_bridges_count"] == m["verified_bridges_count"]
    assert m2["cross_domain_isomorphisms_count"] == m["cross_domain_isomorphisms_count"]
    assert m2["domains_bridged"] == m["domains_bridged"]
