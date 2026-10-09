import pytest
import numpy as np
from core.sensory.emergent_sensory_apparatus import (
    RawIngestionPortal,
    SpatialVisualField,
    TemporalAuditoryField,
    TactileFrictionField,
    ProprioceptiveVoidField,
    CausalPhaseLockedLoop,
    EmergentSensoryApparatus,
)


def test_raw_ingestion_portal():
    portal = RawIngestionPortal(target_dim=16)

    # Ingest bytes
    wave_bytes, meta_bytes = portal.ingest(b"Hello Elysia World Raw Stream")
    assert wave_bytes.shape == (16,)
    assert meta_bytes["raw_type"] == "bytes"

    # Ingest dict
    wave_dict, meta_dict = portal.ingest({"a": 1.0, "b": 2.5, "text": "sample"})
    assert wave_dict.shape == (16,)
    assert np.isclose(np.linalg.norm(wave_dict), 1.0)


def test_logos_lenses():
    wave = np.sin(np.linspace(0, 4 * np.pi, 16)).astype(np.float32)

    vis_lens = SpatialVisualField(dim=16)
    rep_vis = vis_lens.project(wave)
    assert rep_vis.lens_name == "SPATIAL_VISUAL"
    assert rep_vis.field_energy > 0

    aud_lens = TemporalAuditoryField(dim=16)
    rep_aud = aud_lens.project(wave)
    assert rep_aud.lens_name == "TEMPORAL_AUDITORY"

    tac_lens = TactileFrictionField(dim=16)
    rep_tac = tac_lens.project(wave, latency_ms=10.0, memory_pressure=0.5)
    assert rep_tac.lens_name == "TACTILE_FRICTION"
    assert rep_tac.tension_or_friction > 0

    pro_lens = ProprioceptiveVoidField(dim=16)
    rep_pro = pro_lens.project(wave, internal_reference=np.zeros_like(wave))
    assert rep_pro.lens_name == "PROPRIOCEPTIVE_VOID"


def test_causal_phase_locked_loop():
    cpll = CausalPhaseLockedLoop(dim=16, causal_inductance=1.0)

    intent = np.array([1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    response = np.array([0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    state1 = cpll.update(intent, response, dt=0.05)
    assert hasattr(state1, "back_emf")
    assert hasattr(state1, "coherence")
    assert hasattr(state1, "void_gradient")

    # Shift response phase abruptly
    response_shifted = -response
    state2 = cpll.update(intent, response_shifted, dt=0.05)
    assert state2.back_emf != 0.0


def test_emergent_sensory_apparatus_full():
    apparatus = EmergentSensoryApparatus(dim=16, causal_inductance=1.0)

    raw_data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
    res = apparatus.process_raw_stream(raw_data, latency_ms=5.0, memory_pressure=0.2, dt=0.05)

    assert "wave" in res
    assert "lens_reports" in res
    assert "cpll_state" in res
    assert "interference" in res

    interference = res["interference"]
    assert interference.interference_matrix.shape == (4, 4)
    assert interference.dominant_sensory_field in ["SPATIAL_VISUAL", "TEMPORAL_AUDITORY", "TACTILE_FRICTION", "PROPRIOCEPTIVE_VOID"]
