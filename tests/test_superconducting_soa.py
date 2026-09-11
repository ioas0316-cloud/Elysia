import pytest
import math
import numpy as np

ce = pytest.importorskip("causal_engine")

def test_superconducting_soa_field_initialization():
    field = ce.SuperconductingSoAField(64)
    assert field.num_cells == 64
    assert len(field.lattice_phase) == 64
    assert len(field.signal_phase) == 64
    assert len(field.demarcation_wall) == 64
    assert len(field.macro_potential) == 64

def test_stage_1_demarcation_and_telos():
    field = ce.SuperconductingSoAField(64)

    # Set demarcation wall
    wall = list(field.demarcation_wall)
    for i in range(16, 21):
        wall[i] = 1.0
    field.demarcation_wall = wall

    # Set telos gradient
    telos = list(field.gradient_telos)
    for i in range(16):
        telos[i] = 0.5
    field.gradient_telos = telos

    ce.step_superconducting_transport(field, 0.1, 0.05, 0.02, 0.05, 0.1)

    assert field.micro_velocity[5] > 0.0
    assert field.demarcation_wall[18] == 1.0

def test_stage_2_phase_locking_zero_scattering():
    field = ce.SuperconductingSoAField(64)

    lattice_phase = [1.23] * 64
    field.lattice_phase = lattice_phase

    signal_phase = list(field.signal_phase)
    signal_phase[5] = 1.24  # Phase-locking
    signal_phase[15] = 4.00 # Decoherence
    field.signal_phase = signal_phase

    signal_amp = list(field.signal_amplitude)
    signal_amp[5] = 100.0
    signal_amp[15] = 100.0
    field.signal_amplitude = signal_amp

    ce.step_superconducting_transport(field, 0.2, 0.05, 0.02, 0.05, 0.1)

    assert field.coherence_gate[5] == 1.0
    assert field.coherence_gate[15] < 0.5
    assert field.signal_amplitude[6] > field.signal_amplitude[16]

def test_stage_3_hysteresis_engraving():
    field = ce.SuperconductingSoAField(64)

    lp = list(field.lattice_phase)
    sp = list(field.signal_phase)
    cg = list(field.coherence_gate)
    sa = list(field.signal_amplitude)

    lp[10] = 0.5
    sp[10] = 0.5
    cg[10] = 1.0
    sa[10] = 50.0

    field.lattice_phase = lp
    field.signal_phase = sp
    field.coherence_gate = cg
    field.signal_amplitude = sa

    for _ in range(10):
        ce.step_superconducting_transport(field, 0.1, 0.1, 0.05, 0.05, 0.1)

    assert field.macro_potential[10] > 0.0

def test_topological_field_2d():
    field = ce.TopologicalField2D(16, 16)
    assert field.width == 16
    assert field.height == 16

    phase = list(field.phase)
    amp = list(field.amplitude)
    for y in range(1, 15):
        for x in range(1, 15):
            idx = y * 16 + x
            dx = float(x) - 8.0
            dy = float(y) - 8.0
            phase[idx] = math.atan2(dy, dx) + math.pi
            amp[idx] = 10.0

    field.phase = phase
    field.amplitude = amp

    ce.step_multidim_topological_transport(field, 0.1, 0.1)

    assert field.vorticity[8 * 16 + 8] >= 0.0
