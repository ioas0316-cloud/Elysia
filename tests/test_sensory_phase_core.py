import pytest
import math
import torch
import elysia_phase_lock_cuda as eplc

def test_sensory_phase_config_default():
    cfg = eplc.SensoryPhaseConfig()
    assert cfg.lambda_1 == pytest.approx(1.0)
    assert cfg.lambda_2 == pytest.approx(0.5)
    assert cfg.lambda_3 == pytest.approx(1.0)
    assert cfg.kappa_1 == pytest.approx(1.0)
    assert cfg.alpha == pytest.approx(0.5)
    assert cfg.beta == pytest.approx(0.3)
    assert cfg.gamma_max == pytest.approx(3.0)
    assert cfg.phi_liquid == pytest.approx(0.4)
    assert cfg.phi_solid == pytest.approx(0.8)
    assert cfg.tau_shear == pytest.approx(2.0)

def test_sensory_phase_pipeline_execution():
    pipeline = eplc.SensoryPhaseCorePipeline()

    inputs = []

    # Node 0: Perfectly aligned motion & low noise -> High phase lock (Solid)
    in0 = eplc.SensoryStreamInput()
    in0.position_tension = eplc.Float4(0.0, 0.0, 0.0, 0.5) # x, y, z, tau
    in0.velocity_dtension = eplc.Float4(1.0, 0.0, 0.0, 0.0) # vx, vy, vz, dtau/dt
    in0.audio_spectrum = eplc.Float4(0.2, 0.5, 0.0, 100.0) # AL, AM, AH, w0
    in0.acceleration_grad = eplc.Float4(1.0, 0.0, 0.0, 1.0) # ax, ay, az, |grad AM|
    inputs.append(in0)

    # Node 1: High tension fluctuation -> Shear state
    in1 = eplc.SensoryStreamInput()
    in1.position_tension = eplc.Float4(1.0, 0.0, 0.0, 0.8)
    in1.velocity_dtension = eplc.Float4(0.0, 1.0, 0.0, 3.5) # dtau/dt = 3.5 > tau_shear
    in1.audio_spectrum = eplc.Float4(0.1, 0.1, 0.8, 440.0)
    in1.acceleration_grad = eplc.Float4(0.0, 2.0, 0.0, 0.0)
    inputs.append(in1)

    nodes, diagnostics = pipeline.process_frame_cpu(inputs)

    assert len(nodes) == 2
    assert len(diagnostics) == 2

    # Verify Node 0 phase & solid transition
    phase0 = nodes[0].metric_offdiag[3]
    assert phase0 > 0.8
    assert diagnostics[0].classification_st[0] == 2 # Solid state

    # Verify Node 1 shear transition
    assert diagnostics[1].classification_st[0] == 3 # Shear state

def test_gdi_and_saddle_point_hessian():
    pipeline = eplc.SensoryPhaseCorePipeline()

    inputs = []
    inp = eplc.SensoryStreamInput()
    inp.position_tension = eplc.Float4(-1.0, 0.5, 0.0, 0.2)
    inp.velocity_dtension = eplc.Float4(2.0, -1.0, 0.5, 0.1)
    inp.audio_spectrum = eplc.Float4(0.3, 0.4, 0.1, 220.0)
    inp.acceleration_grad = eplc.Float4(1.0, 0.0, 0.0, 0.5)
    inputs.append(inp)

    nodes, diagnostics = pipeline.process_frame_cpu(inputs)

    gdi = diagnostics[0].position_gdi[3]
    hess_det = diagnostics[0].saddle_hessian[0]
    eig_max = diagnostics[0].saddle_hessian[1]
    eig_min = diagnostics[0].saddle_hessian[2]

    # Check numerical validity
    assert not math.isnan(gdi)
    assert not math.isnan(hess_det)
    assert eig_max >= eig_min
