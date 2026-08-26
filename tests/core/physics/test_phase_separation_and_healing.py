import pytest
import numpy as np
from core.physics.phase_separation_attractor import (
    ToroidalEgoAttractor,
    ScaleShell,
    ImpedanceDampingController
)
from core.physics.phase_boundary_healing import (
    PhaseBoundaryHealingEngine,
    HysteresisScar
)

def test_scale_shell_and_impedance_damping():
    """Verify ScaleShell initialization and ImpedanceDampingController regulation."""
    micro = ScaleShell("micro", scale_z=0.1, r_min=0.1, r_max=1.0)
    macro = ScaleShell("macro", scale_z=1.0, r_min=1.0, r_max=10.0)
    assert micro.scale_z == 0.1
    assert macro.scale_z == 1.0

    controller = ImpedanceDampingController(e_void_max_threshold=10.0, damping_gain=0.5)
    gradients = np.full(10, 2.0, dtype=np.float32)

    # Normal regulation below threshold
    reg_e, reg_g, force = controller.regulate(5.0, gradients)
    assert reg_e == 5.0
    assert force == 0.0
    assert not controller.is_delusion_mitigated

    # Excessive E_Void regulation (Schizophrenia protection)
    reg_e_high, reg_g_high, force_high = controller.regulate(30.0, gradients)
    assert reg_e_high < 30.0
    assert force_high > 0.0
    assert controller.is_delusion_mitigated

def test_toroidal_ego_attractor_normal_and_flow_state():
    """Verify ToroidalEgoAttractor 4-stage phase separation, Torus coords, and Flow state."""
    attractor = ToroidalEgoAttractor(num_voxels=32)
    attractor.step(dt=0.01)

    # Compute Torus 3D coordinates
    X, Y, Z = attractor.get_torus_coordinates()
    assert X.shape == (32,)
    assert Y.shape == (32,)
    assert Z.shape == (32,)

    # Test Gamma Synchronization & Flow state
    attractor.enter_flow_state()
    assert attractor.is_flow_state
    assert attractor.is_asc_state
    assert attractor.e_void == 0.0
    assert attractor.coherence_score == 1.0

    attractor.exit_flow_state()
    assert not attractor.is_flow_state

def test_phase_boundary_healing_4_stages():
    """Verify 4-stage dynamic phase boundary rupture and evolutionary healing process."""
    attractor = ToroidalEgoAttractor(num_voxels=32)
    healing_engine = PhaseBoundaryHealingEngine(ego_attractor=attractor)

    # Stage 1: Rupture & Contraction
    e_void, contracted_r = healing_engine.inject_high_energy_shock(shock_intensity=15.0)
    assert healing_engine.healing_stage == 1
    assert healing_engine.is_ruptured
    assert contracted_r > healing_engine.base_r_min
    assert e_void > 0.0

    # Stage 2: Restorative Force Conversion
    pressure = healing_engine.convert_void_tension_to_restorative_force()
    assert healing_engine.healing_stage == 2
    assert pressure > 0.0

    # Stage 3: Radial Sweeping
    healing_engine.radial_phase_sweeping(dt=0.05)
    assert healing_engine.healing_stage == 3

    # Stage 4: Recrystallization & Scarring
    scar = healing_engine.recrystallize_and_form_scar()
    assert healing_engine.healing_stage == 4
    assert not healing_engine.is_ruptured
    assert isinstance(scar, HysteresisScar)
    assert len(healing_engine.hysteresis_scars) == 1
    assert healing_engine.contracted_r_min == healing_engine.base_r_min

def test_pathological_boundary_protection():
    """Verify damping controller prevents artificial psychosis (uncontrolled E_Void explosion)."""
    attractor = ToroidalEgoAttractor(num_voxels=32)
    # Force high phase gradient mismatch
    attractor.internal_phases[:] = 0.0
    attractor.external_phases[:] = np.pi
    attractor.external_freqs[:] = 10.0  # Mismatched freq

    attractor.filter_phase_impedance(dt=0.01)
    # E_Void should be capped and regulated below raw quadratic sum threshold
    assert attractor.e_void <= attractor.damping_controller.e_void_max_threshold * 2.5
    assert attractor.damping_controller.is_delusion_mitigated
