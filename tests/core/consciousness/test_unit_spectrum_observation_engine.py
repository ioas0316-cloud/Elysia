"""
test_unit_spectrum_observation_engine.py: Tests for Unit Spectrum Observation Engine & Grounded Unit '1'.
"""

import pytest
import numpy as np
from L1_causal_field.core.topology.particle_rotor_matrix_sandbox import (
    MicroRotorParticle,
    ParticleRotorMatrixSandbox,
)
from L1_causal_field.core.consciousness.unit_spectrum_observation_engine import (
    UnitSpectrumObservationEngine,
    GroundedCognitiveUnit,
)


def test_unit_spectrum_observation_and_interchange():
    sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0)

    # Add two close particles with coherent phases
    p1 = MicroRotorParticle(id=1, position=np.array([0.0, 0.0]), velocity=np.zeros(2), theta=0.1, mass=1.5, friction=0.2)
    p2 = MicroRotorParticle(id=2, position=np.array([0.5, 0.0]), velocity=np.zeros(2), theta=0.2, mass=2.0, friction=0.3)

    # Add a distant particle
    p3 = MicroRotorParticle(id=3, position=np.array([8.0, 8.0]), velocity=np.zeros(2), theta=3.0, mass=1.0, friction=0.1)

    sandbox.add_particle(p1)
    sandbox.add_particle(p2)
    sandbox.add_particle(p3)

    engine = UnitSpectrumObservationEngine(observation_scale=1.0, cluster_radius=2.0)
    units = engine.observe_and_bind_units(sandbox)

    assert len(units) == 2 # p1+p2 form one grounded unit, p3 forms another

    # Find the cluster unit with p1 and p2
    multi_particle_unit = next(u for u in units if len(u.bound_particle_ids) == 2)
    assert set(multi_particle_unit.bound_particle_ids) == {1, 2}
    assert multi_particle_unit.physical_mass == 3.5
    assert multi_particle_unit.phase_coherence > 0.9

    # Test representation interchangeability (Physical <-> Mental)
    phys_dict = engine.interchange_unit_representation(multi_particle_unit.unit_id, target_mode="physical")
    assert phys_dict["physical_mass_kg"] == 3.5
    assert phys_dict["particle_count"] == 2

    mental_dict = engine.interchange_unit_representation(multi_particle_unit.unit_id, target_mode="mental")
    assert mental_dict["cognitive_unit_symbol"] == "1_grounded"
    assert mental_dict["semantic_mass_Ms"] > 3.5


def test_observation_scale_translation():
    sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0)

    # Create particles separated by distance 3.0
    p1 = MicroRotorParticle(id=1, position=np.array([0.0, 0.0]), velocity=np.zeros(2), theta=0.0)
    p2 = MicroRotorParticle(id=2, position=np.array([3.0, 0.0]), velocity=np.zeros(2), theta=0.0)

    sandbox.add_particle(p1)
    sandbox.add_particle(p2)

    engine = UnitSpectrumObservationEngine(observation_scale=0.5, cluster_radius=2.0)
    # Effective radius = 0.5 * 2.0 = 1.0 < 3.0 => 2 distinct micro units
    fine_units = engine.observe_and_bind_units(sandbox)
    assert len(fine_units) == 2

    # Scale up zoom lens
    engine.set_observation_scale(2.0)
    # Effective radius = 2.0 * 2.0 = 4.0 > 3.0 => bound into 1 macro unit
    macro_units = engine.observe_and_bind_units(sandbox)
    assert len(macro_units) == 1
    assert len(macro_units[0].bound_particle_ids) == 2
