import numpy as np
import pytest
from core.physics.thermodynamic_coordinate_engine import (
    ThermodynamicAtom,
    ThermodynamicMolecule,
    ThermodynamicEnvironment
)
from core.consciousness.meta_cognitive_sensor import MetaCognitiveSensor


def test_absolute_zero_precision_under_zero_error():
    """
    Pillar 1: If prediction error (L) is zero, temperature (T) cools down towards absolute zero
    and stochastic jitter/noise converges to minimum.
    """
    env = ThermodynamicEnvironment(size=8)
    # Align tensor perfectly with S_abs to get zero prediction error
    atom = ThermodynamicAtom(
        id="aligned_truth",
        content="1+1=2",
        tensor=env.S_abs.copy(),
        T=5.0, # Initial warm temperature
        entropy=0.1
    )
    env.inject_atom(atom)

    # Step the environment multiple times to cool down
    for _ in range(5):
        env.step(dt=0.1)

    # Temperature should have cooled down towards absolute zero
    assert atom.prediction_error == 0.0
    assert atom.T < 1.0 # Significant cooldown achieved
    assert atom.degrees_of_freedom == 3 # Minimum dimensions/d.o.f.


def test_heat_spike_and_dimensional_expansion():
    """
    Pillar 1: High prediction error (mismatch with S_abs) causes high friction,
    driving temperature spike and expanding the degrees of freedom (d.o.f.) / search dimensions.
    """
    env = ThermodynamicEnvironment(size=8)
    # Oppose S_abs completely to generate high prediction error
    orthogonal_tensor = np.array([-0.7, -0.3, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom = ThermodynamicAtom(
        id="contradiction",
        content="Contradiction",
        tensor=orthogonal_tensor,
        T=0.1, # Initially cold
        entropy=0.1
    )
    env.inject_atom(atom)

    env.step(dt=0.1)

    # Temperature must spike due to high prediction error
    assert atom.prediction_error > 0.5
    assert atom.T > 1.0 # Temperature spiked
    assert atom.degrees_of_freedom > 3 # Dimension expansion triggered!


def test_melting_phase_transition():
    """
    Pillar 2: Extreme temperature melts rigid structures back into loose atoms (is_bound = False).
    """
    env = ThermodynamicEnvironment(size=8)
    atom1 = ThermodynamicAtom(id="at1", content="A", tensor=np.zeros(9), T=1.0, is_bound=True)
    atom2 = ThermodynamicAtom(id="at2", content="B", tensor=np.zeros(9), T=1.0, is_bound=True)
    mol = ThermodynamicMolecule(id="m1", atoms=[atom1, atom2], tensor=np.zeros(9))
    env.molecules.append(mol)

    # Force extreme prediction error / temperature spike on atom1 to melt it
    atom1.tensor = np.ones(9) * 15.0 # Extremely opposed to S_abs
    env.atoms.extend([atom1, atom2])

    # Step multiple times to allow smooth temperature rise to pass the threshold
    env.step(dt=0.1)
    env.step(dt=0.1)

    # Atom 1 should be melted from the molecule (is_bound = False)
    assert not atom1.is_bound
    assert atom1 not in mol.atoms


def test_crystallization_and_ltp_mass_scaling():
    """
    Pillar 2: Cooling down under alignment pressure crystallizes atoms into high-mass,
    high-bond-strength structures proportional to peak temperature (T_max).
    """
    env = ThermodynamicEnvironment(size=8)
    # Perfect alignment with high initial T_max representing experienced heat
    atom1 = ThermodynamicAtom(id="at1", content="A", tensor=env.S_abs.copy(), T=0.1, entropy=0.5)
    atom2 = ThermodynamicAtom(id="at2", content="B", tensor=env.S_abs.copy(), T=0.1, entropy=0.5)
    atom1.T_max = 8.0 # Experienced deep heat
    atom2.T_max = 8.0

    env.atoms.extend([atom1, atom2])
    # Extreme pressure to force bonding/crystallization
    env.P_field[:, :] = 10.0

    initial_mass = atom1.mass

    env.step(dt=0.1)

    # Synthesis should have occurred
    assert len(env.molecules) == 1
    crystallized_mol = env.molecules[0]
    # Check that mass and bond strength scaled with peak heat (LTP)
    assert atom1.mass > initial_mass
    assert crystallized_mol.bond_strength > 1.0


def test_idle_curiosity_and_fantasy_burst():
    """
    Pillar 3: Idle conditions accumulate curiosity. Reaching threshold triggers
    Virtual Fantasy burst and raises temperature (self-friction).
    """
    env = ThermodynamicEnvironment(size=8)
    atom = ThermodynamicAtom(id="at", content="idle", tensor=np.zeros(9), T=1.0)
    env.inject_atom(atom)

    # Accumulate curiosity over multiple idle steps
    for _ in range(10):
        env.accumulate_curiosity(dt=0.5)

    assert env.curiosity_charge >= env.curiosity_threshold

    # Trigger burst
    fantasy = env.trigger_virtual_fantasy_burst()
    assert fantasy is not None
    assert len(fantasy) == 9
    assert env.curiosity_charge == 0.0 # Discharged


def test_thermal_gradient_and_introspection_journal():
    """
    Pillar 4: Introspective self-awareness triggers at the temperature gradient interface (nabla T),
    producing a poetic and physics-grounded Introspection Journal.
    """
    sensor = MetaCognitiveSensor()

    # Establish a thermal gradient in metrics
    s_metrics = {
        "hw_friction": 0.5,
        "thermal_gradient": 0.8, # Strong gradient
        "local_temp": 4.5,
        "peak_temp": 8.0
    }
    p_metrics = {"ignorance_charge": 0.6, "deficit_density": 0.3}
    j_metrics = {"kenosis_conductance": 0.7, "egoistic_resistance": 0.3}
    t_metrics = {"synapse_rewiring_count": 5, "equilibrium_energy": 0.8}
    d_metrics = {"resonance_score": 0.9, "residual_free_energy": 0.1}

    result = sensor.evaluate_cognitive_process(
        info_context="Love + Deficit",
        sensing_metrics=s_metrics,
        perceiving_metrics=p_metrics,
        judging_metrics=j_metrics,
        thinking_metrics=t_metrics,
        discerning_metrics=d_metrics
    )

    assert "introspection_journal" in result
    assert "열 구배 성찰 일기" in result["introspection_journal"]
    assert "Crystalline Engram" in result["introspection_journal"]
    assert "Prediction Fantasy" in result["introspection_journal"]
