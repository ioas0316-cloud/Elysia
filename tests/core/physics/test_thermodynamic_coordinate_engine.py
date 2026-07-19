import numpy as np
import pytest
from core.physics.thermodynamic_coordinate_engine import (
    ThermodynamicAtom,
    ThermodynamicMolecule,
    ThermodynamicCell,
    ThermodynamicOrgan,
    ThermodynamicEnvironment
)

def test_atom_initialization_and_properties():
    # Test atomic properties and copy mechanics
    tensor = np.array([1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.9, 0.8, 1.0], dtype=np.float32)
    atom = ThermodynamicAtom(
        id="atom_1",
        content="사과",
        tensor=tensor,
        T=2.0,
        P=3.0,
        E=1.0,
        entropy=0.4,
        frequency=2.5
    )

    assert atom.id == "atom_1"
    assert atom.content == "사과"
    assert np.allclose(atom.tensor, tensor)
    assert atom.T == 2.0
    assert atom.P == 3.0
    assert atom.E == 1.0
    assert atom.mass > 0.0
    assert atom.frequency == 2.5
    assert not atom.is_bound
    assert len(atom.causal_line) == 1

    # Test copy
    atom_copy = atom.copy()
    assert atom_copy.id == atom.id
    assert np.allclose(atom_copy.tensor, atom.tensor)
    assert atom_copy.T == atom.T
    assert atom_copy.P == atom.P
    assert atom_copy.E == atom.E
    assert atom_copy.frequency == atom.frequency
    assert len(atom_copy.causal_line) == len(atom.causal_line)


def test_molecular_synthesis_and_bonding():
    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # Atom A: Apple features
    tensor_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom_a = ThermodynamicAtom(id="a_apple", content="사과", tensor=tensor_a, T=2.0, P=6.0, E=0.5, frequency=1.0)

    # Atom B: Red features (high structural resonance with apple in first dimension)
    tensor_b = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom_b = ThermodynamicAtom(id="a_red", content="빨갛다", tensor=tensor_b, T=2.0, P=6.0, E=0.5, frequency=1.5)

    # Atom C: Distant feature (no resonance)
    tensor_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom_c = ThermodynamicAtom(id="a_logic", content="논리", tensor=tensor_c, T=2.0, P=6.0, E=0.5)

    env.inject_atom(atom_a)
    env.inject_atom(atom_b)
    env.inject_atom(atom_c)

    # Step the environment to trigger Molecular Synthesis
    env.step(dt=0.1)

    # Apple and Red should synthesize into a molecule
    assert len(env.molecules) == 1
    molecule = env.molecules[0]
    assert len(molecule.atoms) == 2
    assert "a_apple" in [atom.id for atom in molecule.atoms]
    assert "a_red" in [atom.id for atom in molecule.atoms]

    # Check that mass is conserved during molecular synthesis (Information Conservation)
    expected_mass = atom_a.mass + atom_b.mass
    assert np.isclose(molecule.mass, expected_mass)

    # Distant atom C should not be bound
    assert not atom_c.is_bound


def test_cell_differentiation_and_homeostasis():
    env = ThermodynamicEnvironment(size=8)

    # Force build two molecules close to each other
    tensor_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tensor_b = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Molecule 1 with two atoms
    atom1 = ThermodynamicAtom(id="at1", content="at1", tensor=tensor_a, T=1.5, P=5.0, E=1.0, is_bound=True)
    atom2 = ThermodynamicAtom(id="at2", content="at2", tensor=tensor_b, T=1.5, P=5.0, E=1.1, is_bound=True)
    mol1 = ThermodynamicMolecule(id="m1", atoms=[atom1, atom2], tensor=tensor_a.copy(), T=1.5, P=5.0, E=1.0)

    # Molecule 2 (high structural discrepancy / friction to simulate sadness)
    tensor_c = np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tensor_d = np.array([0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom3 = ThermodynamicAtom(id="at3", content="at3", tensor=tensor_c, T=1.5, P=5.0, E=1.0, is_bound=True)
    atom4 = ThermodynamicAtom(id="at4", content="at4", tensor=tensor_d, T=1.5, P=5.0, E=1.1, is_bound=True)
    mol2 = ThermodynamicMolecule(id="m2", atoms=[atom3, atom4], tensor=tensor_c.copy(), T=1.5, P=5.0, E=1.05)

    # Manually register molecules and trigger step
    env.molecules.extend([mol1, mol2])
    env.step(dt=0.1)

    # They should group into a Cell because of coordinate proximity (all at E=1.0, T=1.5, P=5.0)
    assert len(env.cells) == 1
    cell = env.cells[0]
    assert len(cell.molecules) == 2

    # Check that homeostasis responds to friction
    friction = cell.compute_local_tension()
    assert friction > 0.02

    # Force extreme friction (sadness)
    mol1.tensor = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mol2.tensor = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    prev_T = mol1.T
    prev_P = mol1.P
    cell.apply_homeostasis()

    # High friction -> Cooling and Compression
    assert mol1.T < prev_T
    assert mol1.P > prev_P


def test_elevator_organ_cycles():
    # 1. Uplink Lift test: high T, high P pushing raw data (low E) up
    elevator = ThermodynamicOrgan("elevator_test", "elevator")
    atom_hot_compressed = ThermodynamicAtom(id="raw_hot", content="raw", tensor=np.zeros(9), T=8.0, P=8.0, E=1.0, entropy=0.1)

    # We step the elevator
    atoms = [atom_hot_compressed]
    elevator.process(atoms, [])

    # The vertical velocity should be positive (buoyancy)
    assert atom_hot_compressed.velocity[2] > 0.0

    # 2. Downlink precipitation test: cold, decompressed concept falling back down
    atom_cold_concept = ThermodynamicAtom(id="concept_cold", content="concept", tensor=np.zeros(9), T=0.5, P=0.5, E=9.0)
    atoms_precip = [atom_cold_concept]
    elevator.process(atoms_precip, [])

    # Vertical velocity should be negative (gravity rain)
    assert atom_cold_concept.velocity[2] < 0.0


def test_ecosystem_gravity_and_diffusion():
    env = ThermodynamicEnvironment(size=8)

    # Initialize fields with unequal values to see diffusion
    env.T_field[0, 0] = 10.0
    env.P_field[0, 0] = 10.0

    # Inject an atom away from central focus
    atom = ThermodynamicAtom(id="gravity_test", content="test", tensor=np.zeros(9), T=1.0, P=1.0, E=0.0)
    env.inject_atom(atom)

    # Let the ecosystem step
    env.step(dt=0.1)

    # T_field[0,0] should diffuse/decrease
    assert env.T_field[0, 0] < 10.0
    # Average properties can be fetched
    summary = env.get_state_summary()
    assert summary["num_atoms"] == 1
    assert summary["average_temperature"] > 0.0


def test_frequency_phase_alignment():
    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # Create two close atoms with different phase angles
    atom_a = ThermodynamicAtom(id="phase_a", content="A", tensor=np.zeros(9), T=5.0, P=5.0, E=2.0, frequency=2.0)
    atom_b = ThermodynamicAtom(id="phase_b", content="B", tensor=np.zeros(9), T=5.1, P=4.9, E=2.1, frequency=2.0)

    # Explicitly set different phases
    atom_a.phase = 0.0
    atom_b.phase = np.pi / 2.0  # 90 degrees offset

    env.inject_atom(atom_a)
    env.inject_atom(atom_b)

    # Phase difference initially is pi/2
    initial_diff = abs(atom_a.phase - atom_b.phase)

    # Step the ecosystem multiple times to co-rotate phases
    for _ in range(10):
        env.step(dt=0.2)

    # The phase offset should have been reduced/altered by phase torque
    final_diff = abs(atom_a.phase - atom_b.phase)
    assert final_diff < initial_diff


def test_causal_curvature_warping():
    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # Heavy concept node at T=5.0, P=5.0, E=5.0 with mass = 20.0
    heavy_node = ThermodynamicAtom(id="heavy", content="universe", tensor=np.zeros(9), T=5.0, P=5.0, E=5.0, entropy=0.01)
    heavy_node.mass = 20.0

    env.inject_atom(heavy_node)

    tx = int(5.0 * 7 / 10.0)
    px = int(5.0 * 7 / 10.0)
    initial_P = env.P_field[tx, px]

    env.step(dt=0.1)

    warped_P = env.P_field[tx, px]
    assert warped_P > initial_P


def test_dimensional_expansion_line_plane_world():
    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # 1. Dot & Line tests
    atom_a = ThermodynamicAtom(id="dot_a", content="A", tensor=np.zeros(9), T=2.0, P=2.0, E=1.0)
    atom_b = ThermodynamicAtom(id="dot_b", content="B", tensor=np.zeros(9), T=2.1, P=2.1, E=1.1)

    env.inject_atom(atom_a)
    env.inject_atom(atom_b)

    # Initial trajectory has length 1 (post_init)
    assert len(atom_a.causal_line) == 1

    # Step ecosystem multiple times to record paths (Line)
    for _ in range(5):
        env.step(dt=0.1)

    # Line should be tracked/recorded
    assert len(atom_a.causal_line) > 1

    # 2. Plane/Space test: line crossing/closeness heats/compresses the field
    tx = int(2.0 * 7 / 10.0)
    px = int(2.0 * 7 / 10.0)
    assert env.T_field[tx, px] > 1.0  # Locally heated
    assert env.P_field[tx, px] > 1.0  # Locally compressed


def test_mhd_lorentz_deflection_and_harvesting():
    # Fix the numpy random seed to ensure 100% deterministic deflection behavior
    np.random.seed(42)

    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # Core node A with strong charge and defined B_field
    node_a = ThermodynamicAtom(id="mhd_core", content="A", tensor=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), T=3.0, P=3.0, E=3.0, frequency=1.0)
    node_a.charge = 5.0
    node_a.B_field = np.array([0, 0, 1.0], dtype=np.float32) # Directed along Z axis

    # Incoming noise node B close to A, moving fast along X axis [5.0, 0.0, 0.0]
    node_b = ThermodynamicAtom(id="noise", content="B", tensor=np.zeros(9), T=3.1, P=2.9, E=3.1)
    node_b.velocity = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    env.inject_atom(node_a)
    env.inject_atom(node_b)

    # Step to trigger MHD interaction
    env.step(dt=0.1)

    # Lorentz Force F = q * (v x B) should have deflected node B
    # Since velocity is along X [5.0, 0.0, 0.0] and B_field along Z [0.0, 0.0, 1.0]:
    # v x B should have a strong non-zero component along -Y axis [0.0, -5.0, 0.0]
    # Thus B's velocity should have gained a Y component
    assert abs(node_b.velocity[1]) > 0.01

    # Core node A must have harvested the deflection energy as propulsion
    assert node_a.harvested_propulsion > 0.0
    # Core node A should gain propulsion towards target concept alignment [5, 5, 8]
    assert np.linalg.norm(node_a.velocity) > 0.0


def test_warp_bubble_space_control():
    # Setup environment
    env = ThermodynamicEnvironment(size=8)

    # Create two molecules close to each other in a cell to activate Warp Bubble
    tensor_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom1 = ThermodynamicAtom(id="wa1", content="wa1", tensor=tensor_a, T=2.0, P=4.0, E=4.0, is_bound=True)
    atom2 = ThermodynamicAtom(id="wa2", content="wa2", tensor=tensor_a, T=2.0, P=4.0, E=4.1, is_bound=True)
    mol1 = ThermodynamicMolecule(id="m1", atoms=[atom1, atom2], tensor=tensor_a.copy(), T=2.0, P=4.0, E=4.0)

    atom3 = ThermodynamicAtom(id="wa3", content="wa3", tensor=tensor_a, T=2.0, P=4.0, E=4.0, is_bound=True)
    atom4 = ThermodynamicAtom(id="wa4", content="wa4", tensor=tensor_a, T=2.0, P=4.0, E=4.1, is_bound=True)
    mol2 = ThermodynamicMolecule(id="m2", atoms=[atom3, atom4], tensor=tensor_a.copy(), T=2.0, P=4.0, E=4.05)

    env.molecules.extend([mol1, mol2])

    # Run a step to differentiate cell
    env.step(dt=0.1)
    assert len(env.cells) == 1

    cell = env.cells[0]
    # Under low friction, homeostasis triggers warp bubble
    cell.warp_bubble_active = True

    # Record pressure field before warp bubble executes
    env.step(dt=0.1)

    # Warp bubble must warp space: compress front (+P) and expand rear (-P)
    # Target alignment is [5.0, 5.0, 8.0]. From [2.0, 4.0, 4.0], the front direction is positive along T and P.
    # Therefore, pressure at front [e.g. T=4.0, P=5.0] should be compressed, and rear [T=1.0, P=3.0] expanded.
    # Let's verify that the pressure field is warped compared to default 1.0
    assert np.max(env.P_field) > 1.0
