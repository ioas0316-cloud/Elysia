"""
Unit tests for 3x3x3 Dynamic Isotropic Wave Lattice Memory Architecture
and Generative Phase Molding Engine.
"""

import pytest
import numpy as np

from core.memory.fractal_cube_lattice_engine import FractalCubeLatticeEngine
from synaptic_architecture.generative_molding_engine import GenerativeMoldingEngine


def test_fractal_cube_lattice_initialization():
    engine = FractalCubeLatticeEngine(num_phase_bins=16, resonance_threshold=0.7)

    # 27 nodes total
    assert len(engine.nodes) == 27

    # Core node (0,0,0)
    assert (0, 0, 0) in engine.nodes
    assert engine.nodes[(0, 0, 0)]["node_type"] == "core"

    # Face, Edge, Vertex counts
    node_types = [n["node_type"] for n in engine.nodes.values()]
    assert node_types.count("core") == 1
    assert node_types.count("face") == 6
    assert node_types.count("edge") == 12
    assert node_types.count("vertex") == 8


def test_isotropic_distance_and_spherical_mapping():
    engine = FractalCubeLatticeEngine()

    # Center (0,0,0) distance to face (1,0,0) is 1.0
    d_face = engine.get_isotropic_distance((0, 0, 0), (1, 0, 0))
    assert pytest.approx(d_face) == 1.0

    # Center (0,0,0) distance to vertex (1,1,1) is sqrt(3)
    d_vertex = engine.get_isotropic_distance((0, 0, 0), (1, 1, 1))
    assert pytest.approx(d_vertex) == np.sqrt(3)

    # Check spherical coordinates for vertex (1,1,1)
    r, theta, phi = engine.spherical_coords[(1, 1, 1)]
    assert pytest.approx(r) == np.sqrt(3)


def test_face_rotor_rotation():
    engine = FractalCubeLatticeEngine()

    # Set known spectrum on face node (1, 0, 0)
    engine.nodes[(1, 0, 0)]["spectrum"][0] = 5.0

    # Rotate z-axis slice at z=0 counter-clockwise 90 degrees (1 turn)
    engine.rotate_face_rotor(axis='z', layer=0, steps=1)

    # Node at (1, 0, 0) should rotate to (0, 1, 0) in xy-plane
    assert engine.nodes[(0, 1, 0)]["spectrum"][0] == 5.0


def test_intrinsic_frame_puzzle_resonance_injection():
    engine = FractalCubeLatticeEngine(num_phase_bins=16)

    test_spectrum = np.random.randn(16)
    matched_coord = engine.inject_wave_fragment(test_spectrum, amplitude=1.5, phase=0.5)

    # Should match one of the 27 coordinates
    assert matched_coord in engine.nodes
    assert np.linalg.norm(engine.nodes[matched_coord]["spectrum"]) > 0


def test_central_core_rectification():
    engine = FractalCubeLatticeEngine(num_phase_bins=16)

    # Inject wave signals into several nodes
    engine.inject_wave_fragment(np.ones(16) * 2.0, amplitude=1.0, phase=0.0)
    engine.inject_wave_fragment(np.ones(16) * 4.0, amplitude=2.0, phase=np.pi / 4.0)

    rect_result = engine.rectify_central_core()

    assert rect_result["macro_rotor_amplitude"] > 0
    assert 0.0 <= rect_result["rectified_phase"] <= 2 * np.pi


def test_retrocausal_tracing():
    engine = FractalCubeLatticeEngine(num_phase_bins=16)

    # Perform operations
    c1 = engine.inject_wave_fragment(np.ones(16) * 3.0)
    engine.rotate_face_rotor(axis='x', layer=1, steps=1)

    assert len(engine.transformation_history) == 2

    # Perform retrocausal back-tracing
    undone_log = engine.retrocausal_trace(steps=2)

    assert len(undone_log) == 2
    assert len(engine.transformation_history) == 0


def test_fractal_scale_expansion():
    engine = FractalCubeLatticeEngine(num_phase_bins=16)
    engine.inject_wave_fragment(np.ones(16) * 5.0)

    higher_lattice = engine.expand_fractal_scale()

    assert isinstance(higher_lattice, FractalCubeLatticeEngine)
    assert len(higher_lattice.nodes) == 27
    assert higher_lattice.nodes[(0, 0, 0)]["amplitude"] > 0


def test_generative_molding_engine():
    lattice = FractalCubeLatticeEngine(num_phase_bins=16)
    lattice.inject_wave_fragment(np.ones(16) * 3.0, amplitude=1.0, phase=0.2)

    molding_engine = GenerativeMoldingEngine(resolution=8)

    # Density field generation
    density = molding_engine.mold_structure_from_lattice(lattice.nodes, lattice.macro_rotor_phase)
    assert density.shape == (8, 8, 8)
    assert np.all(density >= 0.0) and np.all(density <= 1.0)

    # Stress tensor topology
    stress = molding_engine.compute_stress_topology()
    assert stress.shape == (8, 8, 8, 3)

    # Emergent mesh extraction
    mesh = molding_engine.generate_emergent_mesh(density_threshold=0.5)
    assert "vertices" in mesh
    assert "simplex_edges" in mesh
    assert "rigidity_score" in mesh
