import pytest
import numpy as np
from core.physics.causal_engine import CausalNodePy, PrincipleRederivationEngine, PHASE_GAS, PHASE_FLUID, PHASE_CRYSTAL


def test_causal_node_bit_packing():
    # Test CausalNode bit layout encoding and decoding
    node = CausalNodePy.create(phase=PHASE_FLUID, potential=50000, bond_op=0x1234, topo_offset=-5)

    assert node.phase_state == PHASE_FLUID
    assert node.potential == 50000
    assert node.bond_operator == 0x1234
    assert node.topo_offset == -5

    # Test state updates
    node.phase_state = PHASE_CRYSTAL
    assert node.phase_state == PHASE_CRYSTAL

    node.potential = 123456
    assert node.potential == 123456

    node.topo_offset = 12
    assert node.topo_offset == 12


def test_principle_re_derivation_engine_convergence():
    engine = PrincipleRederivationEngine(num_nodes=64, tension_threshold=500.0, entropy_limit=0.1)

    # Initialize high potential difference
    potentials = np.zeros(64, dtype=np.float32)
    potentials[::2] = 2000.0
    potentials[1::2] = 100.0

    engine.initialize_field(potentials=potentials)

    res = engine.re_derive_phenomenon(target_potential_gradient=50.0, max_steps=50)

    assert res["converged_step"] > 0
    assert "final_crystallization" in res
    assert res["final_crystallization"] >= 0.0
