import pytest
import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from synaptic_architecture.resistance_bridge import ResistanceBridge
from core.physics.self_molding_engine import SelfMoldingCausalEngine

def test_resistance_bridge_causal_field_projection():
    """Verifies physical hardware friction projects into 3D CausalField."""
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("v1", "Alpha", np.array([1.0, 0.0, 0.0], dtype=np.float32), position=np.array([0,0,0], dtype=np.float32))
    v2 = InformationVoxel("v2", "Beta", np.array([0.9, 0.1, 0.0], dtype=np.float32), position=np.array([1,0,0], dtype=np.float32))
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("v1", "v2", strength=1.0)

    bridge = ResistanceBridge(causal_field=cf)
    mock_metrics = {"cpu": 0.9, "ram": 0.8, "io_pressure": 0.7, "friction": 0.82}
    
    # Execute projection
    bridge.project_to_causal_field(cf, metrics=mock_metrics)

    # Verify Yellow (Entropy) increased due to hardware friction
    assert v1.chromatic_vector[2] > 0.33
    assert v2.chromatic_vector[2] > 0.33

    # Verify beam tension increased
    beam = cf.beams[0]
    assert beam.current_tension > 0.0

def test_self_outpouring_potential_flow():
    """Verifies Self-Outpouring energy vector flow across voxels (Commandment Rules 11-12)."""
    cf = CausalField(dimensions=3)
    # v1 has high Flux (Red) & high initial potential
    v1 = InformationVoxel("v1", "HighFlux", np.array([1.0, 0.0, 0.0], dtype=np.float32), position=np.array([0,0,0], dtype=np.float32))
    v1.chromatic_vector = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    v1.potential = 5.0

    # v2 has lower potential but higher Yellow (Entropy)
    v2 = InformationVoxel("v2", "HighEntropy", np.array([0.9, 0.1, 0.0], dtype=np.float32), position=np.array([1,0,0], dtype=np.float32))
    v2.chromatic_vector = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    v2.potential = 1.0

    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("v1", "v2", strength=2.0)

    initial_pot_diff = v1.potential - v2.potential

    # Step simulation
    for _ in range(5):
        cf.step(0.1)

    # Potential difference should decrease as v1 pours out to v2
    final_pot_diff = v1.potential - v2.potential
    assert final_pot_diff < initial_pot_diff

    # v1 Red (Flux) should decrease as it converts to Blue (Order)
    assert v1.chromatic_vector[1] > 0.1 # Order increased

def test_topological_healing_and_rewiring():
    """Verifies SelfMoldingCausalEngine heals broken topology by forming new ConnectivityBeams."""
    sm = SelfMoldingCausalEngine(dimensions=3)
    sm.add_information("A", "NodeA", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    sm.add_information("B", "NodeB", np.array([0.95, 0.05, 0.0], dtype=np.float32))
    sm.add_information("C", "NodeC", np.array([0.90, 0.10, 0.0], dtype=np.float32))

    # Link A-B and break the beam manually
    sm.dynamics.link_voxels("A", "B", strength=1.0)
    sm.dynamics.beams[0].is_broken = True

    # Call heal_and_rewire
    new_links = sm.heal_and_rewire(max_new_links=2)

    # Should form new link between resonant unbroken nodes
    assert len(new_links) > 0
    active_beams = [b for b in sm.dynamics.beams if not b.is_broken]
    assert len(active_beams) > 0
