import pytest
import numpy as np
from simulators.causal_grid_sim import CausalTilemapSimulator

def test_causal_loop_formation_and_engram_escape():
    """
    [Causal Loop & Engram Escape Integration Test]
    Verifies:
    1. Without the Oak's Parcel Engram exposed, the agent enters an infinite loop between Pallet Town and Viridian City.
    2. When the Oak's Parcel Engram is exposed in the Causal Field, top-down potential gradient forces the agent to escape the loop and successfully reach Pewter City.
    """
    sim = CausalTilemapSimulator()

    # Step 1: Run 15 steps without Engram exposure -> verify loop behavior
    for _ in range(15):
        sim.step()

    # Check that agent was oscillating between Pallet Town (~0, 0) and Viridian City (~0, 10)
    visited_y = [p[1] for p in sim.visited_positions]
    assert max(visited_y) >= 8.0 # Reached Viridian area
    assert min(visited_y) <= 1.0 # Bounced back to Pallet area

    # Step 2: Expose High-Level Engram ("Oak's Parcel / Delivery to Pewter City")
    sim.set_engram_exposure(True)
    assert sim.field.engrams["oak_parcel"].active is True

    # Step 3: Run another 15 steps with Engram active -> verify loop escape & arrival at Pewter City
    for _ in range(15):
        sim.step()

    final_pos = sim.agent_voxel.position
    dist_to_pewter = np.linalg.norm(final_pos - sim.locations["Pewter City"])

    # Assert agent successfully reaches Pewter City area (within 1.5 units)
    assert dist_to_pewter < 1.5

    # Assert Engram gravitational force was actively applied
    engram_forces = [log["engram_force_norm"] for log in sim.history if log["engram_active"]]
    assert len(engram_forces) > 0
    assert max(engram_forces) > 1.0

if __name__ == "__main__":
    pytest.main([__file__])
