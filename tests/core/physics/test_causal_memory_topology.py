import numpy as np
from core.physics.causal_memory_topology import CausalMemoryTopology

def test_causal_memory_topology():
    mem = CausalMemoryTopology(capacity=50, feature_dim=4)

    slot1 = mem.allocate_object("Object A", [0.5, 0.5, 0.5, 0.5], resonance_freq=2.0)
    slot2 = mem.allocate_object("Object B", [0.51, 0.49, 0.5, 0.5], resonance_freq=2.0)
    slot3 = mem.allocate_object("Object C", [-0.8, -0.8, -0.8, -0.8], resonance_freq=9.0)

    payloads, metrics = mem.fetch_by_resonance([0.5, 0.5, 0.5, 0.5], target_freq=2.0)

    assert "Object A" in payloads
    assert "Object B" in payloads
    assert metrics["pointer_chase_latency"] == 0.0
    assert metrics["cache_miss_rate"] == 0.0
    print("CausalMemoryTopology test passed successfully.")

if __name__ == "__main__":
    test_causal_memory_topology()
