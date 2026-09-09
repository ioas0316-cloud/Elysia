import pytest
from simulators.telos_causal_framework_sim import TelosCausalFrameworkSimulator

def test_telos_causal_framework_simulator():
    sim = TelosCausalFrameworkSimulator(dim=8)
    comparison = sim.compare_frameworks()

    tc = comparison["telos_causal"]
    assert tc["if_branch_evaluations"] == 0
    assert tc["pointer_chase_latency"] == 0.0
    assert tc["cache_miss_rate"] == 0.0
    assert tc["rendering_overdraw"] == 0.0
    assert tc["bus_bytes_transferred"] == 0
    assert tc["context_switches"] == 0
    assert tc["mutex_lock_overhead"] == 0.0

    assert comparison["zero_friction_achieved"] is True
    print("Telos Causal Framework Integrated Simulator Test Passed.")

if __name__ == "__main__":
    test_telos_causal_framework_simulator()
