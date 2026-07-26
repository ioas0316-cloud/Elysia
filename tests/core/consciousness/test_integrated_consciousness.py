import os
import pytest
from core.consciousness.autonomous_loop import ConsciousnessLoop

def test_consciousness_loop_with_self_molding_causal_engine():
    """
    Verifies that ConsciousnessLoop integrates SelfMoldingCausalEngine,
    projects hardware friction to 3D CausalField, molds 3D topology,
    and advances life cycles seamlessly.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    loop = ConsciousnessLoop(corpus_path=data_dir, data_dir=data_dir)

    # 1. Verify 3D Causal Engine initialization
    assert hasattr(loop, "causal_engine")
    assert loop.causal_engine is not None
    assert loop.bridge.causal_field is not None

    # 2. Run 3 life cycles
    for _ in range(3):
        result = loop.process_life_cycle()
        assert "cycle" in result
        assert "hw_friction" in result

    # 3. Check that 3D voxels were populated in SelfMoldingCausalEngine
    topology = loop.causal_engine.dynamics.get_topology()
    assert len(topology["voxels"]) >= 3
