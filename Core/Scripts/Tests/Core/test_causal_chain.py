import py
import torch
import pytest
from Core.Digestion.digestive_system import DigestiveSystem
from Core.L2_Metabolism.Memory.fractal_causality import CausalRole

class MockElysia:
    def __init__(self):
        self.bridge = None
        self.graph = None

def test_active_probing_chain():
    elysia = MockElysia()
    stomach = DigestiveSystem(elysia)
    
    # Create a dummy weight tensor (Linear layer-like)
    weights = torch.randn(10, 128)
    
    # Run Active Probe
    probe_results = stomach.active_probe(weights, "test_layer")
    
    # Verify 4-step keys
    assert "cause" in probe_results
    assert "structure" in probe_results
    assert "function" in probe_results
    assert "reality" in probe_results
    
    print(f"\n[TEST] Probe Cause: {probe_results['cause']}")
    print(f"[TEST] Probe Structure: {probe_results['structure']}")
    print(f"[TEST] Probe Function: {probe_results['function']}")
    print(f"[TEST] Probe Reality: {probe_results['reality']}")

def test_causal_engine_roles():
    from Core.L2_Metabolism.Memory.fractal_causality import FractalCausalityEngine
    engine = FractalCausalityEngine("TestEngine")
    
    # Create a 4-step chain manually
    chain = engine.create_chain("Why", "How", "Result") # Legacy call
    
    # Verify we can use the new roles
    cause = engine.create_node("Seed", depth=0)
    structure = engine.create_node("Geometry", depth=0, parent_id=cause.id, parent_role=CausalRole.STRUCTURE)
    function = engine.create_node("Activation", depth=0, parent_id=structure.id, parent_role=CausalRole.FUNCTION)
    reality = engine.create_node("Node", depth=0, parent_id=function.id, parent_role=CausalRole.REALITY)
    
    assert reality.parent_role == CausalRole.REALITY
    assert function.parent_role == CausalRole.FUNCTION

if __name__ == "__main__":
    pytest.main([__file__])
