import numpy as np
from core.physics.causal_compiler_engine import CausalCompilerEngine

def test_causal_compiler_engine():
    engine = CausalCompilerEngine(state_dim=4)
    instructions = [
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([5.0, 5.0, 5.0, 5.0]),
        np.array([2.0, 2.0, 2.0, 2.0])
    ]

    result = engine.compile_instruction_stream(instructions)
    assert result["if_branch_evaluations"] == 0
    assert result["compiled_friction"] <= result["raw_friction"]
    print("CausalCompilerEngine test passed successfully.")

if __name__ == "__main__":
    test_causal_compiler_engine()
