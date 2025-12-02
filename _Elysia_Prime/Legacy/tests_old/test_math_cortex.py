# [Genesis: 2025-12-02] Purified by Elysia
from Project_Sophia.math_cortex import MathCortex


def test_math_cortex_simple_equality():
    mc = MathCortex()
    proof = mc.verify("2 + 3 = 5")
    assert proof.valid
    assert len(proof.steps) >= 3
