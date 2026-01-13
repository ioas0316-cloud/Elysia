from Core.FoundationLayer.Foundation.math_cortex import MathCortex


def test_math_cortex_symbolic_equality():
    mc = MathCortex()
    proof = mc.symbolic_verify("2*(x + y) = 2*x + 2*y")
    # If sympy unavailable, just consider test skipped-by-logic (valid False with specific verdict)
    if not any(s.action == "error" and "Sympy not available" in s.detail for s in proof.steps):
        assert proof.valid, f"Expected symbolic equality to hold. Steps: {[ (s.action, s.detail) for s in proof.steps ]}"

