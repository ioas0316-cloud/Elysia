"""
Unit test for C++ predictive resonance extension.
"""

import torch
import predictive_resonance_cpp

def test_predictive_resonance_cpp_extension():
    dim = 16
    tau_min = 0.10

    st = torch.randn(dim)

    # 1. Passive match -> Early exit
    x_passive = st.clone()
    _, err_p, gated_p = predictive_resonance_cpp.forward(x_passive, st.clone(), tau_min)
    assert gated_p
    assert err_p < tau_min

    # 2. Active discrepancy -> Volitional active
    x_active = st + torch.randn(dim) * 2.0
    st_updated, err_a, gated_a = predictive_resonance_cpp.forward(x_active, st.clone(), tau_min)
    assert not gated_a
    assert err_a >= tau_min

if __name__ == "__main__":
    test_predictive_resonance_cpp_extension()
    print("C++ EXTENSION TEST PASSED!")
