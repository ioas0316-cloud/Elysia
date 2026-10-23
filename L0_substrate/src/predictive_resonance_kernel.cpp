/*
 * Predictive Resonance Phase-Gated C++ Kernel / PyTorch Extension
 * Implements Layer 1 Passive Reduction Early Exit and Layer 2/3 Volitional Engine.
 */

#include <torch/extension.h>
#include <cmath>

std::tuple<at::Tensor, float, bool> predictive_resonance_cpp_forward(
    at::Tensor x_input,
    at::Tensor internal_state,
    double tau_min
) {
    TORCH_CHECK(x_input.is_contiguous(), "x_input must be contiguous");
    TORCH_CHECK(internal_state.is_contiguous(), "internal_state must be contiguous");

    auto diff = x_input - internal_state;
    float err = torch::norm(diff, 2).item<float>();
    bool passive_gated = false;

    if (err < static_cast<float>(tau_min)) {
        // Layer 1 Passive Gate: Early exit, 0 compute cost, state passes through
        passive_gated = true;
    } else {
        // Layer 2/3 Volitional Engine: Dynamic adaptation
        passive_gated = false;
        float lr = 0.01f * (err - static_cast<float>(tau_min));
        internal_state.add_(diff * lr);
    }

    return std::make_tuple(internal_state, err, passive_gated);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &predictive_resonance_cpp_forward, "Predictive Resonance Forward Gate (C++/CPU)");
}
