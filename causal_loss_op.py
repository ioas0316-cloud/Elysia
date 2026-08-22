import torch
import numpy as np
import causal_engine as ce

class PreisachTensionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_signal, cpp_field, extractor, closed_loop, threshold=0.35):
        """
        u_signal: PyTorch Tensor (shape: [num_nodes], requires_grad=True)
        cpp_field: C++ PreisachTensorFieldSoA instance
        extractor: C++ AttractorExtractionLayer instance
        closed_loop: C++ ClosedLoopCausalEngine instance
        """
        # 1. Zero-Copy / Fast NumPy array passing to C++ SoA field
        u_np = u_signal.detach().cpu().numpy().astype(np.float32)
        cpp_field.set_input_signals_from_numpy(u_np)

        # 2. Execute C++ OpenMP switching update
        ce.update_preisach_field(cpp_field)

        # 3. Extract Macro Causal Graph
        nodes, edges = extractor.extract_causal_graph(cpp_field, threshold)

        tension_val = 0.0
        if len(nodes) > 1:
            current_sr = float(np.mean(cpp_field.get_remanence_as_numpy()))
            target_sr = nodes[-1].current_state_sr
            tension_val = float(np.abs(current_sr - target_sr))

        # Save context for backward surrogate gradient pass
        ctx.save_for_backward(u_signal)
        ctx.tension_val = tension_val
        ctx.cpp_field = cpp_field

        return torch.tensor(tension_val, dtype=torch.float32, device=u_signal.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Surrogate Gradient mechanism replacing non-differentiable switching hysteresis:
        dL_tension / du ≈ grad_output * (1 - tanh(u)^2) * Tension_Val
        """
        u_signal, = ctx.saved_tensors
        tension_val = ctx.tension_val

        u_data = u_signal.detach()
        surrogate_grad = (1.0 - torch.tanh(u_data) ** 2) * tension_val

        return grad_output * surrogate_grad, None, None, None, None
