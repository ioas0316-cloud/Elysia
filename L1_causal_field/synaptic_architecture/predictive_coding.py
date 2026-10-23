import torch
import torch.nn as nn


class PredictiveCodingNet(nn.Module):
    r"""
    Predictive Coding Network minimizing Local Free Energy (F)
    without global scalar loss backpropagation (loss.backward()).

    Dimension-preserving local error tensors:
        \varepsilon_l = r_l - g(W_{l+1} r_{l+1}) \in \mathbb{R}^{D_l}

    2-Timescale Dynamics:
        1. Fast state relaxation (\tau_r / lr_r):
           \frac{\partial r_l}{\partial t} = -\varepsilon_l + W_l^T (\varepsilon_{l-1} \odot g'(W_l r_l))
        2. Slow Hebbian weight learning (\tau_w / lr_w):
           \frac{\partial W_l}{\partial t} = (\varepsilon_{l-1} \odot g'(W_l r_l)) \cdot r_l^T
    """
    def __init__(self, layer_dims, lr_r=0.05, lr_w=0.01):
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.lr_r = lr_r
        self.lr_w = lr_w

        # Top-down generative mapping parameters (W_l: Layer l -> Layer l-1)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(1, self.num_layers):
            # W[i-1] predicts r_{i-1} (dim D_{i-1}) from r_i (dim D_i)
            w = torch.randn(layer_dims[i-1], layer_dims[i]) * 0.1
            b = torch.zeros(layer_dims[i-1])
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

        # Activation function and local derivative
        self.act = torch.tanh
        self.act_deriv = lambda x: 1.0 - torch.tanh(x) ** 2

    def relax_states(self, x, relaxation_steps=30):
        """
        [1. Fast Dynamics (\tau_r)]: State Relaxation
        Given sensory input x (clamped at r_0), latent states r_l iterate to reach
        local equilibrium under top-down and bottom-up error tensor pressures.
        """
        batch_size = x.size(0)
        states = [x.clone().detach()]  # r_0 = sensory input (fixed)

        # 1.1 Bottom-up initial pass
        curr = x
        for i in range(1, self.num_layers):
            curr = self.act(curr @ self.weights[i-1])
            states.append(curr.clone().detach())

        # 1.2 Fast state relaxation loop
        for _ in range(relaxation_steps):
            preds = []
            errors = []
            pre_acts = []

            # Top-down prediction and dimension-preserving local error tensor collection
            for i in range(1, self.num_layers):
                pre = states[i] @ self.weights[i-1].T + self.biases[i-1]  # (B, D_{i-1})
                hat_r = self.act(pre)
                eps = states[i-1] - hat_r  # Local error tensor \varepsilon_{i-1} in R^{D_{i-1}}

                pre_acts.append(pre)
                preds.append(hat_r)
                errors.append(eps)

            # Update latent states r_l (for l = 1 .. num_layers - 1)
            for i in range(1, self.num_layers):
                # 1) Top-level error (inhibition from upper layer)
                top_error = errors[i] if i < self.num_layers - 1 else 0.0

                # 2) Bottom-level error propagation (\varepsilon_{i-1} * g'(pre) * W)
                grad_pre = errors[i-1] * self.act_deriv(pre_acts[i-1])
                bottom_error_prop = grad_pre @ self.weights[i-1]

                # Local free energy gradient w.r.t. r_i
                dr = top_error - bottom_error_prop

                # Fast state update
                states[i] = states[i] - self.lr_r * dr

        return states, errors, pre_acts

    def update_weights(self, states, errors, pre_acts):
        """
        [2. Slow Dynamics (\tau_w)]: Slow Hebbian Weight Learning
        Updates weights using local outer product of error tensor and state
        without global autograd graph traversal.
        """
        batch_size = states[0].size(0)
        with torch.no_grad():
            for i in range(1, self.num_layers):
                grad_pre = errors[i-1] * self.act_deriv(pre_acts[i-1])  # (B, D_{i-1})

                # Local Hebbian weight gradient: \Delta W \propto \varepsilon_{i-1} \otimes r_i
                dW = (grad_pre.T @ states[i]) / batch_size
                db = grad_pre.mean(dim=0)

                # Local weight update
                self.weights[i-1].add_(dW, alpha=self.lr_w)
                self.biases[i-1].add_(db, alpha=self.lr_w)

    def compute_free_energy(self, errors):
        """Calculates total local Free Energy (sum of squared error tensor norms)."""
        free_energy = sum([0.5 * torch.sum(eps ** 2).item() for eps in errors])
        return free_energy
