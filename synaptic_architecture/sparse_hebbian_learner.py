import torch
import torch.nn as nn


class SparseHebbianCausalLearner(nn.Module):
    """
    Sparse Hebbian Causal Learner (SCHL)
    Learns causal coupling matrix J online without backpropagation or gradients,
    using temporal asymmetric correlations (STDP), competitive lateral inhibition,
    and L1 non-linear dynamic sparsity decay.
    """
    def __init__(self, num_switches: int, lr: float = 0.05, sparsity_gamma: float = 0.02, threshold: float = 0.1):
        super().__init__()
        self.N = num_switches
        self.lr = lr
        self.gamma = sparsity_gamma  # Sparsity decay coefficient
        self.threshold = threshold  # Hard threshold for sparsification

        # Learned causal coupling matrix J (no gradient required)
        self.J = nn.Parameter(torch.zeros(num_switches, num_switches), requires_grad=False)

        # Buffer for previous step discrete switch state (t-1)
        self.register_buffer("prev_switches", torch.zeros(num_switches))

    def update_causal_matrix(self, current_switches: torch.Tensor) -> torch.Tensor:
        """
        Updates J based on state transitions from t-1 to t.
        Args:
            current_switches: [N] (+1 or -1 discrete switch states at t)
        Returns:
            J: [N, N] updated sparse causal matrix
        """
        curr = current_switches.unsqueeze(1)    # [N, 1] (current result s_i)
        prev = self.prev_switches.unsqueeze(0)  # [1, N] (previous cause s_j)

        # 1. Temporal Asymmetric Hebbian Update (s_i(t) * s_j(t-1))
        causal_correlation = torch.matmul(curr, prev)  # [N, N]

        # 2. Competitive Lateral Inhibition
        current_predictions = torch.matmul(self.J, self.prev_switches).unsqueeze(1)
        competition = curr * current_predictions

        # 3. Non-linear Sparsity Decay
        sparsity_decay = self.gamma * torch.abs(self.J) * self.J

        # 4. Weight Delta Computation
        dJ = self.lr * (causal_correlation - 0.2 * competition - sparsity_decay)

        # Prevent self-loops
        dJ.fill_diagonal_(0.0)
        self.J.add_(dJ)

        # 5. Hard Thresholding for Sparsity
        with torch.no_grad():
            self.J.copy_(torch.where(torch.abs(self.J) > self.threshold, self.J, torch.zeros_like(self.J)))

        # Update previous state buffer
        self.prev_switches.copy_(current_switches)
        return self.J.clone()

    def get_sparse_causal_graph(self) -> torch.Tensor:
        """Returns clone of current learned sparse causal graph J."""
        return self.J.clone()
