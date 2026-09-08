import torch
import torch.nn as nn
from typing import Dict, Any


class HierarchicalHebbianCausalEngine(nn.Module):
    """
    2-Tier Hierarchical Hebbian Causal Architecture
    Connects Tier 1 (micro real-time causal switches) and Tier 2 (macro abstract concept nodes)
    with bottom-up pattern chunking and top-down context modulation.
    """
    def __init__(self, num_micro: int, num_macro: int, lr_micro: float = 0.1, lr_macro: float = 0.05):
        super().__init__()
        self.N = num_micro
        self.M = num_macro

        # Tier 1: Micro causal coupling matrix J_micro
        self.register_buffer("J_micro", torch.zeros(num_micro, num_micro))
        self.register_buffer("prev_micro", torch.zeros(num_micro))
        self.lr_micro = lr_micro

        # Bottom-up chunking matrix (pattern detector mapping micro -> macro)
        self.chunk_weights = nn.Parameter(torch.randn(num_macro, num_micro) * 0.1, requires_grad=False)

        # Tier 2: Macro causal coupling matrix J_macro
        self.register_buffer("J_macro", torch.zeros(num_macro, num_macro))
        self.register_buffer("prev_macro", torch.zeros(num_macro))
        self.lr_macro = lr_macro

    def forward_step(self, current_micro: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            current_micro: [N] Micro switch state (+1 or -1)
        Returns:
            dictionary containing updated micro and macro state information
        """
        # 1. Tier 1 (Micro layer) Hebbian update
        curr_m = current_micro.unsqueeze(1)    # [N, 1]
        prev_m = self.prev_micro.unsqueeze(0)  # [1, N]

        dJ_micro = self.lr_micro * (torch.matmul(curr_m, prev_m) - 0.05 * torch.abs(self.J_micro) * self.J_micro)
        dJ_micro.fill_diagonal_(0.0)
        self.J_micro.add_(dJ_micro)

        # 2. Bottom-up Chunking: Micro -> Macro transformation
        macro_raw = torch.matmul(self.chunk_weights, current_micro)
        current_macro = torch.sign(macro_raw)
        current_macro = torch.where(current_macro == 0, torch.ones_like(current_macro), current_macro)

        # 3. Tier 2 (Macro layer) Long-term Hebbian learning
        curr_M = current_macro.unsqueeze(1)    # [M, 1]
        prev_M = self.prev_macro.unsqueeze(0)  # [1, M]

        dJ_macro = self.lr_macro * (torch.matmul(curr_M, prev_M) - 0.05 * torch.abs(self.J_macro) * self.J_macro)
        dJ_macro.fill_diagonal_(0.0)
        self.J_macro.add_(dJ_macro)

        # Update state buffers
        self.prev_micro.copy_(current_micro)
        self.prev_macro.copy_(current_macro)

        return {
            "micro_switches": current_micro,
            "macro_concepts": current_macro,
            "active_macro_count": (current_macro > 0).sum().item()
        }
