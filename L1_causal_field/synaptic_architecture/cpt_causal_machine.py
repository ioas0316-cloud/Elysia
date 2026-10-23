import torch
import torch.nn as nn


class PhaseTransitionCausalMachine(nn.Module):
    """
    Critical Phase-Transition Causal Machine (CPT-CM)
    Applies thermodynamic phase transitions (Landau-Ginzburg) to attenuate sub-layer noise
    and collapse macro states into discrete +/-1 causal switches.
    """
    def __init__(self, num_switches: int, causal_graph_J: torch.Tensor, T_critical: float = 1.0):
        super().__init__()
        self.N = num_switches
        self.T_c = T_critical

        # Causal coupling matrix J (J_ij > 0: excitation, J_ij < 0: inhibition)
        self.register_buffer("J", causal_graph_J.clone())

        # Macro discrete causal switch state (+1.0 or -1.0)
        self.register_buffer("switches", torch.ones(num_switches, dtype=torch.float32))

    def observe_and_transition(self, h_local: torch.Tensor, T_current: float) -> torch.Tensor:
        """
        Args:
            h_local: [N] Local stimulus field from lower sub-tensor layers
            T_current: Current cognitive system temperature
        Returns:
            switches: [N] Updated discrete causal switches (+1 or -1)
        """
        # 1. Causal Field Computation: J * switches + h_local
        causal_field = torch.matmul(self.J, self.switches) + h_local

        # 2. Effective inverse temperature beta
        beta = 1.0 / max(T_current, 1e-5)

        # 3. Thermodynamic Spin Collapse Phase Transition
        if T_current < self.T_c:
            # [Crystalline Phase]: Lower noise filtered -> Instant spin collapse (sign function)
            new_switches = torch.sign(causal_field)
            # Preserve state if field is exactly 0
            new_switches = torch.where(new_switches == 0, self.switches, new_switches)
        else:
            # [Fluid Phase]: Soft probabilistic exploration
            prob_up = torch.sigmoid(2.0 * beta * causal_field)
            new_switches = torch.where(torch.rand_like(prob_up) < prob_up, 1.0, -1.0)

        self.switches.copy_(new_switches)
        return self.switches

    def get_macro_state_id(self) -> int:
        """Bit-packs discrete switch array into a single integer macro state ID."""
        binary_bits = ((self.switches + 1) // 2).to(torch.int64)
        weights = 2 ** torch.arange(len(binary_bits), device=binary_bits.device)
        return torch.sum(binary_bits * weights).item()
