import torch
import logging
from typing import List, Dict, Optional
from Core.L1_Foundation.Foundation.unified_monad import UnifiedMonad

try:
    import jax
    import jax.numpy as jnp
    from Core.L1_Foundation.Logic.jax_kernel_fusion import compute_inter_monad_proximity, calculate_synaptic_flow
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger("MonadicCircuitry")

class MonadicCircuitry:
    """
    [PHASE 25.2] MONADIC CIRCUITRY ENGINE
    =====================================
    The 'Nervous System' of the HyperCosmos.
    Manages the active flow of energy between resonant monads.
    """
    def __init__(self, conductivity_base: float = 0.05, decay_rate: float = 0.01):
        self.conductivity_base = conductivity_base
        self.decay_rate = decay_rate
        self.active_fires: List[str] = [] # Names of monads currently 'Firing'

    def process_weave(self, monads: List[UnifiedMonad], field_intensity: torch.Tensor):
        """
        Updates the synaptic connections and transfers energy.
        Called during each Pulse.
        """
        if not monads: return

        # 1. Gather tensors explicitly
        vectors = torch.stack([m.vector.data for m in monads])
        masses = torch.tensor([[m.mass] for m in monads], device=vectors.device)

        if HAS_JAX:
            # 2. XLA INTERFACE: Gather data
            v_np = vectors.detach().cpu().numpy()
            m_np = masses.detach().cpu().numpy()
            
            # 3. KERNEL EXECUTION
            v_jax = jnp.array(v_np)
            m_jax = jnp.array(m_np)
            
            prox_jax = compute_inter_monad_proximity(v_jax)
            flow_jax = calculate_synaptic_flow(prox_jax, m_jax, self.conductivity_base)
            
            # 4. SYNC BACK
            circuit_flow = torch.from_numpy(np.array(jax.device_get(flow_jax))).to(vectors.device)
        else:
            # Legacy Fallback
            dist_matrix = torch.cdist(vectors.unsqueeze(0), vectors.unsqueeze(0)).squeeze(0)
            proximity = torch.exp(-dist_matrix * 2.0)
            circuit_flow = torch.mm(proximity, masses)
        
        avg_mass = float(masses.mean())
        firing_threshold = avg_mass * 2.0
         
        for i, m in enumerate(monads):
            # Apply flow
            flow_val = float(circuit_flow[i])
            m.mass += flow_val * self.conductivity_base * m.conductivity
            
            # Firing Logic
            if m.mass > firing_threshold:
                if m.name not in self.active_fires:
                    self.active_fires.append(m.name)
                    logger.info(f"?뵦 [CIRCUIT] '{m.name}' FIRED!")
            else:
                if m.name in self.active_fires:
                    self.active_fires.remove(m.name)

            # Natural Decay
            m.mass *= (1.0 - self.decay_rate)

    def get_circuit_status(self) -> str:
        return f"Active Fires: {len(self.active_fires)}"
