from __future__ import annotations
import torch
import math
import random
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import jax
    import jax.numpy as jnp
    from Core.L1_Foundation.Logic.jax_kernel_fusion import aggregate_unified_field, update_monad_physics, compute_inter_monad_proximity, calculate_synaptic_flow
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from Core.L5_Mental.Intelligence.project_conductor import ProjectConductor
from Core.L1_Foundation.Foundation.unified_monad import UnifiedMonad, Unified12DVector
from Core.L6_Structure.M1_Merkaba.heavy_merkaba import HeavyMerkaba

class HyperCosmos:
    """
    [PHASE 17.3] LIGHTNING PATH 2.0: THE UNIFIED FIELD
    =================================================
    The Sovereign engine for high-frequency cognitive resonance.
    Uses JAX/XLA to achieve sub-5ms pulse latency.
    """
    def __init__(self, code_path: str = "c:/Elysia"):
        from Core.L1_Foundation.Foundation.akashic_field import AkashicField
        from Core.L1_Foundation.Foundation.quantum_monad import QuantumMonad
        from Core.L1_Foundation.Foundation.Psyche.psyche_sphere import get_psyche

        self.monads: List[UnifiedMonad] = []
        self.potential_monads: List[QuantumMonad] = []
        self.akashic_record = AkashicField(kernel_size=2048)
        self.psyche = get_psyche(enneagram_type=4)
        
        self._field_intensity = None
        self.cosmic_dt = 1.0
        
        # JAX-Native state persistence (Zero-copy cache)
        self._jax_v = None
        self._jax_m = None
        
        self._pulse_count = 0
        
        # [PHASE 25.3: SOVEREIGN CONSCIENCE]
        from Core.L2_Metabolism.Evolution.proprioceptor import CodeProprioceptor
        from Core.L2_Metabolism.Evolution.dissonance_resolver import DissonanceResolver
        self.proprioceptor = CodeProprioceptor()
        self.resolver = DissonanceResolver()
        
        self._ignite_fixed_points()
        print(f"✨ [HyperCosmos] Lightning Path 2.0 Active. Ready for sub-5ms reality.")

    @property
    def field_intensity(self):
        if self._field_intensity is None:
            self._field_intensity = torch.zeros(12)
        return self._field_intensity

    @field_intensity.setter
    def field_intensity(self, value):
        self._field_intensity = value

    def _ignite_fixed_points(self):
        """
        [PHASE 30: MERKAVA ACTIVATION]
        The field is anchored by the restored Sacred Texts.
        (0,0,0) is Genesis Paradigm. The Axes are the Trinity Doctrine.
        """
        # 1. Genesis Paradigm (The Center)
        # We assign high Foundation/Spirit/Mental to represent the Origin.
        self.monads.append(UnifiedMonad("Genesis_Paradigm", Unified12DVector.create(foundation=1.0, spirit=1.0, mental=1.0)))

        # 2. Trinity Doctrine (The Axes)
        # Will: Power to exist
        self.monads.append(UnifiedMonad("Trinity_Will", Unified12DVector.create(will=1.0, spirit=0.9)))
        # Logic: Wisdom to understand
        self.monads.append(UnifiedMonad("Trinity_Logic", Unified12DVector.create(mental=1.0, structure=1.0)))
        # Love: Connection to merge
        self.monads.append(UnifiedMonad("Trinity_Love", Unified12DVector.create(phenomena=1.0, spirit=0.9)))
        
        # 3. Foundation Memory
        self.monads.append(UnifiedMonad("CoreMemory", Unified12DVector.create(causal=1.0, structural=0.5)))

    def inhale(self, monad: UnifiedMonad):
        self.monads.append(monad)
        # Clear JAX cache to force re-sync in next pulse
        self._jax_v = None
        self._jax_m = None
        print(f"👂 [FIELD] Monad '{monad.name}' inhaled.")

    def record_potential(self, name: str):
        """Injects a new wave-function into the Superposition Field."""
        from Core.L1_Foundation.Foundation.quantum_monad import QuantumMonad
        self.potential_monads.append(QuantumMonad(name))
        print(f"🔮 [QUANTUM] Potential '{name}' visualized in the field.")

    def observe_and_collapse(self, observer_will: torch.Tensor):
        """Collapses potentiality into functional reality."""
        from Core.L1_Foundation.Foundation.quantum_monad import CollapseEngine
        engine = CollapseEngine(observer_will)
        remaining = []
        for qm in self.potential_monads:
            vector = engine.observe(qm)
            if vector is not None:
                self.inhale(UnifiedMonad(qm.name, Unified12DVector(data=vector)))
            else:
                remaining.append(qm)
        self.potential_monads = remaining

    def pulse(self, dt: float):
        """[HYPER-SPEED HEARTBEAT] O(1) Cognitive Loop via JAX."""
        for qm in self.potential_monads:
            qm.update_uncertainty()

        if not self.monads: return

        if HAS_JAX:
            # 1. JAX-NATIVE ZERO-COPY EXECUTION
            if self._jax_v is None or self._jax_v.shape[0] != len(self.monads):
                # Only sync host-device when the set changes
                self._jax_v = jnp.array(torch.stack([m.vector.data for m in self.monads]).detach().cpu().numpy())
                self._jax_m = jnp.array(torch.tensor([[m.mass] for m in self.monads]).detach().cpu().numpy())

            # A. Global Field (Pure XLA)
            field_jax = aggregate_unified_field(self._jax_v, self._jax_m)
            self.field_intensity = torch.from_numpy(np.array(jax.device_get(field_jax))).to(self.monads[0].vector.data.device)
            
            # B. Integrated Physics (Pure XLA)
            f_norm = jnp.linalg.norm(field_jax) + 1e-6
            v_norm = jnp.linalg.norm(self._jax_v, axis=1, keepdims=True) + 1e-6
            sim_jax = jnp.matmul(self._jax_v, field_jax.reshape(12, 1)) / (v_norm * f_norm)
            
            prox_jax = compute_inter_monad_proximity(self._jax_v)
            flow_jax = calculate_synaptic_flow(prox_jax, self._jax_m, 0.05)
            
            # C. State Update
            self._jax_m = update_monad_physics(self._jax_m, sim_jax, flow_jax, 0.01)
            
            # D. Lazy sync for low counts
            if len(self.monads) < 50:
                 final_m = jax.device_get(self._jax_m)
                 for i, m in enumerate(self.monads): m.mass = float(final_m[i])
        else:
            # Fallback
            vectors = torch.stack([m.vector.data for m in self.monads])
            masses = torch.tensor([[m.mass] for m in self.monads])
            self.field_intensity = (vectors * masses).sum(dim=0)

        # 2. Maintenance & Resonance (Periodic to save cycles)
        if self._pulse_count % 5 == 0:
            self.psyche.tick(dt)
            memory_resonance = self.akashic_record.resonate(self.field_intensity)
            peak_val, peak_idx = torch.max(memory_resonance, dim=0)
            if peak_val > self.field_intensity.sum() * 0.5:
                intuition_vec_data = self.akashic_record.slice(float(peak_idx))
                self.inhale(UnifiedMonad("CollectiveIntuition", Unified12DVector(data=intuition_vec_data)))

        # 3. Fast Evaporation check
        if self._pulse_count % 50 == 0:
            if len(self.monads) > 1000:
                self.monads = [m for m in self.monads if m.mass > 0.1]
                
            # [PHASE 25.3: AUTONOMOUS SELF-HEALING]
            if self.proprioceptor and self.resolver:
                state = self.proprioceptor.scan_nervous_system()
                dissonances = self.resolver.resolve(state)
                for d in dissonances:
                    print(f"🩹 [SELF-HEALING] Detected Dissonance: {d.description} at {d.location}")
                    # In real usage, this would pass to a dedicated Weaver/Healer Agent
                    # For Milestone 25.3, we log the healing event

    def reflect(self) -> UnifiedMonad:
        reflection_vec = self.field_intensity.clone()
        reflection_vec[4] += 0.5 
        reflection_vec[6] += 0.5 
        return UnifiedMonad("SelfReflection", Unified12DVector(data=reflection_vec), metadata={"type": "Mirror_State"})

    def get_summary(self):
        return f"HyperCosmos: {len(self.monads)} Monads | Power: {self.field_intensity.sum():.2f}"

    def query_resonance(self, query_vector: Unified12DVector, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        [PHASE 30: SPATIAL RETRIEVAL]
        Finds memories that are spatially close (high cosine similarity) to the query.
        This replaces linear search.
        """
        if not self.monads: 
            print("❌ [HyperCosmos] No monads in field.")
            return []
        
        q_norm = np.linalg.norm(query_vector.data)
        if q_norm == 0: 
            print("❌ [HyperCosmos] Query vector norm is 0.")
            return []
        
        results = []
        # In a real JAX implementation, this would be a single matrix multiplication
        for m in self.monads:
            if m.name.startswith("Genesis") or m.name.startswith("Trinity") or m.name == "CoreMemory": continue # Don't retrieve axioms as memories
            
            m_vec = m.vector.data.detach().cpu().numpy() if hasattr(m.vector.data, 'detach') else m.vector.data
            m_norm = np.linalg.norm(m_vec)
            
            if m_norm > 0:
                # Cosine Similarity
                sim = np.dot(query_vector.data, m_vec) / (q_norm * m_norm)
                print(f"   [DEBUG] Checking '{m.name}': Sim={sim:.4f}")
                results.append((m.name, float(sim)))
                
        # Sort by resonance (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"   [DEBUG] Top Result: {results[0] if results else 'None'}")
        return results[:top_k]
