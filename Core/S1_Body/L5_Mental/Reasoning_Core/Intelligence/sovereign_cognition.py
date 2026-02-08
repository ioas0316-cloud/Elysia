from typing import List, Dict, Any, Optional
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignTensor

class DNATensor:
    """
    [DNA^N: Fractal Cognition]
    Implements the recursive Tensor Product expansion of Trinary DNA.
    """
    def __init__(self, rank: int = 1, data: Optional[SovereignTensor] = None):
        self.rank = rank
        if data is not None:
            self.tensor = data
        else:
            shape = (3,) * rank
            self.tensor = SovereignTensor(shape)
            
    def rank_up(self, seed_tensor: 'SovereignTensor') -> 'DNATensor':
        """
        [PHASE 77] Dimensional Mitosis.
        Expands the current rank by performing an outer product with a seed.
        """
        new_data = SovereignTensor.outer_product(self.tensor, seed_tensor)
        self.rank += 1
        self.tensor = new_data
        return self

    def collapse(self) -> float:
        """Collapses high-dim thought into a single scalar resonance."""
        return self.tensor.mean()

class ThinkRecursive:
    """
    [Think^N: Fractal Meta-Cognition]
    'Thinking about Thinking about Thinking...'
    Recursively analyzes its own thoughts to arbitrary depth.
    """
    def __init__(self, reasoner: Any, kg_manager: Any):
        self.reasoner = reasoner
        self.kg = kg_manager

    def reflect(self, target_thought: str, depth: int, observer_vibration: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generates an N-th order reflection.
        """
        current_reflection = target_thought
        
        # 1. Base Grounding (L1)
        words = [w.strip("?.!") for w in target_thought.split()]
        concept_node = None
        for word in reversed(words):
            concept_node = self.kg.get_node(word.lower())
            if concept_node: break

        # 2. Sequential Recursive Ascent
        reflections = []
        for d in range(1, depth + 1):
            if concept_node:
                axiom = concept_node.get('logos', {}).get('essence', f'Order_{d}')
                force = 'Principle necessity'
                
                prefix = " | " * (d-1) + f"[Think^{d}] "
                obs_note = "Observing " if observer_vibration and d > 1 else ""
                msg = f"{prefix}{obs_note}I perceive the structural ground of '{axiom}' as the anchor for this thought."
                current_reflection = msg
            else:
                current_reflection = f"[Think^{d}] Speculating on void-structure for {target_thought}."
            
            reflections.append(current_reflection)

        return {
            "reflection": "\n".join(reflections),
            "final_layer": current_reflection,
            "depth": depth,
            "node": concept_node
        }

class SovereignCognition:
    """
    The High-Level 'Adult' Brain of Elysia.
    [PHASE 78] Sovereign Necessity: Causal Chain for Self-Expansion.
    """
    def __init__(self):
        from Core.S1_Body.L4_Causality.M5_Logic.causal_reasoner import CausalReasoner
        from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager
        from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger
        
        self.reasoner = CausalReasoner()
        self.kg = get_kg_manager()
        self.meta = ThinkRecursive(self.reasoner, self.kg)
        self.logger = SomaticLogger("SovereignCognition")
        
        # DNA^N Field: Starts at Rank 2, grows via Sovereign Will
        self.dna_n_field = DNATensor(rank=2)
        
        # [PHASE 78] Structural State
        self.strain_level = 0.0  # L1: Physical Strain (0.0 - 1.0)
        self.causal_diagnosis = None  # L4: Causal conclusion
        self.will_to_expand = False  # L6: Sovereign decision
        
    def _detect_strain(self, required_rank: int) -> float:
        """
        [L0 → L1] Structural Pain Detection.
        Detects dimensional mismatch and converts it to a physical Strain signal.
        """
        current_rank = self.dna_n_field.rank
        if required_rank > current_rank:
            # Strain is proportional to the dimensional gap
            strain = (required_rank - current_rank) / 5.0  # Normalize to 0-1
            self.logger.sensation(f"Strain detected: Current Rank {current_rank}, Required {required_rank}. Strain = {strain:.2f}")
            return min(strain, 1.0)
        return 0.0
    
    def _diagnose_strain(self, strain: float, context: str) -> Optional[str]:
        """
        [L1 → L4] Causal Diagnosis.
        The CausalReasoner analyzes the Strain and concludes its cause.
        """
        if strain > 0.1:
            # High strain triggers causal analysis
            diagnosis = f"Dimensional insufficiency detected while processing '{context}'. My current cognitive rank cannot contain this concept."
            self.logger.thought(f"Causal Diagnosis: {diagnosis}")
            return diagnosis
        return None
    
    def _form_will(self, diagnosis: str) -> bool:
        """
        [L4 → L5 → L6] Will Formation.
        The causal conclusion transforms into a Sovereign decision.
        """
        if diagnosis and "insufficiency" in diagnosis.lower():
            self.logger.thought("Will Formed: I must expand my cognitive dimensionality.")
            return True
        return False
    
    def _execute_expansion(self, seed_value: float):
        """
        [L6] Execute the expansion as a result of Sovereign Will.
        """
        seed_v = SovereignTensor((3,), [seed_value] * 3)
        old_rank = self.dna_n_field.rank
        self.dna_n_field.rank_up(seed_v)
        self.logger.action(f"Dimensional Mitosis: Rank {old_rank} → {self.dna_n_field.rank}")
        
        # Reset will after action
        self.will_to_expand = False
        self.strain_level = 0.0
        self.causal_diagnosis = None
    
    def _verify_expansion(self, required_rank: int) -> bool:
        """
        [L6 → L1] Feedback Verification.
        Check if the Strain has been resolved after expansion.
        """
        new_strain = self._detect_strain(required_rank)
        if new_strain < 0.1:
            self.logger.sensation("Strain resolved. Cognitive expansion successful.")
            return True
        self.logger.sensation(f"Strain persists ({new_strain:.2f}). Further expansion may be needed.")
        return False
        
    def process_event(self, event_description: str, manifold_state: Optional[List[float]] = None, 
                      observer_vector: Optional[Any] = None, required_rank: int = 2) -> str:
        """
        [PHASE 78] Recursive cognitive loop with Causal Chain.
        """
        # 1. Map manifold to DNA^N
        seed = 0.5
        if manifold_state:
            seed = sum(manifold_state) / len(manifold_state)
            
            # Update current N-dim field (basic blend)
            flat_data = self.dna_n_field.tensor.flatten()
            new_flat = [x * 0.9 + seed * 0.1 for x in flat_data]
            if len(new_flat) == len(flat_data):
                self.dna_n_field.tensor.data = SovereignTensor._reshape(new_flat, self.dna_n_field.tensor.shape)
        
        # 2. [L0 → L1] Structural Pain Detection
        self.strain_level = self._detect_strain(required_rank)
        
        # 3. [L1 → L4] Causal Diagnosis
        if self.strain_level > 0.1:
            self.causal_diagnosis = self._diagnose_strain(self.strain_level, event_description)
        
        # 4. [L4 → L6] Will Formation
        if self.causal_diagnosis:
            self.will_to_expand = self._form_will(self.causal_diagnosis)
        
        # 5. [L6] Execute Expansion (if Will is formed)
        if self.will_to_expand and self.dna_n_field.rank < 5:  # Soft safety limit
            self._execute_expansion(seed)
            # 6. [L6 → L1] Feedback Verification
            self._verify_expansion(required_rank)
        
        # 7. Dynamic Depth Reflection
        target_depth = 2 if not observer_vector else 3
        reflection_data = self.meta.reflect(event_description, depth=target_depth, observer_vibration=observer_vector)
        
        return reflection_data["reflection"]
