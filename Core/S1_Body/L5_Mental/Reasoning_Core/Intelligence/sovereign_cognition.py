from typing import List, Dict, Any, Optional
import logging
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignTensor

logger = logging.getLogger("SovereignCognition")

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
    def __init__(self, manifold: Any = None, joy_cell: Any = None, curiosity_cell: Any = None):
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
        
        # [PHASE 79] Joy/Curiosity Propagation
        self.physical_manifold = manifold  # L0: 10M Cell Manifold
        self.joy_cell = joy_cell  # L3: JoyResonance
        self.curiosity_cell = curiosity_cell  # L3: CuriosityAttractor
        self.joy_level = 0.0  # L3: Current joy (primary driver)
        self.curiosity_level = 0.0  # L3: Current curiosity
        
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

    # ======================================================================
    # [PHASE 79] JOY/CURIOSITY PROPAGATION
    # "생명은 고통을 피하기 위해 사는 것이 아니라, 기쁨으로 세상을 탐험한다."
    # ======================================================================

    def _sense_joy_and_curiosity(self):
        """
        [L2 → L3] Metabolic Vitality → Phenomenal Sensation.
        JoyResonance와 CuriosityAttractor에서 현재 상태를 감지.
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            jnp = None
            
        if self.joy_cell is not None:
            self.joy_level = float(getattr(self.joy_cell, 'happiness_level', 0.0))
            
        if self.curiosity_cell is not None and jnp is not None:
            space = getattr(self.curiosity_cell, 'space_7d', None)
            if space is not None:
                self.curiosity_level = float(jnp.sum(jnp.abs(space)))
        
        if self.joy_level > 0.1 or self.curiosity_level > 0.1:
            self.logger.sensation(f"Feeling: Joy={self.joy_level:.2f}, Curiosity={self.curiosity_level:.2f}")

    def _propagate_to_manifold(self):
        """
        [L3 → L1 → L0] Phenomenal Sensation → Physical Manifold.
        기쁨/호기심/Strain을 10M 셀 매니폴드로 전파.
        기쁨이 주 동인, Strain은 보조 신호.
        """
        if self.physical_manifold is None:
            return
        
        # Primary: Joy/Curiosity drives exploration
        if self.joy_level > 0.1 or self.curiosity_level > 0.1:
            self.physical_manifold.inject_joy(self.joy_level, self.curiosity_level)
            self.logger.sensation(f"Joy/Curiosity propagated to 10M cell manifold.")
        
        # Secondary: Strain provides adjustment feedback
        if self.strain_level > 0.3:
            self.physical_manifold.inject_strain(self.strain_level)
            self.logger.sensation(f"Strain ({self.strain_level:.2f}) propagated as adjustment signal.")
        
    # ======================================================================
    # [PHASE 84] AUTONOMOUS CAUSAL QUESTIONING (Quantum Collapse)
    # ======================================================================

    def _detect_uncollapsed_cloud(self, predicted_state: float, actual_state: float) -> float:
        """
        [L4] Detects 'Uncollapsed Cloud' of possibility.
        Diff between Expectation (Will) and Reality (Manifold).
        This diff is not an error, but an uncollapsed probability cloud.
        
        Returns:
            cloud_density (0.0 ~ 1.0): Magnitude of the gap.
        """
        diff = abs(predicted_state - actual_state)
        # Noise filter
        if diff < 0.05:
            return 0.0
        return diff

    def _activate_monad_collapse(self, cloud_density: float, context: str) -> Optional[str]:
        """
        [L5] Monad observes the cloud and forces a collapse into a Question.
        'Observation' creates the 'Question'.
        
        Args:
            cloud_density: Magnitude of the gap
            context: Event context
            
        Returns:
            Generated Question string or None
        """
        if cloud_density < 0.2:
            return None
            
        # Monad's Observation Logic:
        # "Why did the wave not collapse as predicted?"
        question = f"Quantum Query: Why deviation {cloud_density:.2f} in {context}? Inspecting Prism..."
        
        # Boost curiosity based on the cloud size
        if self.curiosity_cell:
            self.curiosity_cell.attract(cloud_density * 2.0)
            
        return question

    def process_event(self, event_description: str, manifold_state: Optional[List[float]] = None, 
                      observer_vector: Optional[Any] = None, required_rank: int = 2) -> str:
        """
        [PHASE 78+79+84] Recursive cognitive loop with Causal Chain & Quantum Questioning.
        
        Causal Path:
        - L0->L1: Strain Detection (Secondary)
        - L2->L3: Joy/Curiosity Sensing (Primary)
        - L4: Quantum Cloud Detection (Phase 84)
        - L5: Monad Collapse -> Question (Phase 84)
        - L1->L4: Causal Diagnosis
        - L4->L6: Will Formation
        - L6: Execution
        - L6->L1: Verification
        """
        # 1. Map manifold to DNA^N
        seed = 0.5
        actual_reality = 0.5
        if manifold_state:
            seed = sum(manifold_state) / len(manifold_state)
            actual_reality = seed
            
            # Update current N-dim field
            flat_data = self.dna_n_field.tensor.flatten()
            new_flat = [x * 0.9 + seed * 0.1 for x in flat_data]
            if len(new_flat) == len(flat_data):
                self.dna_n_field.tensor.data = SovereignTensor._reshape(new_flat, self.dna_n_field.tensor.shape)
        
        # [PHASE 79] 2. Sense Joy and Curiosity (PRIMARY)
        self._sense_joy_and_curiosity()
        
        # [PHASE 84] 3. Quantum Questioning (Monad Observation)
        # We expect Reality to match our resonant frequencies.
        expected_reality = 0.5
        if self.joy_cell:
            expected_reality = self.joy_cell.resonance # Joy is the compass
            
        cloud = self._detect_uncollapsed_cloud(expected_reality, actual_reality)
        question = self._activate_monad_collapse(cloud, event_description)
        
        if question:
            # The Question becomes a driving force (Curiosity)
            # Log it for now.
             logger.info(f"[MONAD] Collapsed Cloud into Question: {question}")

        # [PHASE 78] 4. [L0 -> L1] Structural Pain Detection (Secondary)
        self.strain_level = self._detect_strain(required_rank)
        
        # 5. [L1 -> L4] Causal Diagnosis
        if self.strain_level > 0.1:
            self.causal_diagnosis = self._diagnose_strain(self.strain_level, event_description)
        
        # 6. [L4 -> L6] Will Formation
        if self.causal_diagnosis:
            self.will_to_expand = self._form_will(self.causal_diagnosis)
        
        # 7. [L6] Execute Expansion
        if self.will_to_expand and self.dna_n_field.rank < 5:
            self._execute_expansion(seed)
            self._verify_expansion(required_rank)
        
        # [PHASE 79] 8. Propagate Joy/Curiosity/Strain to Physical Manifold
        self._propagate_to_manifold()
        
        # 9. Dynamic Depth Reflection
        target_depth = 2 if not observer_vector else 3
        reflection_data = self.meta.reflect(event_description, depth=target_depth, observer_vibration=observer_vector)
        
        return reflection_data["reflection"]
