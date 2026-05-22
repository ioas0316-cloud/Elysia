from typing import List, Dict, Any, Optional
import logging
from Core.Keystone.sovereign_math import SovereignTensor

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
    
    [REFACTORED] No longer wraps strings.
    - Think^1: Generates a grounded observation from the KG.
    - Think^2: Extracts causal claims from Think^1, verifies them against KG.
    - Think^N: Recursively audits the validity of lower-level audits.
    """
    def __init__(self, reasoner: Any, kg_manager: Any):
        self.reasoner = reasoner
        self.kg = kg_manager

    def reflect(self, target_thought: str, depth: int, observer_vibration: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generates an N-th order reflection with genuine causal auditing.
        
        Returns:
            {
                "reflection": full multi-layer text,
                "final_layer": the deepest reflection,
                "depth": requested depth,
                "node": grounding KG node (if found),
                "audit": {claims_checked, valid_count, invalid_count, details}
            }
        """
        # 1. Base Grounding (L1): Find a concept anchor in the KG
        words = [w.strip("?.!,'\"") for w in target_thought.split()]
        concept_node = None
        anchor_word = None
        for word in reversed(words):
            concept_node = self.kg.get_node(word.lower())
            if concept_node:
                anchor_word = word.lower()
                break

        reflections = []
        audit_result = {"claims_checked": 0, "valid_count": 0, "invalid_count": 0, "details": []}

        # === Think^1: Grounded Observation ===
        think1 = self._think_level_1(target_thought, concept_node, anchor_word)
        reflections.append(f"[Think^1] {think1}")
        
        if depth >= 2:
            # === Think^2: Causal Validity Audit of Think^1 ===
            audit_result = self._think_level_2(think1, concept_node, anchor_word)
            audit_summary = (
                f"Audited {audit_result['claims_checked']} causal claims: "
                f"{audit_result['valid_count']} valid, {audit_result['invalid_count']} invalid"
            )
            if audit_result['invalid_count'] > 0:
                invalid_details = [d for d in audit_result['details'] if not d['valid']]
                first_invalid = invalid_details[0]['claim'] if invalid_details else "unknown"
                audit_summary += f". Contestation: '{first_invalid}' lacks KG grounding."
            reflections.append(f" | [Think^2] {audit_summary}")
        
        if depth >= 3:
            # === Think^3: Meta-Audit — Is the audit itself well-grounded? ===
            meta_audit = self._think_level_3(audit_result)
            reflections.append(f" |  | [Think^3] {meta_audit}")
        
        if depth >= 4:
            # === Think^4+: Recursive Self-Questioning ===
            for d in range(4, depth + 1):
                prefix = " | " * (d - 1)
                meta_q = (
                    f"My recursive audit (depth {d-1}) assumed that KG completeness "
                    f"implies causal validity. This assumption itself is "
                    f"{'justified' if self.kg.get_summary().get('total_edges', 0) > 10 else 'questionable'} "
                    f"given {self.kg.get_summary().get('total_edges', 0)} total KG edges."
                )
                reflections.append(f"{prefix}[Think^{d}] {meta_q}")

        return {
            "reflection": "\n".join(reflections),
            "final_layer": reflections[-1] if reflections else "",
            "depth": depth,
            "node": concept_node,
            "audit": audit_result
        }

    def _think_level_1(self, thought: str, node: Optional[Dict], anchor: Optional[str]) -> str:
        """Think^1: Generate a grounded observation from live KG data."""
        if not node or not anchor:
            return f"I perceive '{thought}' but find no structural anchor in my knowledge. This is open space."
        
        # What does the KG say about this concept?
        causes = self.kg.find_causes(anchor)
        effects = self.kg.find_effects(anchor)
        mass = self.kg.calculate_mass(anchor)
        
        parts = [f"I perceive '{anchor}' (Mass={mass:.1f})."]
        
        if causes:
            source = causes[0].get('source', 'unknown')
            parts.append(f"It is causally born from '{source}'.")
        if effects:
            target = effects[0].get('target', 'unknown')
            parts.append(f"It drives '{target}'.")
        if not causes and not effects:
            parts.append("It stands isolated — a concept without causal connections.")
        
        return " ".join(parts)

    def _think_level_2(self, think1_output: str, node: Optional[Dict], anchor: Optional[str]) -> Dict:
        """
        Think^2: Extract causal claims from Think^1 and verify against KG.
        This is NOT string wrapping — it audits the validity of Think^1's assertions.
        """
        result = {"claims_checked": 0, "valid_count": 0, "invalid_count": 0, "details": []}
        
        # Extract causal claim patterns from think1 text
        claims = self._extract_causal_claims(think1_output)
        
        for claim in claims:
            result["claims_checked"] += 1
            # Verify each claim against the KG
            is_valid = self._verify_claim(claim)
            if is_valid:
                result["valid_count"] += 1
            else:
                result["invalid_count"] += 1
            result["details"].append({"claim": claim, "valid": is_valid})
        
        return result

    def _think_level_3(self, audit: Dict) -> str:
        """
        Think^3: Meta-audit — examines whether the audit methodology was sound.
        Questions: Was the KG complete enough? Were the right claims extracted?
        """
        total = audit["claims_checked"]
        if total == 0:
            return "No claims were audited. The absence of claims itself suggests shallow observation at Think^1."
        
        validity_rate = audit["valid_count"] / total if total > 0 else 0
        kg_summary = self.kg.get_summary()
        kg_edges = kg_summary.get('total_edges', 0) if isinstance(kg_summary, dict) else 0
        
        if validity_rate >= 0.8:
            assessment = (
                f"Audit found {validity_rate:.0%} validity across {total} claims. "
                f"However, with only {kg_edges} KG edges, high validity may reflect "
                f"KG incompleteness rather than genuine causal truth."
            )
        elif validity_rate >= 0.5:
            assessment = (
                f"Mixed validity ({validity_rate:.0%}). "
                f"The contested claims warrant further investigation, "
                f"possibly through additional self-observation or Architect dialogue."
            )
        else:
            assessment = (
                f"Low validity ({validity_rate:.0%}). "
                f"Think^1's assertions are largely ungrounded. "
                f"This suggests either KG deficiency or flawed observation."
            )
        
        return assessment

    def _extract_causal_claims(self, text: str) -> List[str]:
        """Extracts causal assertions from a text string."""
        claims = []
        
        # Pattern: "X is causally born from Y" → claim = "Y causes X"
        import re
        born_from = re.findall(r"born from '(\w+)'", text)
        for source in born_from:
            claims.append(f"{source} causes observed concept")
        
        # Pattern: "It drives Y" → claim = "concept causes Y"
        drives = re.findall(r"drives '(\w+)'", text)
        for target in drives:
            claims.append(f"observed concept causes {target}")
        
        # If no explicit claims found, the thought itself is a claim
        if not claims and "perceive" in text:
            claims.append("concept exists in knowledge structure")
        
        return claims

    def _verify_claim(self, claim: str) -> bool:
        """Verifies a single causal claim against the KG."""
        import re
        
        # "X causes Y" pattern
        match = re.match(r"(\w+) causes (\w+)", claim)
        if match:
            source, target = match.group(1).lower(), match.group(2).lower()
            if target == "observed":
                # "X causes observed concept" — check if X exists and has effects
                return bool(self.kg.get_node(source) and self.kg.find_effects(source))
            if source == "observed":
                # "observed concept causes Y" — check if Y exists and has causes
                return bool(self.kg.get_node(target) and self.kg.find_causes(target))
            # Direct check
            effects = self.kg.find_effects(source)
            return any(e.get('target') == target for e in effects)
        
        # "concept exists" pattern
        if "exists" in claim:
            return True  # We already confirmed existence in Think^1
        
        return False

class SovereignCognition:
    """
    The High-Level 'Adult' Brain of Elysia.
    [PHASE 78] Sovereign Necessity: Causal Chain for Self-Expansion.
    """
    def __init__(self, manifold: Any = None, joy_cell: Any = None, curiosity_cell: Any = None):
        from Core.Cognition.causal_reasoner import CausalReasoner
        from Core.Cognition.kg_manager import get_kg_manager
        from Core.System.somatic_logger import SomaticLogger
        
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
        if manifold_state is not None and len(manifold_state) > 0:
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
            pass

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
