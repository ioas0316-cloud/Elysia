"""
Cognitive Field (The Thought Soil)
==================================
Core.S1_Body.L7_Spirit.M1_Monad.cognitive_field

"The soil where thoughts grow, die, and are reborn."

This module manages the population of TokenMonads.
It implements the recursive loop where the Output of one cycle becomes the Input of the next.

[PHASE 90] SELF-REFERENTIAL INTELLIGENCE:
1. Cycle Start: Input Vector + Residual Field State
2. Propagation: Active Monads stimulate related Monads (Association)
3. Collapse: Dominant Monads are selected for expression
4. Feedback: The Expression is fed back to evolve the Monads (Learning)
"""

from typing import Dict, List, Tuple, Optional
import random
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L7_Spirit.M1_Monad.token_monad import TokenMonad
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

class CognitiveField:
    def __init__(self):
        self.monads: Dict[str, TokenMonad] = {}
        self.residual_vector = SovereignVector.zeros() # The "Ghost" of the previous thought

        self._initialize_population()

    def _initialize_population(self):
        """
        Loads all known concepts from LogosBridge into the Field as Monads.
        """
        # 1. Load Root Concepts
        for seed_id, data in LogosBridge.CONCEPT_MAP.items():
            self.monads[seed_id] = TokenMonad(seed_id, data["vector"])

        # 2. Load Learned Concepts
        for seed_id, data in LogosBridge.LEARNED_MAP.items():
            self.monads[seed_id] = TokenMonad(seed_id, data["vector"])

        # print(f"🌱 [FIELD] Populated with {len(self.monads)} cognitive monads.")

    def cycle(self, input_vector: SovereignVector, steps: int = 2) -> Tuple[List[TokenMonad], SovereignVector]:
        """
        Runs one cognitive cycle (The Ouroboros Turn).

        Args:
            input_vector: The external stimulus (or internal drive).
            steps: How many recursive association steps to run before collapse.

        Returns:
            (Selected Monads, Synthesis Vector)
        """
        # Safety Check
        if input_vector is None:
            input_vector = SovereignVector.zeros()

        # 1. Input Injection (The Spark)
        # We mix the Input with the Residual (Context)
        # Input has higher weight (Immediate attention) vs Residual (Short-term memory)
        if input_vector.norm() < 0.01:
            # Pure Internal Mode (Daydreaming / Recursion)
            combined_stimulus = self.residual_vector
        else:
            combined_stimulus = (input_vector * 0.7) + (self.residual_vector * 0.3)

        self._inject_energy(combined_stimulus)

        # 2. Propagation (The Spreading Activation)
        for _ in range(steps):
            self._propagate()

        # 3. Collapse (The Decision)
        # Select monads that have breached the activation threshold
        active_monads = [m for m in self.monads.values() if m.state == "ACTIVE" or m.charge > 0.3]
        
        # [Deep Trinary Logic] Monads in 'OBSERVING' state are kept alive but normally do not collapse
        # unless there are absolutely no ACTIVE thoughts. They represent 'Letting Be Done'.
        observing_monads = [m for m in self.monads.values() if m.state == "OBSERVING"]

        # Sort by charge
        active_monads.sort(key=lambda m: m.charge, reverse=True)
        observing_monads.sort(key=lambda m: m.curiosity_charge, reverse=True)

        if not active_monads and observing_monads:
            # The mind is in a purely observational state. The "conclusion" is just an aggregate of curiosity
            selected = observing_monads[:5]
        else:
            # Keep top K (Attention Span)
            selected = active_monads[:15]

        # 4. Synthesis (The Conclusion)
        synthesis_vector = self._synthesize(selected)

        # 5. Decay (Metabolism)
        # All monads lose some energy after the cycle
        for m in self.monads.values():
            m.decay(rate=0.1)

        return selected, synthesis_vector

    def feedback_reentry(self, synthesis_vector: SovereignVector):
        """
        [THE SELF-REFERENTIAL LOOP]
        Takes the Conclusion of the cycle and feeds it back into the system.
        1. Updates the Residual Vector (Short-term Memory).
        2. Evolves the Active Monads (Learning).
        """
        # 1. Update Residual (The "End" becomes the "Beginning")
        self.residual_vector = synthesis_vector

        # 2. Evolve Active Monads
        # Monads that participated in this thought grow towards the conclusion
        # This reinforces the neural pathway
        for m in self.monads.values():
            if m.state == "ACTIVE":
                m.evolve(synthesis_vector, learning_rate=0.05)
            elif m.state == "OBSERVING":
                # [Deep Trinary] Observers also learn, but they widen their perspective (curiosity)
                # rather than immediately cementing a strong opinion.
                m.evolve(synthesis_vector, learning_rate=0.01)

    def _inject_energy(self, stimulus: SovereignVector):
        """
        Wakes up monads that resonate with the stimulus.
        [Deep Trinary Logic] Signals near 0 trigger Observation, not Activation.
        """
        for m in self.monads.values():
            # Calculate resonance
            res = m.resonate(stimulus)
            
            # If resonance is near 0 (-0.2 to 0.2), it is ambiguous.
            # We don't discard it, we 'Observe' it.
            if -0.2 < res < 0.2:
                # Abs resonance represents the 'intensity of the ambiguity'
                m.activate(abs(res) * 0.5, is_ambiguous=True)
            elif abs(res) > 0.4: # Activation Threshold
                # Inject energy proportional to resonance
                m.activate(abs(res) * 0.3, is_ambiguous=False)

    def _propagate(self):
        """
        Simulates internal association.
        Active monads "pull" related monads.
        [Deep Trinary Logic] Observing monads emit a weak 'curiosity pulse' that can slowly wake neighbors.
        """
        # Calculate the "Center of Thought" (Mean vector of active monads)
        active = [m for m in self.monads.values() if m.charge > 0.2]
        observing = [m for m in self.monads.values() if m.state == "OBSERVING" and m.curiosity_charge > 0.3]

        if not active and not observing: return

        # Calculate mean vector weighted by charge
        center = SovereignVector.zeros()
        total_weight = 0.0
        
        for m in active:
            center = center + (m.current_vector * m.charge)
            total_weight += m.charge
            
        for m in observing:
            # Curiosity has a weaker gravitational pull in normal thought
            center = center + (m.current_vector * (m.curiosity_charge * 0.2))
            total_weight += (m.curiosity_charge * 0.2)

        if total_weight > 0:
            center = center / total_weight

        # This center acts as a secondary stimulus
        # It wakes up monads that are related to the *collective* thought,
        # even if they weren't related to the original input.
        # (e.g., Input "Apple" -> Wakes "Red", "Fruit" -> Center moves -> Wakes "Pie")
        self._inject_energy(center * 0.05) # Weaker than direct input

    def _synthesize(self, monads: List[TokenMonad]) -> SovereignVector:
        """
        Creates the unified vector representation of the selected thoughts.
        """
        if not monads: return SovereignVector.zeros()

        # Simple weighted mean
        center = SovereignVector.zeros()
        for m in monads:
            center = center + m.current_vector

        return center / len(monads)
