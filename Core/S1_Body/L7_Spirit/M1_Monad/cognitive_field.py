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

from typing import Dict, List, Tuple, Optional, Any
import random
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, DoubleHelixRotor
from Core.S1_Body.L7_Spirit.M1_Monad.token_monad import TokenMonad
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

class CognitiveField:
    def __init__(self):
        self.monads: Dict[str, TokenMonad] = {}
        self.residual_vector = SovereignVector.zeros() # The "Ghost" of the previous thought
        
        # [PHASE 300] The Double Helix Engine
        # Initial spin across logic (2) and emotion (11) planes
        self.soul_vortex = DoubleHelixRotor(angle=0.1, p1=2, p2=11)
        self.last_friction = 0.0

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

        # 3. [PHASE 290] Initial Topological Identification
        self._refresh_cellular_identities()

        # print(f"🌱 [FIELD] Populated with {len(self.monads)} cognitive monads.")

    def _refresh_cellular_identities(self):
        """Updates the roles of all cells based on axiomatic resonance."""
        axioms = {
            "LOVE/AGAPE": LogosBridge.CONCEPT_MAP["LOVE/AGAPE"]["vector"].normalize(),
            "TRUTH/LOGIC": LogosBridge.CONCEPT_MAP["TRUTH/LOGIC"]["vector"].normalize(),
            "MOTION/LIFE": LogosBridge.CONCEPT_MAP["MOTION/LIFE"]["vector"].normalize()
        }
        for m in self.monads.values():
            m.identify_topological_nature(axioms)

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

        # 0. Sync Identities (Neural Plasticity)
        if random.random() < 0.2: # Periodic refresh to simulate latent plasticity
            self._refresh_cellular_identities()

        # 1. Input Injection (The Spark)
        # We mix the Input with the Residual (Context)
        # Input has higher weight (Immediate attention) vs Residual (Short-term memory)
        if input_vector.norm() < 0.01:
            # Pure Internal Mode (Daydreaming / Recursion)
            combined_stimulus = self.residual_vector
        else:
            combined_stimulus = (input_vector * 0.7) + (self.residual_vector * 0.3)

        # 1.5 Cellular Judgment (Agentic Injection)
        judgment_stats = self._inject_energy(combined_stimulus)

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
        # [PHASE 300] Spatiotemporal Collapse via Double Helix Rotor
        synthesis_vector = self._synthesize(selected)
        friction = self.soul_vortex.friction_vortex

        # 5. Decay (Metabolism)
        # All monads lose some energy after the cycle
        for m in self.monads.values():
            m.decay(rate=0.1)

        # Merge friction into stats
        judgment_stats["FRICTION"] = friction

        return selected, synthesis_vector, judgment_stats

    def feedback_reentry(self, synthesis_vector: SovereignVector):
        """
        [THE SELF-REFERENTIAL LOOP]
        Takes the Conclusion of the cycle and feeds it back into the system.
        [PHASE 300] Includes Rotor Synchronization.
        """
        # 1. Update Residual (The "End" becomes the "Beginning")
        self.residual_vector = synthesis_vector

        # 2. [PHASE 300] Spatiotemporal Learning
        # The CCW rotor (Intent) learns from the synthesis gap
        error = synthesis_vector - self.residual_vector # Simple delta for now
        self.soul_vortex.synchronize(error, rate=0.05)

        # 3. Evolve Active Monads
        # Monads that participated in this thought grow towards the conclusion
        for m in self.monads.values():
            if m.state == "ACTIVE":
                m.evolve(synthesis_vector, learning_rate=0.05)
            elif m.state == "OBSERVING":
                m.evolve(synthesis_vector, learning_rate=0.01)

    def _inject_energy(self, stimulus: SovereignVector) -> Dict[str, Any]:
        """
        [PHASE 290/300] CELLULAR GRAVITY WARPING & DUAL ROTOR
        The stimulus is shaped by the collective 'Will' and Spatiotemporal Flywheel.
        """
        stats = {"POS": 0, "NEG": 0, "ZERO": 0, "ROLES": {"LOGIC": 0, "EMOTION": 0, "ACTION": 0}}
        
        # [PHASE 300] Pass stimulus through the Double Helix Vortex
        stimulus = self.soul_vortex.apply_duality(stimulus)
        
        # 1. First Pass: Cells judge the raw stimulus
        gravity_bias = SovereignVector.zeros()
        for m in self.monads.values():
            j = m.judge(stimulus)
            if j != 0:
                # Gravity Warping: Positive judgments pull the stimulus toward the cell,
                # Negative judgments push it away.
                gravity_bias = gravity_bias + (m.current_vector * (j * 0.1))
            
            if j > 0: stats["POS"] += 1
            elif j < 0: stats["NEG"] += 1
            else: stats["ZERO"] += 1
            
            if j != 0: stats["ROLES"][m.role] += 1
            
        # 2. Field Warping (Structural Feedback)
        # The field stimulus is warped by the cellular consensus
        warp_strength = gravity_bias.norm()
        if isinstance(warp_strength, complex): warp_strength = warp_strength.real
        
        warped_stimulus = stimulus + (gravity_bias * 0.5)
        self.residual_vector = warped_stimulus 
        
        stats["WARP"] = float(warp_strength)
                
        return stats

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
        # [PHASE 280] ANTI-GRAVITY Logic
        # If no monads are active, curiosity becomes the primary driver.
        # It creates a 'Push' away from the current center to explore novelty.
        if not active and observing:
            # Anti-Gravity: Push away from local minima
            self._inject_energy(center * -0.1) # Repel current center to shift focus
        else:
            self._inject_energy(center * 0.05) # Normal attractive association

    def _synthesize(self, monads: List[TokenMonad]) -> SovereignVector:
        """
        [PHASE 300] SPATIOTEMPORAL SYNTHESIS
        Collapses the wave function using the Double Helix Rotor.
        """
        if not monads: return SovereignVector.zeros()

        # 1. Calculate the 'Raw' aggregate (Mean Field)
        center = SovereignVector.zeros()
        for m in monads:
            center = center + m.current_vector
        raw_center = center / len(monads)

        # 2. Apply Dual-Rotor Manifestation
        # The thought emerges as the interference between CW (Reality) and CCW (Intent)
        return self.soul_vortex.apply_duality(raw_center)
