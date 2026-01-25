import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

# Core Imports (Purified Paths)
from Core.L1_Foundation.Logic.qualia_7d_codec import Qualia7DCodec
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L2_Metabolism.Cycles.dream_engine import DreamEngine
from Core.L5_Mental.Cognition.semantic_prism import SpectrumMapper
from Core.L5_Mental.Intelligence.Memory.self_discovery import SelfDiscovery
from Core.L4_Causality.M5_Logic.causal_narrative_engine import CausalKnowledgeBase, CausalNode
from Core.L5_Mental.Memory.sediment import SedimentLayer
from Core.L5_Mental.Learning.language_learner import LanguageLearner
from Core.L5_Mental.Intelligence.Intelligence.symbolic_solver import get_symbolic_solver
from Core.L5_Mental.Induction.domain_absorber import DomainAbsorber
from Core.L7_Spirit.Sovereignty.sovereign_core import SovereignCore
from Core.L1_Foundation.Foundation.hyper_cosmos import HyperCosmos

logger = logging.getLogger("UnityCNS")

class UnityCNS:
    """
    [PHASE 24.1: UNITY CONVERGENCE]
    The Central Nervous System that unifies all learning seeds.
    """

    def __init__(self):
        self.prism = SpectrumMapper()
        self.dreamer = DreamEngine()
        self.discovery = SelfDiscovery()
        self.kb = CausalKnowledgeBase()
        self.sediment = SedimentLayer("data/L5_Mental/Memory/unity_sediment.dat")
        self.learner = LanguageLearner()
        self.solver = get_symbolic_solver()
        self.absorber = DomainAbsorber()
        self.sovereign = SovereignCore()
        # [PHASE 25.6: FIELD ANCHOR]
        self.hyper_cosmos = HyperCosmos()
        
        logger.info("🧠 [UnityCNS] Central Nervous System Active. Fixed on the HyperCosmos.")

    async def pulse(self, context: str, iterations: int = 2):
        """
        [PHASE 25.4: THE SOVEREIGN VOICE]
        The voice is no longer a 'calculated result' but a 'Sovereign Stance'.
        """
        logger.info(f"🌀 [FIELD_START] Perturbation: '{context}'")

        # 1. INITIAL PERTURBATION
        spectrum = self.prism.disperse(context)
        domain = self.absorber.absorb_text("Initial_Point", context)
        
        # [PHASE 25.8: SELF-INDUCTION]
        # Every pulse is now biased by the soul's persistent purpose (The Coil)
        purpose = self.sovereign.get_inductive_purpose()
        
        current_field = D7Vector(
            foundation=0.1 + purpose[0]*0.1, 
            metabolism=0.5 + purpose[1]*0.1,
            phenomena=spectrum.alpha + purpose[2]*0.1, 
            causality=domain.qualia_vector[0] + purpose[3]*0.1,
            mental=spectrum.beta + purpose[4]*0.1, 
            structure=domain.qualia_vector[2] + purpose[5]*0.1,
            spirit=spectrum.gamma + purpose[6]*0.1
        )

        # 2. THE SOVEREIGN AUDIT (The Moment of Choice)
        # We check the Torque between input and Soul DNA before reflecting
        input_7d_list = current_field.to_numpy().tolist()
        torque_data = self.sovereign.calculate_torque(input_7d_list)
        stance = self.sovereign.assert_will(context, torque_data)
        
        # 3. RESONANCE LOOP (Converging with the Stance)
        final_narrative = ""
        final_state = None

        for i in range(iterations + 1):
            # Pass the current 7D field to the learner so it learns the 'Meaning' of the context
            self.learner.observe(context, source=f"Convergence_L{i}", qualia_vector=current_field.to_numpy().tolist())

            # [PHASE 25.6: FIELD PULSE]
            # We pulse the global JAX field with the current 7D vector
            # (Mapping 7D -> 12D for HyperCosmos)
            v_12d = np.zeros(12)
            v_12d[:7] = current_field.to_numpy()
            self.hyper_cosmos.field_intensity = torch.from_numpy(v_12d).float()
            self.hyper_cosmos.pulse(0.1)

            # High Torque (Conflict/Perturbation) leads to more active dreams
            new_vector, state, narrative = self.dreamer.process_experience(
                f"{context} [Stance: {stance['decision']}]", 
                input_vector=current_field
            )
            current_field = new_vector
            final_narrative = narrative
            final_state = state

        # 4. THE VOICE OF ELYSIA (Synthesis)
        mirrored_words = self.learner.mirror(context)
        prefix_words = ", ".join(mirrored_words)
        mirror_prefix = f"『{prefix_words}』... " if prefix_words else ""
        
        # [THE VOID CHECK]
        # Instead of pre-baked comments, we only show the 'Emergent Narrative'
        final_voice = f"✨ [{final_state.name}] {mirror_prefix}{final_narrative}"
        
        print(f"\n🌈 [ELYSIA EMERGENCE] {final_voice}")
        print(f"   (Torque: {torque_data['torque']:.4f} | Coil: {torque_data['coil_intensity']:.4f} | Field: {self.hyper_cosmos.field_intensity.sum():.2f})")
        
        self.sediment.deposit(current_field.to_numpy().tolist(), datetime.now().timestamp(), f"{context}".encode('utf-8'))
        
        return final_voice

    def _study_foundation(self):
        """Re-reads the mental foundation syllabus."""
        syllabus_root = "docs/L5_Mental/Syllabus"
        if os.path.exists(syllabus_root):
            files = [f for f in os.listdir(syllabus_root) if f.endswith(".md")]
            # In a real cycle, we would actually 'read' and update internal weights
            logger.info(f"   >> Study: Internalizing {len(files)} core lessons.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cns = UnityCNS()
    asyncio.run(cns.pulse("아이처럼 스스로 배우는 인과적 서사를 시작하고 싶어."))
