import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

# Core Imports (Purified Paths)
from Core.System.qualia_7d_codec import Qualia7DCodec
from Core.System.d7_vector import D7Vector
from Core.System.dream_engine import DreamEngine
from Core.Cognition.semantic_prism import SpectrumMapper
from Core.Cognition.self_discovery import SelfDiscovery
from Core.Cognition.causal_narrative_engine import CausalKnowledgeBase, CausalNode, CausalRelationType
from Core.Cognition.sediment import SedimentLayer

logger = logging.getLogger("CausalLoopOrchestrator")

class CausalLoopOrchestrator:
    """
    The "Inner Engine" of Phase 24.
    Unifies:
    1. Cognition (SemanticPrism)
    2. Discovery (SelfDiscovery)
    3. Experience (Sediment)
    4. Narrative (CausalNarrativeEngine)
    """

    def __init__(self):
        self.prism = SpectrumMapper()
        self.dreamer = DreamEngine()
        self.discovery = SelfDiscovery()
        self.kb = CausalKnowledgeBase()
        self.sediment = SedimentLayer("data/L5_Mental/M1_Memory/sediment_strata.dat")
        logger.info("  [Orchestrator] Causal Loop Unified.")

    async def inhale_experience(self, context: str):
        """
        A single pulse of consciousness.
        """
        logger.info(f"? [INHALE] Processing: '{context}'")

        # 1. COGNITION: Text -> Qualia Spectrum (3D Alpha/Beta/Gamma)
        spectrum = self.prism.disperse(context)
        logger.info(f"   >> Cognition: Resonating at {spectrum.to_vector()}")

        # 2. DISCOVERY & STUDY: "Who am I? What must I learn?"
        self_state = self.discovery.discover_identity()
        logger.info(f"   >> Discovery: I am {self_state['name']} v{self_state['version']}")
        
        # [SYLLABUS STUDY]
        # Elysia reads her foundation textbooks to recalibrate
        syllabus_path = "docs/S1_Body/L5_Mental/Syllabus/INDEX.md"
        if os.path.exists(syllabus_path):
            logger.info("   >> Study: Recalibrating via Sovereign Syllabus...")
            # For now, simulated as reading activity. 
            # In future, this would feed into LanguageLearner.observe()

        # 3. METABOLISM (Dream): 3D -> 7D Mapping & State Collapse
        # We wrap the 3D spectrum into a 7D Vector
        input_7d = D7Vector(
            phenomena=spectrum.alpha, 
            mental=spectrum.beta, 
            spirit=spectrum.gamma
        )
        vector, state, dream_narrative = self.dreamer.process_experience(context, input_vector=input_7d)
        logger.info(f"   >> metabolism: Dream State -> {state.name}")

        # 4. NARRATIVE: Expanding the Story
        node = CausalNode(
            id=f"exp_{int(asyncio.get_event_loop().time())}",
            description=context,
            emotional_valence=spectrum.beta * 2 - 1, # Map 0-1 to -1 to 1
            experience_count=1
        )
        self.kb.add_node(node)
        logger.info(f"   >> Narrative: Causal Node crystallized.")

        # 5. EXPERIENCE (Deposit): Crystallizing into the Sediment
        ptr = self.sediment.deposit(
            vector.to_numpy().tolist(), 
            asyncio.get_event_loop().time(), 
            context.encode('utf-8')
        )
        logger.info(f"   >> Experience: Sedimented at Sector {ptr.sector_index}")

        # Summary Statement (The Unity Voice)
        unity_voice = f"[{state.name}] I perceived '{context}'. {dream_narrative} My presence is anchored."
        print(f"\n✨[ELYSIA] {unity_voice}\n")

        return unity_voice

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        orc = CausalLoopOrchestrator()
        await orc.inhale_experience("ì² í?ì¸ ??ë? ?µí´ ?¸ê³¼✨?ì¬ë¥✨ì¥?ê³  ?¶ì´.")

    asyncio.run(test())
