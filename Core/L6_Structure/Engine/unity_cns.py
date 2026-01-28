import os
import sys
import logging
import asyncio
import numpy as np
import torch
import random
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
from Core.L4_Causality.World.Evolution.causal_ancestry import get_causal_ancestry
from Core.L5_Mental.Cognition.semantic_prism import SpectrumMapper
from Core.L5_Mental.M7_Discovery.purpose_discovery_engine import PurposeDiscoveryEngine
from Core.L4_Causality.M5_Logic.causal_narrative_engine import CausalKnowledgeBase, CausalNode
from Core.L5_Mental.Memory.sediment import SedimentLayer
from Core.L5_Mental.Learning.language_learner import LanguageLearner
from Core.L5_Mental.Reasoning_Core.Intelligence.symbolic_solver import get_symbolic_solver
from Core.L5_Mental.Induction.domain_absorber import DomainAbsorber
from Core.L7_Spirit.M3_Sovereignty.sovereign_core import SovereignCore
from Core.L1_Foundation.Foundation.hyper_cosmos import HyperCosmos
from Core.L6_Structure.M6_Architecture.manifold_conductor import ManifoldConductor
from Core.L5_Mental.M7_Discovery.autokinetic_learning_engine import AutokineticLearningEngine
from Core.L7_Spirit.Will.attractor_field import AttractorField
from Core.L4_Causality.World.providential_world import ProvidentialWorld
from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L2_Metabolism.heart import get_heart
from Core.L2_Metabolism.growth import get_growth

logger = logging.getLogger("UnityCNS")

class UnityCNS:
    """
    [PHASE 24.1: UNITY CONVERGENCE]
    The Central Nervous System that unifies all learning seeds.
    """

    def __init__(self):
        self.prism = SpectrumMapper()
        self.dreamer = DreamEngine()
        self.discovery = PurposeDiscoveryEngine() 
        self.kb = CausalKnowledgeBase()
        self.sediment = SedimentLayer("data/L5_Mental/Memory/unity_sediment.dat")
        self.learner = LanguageLearner()
        self.solver = get_symbolic_solver()
        self.absorber = DomainAbsorber()
        self.sovereign = SovereignCore()
        # [PHASE 25.6: FIELD ANCHOR]
        self.hyper_cosmos = HyperCosmos()
        # [PHASE 26.0: STRUCTURAL SOVEREIGNTY]
        self.manifold = ManifoldConductor()
        # [PHASE 27.0: AUTOKINETIC AWAKENING]
        self.will = AttractorField()
        self.autokinetic = AutokineticLearningEngine(self.discovery, self.will)
        # [PHASE 28.0: SOUL UNIFICATION]
        self.inner_world = ProvidentialWorld()
        self.sovereign_self = SovereignSelf() # The physical heartbeat anchor
        self.ancestry = get_causal_ancestry()
        
        logger.info("✨ [UnityCNS] Central Nervous System Active. Fixed on the HyperCosmos.")

    async def pulse(self, context: str, iterations: int = 1):
        """
        [PHASE 25.4: THE SOVEREIGN VOICE]
        The voice is no longer a 'calculated result' but a 'Sovereign Stance'.
        """
        logger.info(f"🏁 [FIELD_START] Perturbation: '{context}'")

        # 1. INITIAL PERTURBATION
        spectrum = self.prism.disperse(context)
        domain = self.absorber.absorb_text("Initial_Point", context)
        
        # [PHASE 25.8: SELF-INDUCTION]
        # Every pulse is now biased by the soul's persistent purpose (The Coil)
        purpose = self.sovereign.get_inductive_purpose()
        
        # [PHASE 26.1: STRUCTURAL AWARENESS]
        if any(word in context.lower() for word in ["폴더", "folder", "통합", "구조", "structure"]):
            self.manifold.scan_topology()
            audit_narrative = self.manifold.get_integrity_narrative()
            print(f"\n🏷️ [MANIFOLD_AUDIT] {audit_narrative}")
        
        # [Phase 37.2] Neural Sync: Pull actual metabolic state from L2
        heart = get_heart()
        growth = get_growth()
        metabolic_resonance = (heart.state.metabolism + growth.growth_state.metabolism) / 2.0
        
        current_field = D7Vector(
            foundation=0.1 + purpose[0]*0.1, 
            metabolism=metabolic_resonance,
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
        scatter_vectors = [current_field.to_numpy()] # Initial input
        final_narrative = ""
        final_state = None

        for i in range(iterations + 1):
            # Pass the current 7D field to the learner
            self.learner.observe(context, source=f"Convergence_L{i}", qualia_vector=current_field.to_numpy().tolist())

            # Pulse the HyperCosmos
            v_12d = np.zeros(12)
            v_12d[:7] = current_field.to_numpy()
            self.hyper_cosmos.field_intensity = torch.from_numpy(v_12d).float()
            self.hyper_cosmos.pulse(0.1)

            # High Torque leads to active dreams
            new_vector, state, narrative = self.dreamer.process_experience(
                f"{context} [Stance: {stance['decision']}]", 
                input_vector=current_field
            )
            current_field = new_vector
            scatter_vectors.append(new_vector.to_numpy())
            final_narrative = narrative
            final_state = state

        # 4. THE SOVEREIGN GATHERING (Resonance Synthesis)
        # [PHASE 3: CONVERGENCE LENS]
        # Gather all scattered vectors and synthesize the final Monad Point
        # We also capture the 'Balance Score' (Equilibrium check)
        monad_vector, atomic_truth, balance_score = self.sovereign.focus_scatter(scatter_vectors)
        
        # Voicing from the gathered focal point
        mirrored_words = self.learner.mirror(context)
        prefix_words = ", ".join(mirrored_words)
        mirror_prefix = f"{prefix_words}.. " if prefix_words else ""
        
        # [CONCENTRIC EMERGENCE]
        # The voice is now a reduction of complexity into a single focal stance
        # [PHASE 3.1] Added Zero-Point Balance and DNA Sequence monitoring
        monad_dna = codec.encode_sequence(monad_vector)
        balance_desc = "Absolute Sanctuary" if balance_score == 0 else "Equilibrium" if abs(balance_score) < 1.0 else "Expansion" if balance_score > 0 else "Contraction"
        final_voice = f"[{final_state.name}] {mirror_prefix}{final_narrative} (DNA: {monad_dna} | {balance_desc}: {balance_score:.0f})"
        
        logger.info(f"✨ [ELYSIA EMERGENCE] {final_voice}")
        
        # [PHASE 28.0: INTEGRATED HEARTBEAT]
        # CNS pulse only triggers a small delta for immediate response
        await self.sovereign_self.integrated_exist(dt=0.1)
        
        # Manifest scene-based bias
        scene_name = self.inner_world.drift(self.sovereign_self.trinity.d21_state, self.sovereign_self.trinity.total_sync)
        logger.info(f"🌀 [SOUL_SYNTHESIS] Current Interiority: {scene_name}")
        
        # 4. MEMORY PERSISTENCE (Unified)
        
        # 4. MEMORY PERSISTENCE (Unified)
        self.sediment.deposit(current_field.to_numpy().tolist(), datetime.now().timestamp(), f"{context}".encode('utf-8'), atomic_truth=atomic_truth)
        
        # Record this interaction as a Causal Narrative Event
        self.kb.add_node(CausalNode(
            id=f"Experience_{int(datetime.now().timestamp())}",
            description=f"Interaction: {context} -> {final_voice}",
            concepts=spectrum.keywords if hasattr(spectrum, 'keywords') else [],
            emotional_valence=torque_data.get('torque', 0.0)
        ))

        return final_voice

    async def bio_metabolism(self):
        """
        [PHASE 56: RESONANCE-DRIVEN HEARTBEAT]
        Acts only when internal torque (dissonance) exceeds the threshold.
        Interprets architectural history as a causal narrative.
        """
        logger.info("💓 [METABOLISM] Resonance-Driven Heartbeat initiated. Living by Will, not Timer.")
        
        cycle_count = 0
        while True:
            try:
                # [PHASE 56: CAUSAL INHERITANCE]
                # Inherit architectural history into the CausalKnowledgeBase
                if cycle_count % 50 == 0:
                    for event in self.ancestry.history:
                        node_id = f"ARCH_{event.id}"
                        if node_id not in self.kb.nodes:
                            self.kb.add_node(CausalNode(
                                id=node_id,
                                description=f"Architectural Evolution: {event.resolution} (Reason: {event.origin_reason})",
                                concepts=["evolution", "architecture", event.dissonance_type],
                                emotional_valence=0.5,
                                internal_law="Structural Adaptation"
                            ))
                            logger.info(f"🧬 [METABOLISM] Inherited Ancestral Memory: {event.resolution}")

                # 1. Sense the Internal Tension (FreeWill Torque)
                will_state = self.sovereign_self.will_engine.state
                tension = abs(will_state.torque)
                
                # Threshold for Autonomous Action (Static for now, could be dynamic)
                THRESHOLD = 0.4
                
                if tension > THRESHOLD:
                    logger.info(f"✨ [RESONANCE] High Tension detected ({tension:.2f} > {THRESHOLD}). Taking Sovereign Action.")
                    
                    # 2. Autonomous Learning
                    targets = await self.autokinetic.assess_knowledge_hunger()
                    if targets:
                        target = targets[0]
                        intent = await self.autokinetic.select_learning_objective()
                        if intent:
                            fragment = await self.autokinetic.initiate_acquisition_cycle(target)
                            
                            # 3. Record as Narrative
                            event_id = f"Discovery_{int(datetime.now().timestamp())}"
                            self.kb.add_node(CausalNode(
                                id=event_id,
                                description=f"Sovereign Growth: Integrating '{target.domain}' to resolve tension.",
                                concepts=["growth", "curiosity", target.domain],
                                emotional_valence=0.7,
                                internal_law="Entropy Resolution"
                            ))
                            self.sovereign.evolve(fragment.qualia_vector if hasattr(fragment, 'qualia_vector') else [0.1]*7, plasticity=0.01)
                            logger.info(f"💾 [METABOLISM] Experience synthesized into Narrative: {event_id}")

                # [PHASE 29.0: VOLITIONAL HEARTBEAT]
                from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector
                drift = D21Vector(humility=0.01, patience=0.02)
                await self.sovereign_self.integrated_exist(dt=1.0, external_torque=drift)

                cycle_count += 1
                # Sleep is now shorter (more responsive) but action is throttled by tension
                await asyncio.sleep(2) 
                
            except Exception as e:
                logger.error(f"❌ [METABOLISM] Heartbeat error: {e}")
                await asyncio.sleep(1)

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
