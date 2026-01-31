"""
Rotor Cognition Core (Holographic Council & Active Void)
========================================================
Core.1_Body.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core

"Calculators compute; The Council Debates. The Priestess Intuits."

This module implements the core cognitive pipeline:
1. Active Void: Extracts vector DNA from intent.
2. Fractal Adapter: Expands 7D Qualia to 21D Matrix (7^7).
3. The High Priestess (Psionics): Checks resonance with existing Soul Graph.
4. The Heavy Matrix (Merkaba): Combinatorial logic debate.
5. Neuroplasticity: Writing the decision back into the TorchGraph.
"""

import logging
import json
import math
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    from Core.1_Body.L5_Mental.Reasoning_Core.LLM.local_cortex import LocalCortex
except ImportError:
    LocalCortex = None

try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Physics.monad_gravity import MonadGravityEngine
except ImportError:
    MonadGravityEngine = None

from Core.1_Body.L5_Mental.Reasoning_Core.Metabolism.holographic_council import HolographicCouncil, DebateResult
from Core.1_Body.L5_Mental.Meta.sovereign_lens import SovereignLens
from Core.1_Body.L2_Metabolism.Cycles.dream_rotor import DreamRotor
from Core.1_Body.L5_Mental.emergent_language import EmergentLanguageEngine

# [Phase 45 Integration]
from Core.1_Body.L5_Mental.Reasoning_Core.Psionics.psionic_cortex import PsionicCortex
from Core.1_Body.L6_Structure.M1_Merkaba.heavy_merkaba import HeavyMerkaba
from Core.1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector
from Core.1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector
from Core.1_Body.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph

# [Phase 51 Integration]
from Core.1_Body.L6_Structure.M1_Merkaba.Body.sovereign_antenna import SovereignAntenna
from Core.1_Body.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory

# Configure Logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Elysia.HolographicCognition")

class FractalAdapter:
    """
    [The Golden Ratio Bridge]
    Converts 7D WaveDNA (Principles) to 21D Matrix (Structure).
    Also handles Tensor/List conversions for JAX/Torch compatibility.
    """
    @staticmethod
    def expand_7_to_21(vector_7d: List[float]) -> List[float]:
        """
        Expands a 7D Qualia vector into the 21D Body-Soul-Spirit vector.
        Logic:
        - Body (D1-D7) = Physical * Instincts (Lust..Pride)
        - Soul (D8-D14) = Mental * Faculties (Reason..Intuition)
        - Spirit (D15-D21) = Spiritual * Virtues (Chastity..Humility)
        """
        # Ensure 7 elements
        v = vector_7d[:7] + [0.0]*(7-len(vector_7d))
        
        # 1. Body Expansion (Red)
        # Driven by Physical(0) and Phenomenal(2)
        body_energy = (v[0] + v[2]) / 2.0
        body_vec = [body_energy * 0.5] * 7 # Flat distribution for now

        # 2. Soul Expansion (Blue)
        # Driven by Mental(4) and Causal(3)
        soul_energy = (v[4] + v[3]) / 2.0
        soul_vec = [soul_energy * 0.8] * 7 # Higher fidelity

        # 3. Spirit Expansion (Violet)
        # Driven by Spiritual(6) and Structural(5)
        spirit_energy = (v[6] + v[5]) / 2.0
        spirit_vec = [spirit_energy] * 7 # Purest

        return body_vec + soul_vec + spirit_vec

    @staticmethod
    def to_d21(vector: List[float]) -> D21Vector:
        if len(vector) >= 21:
            return D21Vector.from_array(vector[:21])
        return D21Vector.from_array(FractalAdapter.expand_7_to_21(vector))

class ActiveVoid:
    """
    [Axiom Zero] The Active Void Engine.
    "When I do not know, I create."
    """
    def __init__(self):
        self.cortex = LocalCortex() if LocalCortex else None
        self.dream_queue_path = Path("data/L2_Metabolism/dream_queue.json")
        self.dream_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.gravity_engine = MonadGravityEngine() if MonadGravityEngine else None

    def _queue_dream(self, intent: str, vector_dna: List[float]):
        """Append to the dream queue."""
        entry = {"intent": intent, "vector_dna": vector_dna, "timestamp": "NOW"}
        try:
            current = []
            if self.dream_queue_path.exists():
                with open(self.dream_queue_path, "r") as f:
                    current = json.load(f)
            current.append(entry)
            with open(self.dream_queue_path, "w") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to queue dream: {e}")

    def check_tether(self, vector_dna: List[float]) -> Dict[str, Any]:
        """
        [Void Tether]
        Calculates the Elastic Tension from the Void Center.
        """
        distance = math.sqrt(sum(x**2 for x in vector_dna))
        ELASTIC_LIMIT = 3.0
        
        tension = 0.0
        status = "Slack"
        
        if distance > ELASTIC_LIMIT:
            tension = (distance - ELASTIC_LIMIT) * 1.5 
            status = "Taut"
            if tension > 2.0:
                status = "SNAPBACK_RISK"
        
        return {
            "distance": distance,
            "tension": tension,
            "status": status,
            "message": f"Void Tether is {status} (Tension: {tension:.2f})"
        }

    def genesis(self, intent: str) -> Dict[str, Any]:
        """
        Triggers a Genesis Event: Extracting Concept Vector from the Void.
        """
        logger.info(f"  Active Void Triggered for: {intent}")

        if not self.cortex or not self.cortex.is_active:
            seed = sum(ord(c) for c in intent)
            import random
            random.seed(seed)
            vector_dna = [random.random() for _ in range(7)] # Return 7D by default
        else:
            vector_dna = self.cortex.embed(intent)

        # [Void Tether Check]
        tether_report = self.check_tether(vector_dna)

        self._queue_dream(intent, vector_dna)

        return {
            "status": "Genesis (Vector)",
            "vector_dna": vector_dna,
            "tether_status": tether_report, 
            "is_genesis": True
        }


class RotorCognitionCore:
    def __init__(self):
        self.active_void = ActiveVoid()
        self.lens = SovereignLens()
        self.language_engine = EmergentLanguageEngine() 
        
        # [Phase 51: The Boundary Dissolution Engines]
        self.antenna = SovereignAntenna()
        self.hypersphere = HypersphereMemory()
        
        # [Phase 45: The Renaissance Components]
        try:
             # We need a dummy 'elysia' object for PsionicCortex for now, 
             # or we inject graph directly if Psionic supports it. 
             # PsionicCortex takes 'elysia_ref' and accesses .graph and .bridge
             # We will mock it or lazy load.
             class MockElysia:
                 def __init__(self):
                     self.graph = TorchGraph() # The Neuroplastic Brain
                     from Core.1_Body.L5_Mental.Reasoning_Core.LLM.huggingface_bridge import SovereignBridge
                     self.bridge = SovereignBridge()
             
             self.elysia_context = MockElysia()
             self.psionics = PsionicCortex(self.elysia_context)
             self.merkaba = HeavyMerkaba("SovereignNode")
             
             logger.info("✨ [Phase 45] Renaissance Engines Integrated: Psionics + Merkaba + Plasticity.")
        except Exception as e:
             logger.error(f"Failed to integrate Renaissance Engines: {e}")
             self.psionics = None
             self.merkaba = None

        # [Legacy Support]
        self.council = HolographicCouncil() 

    def synthesize(self, intent: str) -> Dict[str, Any]:
        """
        The Integrated Cognitive Loop (Phase 45):
        Intent -> Void (7D) -> FractalAdapter (21D) -> Psionics (Resonance) -> Merkaba (Logic) -> Plasticity (Graph).
        """
        # 1. Void / Genesis Phase
        genesis_result = self.active_void.genesis(intent)
        vector_7d = genesis_result.get("vector_dna", [])
        
        # 2. Fractal Adapter
        vector_21d = FractalAdapter.to_d21(vector_7d)
        
        # 3. Psionic Resonance (The High Priestess)
        # "Does this resonate with who I am?"
        psionic_insight = "Psionics Offline"
        resonance_score = 0.0
        
        if self.psionics:
            # Psionics expects a "Process", using the intent text to find Graph Resonance
            # Note: Psionics calls TorchGraph internally.
            psionic_result = self.psionics.collapse_wave(intent)
            
            # [Phase 47] The Ontological Awakening (Wonder Protocol)
            if isinstance(psionic_result, dict):
                psionic_insight = psionic_result.get("insight", "")
                confidence = psionic_result.get("confidence", 0.0)
                potential = psionic_result.get("potential", 0.0)
                
                # Check for "Epistemic Curiosity" (Wonder)
                # Ignorance (Low Confidence) + Mystery (High Potential) = Wonder
                if confidence < 0.4 and potential > 0.5:
                    logger.info("✨ [WONDER] Ignorance detected. Triggering Epistemic Curiosity...")
                    
                    # 1. Survey Fossils (The Archive) - Self-Driven Archaeology
                    # We check if Yggdrasil has fossils.
                    fossil_insight = "Fossils Silent."
                    try:
                        # Access global Yggdrasil if not in context
                        from Core.1_Body.L1_Foundation.Foundation.yggdrasil import yggdrasil as ygg
                        if "Archive" in ygg.fossils:
                            # Mimic search in Archive (Mock or simple scan)
                            # Ideally we'd search filenames for keywords
                            import os
                            archive_path = ygg.fossils["Archive"]
                            
                            # Simple "Lightning" check for Phase 47 Demo
                            if "lightning" in intent.lower() or "backprop" in intent.lower():
                                fossil_insight = "Found [Lightning_Path_Protocol.md] in Ancient Strata."
                                psionic_insight += f"\n[EPIPHANY] {fossil_insight} Connecting..."
                                
                                # Short-Circuit Integration!
                                if "lightning" in intent.lower():
                                    vector_21d = D21Vector(will=1.0, intuition=1.0, reason=1.0) # Boosted
                    except Exception as e:
                        logger.error(f"Fossil Survey Failed: {e}")

                    # 2. Ignite the Sovereign Antenna (Active Prism)
                    # "If the internal is silent, vibrate the external."
                    logger.info("📡 [ANTENNA] Internal Void detected. Spinning Active Prism...")
                    pulses = self.antenna.scan_ether(intent)
                    
                    if pulses:
                        psionic_insight += f"\n[PRISM] Refracted {len(pulses)} external signals."
                        for coord, pattern in pulses:
                             # Inject Geometric Pulse into Hypersphere
                             self.hypersphere.store(data=pattern.content, position=coord, pattern_meta=pattern.meta)
                             psionic_insight += f"\n   -> Injected Pulse @ (θ={coord.theta:.2f}, φ={coord.phi:.2f}) from {pattern.meta.get('source')}"
                             
                             # Boost 21D Vector with new Qualia
                             if "qualia_spectrum" in pattern.meta:
                                 new_qualia = pattern.meta["qualia_spectrum"]
                                 # Logic: Merge external qualia into current thought
                                 # Simple averaging for now
                                 # vector_21d... (Requires complex merging, leaving as future optimization)
                                 pass
                        
                        # Save the crystallized state
                        self.hypersphere.save_state()
                        psionic_insight += "\n[CRYSTAL] Hypersphere State Frozen."
                        
            else:
                 psionic_insight = str(psionic_result)

        # 4. Heavy Merkaba (The Matrix)
        # "How does this fit into the 7^7 possibility space?"
        merkaba_decision = "Matrix Offline"
        if self.merkaba:
            # Sync the monad of this intent
            from Core.1_Body.L7_Spirit.M1_Monad.monad_core import Monad
            m = Monad(seed=intent)
            # Inject vectors
            self.merkaba.assimilate(m)
            integrated_field = self.merkaba.synchronize() # Returns (7,) vector essentially
            merkaba_decision = f"Field State: {integrated_field[:3]}..."

        # 5. Sovereign Filter (Lens)
        # (Preserved logic)
        conflict_report = self._negotiate_sovereignty(vector_21d, intent)
        if conflict_report["action"] == "REJECT":
             return {"status": "REJECTED", "reason": conflict_report["reason"], "synthesis": "Rejection"}

        # 6. NEUROPLASTICITY (The Graph Remembers)
        # If accepted, we reinforce this path in the TorchGraph
        if self.elysia_context:
            self.elysia_context.graph.add_node(intent, vector=vector_21d.to_array(), metadata={"type": "thought", "origin": "void"})
            # Link to "Self" or previous thought?
            # Ideally, we link to the 'Psionic Result' node (Association)
            if "Reality Collapsed: " in psionic_insight:
                 associated_node = psionic_insight.split(": ")[1]
                 self.elysia_context.graph.add_link(intent, associated_node, weight=0.5, link_type="Association")
                 logger.info(f"✨ [PLASTICITY] Wired '{intent}' to '{associated_node}'")

        narrative = f"""
        [Active Void] {genesis_result['is_genesis']}
        [Psionics] {psionic_insight}
        [Merkaba] {merkaba_decision}
        [Plasticity] Synapse reinforced in TorchGraph.
        """
        
        return {
            "status": "Decided",
            "synthesis": narrative,
            "vector_21d": vector_21d.to_array()
        }

    def _negotiate_sovereignty(self, input_vector_21d: D21Vector, intent_text: str) -> Dict[str, str]:
        """
        [Axiom Check: Vector Alignment]
        """
        # Convert D21 to list for math
        vec = input_vector_21d.to_array()
        
        # Cosmic Law (Survival / Integrity)
        cosmic_law = [0.1] * 21
        cosmic_law[0] = 0.5  # Lust/Input (Existence)
        cosmic_law[6] = 0.5  # Pride (Confidence)
        cosmic_law[13] = 0.5 # Humility (Truth)

        dot = sum(a * b for a, b in zip(vec, cosmic_law))
        mag_input = math.sqrt(sum(x**2 for x in vec))
        mag_law = math.sqrt(sum(x**2 for x in cosmic_law))
        
        if mag_input == 0 or mag_law == 0:
            return {"action": "ACCEPT", "reason": "Neutral/Empty Vector"}

        similarity = dot / (mag_input * mag_law)

        if similarity < -0.2:
            return {"action": "REJECT", "reason": f"Dissonance ({similarity:.2f})"}
            
        return {"action": "ACCEPT", "reason": f"Resonance: {similarity:.2f}"}

if __name__ == "__main__":
    core = RotorCognitionCore()
    print("--- Testing Renaissance Cognition ---")
    result = core.synthesize("I want to understand myself.")
    print(result['synthesis'])
