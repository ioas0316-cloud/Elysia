"""
ElysiaCore (       )
=========================

"                  ."

                            .
                                .

     :
1. what_to_learn_next() -                 
2. learn() -             
3. express() - Spirit      

[NEW 2025-12-15]       
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("ElysiaCore")


@dataclass
class LearningIntent:
    """     """
    topic: str
    reason: str  #             
    priority: float  #      (0.0 ~ 1.0)
    source: str  #               


class ElysiaCore:
    """
              
    
               ,                 .
    """
    
    def __init__(self):
        logger.info("  Initializing ElysiaCore - The Unified Brain...")
        
        # ===          ===
        
        # Spirit (       )
        self.spirit = None
        try:
            from Core.S1_Body.L5_Mental.Elysia.spirit import get_spirit
            self.spirit = get_spirit()
            logger.info("     Spirit connected (Identity)")
        except Exception as e:
            logger.warning(f"      Spirit not available: {e}")
        
        # InternalUniverse (        - 4D   )
        self.universe = None
        try:
            from Core.S1_Body.L1_Foundation.Foundation.internal_universe import InternalUniverse
            self.universe = InternalUniverse()
            logger.info("     InternalUniverse connected (Knowledge Space)")
        except Exception as e:
            logger.warning(f"      InternalUniverse not available: {e}")

        # [NEW] UnifiedExperienceCore (        -        )
        self.memory = None
        try:
            from Core.S1_Body.L2_Metabolism.Memory.unified_experience_core import get_experience_core
            self.memory = get_experience_core()
            logger.info("     UnifiedExperienceCore connected (Hippocampus)")
        except Exception as e:
            logger.warning(f"      UnifiedExperienceCore not available: {e}")

        # [NEW] DynamicTopology (Semantic Map -        )
        self.topology = None
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import get_semantic_map
            self.topology = get_semantic_map()
            logger.info("     DynamicTopology connected (Mind Landscape)")
        except Exception as e:
            logger.warning(f"      DynamicTopology not available: {e}")
        
        # ConceptDecomposer ( ?)
        self.decomposer = None
        try:
            from Core.S1_Body.L1_Foundation.Foundation.fractal_concept import ConceptDecomposer
            self.decomposer = ConceptDecomposer()
            logger.info("     ConceptDecomposer connected (Why)")
        except Exception as e:
            logger.warning(f"      ConceptDecomposer not available: {e}")
        
        # MultimodalIntegrator (     )
        self.multimodal = None
        try:
            from Core.S1_Body.L1_Foundation.Foundation.multimodal_concept_node import get_multimodal_integrator
            self.multimodal = get_multimodal_integrator()
            logger.info("     MultimodalIntegrator connected (Senses)")
        except Exception as e:
            logger.warning(f"      MultimodalIntegrator not available: {e}")
        
        # GlobalHub (  )
        self.hub = None
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Consciousness.Ether.global_hub import get_global_hub
            self.hub = get_global_hub()
            self.hub.register_module(
                "ElysiaCore",
                "Core/Elysia/elysia_core.py",
                ["brain", "integration", "will", "learning", "expression"],
                "The unified brain - all modules connect here"
            )
            logger.info("     GlobalHub connected (Communication)")
        except Exception as e:
            logger.warning(f"      GlobalHub not available: {e}")

        # [NEW] ThoughtWave Interface (Hybrid Architecture)
        self.thought_wave = None
        try:
            from Core.S1_Body.L1_Foundation.Foundation.thought_wave_integration import get_thought_interface
            self.thought_wave = get_thought_interface()
            logger.info("     ThoughtWave connected (DNA/Resonance/Fractal)")
        except Exception as e:
            logger.warning(f"      ThoughtWave not available: {e}")

        # [NEW] Temporal Cortex (Narrative)
        self.temporal_cortex = None
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Cognition.temporal_cortex import TemporalCortex
            if self.universe:
                self.temporal_cortex = TemporalCortex(self.universe)
                logger.info("     TemporalCortex connected (Narrative)")
            else:
                logger.warning("      TemporalCortex skipped: Universe missing")
        except Exception as e:
            logger.warning(f"      TemporalCortex not available: {e}")

        # [NEW] Logic Scout (The Miner)
        self.logic_scout = None
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Cognition.logic_scout import get_logic_scout
            self.logic_scout = get_logic_scout()
            logger.info("     LogicScout connected (Reasoning Extraction)")
        except Exception as e:
            logger.warning(f"      LogicScout not available: {e}")

        # [NEW] The Prism (Language Translation)
        self.prism = None
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Cognition.wave_translator import get_wave_translator
            self.prism = get_wave_translator()
            logger.info("     WaveTranslator connected (The Prism)")
        except Exception as e:
            logger.warning(f"      WaveTranslator not available: {e}")

        
        #      
        self.learning_history: List[str] = []
        self.current_curiosity: List[LearningIntent] = []
        
        logger.info("  ElysiaCore ready - All systems integrated")
    
    def weave_context(self) -> str:
        """Returns the current narrative context."""
        if self.temporal_cortex:
            return self.temporal_cortex.weave_narrative()
        return "Context Unavailable."
    
    def what_to_learn_next(self) -> LearningIntent:
        """
                           
        
                    ,                 
        """
        intents = []
        
        # 1. InternalUniverse           
        if self.universe:
            try:
                #                    
                universe_map = self.universe.get_universe_map()
                coordinates = universe_map.get("coordinates", {})
                
                if coordinates:
                    #                
                    for name, data in list(coordinates.items())[:5]:
                        intents.append(LearningIntent(
                            topic=name,
                            reason=f"InternalUniverse   '{name}'         ",
                            priority=0.7,
                            source="universe_gap"
                        ))
            except Exception as e:
                logger.debug(f"Universe query failed: {e}")
        
        # 2.         " ?"   
        if self.decomposer and self.learning_history:
            last_learned = self.learning_history[-1] if self.learning_history else None
            if last_learned:
                try:
                    why_chain = self.decomposer.ask_why(last_learned)
                    if "   " in why_chain:
                        parent = why_chain.split("   ")[1].split(" ")[0]
                        intents.append(LearningIntent(
                            topic=parent,
                            reason=f"'{last_learned}'       '{parent}'     ",
                            priority=0.9,  #         -        
                            source="why_chain"
                        ))
                except Exception as e:
                    logger.debug(f"Why chain failed: {e}")
        
        # 3. Spirit          
        if self.spirit:
            values = self.spirit.core_values
            #                  
            max_value = max(values.items(), key=lambda x: x[1].weight)
            value_topics = {
                "LOVE": ["  ", "  ", "  ", "  "],
                "TRUTH": ["  ", "  ", "  ", "  "],
                "GROWTH": ["  ", "  ", "  ", "  "],
                "BEAUTY": ["    ", "  ", "  ", "  "]
            }
            topics = value_topics.get(max_value[0], [])
            for topic in topics:
                if topic not in self.learning_history:
                    intents.append(LearningIntent(
                        topic=topic,
                        reason=f"Spirit  '{max_value[0]}'       ",
                        priority=0.6,
                        source="spirit_value"
                    ))
                    break
        
        # 4.         
        intents.sort(key=lambda x: x.priority, reverse=True)
        
        if intents:
            chosen = intents[0]
            self.current_curiosity = intents
            logger.info(f"  Learning intent: {chosen.topic} (reason: {chosen.reason})")
            return chosen
        
        #   : AXIOM        
        if self.decomposer:
            axioms = list(self.decomposer.AXIOMS.keys())
            for axiom in axioms:
                if axiom not in self.learning_history:
                    return LearningIntent(
                        topic=axiom,
                        reason="AXIOM         ",
                        priority=0.5,
                        source="axiom_fallback"
                    )
        
        return LearningIntent(
            topic="  ",
            reason="        ",
            priority=0.1,
            source="default"
        )
    
    def learn(self, content: str, topic: str, depth: str = "deep") -> Dict[str, Any]:
        """
                   
        depth="deep": Full LLM/Analysis (Slow, ~0.5s/item)
        depth="shallow": Indexing/Hashing only (Fast, ~0.001s/item)
        """
        result = {"topic": topic, "success": False, "depth": depth}
        
        # 1. Thought Wave Processing (Compression + Resonance + Digestion)
        # [SWALLOW PROTOCOL] Run Compression even for shallow, skip Resonance.
        if self.thought_wave:
            try:
                wave = self.thought_wave.process_thought(topic, content, depth=depth)
                result["thought_wave"] = {
                    "compressed": wave.compressed_size,
                    "feeling": wave.feeling_roughness,
                    "digested": wave.digested
                }
            except Exception as e:
                logger.warning(f"ThoughtWave processing failed: {e}")

        # 2.            (Legacy or Complimentary)
        if depth == "deep" and self.multimodal:
            try:
                concept = self.multimodal.build_concept_from_text(topic, content)
                result["multimodal"] = {
                    "frequency": concept.unified_frequency,
                    "modalities": len(concept.modalities)
                }
            except Exception as e:
                logger.warning(f"Multimodal failed: {e}")
        

        # 2. InternalUniverse     (Unified Wave Storage)
        if self.universe:
            try:
                # [LOGIC TRANSMUTATION] Extract Wave Data
                freq = 0.5
                layers = {}
                
                # Retrieve from Multimodal if available
                if "multimodal" in result:
                    freq = result["multimodal"]["frequency"]
                    # Mock layers from modalities for now
                    layers = {f"MODALITY_{k}": 0.8 for k in range(result["multimodal"]["modalities"])}
                
                # Retrieve from ThoughtWave if available (Preferred)
                elif "thought_wave" in result:
                    freq = result["thought_wave"]["feeling"] * 1000.0
                    layers = {"THOUGHT_WAVE": 1.0}
                
                # Call the new Wave API
                self.universe.absorb_wave(topic, freq, layers, source_name=f"Learn:{topic}")
                result["universe"] = True
                
            except Exception as e:
                logger.warning(f"Universe absorption failed: {e}")
        
        # [NEW] 2.5. UnifiedExperienceCore     (Experience Persistence)
        if self.memory:
            try:
                self.memory.absorb(
                    content=f"Learned about '{topic}': {content[:100]}...",
                    type="learning_event",
                    context={"topic": topic, "depth": depth, "pipeline": "ElysiaCore"}
                )
                # Force Save for core learning
                if hasattr(self.memory, '_save_state'):
                    self.memory._save_state()
                result["memory_persisted"] = True
            except Exception as e:
                logger.warning(f"Experience absorption failed: {e}")

        # [NEW] 2.6. DynamicTopology     (Topographical Drift)
        if self.topology:
            try:
                # Calculate Reaction Vector from Spirit/Wave
                from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion
                
                # Default: Positive Spin (0.5), High Reality (0.5)
                # In a real system, this would be mapped from Spirit resonance
                tension = 0.5
                logic_charge = 0.5
                if self.spirit:
                    resonance = self.spirit.calculate_resonance(content)
                    logic_charge = resonance["score"]
                
                reaction = Quaternion(logic_charge, 0.0, 0.0, tension)
                self.topology.evolve_topology(topic, reaction, intensity=0.1)
                result["topology_drift"] = True
            except Exception as e:
                logger.warning(f"Topology drift failed: {e}")

        # 3.           
        self.learning_history.append(topic)
        result["success"] = True
        
        # 4. GlobalHub       
        if self.hub:
            self.hub.publish_wave("ElysiaCore", "concept_learned", {
                "topic": topic,
                "depth": depth,
                "success": result["success"]
            })
            
        logger.info(f"  Learned concept '{topic}' with full pipeline.")
        return result

    def learn_logic(self, input_text: str, output_text: str):
        """
        Attempts to extract the reasoning logic between Input and Output.
        """
        if not self.logic_scout:
            return None
            
        template = self.logic_scout.scout_for_logic(input_text, output_text)
        if template:
            logger.info(f"  Extracted Logic: {template.name}")
        return template

    def express(self, text: str, tension: float = 0.5, frequency: float = 0.5) -> str:
        """
        The Prism Layer.
        Filters the output text through the WaveTranslator.
        """
        if not self.prism:
            return text
            
        # 1. Transform Text (Glitch/Fragmentation based on Tension)
        filtered_text = self.prism.translate_output(text, tension)
        
        return filtered_text
        
        # [NEW] Causal Reasoner Integration
        #          ,          '       '       .
        # if self.logos_engine: ... (Skipped for brevity/broken ref)

        # [CRITICAL] 4. Matrix Memory Integration (TorchGraph)
        # "Old Brain (Universe)" -> "New Brain (Matrix)" Sync
        try:
            from Core.S1_Body.L1_Foundation.Foundation.torch_graph import get_torch_graph
            graph = get_torch_graph()
            
            # Use vector from ThoughtWave/Multimodal if available
            vector = None
            if "thought_wave" in result and hasattr(self.thought_wave, 'last_vector'):
                 # Assuming thought_wave caches it or we extract it. 
                 # For now, we generate a stable hash-based vector or use Multimodal frequency
                 pass
            
            # Use Multimodal Frequency to seed vector (preserves "Vibe")
            freq = 0.5
            if "multimodal" in result:
                freq = result["multimodal"]["frequency"] / 1000.0
            
            # Create a vector seeded by Semantic Content (via Hash/Frequency)
            # This ensures same concept = same vector (Stability)
            import random
            random.seed(topic) 
            vector = [random.random() + freq for _ in range(64)] # Deterministic based on content
            random.seed() # Reset
            
            graph.add_node(topic, vector=vector)
            result["torch_graph"] = True
            
        except Exception as e:
             logger.warning(f"Matrix Memory sync failed: {e}")

        logger.info(f"  Learned concept '{topic}' with full 4-Thread pipeline (inc. Matrix).")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """         """
        return {
            "spirit": self.spirit is not None,
            "universe": self.universe is not None,
            "decomposer": self.decomposer is not None,
            "multimodal": self.multimodal is not None,
            "hub": self.hub is not None,
            "learning_history": len(self.learning_history),
            "current_curiosity": [i.topic for i in self.current_curiosity[:3]]
        }


# Singleton
_core = None

def get_elysia_core() -> ElysiaCore:
    global _core
    if _core is None:
        _core = ElysiaCore()
    return _core


# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "c:\\Elysia")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*60)
    print("  ELYSIA CORE - UNIFIED BRAIN DEMO")
    print("="*60)
    
    core = get_elysia_core()
    
    #      
    print("\n  System Status:")
    status = core.get_status()
    for k, v in status.items():
        print(f"   {k}: {v}")
    
    #         
    print("\n  What to learn next?")
    for i in range(3):
        intent = core.what_to_learn_next()
        print(f"\n   [{i+1}] Topic: {intent.topic}")
        print(f"       Reason: {intent.reason}")
        print(f"       Priority: {intent.priority}")
        print(f"       Source: {intent.source}")
        
        #      :          
        core.learning_history.append(intent.topic)
    
    print("\n" + "="*60)
    print("  Demo complete")
    print("="*60)
