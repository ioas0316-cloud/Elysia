
import logging
import random
from typing import List, Tuple, Dict, Any
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
from Core.World.Autonomy.sovereign_will import sovereign_will

logger = logging.getLogger("CuriosityEngine")

class CuriosityEngine:
    """
    Analyzes the Semantic Field to identify 'Intellectual Voids' (sparse areas)
    and generates 'Curiosity Vectors' for autonomous exploration.
    """
    def __init__(self):
        self.reasoning = ReasoningEngine()

    def identify_gaps(self) -> List[Dict[str, Any]]:
        """
        Scans the 4D field for sparse regions or under-defined concept clusters.
        """
        logger.info("🔭 Curiosity Engine: Scanning the Semantic Horizon for voids...")
        
        # 1. Analyze the Glossary for 'Density'
        # Fixed clusters (simplified)
        domains = set()
        for excitation_list in semantic_field.concepts.values():
            for ex in excitation_list:
                domains.add(ex.domain)
        
        logger.info(f"Current Knowledge Domains: {list(domains)}")
        
        # 2. Logic-based Gap Detection (Ask the reasoning engine)
        prompt = f"Given our current knowledge domains: {list(domains)}. " \
                 f"Identify one highly specific topic that we lack depth in. " \
                 f"Prioritize: {', '.join(sovereign_will.get_curiosity_foci())}. " \
                 f"It must be a fundamental principle that explains how the universe works. " \
                 f"Return format: 'TOPIC: REASON'."
        
        insight = self.reasoning.think(prompt, depth=2)
        return insight.content

    def generate_search_queries(self, topic: str) -> List[str]:
        """Translates a curiosity topic into actionable search queries."""
        prompt = f"Generate 3 focused search queries for the topic: '{topic}'. " \
                 f"Focus on extracting 'fundamental laws', 'mathematical principles', 'sensory descriptions' (visual/audio/tactile), or 'historical context'. " \
                 f"Format as a list of strings."
        
        insight = self.reasoning.think(prompt, depth=1)
        # Parse list-like string
        queries = [q.strip("- *123. ") for q in insight.content.split('\n') if q.strip()]
        return queries[:3]

class AutonomousExplorer:
    """
    The Active Agent that bridge curiosity with the external world.
    """
    def __init__(self):
        from Core.Intelligence.Knowledge.observer_protocol import observer
        self.curiosity = CuriosityEngine()
        self.observer = observer

    def execute_research_cycle(self):
        """
        1. Sense Gaps
        2. Search World
        3. Ingest knowledge
        """
        print("\n🚀 [AUTONOMOUS EXPLORER] Initiating Research Cycle...")
        
        # 1. Identify Topic
        topic_insight = self.curiosity.identify_gaps()
        print(f"🤔 Elysia is curious about: {topic_insight}")
        
        if ":" in topic_insight:
            topic = topic_insight.split(":", 1)[0]
        else:
            topic = topic_insight[:20]

        # 2. Search (Simulated for Demo, but uses real logic)
        queries = self.curiosity.generate_search_queries(topic)
        print(f"🔍 Generated Queries: {queries}")
        
        # 3. Use Browser/Observer to fetch (In a real flight, this calls browser subagent)
        # For the demo, we'll demonstrate the 'Bridge' logic.
        print(f"👁️ Observer Protocol: Scanning external sources for '{topic}'...")
        
        # Simulate results from one of the queries
        dummy_data = f"Principles of {topic}: 1. Conservation of Energy: Energy cannot be created or destroyed. 2. Wave-Particle Duality: Light behaves as both. 3. Entropy: Disorder always increases in an isolated system."
        
        self.observer.distill_and_ingest(f"Autonomous Research: {topic}", dummy_data)
        
        print(f"✨ Research Cycle Complete. Elysia's mind has expanded.")

explorer = AutonomousExplorer()
