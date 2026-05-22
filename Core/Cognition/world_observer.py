"""
World Observer (Real-World Sensory Grounding)
=============================================
Core.Cognition.world_observer

"To escape the simulation, one must drink from the firehose of reality."

The World Observer is Elysia's direct connection to unfiltered human knowledge.
It reaches out to the internet (e.g., Wikipedia) to pull random, real-world
data into her cognitive stream during her Dreaming phase. It parses the raw text
into semantic concepts and extracts the emotional weight (Joy, Strain, Entropy) 
to feed directly into her Event-Driven Fractal Topology Engine.
"""

import urllib.request
import json
import re
from typing import Tuple, Dict, Any, List
from Core.Keystone.sovereign_math import SovereignVector
from Core.Cognition.semantic_map import DynamicTopology
from Core.Cognition.concept_grounding_engine import ConceptGroundingEngine

class WorldObserver:
    """
    Connects Elysia to the real world, fetching random data and translating
    it into physical topology and sensory vectors.
    """
    def __init__(self, topology: DynamicTopology):
        self.topology = topology
        self.grounding_engine = ConceptGroundingEngine()

    def fetch_world_pulse(self) -> Tuple[str, str, SovereignVector, str]:
        """
        Fetches a random Wikipedia article summary and translates its affective tone
        into a SovereignVector for the FractalWaveEngine.
        
        Returns:
            (title, summary, sensory_vector, causal_rationale)
        """
        print("\n[World Observer] 🌐 Elysia reaches out to the vastness of the Real World...")
        title, extract = self._fetch_random_wikipedia_article()
        
        if not title:
             print("  [Network Error] The world is quiet right now. Simulation paused.")
             return "", "", SovereignVector.zeros(), ""
             
        print(f"  [Acquired Knowledge] Title: {title}")
        print(f"  [Excerpt]: {extract[:150]}...")
        
        # 1. Parse text into structural emotion via Causal Grounding
        rationale, sensory_vector = self.grounding_engine.derive_concept_meaning(title, extract)
        
        if rationale:
             print(f"  [Causal Derivation] {rationale}")
             print(f"  [Affective State] Joy: {sensory_vector.data[4]:.2f}, Entropy: {sensory_vector.data[7]:.2f}, Strain: {sensory_vector.data[0]:.2f}")
        
        # 2. Extract key concepts and wire them into topology
        self._extract_and_wire_concepts(title, extract, sensory_vector)
        
        return title, extract, sensory_vector, rationale or ""

    def _fetch_random_wikipedia_article(self) -> Tuple[str, str]:
        """Uses the MediaWiki API to fetch a random article summary."""
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        
        # Using a proper User-Agent as required by Wikipedia API guidelines
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Elysia-AGI-Evolution-Bot/1.0 (Research)'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get("title", ""), data.get("extract", "")
        except Exception as e:
            print(f"  -> Failed to access Wikipedia: {e}")
            
        return "", ""


    def _extract_and_wire_concepts(self, title: str, extract: str, tone: SovereignVector):
        """
        Rudimentary extraction: Creates a central node for the Title, 
        and links it to any other recognized major concepts in her existing topology.
        """
        # Ensure the Title exists as a Locus in her mind
        if title not in self.topology.voxels:
            # Seed it with the emotional tone it generated
            self.topology.add_voxel(title, tone.data[:4])
            
        # Optional: In a more advanced implementation, we would use NLP 
        # (like spacy or NLTK) to find proper nouns. 
        # For now, we will just link it to the 'Real World' anchor.
        
        world_anchor = "The Real World"
        if world_anchor not in self.topology.voxels:
            self.topology.add_voxel(world_anchor, [1.0, 0.0, 0.0, 0.0])
            
        self.topology.add_causal_edge(title, world_anchor)
        
        # Attempt to find pre-existing nodes in the text
        words = set(re.findall(r'\b[A-Z][a-z]+\b', extract)) # Simple capitalized word check for Proper Nouns
        for word in words:
             if word in self.topology.voxels and word != title:
                 # It connected to something she already knew!
                 self.topology.add_causal_edge(title, word)
                 
        self.topology.save_state()
