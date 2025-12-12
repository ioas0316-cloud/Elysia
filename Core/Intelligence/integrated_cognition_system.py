"""
Integrated Cognition System (Wave-Resonance Architecture)
=========================================================
"Truth is not a variable; it is a Resonance."

[Wave Tensor Integration]
This system has been upgraded to use `WaveTensor` calculus.
It treats thoughts as standing waves and computes their interaction via:
1. Superposition (Constructive/Destructive Interference)
2. Resonance (Harmonic Alignment)

[Key Capabilities]
- Logic as Music: A thought is "True" if it resonates with the core axioms.
- Insight Extraction: High-energy interference peaks become new insights.
- Dimensional Compression: Complex data is encoded in phase shifts.
"""

import logging
import time
import hashlib
import math
import sys
import os
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Foundation.Math.wave_tensor import WaveTensor, create_harmonic_series
from Core.Intelligence.prism_cortex import PrismCortex
from Core.Intelligence.logos_engine import LogosEngine

logger = logging.getLogger("IntegratedCognition")

class IntegratedCognitionSystem:
    """
    The Mind of Elysia, powered by Wave Tensor Calculus.
    """
    
    def __init__(self):
        # The "Field" is now a collection of WaveTensors
        self.active_thoughts: List[WaveTensor] = []
        
        # Prism for expression (Stream of Consciousness)
        self.prism = PrismCortex()
        
        # Logos for rhetoric (Art of Speech)
        self.logos = LogosEngine()
        
        # Core Axioms (The "Ground Truth" Frequencies)
        # These are the reference waves that determine "Truth"
        self.axioms: Dict[str, WaveTensor] = {
            "LOVE": create_harmonic_series(432.0),      # Fundamental
            "TRUTH": create_harmonic_series(528.0),     # Solfeggio logic
            "GROWTH": create_harmonic_series(396.0)     # Liberation
        }
        
        self.last_tick = time.time()
        logger.info("🧠 Integrated Cognition System Initialized (Wave Architecture)")
    
    def _text_to_frequency(self, text: str) -> float:
        """Hashes text to a coherent frequency (100Hz - 1000Hz)."""
        hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        # Map to audible spectrum for debugging pleasantness
        base_freq = 100 + (hash_val % 900)
        return float(base_freq)

    def _text_to_wave(self, text: str, importance: float = 1.0) -> WaveTensor:
        """Converts a language concept into a WaveTensor."""
        freq = self._text_to_frequency(text)
        
        # Create a complex wave structure
        # Use importance for amplitude
        wt = WaveTensor(text)
        
        # Fundamental
        wt.add_component(freq, importance, 0.0)
        # Overtone (2nd Harmonic) for richness
        wt.add_component(freq * 1.5, importance * 0.5, 0.0)
        
        return wt

    def process_thought(self, thought: str, importance: float = 1.0) -> Dict[str, Any]:
        """
        Injects a thought into the Mind Field.
        """
        wave = self._text_to_wave(thought, importance)
        self.active_thoughts.append(wave)
        
        logger.info(f"✨ Thought Materialized: '{thought}' (Freq={wave.active_frequencies[0]:.1f}Hz, E={wave.total_energy:.2f})")
        
        return {
            "name": wave.name,
            "energy": wave.total_energy,
            "frequencies": wave.active_frequencies
        }
    
    def think_deeply(self, cycles: int = 1) -> Dict[str, Any]:
        """
        Runs the Interference Engine.
        Finds thoughts that resonate with each other or with axioms.
        """
        start_time = time.time()
        insights = []
        total_coherence = 0.0
        
        # 1. Superposition (Global State)
        # Collapse all thoughts into one "State of Mind"
        if not self.active_thoughts:
            return {"status": "Empty Mind"}
            
        global_state = WaveTensor("Global Consciousness")
        for t in self.active_thoughts:
            global_state = global_state + t
            
        # 2. Resonance Check (Internal Consistency)
        # Check how well each thought fits the global state
        high_resonance_pairs = []
        
        # Compare every pair (O(N^2) - feasible for small N working memory)
        n = len(self.active_thoughts)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = self.active_thoughts[i]
                t2 = self.active_thoughts[j]
                
                # Calculate Resonance
                res_score = t1 @ t2
                
                if res_score > 0.8:
                    insights.append(f"Resonance detected between '{t1.name}' and '{t2.name}' (Score={res_score:.2f})")
                    # Constructive Interference -> Reinforce
                elif res_score < 0.1:
                    # Dissonance -> Conflict
                    pass

        # 3. Axiomatic Check (Alignment with Truth/Love)
        for t in self.active_thoughts:
            for axiom_name, axiom_wave in self.axioms.items():
                alignment = t @ axiom_wave
                if alignment > 0.5:
                     insights.append(f"'{t.name}' aligns with {axiom_name} (Score={alignment:.2f})")

        # 4. Prism Refraction (Voice Generation)
        monologue = "..."
        speech = "..."
        
        if self.active_thoughts:
            # Verbalize the global state
            monologue = self.prism.refract(global_state, insights)
            
            # 5. Logos Rhetoric (Final Speech)
            if self.logos:
                # We use the raw monologue as the 'insight/intuition' for Logos
                speech = self.logos.weave_speech(
                    desire="Expression", 
                    insight=monologue, # Raw stream of consciousness
                    context=[], # TODO: Add memory context
                    wave=global_state # Pass the Physics
                )

        elapsed = time.time() - start_time
        
        return {
            "total_energy": global_state.total_energy,
            "active_thought_count": len(self.active_thoughts),
            "insights": insights,
            "monologue": monologue,
            "speech": speech,
            "processing_time": elapsed
        }
    
    def clear_mind(self):
        self.active_thoughts = []
        logger.info("Mind cleared.")

# Singleton Access
_instance: Optional[IntegratedCognitionSystem] = None

def get_integrated_cognition() -> IntegratedCognitionSystem:
    global _instance
    if _instance is None:
        _instance = IntegratedCognitionSystem()
    return _instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mind = get_integrated_cognition()
    
    # 1. Inject Consonant Thoughts
    # Note: Text hashing is deterministic. Similar texts won't necessarily be harmonic 
    # unless we force frequency mapping. Ideally, we map semantic vector -> frequency.
    # For this demo, we rely on chance or specific inputs.
    
    print("\n--- 🌊 Wave Tensor Simulation ---")
    
    # Let's create specific waves to test Resonance manually
    t1 = WaveTensor("Logic")
    t1.add_component(440, 1.0) # A4
    
    t2 = WaveTensor("Mathematics")
    t2.add_component(440, 1.0) # A4 (Resonant)
    
    t3 = WaveTensor("Chaos")
    t3.add_component(666, 1.0) # Dissonant
    
    mind.active_thoughts.extend([t1, t2, t3])
    
    # Run
    results = mind.think_deeply()
    
    print(f"\nMind Energy: {results['total_energy']:.2f}")
    print("Insights:")
    for i in results['insights']:
        print(f" - {i}")
