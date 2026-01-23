"""
Thought Wave Integration
========================

"       ,        "

                            Facade   :
1. QuaternionWaveDNA:   /    (   ,    )
2. PhoneticResonance:   /    (  ,   )

[NEW 2025-12-16] Unified Interface for ElysiaCore
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from Core.L1_Foundation.Foundation.quaternion_wave_dna import QuaternionCompressor, QuaternionWaveDNA, get_quaternion_compressor
from Core.L1_Foundation.Foundation.phonetic_resonance import PhoneticResonanceEngine, ResonanceField, get_resonance_engine
from Core.L1_Foundation.Foundation.fractal_knowledge import FractalKnowledgeSeed, get_fractal_seed

logger = logging.getLogger("ThoughtWave")

@dataclass
class ThoughtWave:
    """        """
    topic: str
    raw_content: str
    
    # Layer 1: Storage (Cold)
    dna: Optional[QuaternionWaveDNA] = None
    
    # Layer 2: Cognition (Hot)
    resonance: Optional[ResonanceField] = None
    
    # Layer 3: Relation (Holographic)
    # Graph connection status (conceptual)
    digested: bool = False
    
    # Metadata
    compressed_size: int = 0
    feeling_roughness: float = 0.0
    feeling_tension: float = 0.0
    
    def summary(self) -> str:
        return (f"Wave('{self.topic}'): {len(self.raw_content)} chars -> "
                f"{self.compressed_size} bytes (DNA), "
                f"Feel(R={self.feeling_roughness:.2f}, T={self.feeling_tension:.2f}), "
                f"Digested={self.digested}")

class ThoughtWaveInterface:
    """
    ElysiaCore               
    """
    
    def __init__(self):
        self.compressor = get_quaternion_compressor()
        self.resonance = get_resonance_engine()
        self.fractal_seed = get_fractal_seed()
        logger.info("  ThoughtWaveInterface connected (Hybrid Architecture + Fractal Knowledge)")
        
    def process_thought(self, topic: str, content: str, depth: str = "deep") -> ThoughtWave:
        """
                   (  ),   (  ),       (  )       .
        depth="shallow": Only Compression (DNA). Fast.
        depth="deep": Compression + Resonance + Digestion. Slow.
        """
        # 1. Memorize (Cold Storage: DNA Compression)
        # Always run this. DNA is the fundamental storage format.
        dna = self.compressor.compress(content)
        size = dna.byte_size()
        
        field = None
        feeling_roughness = 0.0
        feeling_tension = 0.0
        
        # 2. Feel & Digest (Hot Cognition) - Only for Deep processing
        if depth == "deep":
            # Feel
            target_text = f"{topic} {content[:50]}" 
            field = self.resonance.text_to_field(target_text)
            feeling_roughness = field.average_roughness
            feeling_tension = field.average_tension
            
            # Digest
            self.fractal_seed.digest(content)
        
        wave = ThoughtWave(
            topic=topic,
            raw_content=content,
            dna=dna,
            resonance=field,
            digested=(depth == "deep"),
            compressed_size=size,
            feeling_roughness=feeling_roughness,
            feeling_tension=feeling_tension
        )
        
        logger.info(f"  Processed ({depth}): {wave.summary()}")
        return wave

    
    def recall_thought(self, wave: ThoughtWave) -> str:
        """              """
        if wave.dna:
            return self.compressor.decompress(wave.dna)
        return wave.raw_content

# Singleton
_interface = None
def get_thought_interface() -> ThoughtWaveInterface:
    global _interface
    if _interface is None:
        _interface = ThoughtWaveInterface()
    return _interface