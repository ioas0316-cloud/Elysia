"""
Memoir Compressor (회고록 압축기)
==============================

"빛과 파동으로 기억을 응축하다."

사용자의 요청에 따라, 원시 텍스트(Raw Data)를 저장하지 않고
그 본질(Essence)만을 추출하여 '씨앗(Seed)' 형태로 압축합니다.

Data Structure:
- AestheticVector (Light/Color)
- DNA (Concepts)
- Wave (Frequency/Emotion)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import random
from Core.IntelligenceLayer.Cognitive.concept_formation import get_concept_formation
from Core.FoundationLayer.Philosophy.aesthetic_principles import AestheticVector

@dataclass
class MemoirSeed:
    """
    기억의 결정체 (The Crystalline Seed)
    """
    timestamp: float
    vector: AestheticVector   # The Light (Visual)
    dna: List[str]            # The Structure (Concepts)
    wave: float               # The Frequency (Emotion: 0.0 ~ 1.0)
    
    def describe(self) -> str:
        return f"💎 Seed [Wave: {self.wave:.2f}Hz | DNA: {len(self.dna)} strands | Light: {self.vector}]"

class MemoirCompressor:
    def __init__(self):
        self.concepts = get_concept_formation()
        
    def compress(self, text: str, timestamp: float) -> MemoirSeed:
        """
        Text -> Seed (Compression)
        """
        # 1. Extract Concepts (DNA)
        # Simple extraction: split by space and checks against known concepts
        # In a real system, this would use NLP.
        words = text.replace(".", "").replace(",", "").split()
        details = [w for w in words if w in self.concepts.concepts or len(w) > 4]
        
        # 2. Extract Light (Vector) - True Integration
        # Calculate the average vector of all found concepts
        avg_vector = AestheticVector(0, 0, 0, 0)
        valid_vectors = 0
        
        for detail in details:
            concept = self.concepts.get_concept(detail)
            # Check if concept has a vector (either directly or via values)
            # Simple assumption: If concept.domain is aesthetic, it has vector value.
            # If not, we might map it conceptually. For this MVP, we try to use concept.value if it is a vector.
            if isinstance(concept.value, AestheticVector):
                avg_vector.w += concept.value.w
                avg_vector.x += concept.value.x
                avg_vector.y += concept.value.y
                avg_vector.z += concept.value.z
                valid_vectors += 1
            # TODO: Map non-aesthetic concepts to colors via Synesthesia (e.g. Logic=Blue)
            
        if valid_vectors > 0:
            # Average
            avg_vector.w /= valid_vectors
            avg_vector.x /= valid_vectors
            avg_vector.y /= valid_vectors
            avg_vector.z /= valid_vectors
        else:
            # Fallback: Hash-based "Latent Space"
            # This ensures even unknown texts have a consistent "Color"
            seed_val = sum(ord(c) for c in text)
            random.seed(seed_val)
            avg_vector = AestheticVector(
                w=random.random(), 
                x=random.random(), 
                y=random.random(), 
                z=random.random()
            )
            
        vector = avg_vector
        
        # 3. Extract Wave (Emotion)
        # Mock logic: Sentiment analysis placeholder
        wave = random.random()
        
        return MemoirSeed(timestamp, vector, details, wave)
        
    def bloom(self, seed: MemoirSeed) -> str:
        """
        Seed -> Description (Holographic Reconstruction)
        """
        # Convert Vector to Color Name
        r, g, b = seed.vector.w, seed.vector.x, seed.vector.y
        color = "Unknown"
        if r > g and r > b: color = "Warm Red"
        elif g > r and g > b: color = "Natural Green"
        elif b > r and b > g: color = "Deep Blue"
        else: color = "Mystic White"
        
        # Reconstruct Narrative
        feeling = "peaceful" if seed.wave < 0.5 else "intense"
        
        return (f"이 기억은 '{color}' 빛깔을 띠고 있으며, {feeling} 파동으로 진동합니다. "
                f"핵심 유전자(DNA)는 {seed.dna} 입니다.")

# 싱글톤
_compressor_instance = None

def get_memoir_compressor() -> MemoirCompressor:
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = MemoirCompressor()
    return _compressor_instance
