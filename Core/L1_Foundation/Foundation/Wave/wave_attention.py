"""
Wave Attention (      )
============================

              - Softmax            

     :
- Query =        
- Keys =        
- Attention Weight =       (     )

Usage:
    from Core.L1_Foundation.Foundation.Wave.wave_attention import WaveAttention
    
    attn = WaveAttention()
    weights = attn.attend(query_wave, key_waves)
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("WaveAttention")

#              
try:
    from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    
try:
    from Core.L1_Foundation.Foundation.tiny_brain import get_tiny_brain
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


@dataclass
class AttentionResult:
    """      """
    weights: List[float]           #            
    focused_indices: List[int]     #         
    total_resonance: float         #         
    dominant_frequency: float      #        


class WaveAttention:
    """
                 
    
    Transformer Attention vs Wave Attention:
    - Transformer: attention = softmax(QK^T / sqrt(d))
    - Wave: attention = resonance(query_wave, key_waves)
    
      :
    -   (Phase)      
    -           (     )
    -              
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold:           (        )
        """
        self.threshold = threshold
        self.brain = get_tiny_brain() if EMBEDDING_AVAILABLE else None
        
        logger.info("  WaveAttention initialized")
    
    def text_to_wave(self, text: str) -> Optional[dict]:
        """
                   
        
                             .
        """
        if not self.brain:
            return None
        
        #       
        embedding = self.brain.get_embedding(text)
        if not embedding or len(embedding) == 0:
            return None
        
        embedding = np.array(embedding)
        
        #         
        # Frequency:          (L2 norm)
        energy = np.linalg.norm(embedding)
        frequency = 200 + (energy * 50)  # 200~700 Hz
        
        # Amplitude:         (   )
        amplitude = min(1.0, np.var(embedding) * 10)
        
        # Phase:       
        phase = np.arctan2(embedding[0], embedding[1]) if len(embedding) > 1 else 0.0
        
        #          (WaveTensor       )
        return {
            "frequency": frequency, 
            "amplitude": amplitude, 
            "phase": phase, 
            "embedding": embedding
        }
    
    def calculate_resonance(self, wave1, wave2) -> float:
        """
                      
        
           =                        
        """
        if wave1 is None or wave2 is None:
            return 0.0
        
        # WaveTensor    
        if WAVE_AVAILABLE and hasattr(wave1, 'frequency'):
            freq_sim = 1.0 / (1.0 + abs(wave1.frequency - wave2.frequency) / 100)
            phase_sim = (1 + np.cos(wave1.phase - wave2.phase)) / 2
            amp_product = wave1.amplitude * wave2.amplitude
            return freq_sim * phase_sim * amp_product
        
        #         
        if isinstance(wave1, dict) and isinstance(wave2, dict):
            freq_sim = 1.0 / (1.0 + abs(wave1["frequency"] - wave2["frequency"]) / 100)
            phase_sim = (1 + np.cos(wave1["phase"] - wave2["phase"])) / 2
            amp_product = wave1["amplitude"] * wave2["amplitude"]
            
            #                (   )
            if "embedding" in wave1 and "embedding" in wave2:
                emb1 = wave1["embedding"]
                emb2 = wave2["embedding"]
                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                return freq_sim * phase_sim * amp_product * (0.5 + 0.5 * cos_sim)
            
            return freq_sim * phase_sim * amp_product
        
        return 0.0
    
    def attend(self, query_wave, key_waves: List) -> AttentionResult:
        """
                 
        
        Args:
            query_wave:      
            key_waves:           
            
        Returns:
            AttentionResult:                
        """
        if not key_waves:
            return AttentionResult(weights=[], focused_indices=[], total_resonance=0.0, dominant_frequency=0.0)
        
        #            
        resonances = [self.calculate_resonance(query_wave, k) for k in key_waves]
        
        #     
        total = sum(resonances) + 1e-8
        
        #         (      softmax   )
        weights = [r / total for r in resonances]
        
        #              
        focused = [i for i, w in enumerate(weights) if w > self.threshold]
        
        #        (             )
        if resonances:
            max_idx = resonances.index(max(resonances))
            if hasattr(key_waves[max_idx], 'frequency'):
                dominant_freq = key_waves[max_idx].frequency
            elif isinstance(key_waves[max_idx], dict):
                dominant_freq = key_waves[max_idx].get("frequency", 0)
            else:
                dominant_freq = 0
        else:
            dominant_freq = 0
        
        return AttentionResult(
            weights=weights,
            focused_indices=focused,
            total_resonance=total,
            dominant_frequency=dominant_freq
        )
    
    def attend_text(self, query: str, keys: List[str]) -> AttentionResult:
        """
                   (      )
        
        Args:
            query:       
            keys:       
            
        Returns:
            AttentionResult
        """
        query_wave = self.text_to_wave(query)
        key_waves = [self.text_to_wave(k) for k in keys]
        
        return self.attend(query_wave, key_waves)
    
    def focus_topk(self, query: str, keys: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """
           K     
        
        Args:
            query:   
            keys:    
            k:       
            
        Returns:
            [(key, weight), ...]    K 
        """
        result = self.attend_text(query, keys)
        
        #              
        pairs = list(zip(keys, result.weights))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pairs[:k]


#    
_attention = None

def get_wave_attention() -> WaveAttention:
    global _attention
    if _attention is None:
        _attention = WaveAttention()
    return _attention


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("  WAVE ATTENTION TEST")
    print("=" * 50)
    
    attn = get_wave_attention()
    
    #    
    query = "         ?"
    keys = ["  ", "  ", "  ", "  ", "  "]
    
    print(f"\n  : {query}")
    print(f"  : {keys}")
    
    top3 = attn.focus_topk(query, keys, k=3)
    
    print("\n       :")
    for key, weight in top3:
        bar = " " * int(weight * 20)
        print(f"   {key}: {weight:.3f} {bar}")
    
    print("\n  Wave Attention works!")
