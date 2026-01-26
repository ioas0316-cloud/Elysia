"""
Khala Wave Communication -           
                                                                        

"            .                     
                               "

         -        :
                    ,                  .
           ,        .
                    .

                                                                        

     :
1.               (     )
2.       ,   ,            
3.                        
4.         '  /  '   (      )

"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random
import time as py_time

# ============================================================================
#   
# ============================================================================

#      
WAVE_DIMENSIONS = 8  #        (  ,   ,     )
RESONANCE_THRESHOLD = 0.6  #       '  '             
PATTERN_MEMORY_SIZE = 100  #                    
NAMING_THRESHOLD = 5  #                   


@dataclass
class KhalaWave:
    """
          -             
    
        '  '     .            /  /        .
                 ,                .
    
    8         :
    - dim 0:       (  /  )
    - dim 1:       (  /  )
    - dim 2:        (  /  )
    - dim 3:       (  /  )
    - dim 4:       (  /  )
    - dim 5:       (  /  )
    - dim 6:     (   /  )
    - dim 7:    (  /  )
    """
    vector: np.ndarray  # 8        
    timestamp: float = 0.0
    sender_id: str = ""
    
    def __post_init__(self):
        if self.vector is None:
            self.vector = np.zeros(WAVE_DIMENSIONS)
        elif len(self.vector) != WAVE_DIMENSIONS:
            #       
            new_vec = np.zeros(WAVE_DIMENSIONS)
            new_vec[:min(len(self.vector), WAVE_DIMENSIONS)] = self.vector[:WAVE_DIMENSIONS]
            self.vector = new_vec
    
    def resonance_with(self, other: 'KhalaWave') -> float:
        """
                  (0~1)
        
                  = 1.0 (      )
                  = 0.0 (        )
        """
        #        
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        
        similarity = dot / (norm_self * norm_other)
        # -1~1  0~1    
        return (similarity + 1.0) / 2.0
    
    def blend_with(self, other: 'KhalaWave', ratio: float = 0.5) -> 'KhalaWave':
        """         (          )"""
        blended = self.vector * (1 - ratio) + other.vector * ratio
        return KhalaWave(vector=blended, timestamp=max(self.timestamp, other.timestamp))
    
    def signature(self) -> Tuple[int, ...]:
        """       (         ) -       3       """
        quantized = np.round(self.vector * 2) / 2  # -1, -0.5, 0, 0.5, 1
        return tuple(quantized.tolist())


@dataclass  
class WavePattern:
    """
          -            
    
            '  '    .
               .             .
    """
    centroid: np.ndarray  #          
    occurrence_count: int = 0  #      
    first_seen: float = 0.0  #          
    last_seen: float = 0.0  #           
    
    #    (       )
    name: Optional[str] = None
    
    #                    
    frequent_users: Dict[str, int] = field(default_factory=dict)
    
    #       (             )
    contexts: List[str] = field(default_factory=list)
    
    def similarity(self, wave: KhalaWave) -> float:
        """             """
        dot = np.dot(self.centroid, wave.vector)
        norm_c = np.linalg.norm(self.centroid)
        norm_w = np.linalg.norm(wave.vector)
        
        if norm_c < 1e-8 or norm_w < 1e-8:
            return 0.0
        
        sim = dot / (norm_c * norm_w)
        return (sim + 1.0) / 2.0
    
    def update(self, wave: KhalaWave, context: str = ""):
        """              """
        #                
        alpha = 1.0 / (self.occurrence_count + 1)
        self.centroid = (1 - alpha) * self.centroid + alpha * wave.vector
        
        self.occurrence_count += 1
        self.last_seen = wave.timestamp
        
        if wave.sender_id:
            self.frequent_users[wave.sender_id] = self.frequent_users.get(wave.sender_id, 0) + 1
        
        if context and len(self.contexts) < 20:
            self.contexts.append(context)


@dataclass
class KhalaSoul:
    """
          -             
    
              '      '.
                  .
             .
    """
    id: str
    
    #          (8     )
    inner_state: np.ndarray = field(default_factory=lambda: np.random.randn(WAVE_DIMENSIONS) * 0.3)
    
    #            (    '  '   )
    known_patterns: List[WavePattern] = field(default_factory=list)
    
    #               
    connections: Dict[str, float] = field(default_factory=dict)
    
    #   
    waves_sent: int = 0
    waves_received: int = 0
    successful_communications: int = 0
    
    def generate_wave(self, intent: str = "") -> KhalaWave:
        """
              -                   
        
               .        /        .
        """
        #             
        wave_vec = self.inner_state.copy()
        
        #          
        if intent == "greeting":
            wave_vec[0] += 0.5   #   
            wave_vec[2] += 0.5   #   
            wave_vec[7] += 0.3   #      
        elif intent == "question":
            wave_vec[6] -= 0.5   #    
            wave_vec[3] += 0.3   #      
        elif intent == "joy":
            wave_vec[0] += 0.8   #   
            wave_vec[1] += 0.6   #   
            wave_vec[7] += 0.5   #   
        elif intent == "sadness":
            wave_vec[0] -= 0.6   #   
            wave_vec[1] -= 0.4   #   
            wave_vec[7] += 0.4   #   
        elif intent == "fear":
            wave_vec[0] -= 0.4   #   
            wave_vec[1] += 0.7   #   
            wave_vec[2] -= 0.6   #   
        elif intent == "love":
            wave_vec[0] += 0.9   #      
            wave_vec[2] += 0.8   #   
            wave_vec[5] -= 0.3   #      
            wave_vec[7] += 0.7   #   
        elif intent == "curiosity":
            wave_vec[3] += 0.5   #   
            wave_vec[4] += 0.3   #      
            wave_vec[6] -= 0.3   #    
        elif intent == "anger":
            wave_vec[0] -= 0.7   #   
            wave_vec[1] += 0.8   #   
            wave_vec[7] += 0.9   #      
        else:
            #      
            wave_vec += np.random.randn(WAVE_DIMENSIONS) * 0.2
        
        #     (-1 ~ 1)
        wave_vec = np.clip(wave_vec, -1.0, 1.0)
        
        wave = KhalaWave(
            vector=wave_vec,
            timestamp=py_time.time(),
            sender_id=self.id
        )
        
        self.waves_sent += 1
        return wave
    
    def receive_wave(self, wave: KhalaWave, context: str = "") -> float:
        """
              -         '  '  
        
        Returns:
                (0~1)
        """
        self.waves_received += 1
        
        #             
        my_wave = KhalaWave(vector=self.inner_state.copy())
        resonance = my_wave.resonance_with(wave)
        
        #             (     )
        influence = 0.1 * resonance  #              
        self.inner_state = (1 - influence) * self.inner_state + influence * wave.vector
        
        #           
        if wave.sender_id:
            old_conn = self.connections.get(wave.sender_id, 0.0)
            self.connections[wave.sender_id] = old_conn + resonance * 0.1
        
        #         
        if resonance > RESONANCE_THRESHOLD:
            self.successful_communications += 1
        
        #      
        self._remember_pattern(wave, context)
        
        return resonance
    
    def _remember_pattern(self, wave: KhalaWave, context: str = ""):
        """         """
        #               
        best_match = None
        best_sim = 0.0
        
        for pattern in self.known_patterns:
            sim = pattern.similarity(wave)
            if sim > best_sim:
                best_sim = sim
                best_match = pattern
        
        if best_match and best_sim > 0.7:
            #           
            best_match.update(wave, context)
            
            #                
            if best_match.occurrence_count >= NAMING_THRESHOLD and best_match.name is None:
                best_match.name = self._generate_name(best_match)
        else:
            #        
            if len(self.known_patterns) < PATTERN_MEMORY_SIZE:
                new_pattern = WavePattern(
                    centroid=wave.vector.copy(),
                    occurrence_count=1,
                    first_seen=wave.timestamp,
                    last_seen=wave.timestamp
                )
                if wave.sender_id:
                    new_pattern.frequent_users[wave.sender_id] = 1
                if context:
                    new_pattern.contexts.append(context)
                self.known_patterns.append(new_pattern)
    
    def _generate_name(self, pattern: WavePattern) -> str:
        """
                  
        
                               .
            '  '      .
        """
        vec = pattern.centroid
        
        #                 
        # dim 0 (  ):   =     ,   =      
        if vec[0] > 0.3:
            vowel1 = random.choice(['a', 'e', 'i'])
        elif vec[0] < -0.3:
            vowel1 = random.choice(['o', 'u'])
        else:
            vowel1 = random.choice(['a', 'e', 'i', 'o', 'u'])
        
        # dim 1 (  ):   =    ,   =      
        if vec[1] > 0.3:
            consonant1 = random.choice(['k', 't', 'p', 'ch'])
        elif vec[1] < -0.3:
            consonant1 = random.choice(['m', 'n', 'l', 'r'])
        else:
            consonant1 = random.choice(['s', 'h', 'w', 'y'])
        
        # dim 2 (   ):   =    ,   =    
        if vec[2] > 0.3:
            vowel2 = random.choice(['a', 'o'])
        else:
            vowel2 = random.choice(['i', 'u'])
        
        # dim 7 (  ):   =    ,   =     
        if vec[7] > 0.5:
            consonant2 = random.choice(['k', 't', 'n', 'm'])
            name = f"{consonant1}{vowel1}{consonant2}{vowel2}"
        elif vec[7] > 0:
            name = f"{consonant1}{vowel1}{vowel2}"
        else:
            name = f"{consonant1}{vowel1}"
        
        return name
    
    def get_vocabulary(self) -> Dict[str, WavePattern]:
        """        (  )"""
        return {p.name: p for p in self.known_patterns if p.name is not None}
    
    def describe_wave(self, wave: KhalaWave) -> str:
        """       (                   )"""
        for pattern in self.known_patterns:
            if pattern.name and pattern.similarity(wave) > 0.7:
                return pattern.name
        
        #                
        vec = wave.vector
        parts = []
        if vec[0] > 0.3:
            parts.append("positive")
        elif vec[0] < -0.3:
            parts.append("negative")
        if vec[1] > 0.3:
            parts.append("excited")
        elif vec[1] < -0.3:
            parts.append("calm")
        if vec[7] > 0.5:
            parts.append("strong")
        
        return "+".join(parts) if parts else "neutral"


class KhalaNetwork:
    """
            -                  
    
               .
                .
    """
    
    def __init__(self, n_souls: int = 100):
        self.souls: Dict[str, KhalaSoul] = {}
        self.time = 0.0
        
        #                     
        self.shared_patterns: List[WavePattern] = []
        
        #   
        self.total_communications = 0
        self.successful_understandings = 0
        
        #   /      (        )
        self.contexts = [
            "meeting", "parting", "eating", "danger", "safety",
            "sunrise", "sunset", "rain", "storm", "peace",
            "conflict", "cooperation", "discovery", "loss", "gain",
            "birth", "death", "celebration", "mourning", "play",
            "work", "rest", "dream", "nightmare", "love",
            "rejection", "acceptance", "confusion", "clarity", "wonder"
        ]
        
        #      
        for i in range(n_souls):
            soul_id = f"soul_{i}"
            self.souls[soul_id] = KhalaSoul(id=soul_id)
    
    def step(self):
        """      """
        self.time += 1
        
        soul_list = list(self.souls.values())
        n = len(soul_list)
        
        #       (n/2  )
        n_communications = max(1, n // 2)
        
        for _ in range(n_communications):
            #        
            idx1, idx2 = random.sample(range(n), 2)
            soul1 = soul_list[idx1]
            soul2 = soul_list[idx2]
            
            #      
            context = random.choice(self.contexts)
            
            #      
            intents = ["greeting", "question", "joy", "sadness", 
                      "fear", "love", "curiosity", "anger", ""]
            intent = random.choice(intents)
            
            #      
            wave1 = soul1.generate_wave(intent)
            wave2 = soul2.generate_wave(random.choice(intents))
            
            #          
            res1 = soul1.receive_wave(wave2, context)
            res2 = soul2.receive_wave(wave1, context)
            
            self.total_communications += 2
            
            if res1 > RESONANCE_THRESHOLD:
                self.successful_understandings += 1
            if res2 > RESONANCE_THRESHOLD:
                self.successful_understandings += 1
        
        #          (      )
        for soul in soul_list:
            soul.inner_state += np.random.randn(WAVE_DIMENSIONS) * 0.05
            soul.inner_state = np.clip(soul.inner_state, -1.0, 1.0)
    
    def run_simulation(self, ticks: int = 1000, report_interval: int = 100) -> Dict[str, Any]:
        """        """
        start_time = py_time.time()
        
        for tick in range(ticks):
            self.step()
            
            if tick > 0 and tick % report_interval == 0:
                #      
                vocab_sizes = [len(s.get_vocabulary()) for s in self.souls.values()]
                avg_vocab = sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 0
                
                print(f"Tick {tick}: avg_vocabulary={avg_vocab:.1f}, "
                      f"communications={self.total_communications}, "
                      f"understanding_rate={self.successful_understandings/max(1,self.total_communications)*100:.1f}%")
        
        elapsed = py_time.time() - start_time
        
        #         
        all_patterns = defaultdict(list)
        for soul in self.souls.values():
            for pattern in soul.known_patterns:
                if pattern.name:
                    sig = pattern.centroid.round(1).tobytes()
                    all_patterns[sig].append((soul.id, pattern.name, pattern))
        
        #                             =      
        shared_language = {}
        for sig, entries in all_patterns.items():
            if len(entries) > 1:
                #             
                names = [e[1] for e in entries]
                most_common = max(set(names), key=names.count)
                shared_language[most_common] = len(entries)
        
        #      
        vocab_sizes = [len(s.get_vocabulary()) for s in self.souls.values()]
        
        return {
            "ticks_simulated": ticks,
            "elapsed_seconds": elapsed,
            "speed_ticks_per_second": ticks / elapsed if elapsed > 0 else 0,
            "total_souls": len(self.souls),
            "total_communications": self.total_communications,
            "successful_understandings": self.successful_understandings,
            "understanding_rate": self.successful_understandings / max(1, self.total_communications),
            "avg_vocabulary_size": sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 0,
            "max_vocabulary_size": max(vocab_sizes) if vocab_sizes else 0,
            "shared_language_words": len(shared_language),
            "top_shared_words": sorted(shared_language.items(), key=lambda x: -x[1])[:10],
        }


def demo():
    """  """
    print("=" * 70)
    print("Khala Wave Communication -           ")
    print("=" * 70)
    print()
    print("\"                   ")
    print("                               \"")
    print()
    
    network = KhalaNetwork(n_souls=50)
    results = network.run_simulation(ticks=500, report_interval=100)
    
    print()
    print("=" * 70)
    print("  ")
    print("=" * 70)
    for key, value in results.items():
        if 'top_shared' not in str(key):
            print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("       (              )")
    print("=" * 70)
    if results.get('top_shared_words'):
        for word, count in results['top_shared_words']:
            print(f"  \"{word}\" - {count}     ")
    
    #          
    print()
    print("=" * 70)
    print("          (       )")
    print("=" * 70)
    for soul_id, soul in list(network.souls.items())[:3]:
        vocab = soul.get_vocabulary()
        print(f"\n[{soul_id}] {len(vocab)}    :")
        for name, pattern in list(vocab.items())[:5]:
            vec = pattern.centroid
            mood = "  " if vec[0] > 0.3 else ("  " if vec[0] < -0.3 else "  ")
            energy = "  " if vec[1] > 0.3 else ("  " if vec[1] < -0.3 else "  ")
            print(f"  \"{name}\" - {mood}, {energy}, {pattern.occurrence_count}    ")


if __name__ == "__main__":
    demo()
