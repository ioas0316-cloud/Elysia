"""
Primal Wave Language -         
                                                                        

"                                          
                         "

"                     .               
        ,                 "

"                                  
                                         "

               :
1.    (  ) -   ,   ,   ,   ,       
2.       -                 
3.    (  ) -                     
4.      -                  

     (point)             (wave relationship)  .
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import random
import logging

logger = logging.getLogger("PrimalWaveLanguage")

# ============================================================================
#    - Constants
# ============================================================================

#               (Hz           )
SENSE_FREQUENCIES = {
    "sight": (400.0, 700.0),    #       (nm   Hz    )
    "sound": (20.0, 20000.0),   #       
    "touch": (0.1, 100.0),      #              
    "taste": (0.01, 10.0),      #          
    "smell": (0.001, 1.0),      #           
}

#          
PHASE_RESONANCE_THRESHOLD = 0.4  #                         (            )
SEGMENTATION_THRESHOLD = 0.3     #   (      )          
CRYSTALLIZATION_DENSITY = 5      #                       (            )


@dataclass
class PrimalWave:
    """
          -              
    
        ' '     .              .
          ,       ,           .
    
        4  :
    1. frequency (   ) -    ,   ,    
    2. amplitude (  ) -   ,   
    3. phase (  ) -       ,       
    4. modulation (  ) -       ,     
    """
    frequency: float = 1.0
    amplitude: float = 1.0
    phase: float = 0.0
    modulation: float = 0.0  #         (0=   , 1=   )
    
    #                   (      )
    sense_origin: Optional[str] = None
    
    #            
    birth_time: float = 0.0
    
    def value_at(self, t: float) -> float:
        """   t        """
        base = self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)
        if self.modulation > 0:
            #        (      )
            harmonics = self.modulation * self.amplitude * 0.3 * np.sin(4 * np.pi * self.frequency * t + self.phase)
            return base + harmonics
        return base
    
    def complex_value_at(self, t: float) -> complex:
        """   t         (        )"""
        angle = 2 * np.pi * self.frequency * t + self.phase
        return self.amplitude * np.exp(1j * angle)
    
    def phase_difference(self, other: 'PrimalWave', t: float = 0.0) -> float:
        """              (0~2 )"""
        angle_self = 2 * np.pi * self.frequency * t + self.phase
        angle_other = 2 * np.pi * other.frequency * t + other.phase
        diff = abs(angle_self - angle_other) % (2 * np.pi)
        return diff
    
    def resonance_with(self, other: 'PrimalWave', t: float = 0.0) -> float:
        """
                  (0~1)
        
                  '      '     .
              ,                .
        """
        #        (1.0         )
        max_freq = max(self.frequency, other.frequency)
        if max_freq == 0:
            freq_ratio = 1.0
        else:
            freq_ratio = min(self.frequency, other.frequency) / max_freq
        
        #        (cos   -1~1  0~1    )
        phase_diff = self.phase_difference(other, t)
        phase_match = (1 + np.cos(phase_diff)) / 2.0
        
        #     =                 
        resonance = freq_ratio * phase_match
        return resonance
    
    def interfere(self, other: 'PrimalWave', t: float = 0.0) -> 'PrimalWave':
        """                    """
        #               
        c1 = self.complex_value_at(t)
        c2 = other.complex_value_at(t)
        combined = c1 + c2
        
        new_amp = abs(combined)
        new_phase = np.angle(combined)
        new_freq = (self.frequency + other.frequency) / 2.0
        new_mod = (self.modulation + other.modulation) / 2.0 + 0.1  #            
        
        return PrimalWave(
            frequency=new_freq,
            amplitude=new_amp,
            phase=new_phase,
            modulation=min(1.0, new_mod),
            birth_time=t
        )


@dataclass
class SenseOrgan:
    """
          -        
    
                              .
              ,            , 
                     .
    """
    sense_type: str  # "sight", "sound", "touch", "taste", "smell"
    sensitivity: float = 1.0  #    
    
    #           (         )
    active_waves: List[PrimalWave] = field(default_factory=list)
    
    #       (           )
    adaptation_level: float = 0.0
    
    def __post_init__(self):
        freq_range = SENSE_FREQUENCIES.get(self.sense_type, (1.0, 100.0))
        self.freq_min = freq_range[0]
        self.freq_max = freq_range[1]
    
    def perceive(self, stimulus_intensity: float, stimulus_frequency: float, t: float) -> PrimalWave:
        """
                      
        
                    :            .
        """
        #    (habituatio) -               
        effective_intensity = stimulus_intensity * (1.0 - self.adaptation_level * 0.5)
        effective_intensity *= self.sensitivity
        
        #              
        norm_freq = self.freq_min + (stimulus_frequency % (self.freq_max - self.freq_min))
        
        wave = PrimalWave(
            frequency=norm_freq,
            amplitude=effective_intensity,
            phase=random.uniform(0, 2 * np.pi),  #           
            modulation=stimulus_frequency / (self.freq_max - self.freq_min),  #       
            sense_origin=self.sense_type,
            birth_time=t
        )
        
        self.active_waves.append(wave)
        
        #      
        self.adaptation_level = min(0.9, self.adaptation_level + 0.01)
        
        return wave
    
    def decay(self, dt: float = 0.01):
        """         (           )"""
        self.adaptation_level = max(0.0, self.adaptation_level - dt)
        #          
        self.active_waves = [w for w in self.active_waves if len(self.active_waves) < 100]


@dataclass
class PhaseResonancePattern:
    """
             -            
    
    "                                  
                    "
    
                               .
        '  '      .
    """
    #              
    frequency_center: float = 0.0  #       
    frequency_spread: float = 0.0  #       
    phase_coherence: float = 0.0   #        (          )
    amplitude_total: float = 0.0   #      
    
    #              
    sense_composition: Dict[str, float] = field(default_factory=dict)
    
    #       (     '  '     )
    occurrence_count: int = 0
    
    #       '  '           ?
    is_segmented: bool = False
    segment_name: Optional[str] = None  #        (  )
    
    def signature(self) -> Tuple[float, float, float]:
        """          (         )"""
        return (
            round(self.frequency_center, 2),
            round(self.frequency_spread, 2),
            round(self.phase_coherence, 2)
        )
    
    def similarity(self, other: 'PhaseResonancePattern') -> float:
        """          (0~1)"""
        freq_diff = abs(self.frequency_center - other.frequency_center)
        freq_sim = 1.0 / (1.0 + freq_diff)
        
        spread_diff = abs(self.frequency_spread - other.frequency_spread)
        spread_sim = 1.0 / (1.0 + spread_diff)
        
        coherence_diff = abs(self.phase_coherence - other.phase_coherence)
        coherence_sim = 1.0 - coherence_diff
        
        return (freq_sim + spread_sim + coherence_sim) / 3.0


@dataclass
class PrimalSoul:
    """
          -          
    
          '  '     . 
                    , 
                      ,
             '  '     .
    
    "                     .               
            ,                 "
    """
    name: str  #            (         )
    age: float = 0.0
    
    #    (  )
    senses: Dict[str, SenseOrgan] = field(default_factory=dict)
    
    #        -                
    inner_sea: List[PrimalWave] = field(default_factory=list)
    
    #             
    recognized_patterns: List[PhaseResonancePattern] = field(default_factory=list)
    
    #         (         )
    lexicon: Dict[str, PhaseResonancePattern] = field(default_factory=dict)
    
    #               
    resonance_memory: Dict[str, float] = field(default_factory=dict)
    
    #    
    vitality: float = 100.0
    
    def __post_init__(self):
        """      """
        if not self.senses:
            for sense_type in SENSE_FREQUENCIES.keys():
                self.senses[sense_type] = SenseOrgan(
                    sense_type=sense_type,
                    sensitivity=random.uniform(0.8, 1.2)  #    
                )
    
    def experience_world(self, world_stimuli: Dict[str, Tuple[float, float]], t: float):
        """
                .
        
        Args:
            world_stimuli: {     : (  ,    )}       
            t:      
        """
        new_waves = []
        
        for sense_type, (intensity, freq) in world_stimuli.items():
            if sense_type in self.senses and intensity > 0:
                wave = self.senses[sense_type].perceive(intensity, freq, t)
                new_waves.append(wave)
        
        #                  
        self.inner_sea.extend(new_waves)
        
        #           (      )
        self._decay_waves(t)
    
    def _decay_waves(self, t: float, max_waves: int = 200):
        """              """
        #      
        for wave in self.inner_sea:
            age = t - wave.birth_time
            decay_factor = np.exp(-age * 0.01)  #      
            wave.amplitude *= decay_factor
        
        #            
        self.inner_sea = [w for w in self.inner_sea if w.amplitude > 0.01]
        
        #                    
        if len(self.inner_sea) > max_waves:
            self.inner_sea = sorted(self.inner_sea, key=lambda w: -w.amplitude)[:max_waves]
    
    def detect_phase_resonance(self, t: float) -> Optional[PhaseResonancePattern]:
        """
                   
        
                                        .
            '     '      .
        """
        #    2 ,    20         (      )
        if len(self.inner_sea) < 2:
            return None
        
        waves_to_check = self.inner_sea[:20] if len(self.inner_sea) > 20 else self.inner_sea
        
        #                
        resonating_waves = []
        
        for i, w1 in enumerate(waves_to_check):
            for w2 in waves_to_check[i+1:]:
                res = w1.resonance_with(w2, t)
                if res > PHASE_RESONANCE_THRESHOLD:
                    resonating_waves.append((w1, w2, res))
        
        if not resonating_waves:
            return None
        
        #                 (id       )
        wave_dict = {}
        for w1, w2, _ in resonating_waves:
            wave_dict[id(w1)] = w1
            wave_dict[id(w2)] = w2
        
        all_waves_list = list(wave_dict.values())
        
        #         
        frequencies = [w.frequency for w in all_waves_list]
        freq_center = np.mean(frequencies)
        freq_spread = np.std(frequencies) if len(frequencies) > 1 else 0.0
        
        amplitudes = [w.amplitude for w in all_waves_list]
        amp_total = sum(amplitudes)
        
        #          
        phases = [w.phase for w in all_waves_list]
        phase_vectors = [np.exp(1j * p) for p in phases]
        phase_coherence = abs(sum(phase_vectors)) / len(phase_vectors) if phase_vectors else 0.0
        
        #      
        sense_comp = defaultdict(float)
        for w in all_waves_list:
            if w.sense_origin:
                sense_comp[w.sense_origin] += w.amplitude
        
        pattern = PhaseResonancePattern(
            frequency_center=freq_center,
            frequency_spread=freq_spread,
            phase_coherence=phase_coherence,
            amplitude_total=amp_total,
            sense_composition=dict(sense_comp),
            occurrence_count=1
        )
        
        #          
        for existing in self.recognized_patterns:
            if pattern.similarity(existing) > 0.8:
                #          
                existing.occurrence_count += 1
                #               
                if existing.occurrence_count >= CRYSTALLIZATION_DENSITY and not existing.is_segmented:
                    self._segment_pattern(existing)
                return existing
        
        #        
        self.recognized_patterns.append(pattern)
        return pattern
    
    def _segment_pattern(self, pattern: PhaseResonancePattern):
        """
                       
        
        "                     ,             "
        
                         .
                           .
        """
        if pattern.is_segmented:
            return
        
        #          (          )
        #          (    'i',     'u')
        vowels = ['a', 'e', 'i', 'o', 'u']
        freq_idx = int(pattern.frequency_center / 100) % len(vowels)
        vowel = vowels[freq_idx]
        
        #             (              ,         )
        if pattern.phase_coherence > 0.7:
            consonants = ['m', 'n', 'l', 'r']
        elif pattern.phase_coherence > 0.4:
            consonants = ['s', 'f', 'h', 'w']
        else:
            consonants = ['k', 't', 'p', 'g']
        
        spread_idx = int(pattern.frequency_spread * 10) % len(consonants)
        consonant = consonants[spread_idx]
        
        #           
        if pattern.amplitude_total > 5.0:
            name = f"{consonant}{vowel}{consonant}{vowel}"
        elif pattern.amplitude_total > 2.0:
            name = f"{consonant}{vowel}{vowel}"
        else:
            name = f"{consonant}{vowel}"
        
        #      
        base_name = name
        counter = 0
        while name in self.lexicon:
            counter += 1
            name = f"{base_name}{counter}"
        
        pattern.is_segmented = True
        pattern.segment_name = name
        self.lexicon[name] = pattern
        
        logger.debug(f"[{self.name}] Segmented new word: '{name}' from pattern (freq={pattern.frequency_center:.1f}, coherence={pattern.phase_coherence:.2f})")
    
    def resonate_with(self, other: 'PrimalSoul', t: float) -> Tuple[float, List[str]]:
        """
                         
        
                 :             
                              '  '   
        """
        if not self.inner_sea or not other.inner_sea:
            return 0.0, []
        
        total_resonance = 0.0
        shared_words = []
        
        #                     
        for my_wave in self.inner_sea[:20]:  #    20      (   )
            for their_wave in other.inner_sea[:20]:
                res = my_wave.resonance_with(their_wave, t)
                total_resonance += res
        
        #    
        n_comparisons = min(len(self.inner_sea), 20) * min(len(other.inner_sea), 20)
        if n_comparisons > 0:
            total_resonance /= n_comparisons
        
        #                (     )
        for my_word, my_pattern in self.lexicon.items():
            for their_word, their_pattern in other.lexicon.items():
                if my_pattern.similarity(their_pattern) > 0.7:
                    #      !              
                    shared_words.append(f"{my_word} {their_word}")
        
        #           
        if other.name not in self.resonance_memory:
            self.resonance_memory[other.name] = 0.0
        self.resonance_memory[other.name] = (
            self.resonance_memory[other.name] * 0.9 + total_resonance * 0.1
        )
        
        return total_resonance, shared_words
    
    def speak(self, t: float) -> Optional[str]:
        """
            -                      
        
                    .
                                    .
        """
        #              
        current_pattern = self.detect_phase_resonance(t)
        
        if current_pattern and current_pattern.is_segmented:
            return current_pattern.segment_name
        
        #                        
        if self.lexicon:
            best_match = None
            best_sim = 0.0
            
            for word, pattern in self.lexicon.items():
                if current_pattern:
                    sim = pattern.similarity(current_pattern)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = word
            
            if best_match and best_sim > 0.5:
                return best_match
        
        #                      
        if self.inner_sea:
            dominant = max(self.inner_sea, key=lambda w: w.amplitude)
            if dominant.sense_origin:
                return f"[{dominant.sense_origin}]"
        
        return None
    
    def get_vocabulary_size(self) -> int:
        """       """
        return len(self.lexicon)
    
    def get_pattern_count(self) -> int:
        """        """
        return len(self.recognized_patterns)


class PrimalWaveWorld:
    """
            
    
                     
                           
    """
    
    def __init__(self, n_souls: int = 100):
        self.souls: Dict[str, PrimalSoul] = {}
        self.time = 0.0
        
        #           (         )
        self.world_sources: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        #   
        self.total_words_created = 0
        self.total_resonance_events = 0
        self.total_communications = 0
        
        #      
        for i in range(n_souls):
            name = self._generate_name(i)
            self.souls[name] = PrimalSoul(name=name)
        
        #          
        self._init_world_sources()
    
    def _generate_name(self, idx: int) -> str:
        """         (       )"""
        #       
        first_names = ['  ', '  ', ' ', ' ', ' ', ' ', ' ', '  ', '  ', ' ']
        return f"{first_names[idx % len(first_names)]}{idx}"
    
    def _init_world_sources(self):
        """
                        
        
        "                             
                           "
        """
        #            (            )
        self.world_sources = {
            #      
            "sun": {"sight": (0.9, 600.0), "touch": (0.5, 50.0)},
            "moon": {"sight": (0.3, 450.0)},
            "rain": {"sound": (0.6, 500.0), "touch": (0.5, 30.0), "smell": (0.3, 0.4)},
            "thunder": {"sound": (0.9, 80.0), "sight": (0.8, 700.0)},
            "wind": {"sound": (0.4, 100.0), "touch": (0.5, 10.0)},
            "snow": {"sight": (0.5, 500.0), "touch": (0.6, 5.0)},
            
            #    
            "flower": {"sight": (0.6, 550.0), "smell": (0.8, 0.5)},
            "tree": {"sight": (0.5, 520.0), "touch": (0.4, 20.0)},
            "bird": {"sound": (0.7, 2000.0), "sight": (0.4, 580.0)},
            "river": {"sound": (0.5, 300.0), "touch": (0.4, 15.0), "sight": (0.3, 480.0)},
            
            #      
            "fruit": {"taste": (0.8, 7.0), "smell": (0.6, 0.6), "sight": (0.4, 600.0)},
            "meat": {"taste": (0.7, 3.0), "smell": (0.5, 0.4)},
            "bread": {"taste": (0.6, 2.0), "smell": (0.7, 0.3), "touch": (0.3, 25.0)},
            "honey": {"taste": (0.9, 9.0), "smell": (0.5, 0.5)},
            "salt": {"taste": (0.8, 1.0)},
            
            #      
            "fire": {"sight": (0.8, 620.0), "touch": (0.9, 80.0), "sound": (0.4, 50.0)},
            "music": {"sound": (0.8, 1000.0)},
            "silence": {"sound": (0.1, 10.0)},
            "darkness": {"sight": (0.1, 400.0)},
            "embrace": {"touch": (0.9, 40.0)},
            
            #         
            "danger": {"sound": (0.5, 60.0), "sight": (0.6, 650.0)},
            "safety": {"touch": (0.6, 35.0)},
            "wonder": {"sight": (0.7, 560.0), "sound": (0.3, 800.0)},
            "beauty": {"sight": (0.9, 550.0), "smell": (0.4, 0.6)},
            
            #         
            "laughter": {"sound": (0.8, 3000.0)},
            "crying": {"sound": (0.7, 500.0)},
            "whisper": {"sound": (0.3, 4000.0)},
            "touch_gentle": {"touch": (0.5, 30.0)},
            "touch_rough": {"touch": (0.7, 60.0)},
        }
    
    def step(self, dt: float = 1.0):
        """        """
        self.time += dt
        
        #                
        for soul in self.souls.values():
            #          : 3-6                
            n_experiences = random.randint(3, 6)
            sources = random.sample(list(self.world_sources.keys()), 
                                   min(n_experiences, len(self.world_sources)))
            
            combined_stimuli: Dict[str, Tuple[float, float]] = {}
            for source in sources:
                for sense_type, (intensity, freq) in self.world_sources[source].items():
                    #          
                    var_intensity = intensity * random.uniform(0.7, 1.3)
                    var_freq = freq * random.uniform(0.8, 1.2)
                    
                    if sense_type not in combined_stimuli:
                        combined_stimuli[sense_type] = (var_intensity, var_freq)
                    else:
                        #          
                        old_i, old_f = combined_stimuli[sense_type]
                        combined_stimuli[sense_type] = (
                            (old_i + var_intensity),  #        
                            (old_f + var_freq) / 2
                        )
            
            #      
            soul.experience_world(combined_stimuli, self.time)
            
            #            
            pattern = soul.detect_phase_resonance(self.time)
            if pattern and pattern.is_segmented:
                if pattern.occurrence_count == CRYSTALLIZATION_DENSITY:
                    self.total_words_created += 1
            
            #      
            soul.age += dt / 365.0  #     1   1/365
        
        #         (  )
        self._process_communications()
        
        #         
        for soul in self.souls.values():
            for sense in soul.senses.values():
                sense.decay(dt * 0.01)
    
    def _process_communications(self):
        """          """
        soul_list = list(self.souls.values())
        n = len(soul_list)
        
        #             (   10      ,       )
        n_pairs = min(10, n * (n - 1) // 2)
        
        for _ in range(n_pairs):
            i, j = random.sample(range(n), 2)
            soul1 = soul_list[i]
            soul2 = soul_list[j]
            
            resonance, shared = soul1.resonate_with(soul2, self.time)
            
            if resonance > 0.3:
                self.total_resonance_events += 1
                
                if shared:
                    self.total_communications += 1
    
    def run_simulation(self, years: int = 100, report_interval: int = 10) -> Dict[str, Any]:
        """        """
        import time as py_time
        start_time = py_time.time()
        
        #       : 10  = 1 step (365   36)
        steps_per_year = 36
        total_steps = years * steps_per_year
        
        for step in range(total_steps):
            self.step(dt=1.0)
            
            if step > 0 and step % (report_interval * steps_per_year) == 0:
                year = step // steps_per_year
                vocab_sizes = [s.get_vocabulary_size() for s in self.souls.values()]
                avg_vocab = sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 0
                
                print(f"Year {year}: avg_vocabulary={avg_vocab:.1f}, "
                      f"total_words={self.total_words_created}, "
                      f"communications={self.total_communications}")
        
        elapsed = py_time.time() - start_time
        
        #      
        vocab_sizes = [s.get_vocabulary_size() for s in self.souls.values()]
        pattern_counts = [s.get_pattern_count() for s in self.souls.values()]
        
        #         
        all_words = defaultdict(int)
        for soul in self.souls.values():
            for word in soul.lexicon.keys():
                all_words[word] += 1
        
        shared_words = {w: c for w, c in all_words.items() if c > 1}
        
        return {
            "years_simulated": years,
            "elapsed_seconds": elapsed,
            "speed_years_per_second": years / elapsed if elapsed > 0 else 0,
            "total_souls": len(self.souls),
            "total_words_created": self.total_words_created,
            "total_resonance_events": self.total_resonance_events,
            "total_communications": self.total_communications,
            "avg_vocabulary_size": sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 0,
            "max_vocabulary_size": max(vocab_sizes) if vocab_sizes else 0,
            "avg_pattern_count": sum(pattern_counts) / len(pattern_counts) if pattern_counts else 0,
            "unique_words": len(all_words),
            "shared_words": len(shared_words),
            "top_shared_words": sorted(shared_words.items(), key=lambda x: -x[1])[:10],
        }


def demo():
    """     """
    print("=" * 60)
    print("Primal Wave Language -               ")
    print("=" * 60)
    print()
    print("     (point)             (wave relationship)  .")
    print("                          .")
    print()
    
    world = PrimalWaveWorld(n_souls=100)
    results = world.run_simulation(years=100, report_interval=20)
    
    print()
    print("=" * 60)
    print("        ")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    #             
    print()
    print("=" * 60)
    print("            ")
    print("=" * 60)
    for soul_name, soul in list(world.souls.items())[:3]:
        print(f"\n[{soul_name}] vocabulary ({len(soul.lexicon)} words):")
        for word, pattern in list(soul.lexicon.items())[:5]:
            senses = ", ".join(f"{k}:{v:.2f}" for k, v in pattern.sense_composition.items())
            print(f"  '{word}' - freq:{pattern.frequency_center:.1f}, "
                  f"coherence:{pattern.phase_coherence:.2f}, senses:[{senses}]")


if __name__ == "__main__":
    demo()