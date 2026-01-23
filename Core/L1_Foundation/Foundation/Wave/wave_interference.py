"""
Wave Interference Engine (        )
=========================================

"           ,               ."

Phase 10:          (Interference),   (Convergence),
     (Conflict Resolution)       .

     :
- Constructive (     ):       < 90         
- Destructive (     ):       > 90         
- Convergence (  ):              
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum

logger = logging.getLogger("WaveInterference")


class InterferenceType(Enum):
    """     """
    CONSTRUCTIVE = "constructive"   #       (  )
    DESTRUCTIVE = "destructive"     #       (  )
    MIXED = "mixed"                 #      
    NEUTRAL = "neutral"             #    (     )


@dataclass
class Wave:
    """        """
    frequency: float        #     (Hz)
    amplitude: float        #    (0.0 - 1.0)
    phase: float            #    (0 - 2 )
    source: str = ""        #       
    confidence: float = 1.0 #     (0.0 - 1.0)
    
    def to_complex(self) -> complex:
        """            (      )"""
        return self.amplitude * (math.cos(self.phase) + 1j * math.sin(self.phase))
    
    @property
    def energy(self) -> float:
        """        (  ^2    )"""
        return self.amplitude ** 2


@dataclass
class InterferenceResult:
    """     """
    resultant_wave: Wave                #         
    interference_type: InterferenceType #      
    confidence: float                   #        (0-1)
    uncertainty: float                  #         (0-1)
    original_waves: List[Wave] = field(default_factory=list)  #       
    phase_alignment: float = 0.0        #        (0-1)
    
    def is_certain(self, threshold: float = 0.7) -> bool:
        """            """
        return self.confidence >= threshold


class WaveInterference:
    """
             
    
                                   .
    
    Usage:
        engine = WaveInterference()
        waves = [Wave(440, 0.8, 0), Wave(440, 0.6, 0.1)]
        result = engine.calculate_interference(waves)
    """
    
    #           (   )
    CONSTRUCTIVE_THRESHOLD = math.pi / 2   # 90         
    DESTRUCTIVE_THRESHOLD = math.pi / 2    # 90         
    
    def calculate_interference(self, waves: List[Wave]) -> InterferenceResult:
        """
                           .
        
        Args:
            waves:              
            
        Returns:
            InterferenceResult:      
        """
        if not waves:
            return InterferenceResult(
                resultant_wave=Wave(0, 0, 0),
                interference_type=InterferenceType.NEUTRAL,
                confidence=0.0,
                uncertainty=1.0
            )
        
        if len(waves) == 1:
            return InterferenceResult(
                resultant_wave=waves[0],
                interference_type=InterferenceType.NEUTRAL,
                confidence=waves[0].confidence,
                uncertainty=0.0,
                original_waves=waves
            )
        
        # 1.        (          )
        phasor_sum = sum(wave.to_complex() for wave in waves)
        total_amplitude = abs(phasor_sum)
        resultant_phase = math.atan2(phasor_sum.imag, phasor_sum.real)
        
        # 2.        (     )
        total_energy = sum(wave.energy for wave in waves)
        if total_energy > 0:
            resultant_freq = sum(wave.frequency * wave.energy for wave in waves) / total_energy
        else:
            resultant_freq = sum(wave.frequency for wave in waves) / len(waves)
        
        # 3.         
        #          vs            
        simple_sum = sum(wave.amplitude for wave in waves)
        
        if total_amplitude >= simple_sum * 0.9:
            interference_type = InterferenceType.CONSTRUCTIVE
        elif total_amplitude <= simple_sum * 0.3:
            interference_type = InterferenceType.DESTRUCTIVE
        else:
            interference_type = InterferenceType.MIXED
        
        # 4.          
        phase_alignment = self._calculate_phase_alignment(waves)
        
        # 5.       
        #        +       =      
        avg_confidence = sum(w.confidence for w in waves) / len(waves)
        confidence = (phase_alignment * 0.4 + 
                     min(total_amplitude, 1.0) * 0.3 + 
                     avg_confidence * 0.3)
        
        # 6.      = 1 -     (  )
        uncertainty = 1.0 - confidence
        
        #         
        resultant_wave = Wave(
            frequency=resultant_freq,
            amplitude=min(total_amplitude, 1.0),  #    
            phase=resultant_phase % (2 * math.pi),
            source="interference",
            confidence=confidence
        )
        
        logger.info(
            f"  Interference: {len(waves)} waves   "
            f"{interference_type.value} (amp={total_amplitude:.2f}, conf={confidence:.2f})"
        )
        
        return InterferenceResult(
            resultant_wave=resultant_wave,
            interference_type=interference_type,
            confidence=confidence,
            uncertainty=uncertainty,
            original_waves=waves,
            phase_alignment=phase_alignment
        )
    
    def constructive_merge(self, wave_a: Wave, wave_b: Wave) -> Wave:
        """
             :              
        
                       .
        """
        #       (   1.0)
        merged_amplitude = min(wave_a.amplitude + wave_b.amplitude, 1.0)
        
        #          
        total_amp = wave_a.amplitude + wave_b.amplitude
        if total_amp > 0:
            merged_freq = (wave_a.frequency * wave_a.amplitude + 
                          wave_b.frequency * wave_b.amplitude) / total_amp
        else:
            merged_freq = (wave_a.frequency + wave_b.frequency) / 2
        
        #      
        merged_phase = (wave_a.phase + wave_b.phase) / 2
        
        #        (     )
        merged_confidence = min(
            (wave_a.confidence + wave_b.confidence) / 2 * 1.2,  # 20%    
            1.0
        )
        
        logger.debug(f"  Constructive merge: {wave_a.source} + {wave_b.source}")
        
        return Wave(
            frequency=merged_freq,
            amplitude=merged_amplitude,
            phase=merged_phase,
            source=f"{wave_a.source}+{wave_b.source}",
            confidence=merged_confidence
        )
    
    def destructive_cancel(self, wave_a: Wave, wave_b: Wave) -> Wave:
        """
             :             
        
                       .
        """
        #       (   0)
        cancelled_amplitude = abs(wave_a.amplitude - wave_b.amplitude)
        
        #               
        if wave_a.amplitude >= wave_b.amplitude:
            dominant = wave_a
        else:
            dominant = wave_b
        
        #        (           )
        cancelled_confidence = dominant.confidence * 0.5
        
        logger.debug(f"  Destructive cancel: {wave_a.source} vs {wave_b.source}")
        
        return Wave(
            frequency=dominant.frequency,
            amplitude=cancelled_amplitude,
            phase=dominant.phase,
            source=f"{dominant.source}(cancelled)",
            confidence=cancelled_confidence
        )
    
    def converge(self, waves: List[Wave]) -> Wave:
        """
                            
        
        Args:
            waves:         
            
        Returns:
            Wave:          
        """
        if not waves:
            return Wave(0, 0, 0, "empty", 0)
        
        if len(waves) == 1:
            return waves[0]
        
        #          
        total_energy = sum(wave.energy for wave in waves)
        
        if total_energy > 0:
            avg_freq = sum(wave.frequency * wave.energy for wave in waves) / total_energy
            avg_amp = math.sqrt(total_energy / len(waves))  # RMS   
        else:
            avg_freq = sum(wave.frequency for wave in waves) / len(waves)
            avg_amp = 0.0
        
        #          (circular mean)
        x_sum = sum(math.cos(wave.phase) * wave.amplitude for wave in waves)
        y_sum = sum(math.sin(wave.phase) * wave.amplitude for wave in waves)
        avg_phase = math.atan2(y_sum, x_sum)
        
        #       
        avg_confidence = sum(wave.confidence for wave in waves) / len(waves)
        
        sources = ",".join(w.source for w in waves if w.source)
        
        logger.info(f"  Converged {len(waves)} waves   freq={avg_freq:.1f}Hz")
        
        return Wave(
            frequency=avg_freq,
            amplitude=min(avg_amp, 1.0),
            phase=avg_phase % (2 * math.pi),
            source=f"converged({sources[:50]})" if sources else "converged",
            confidence=avg_confidence
        )
    
    def _calculate_phase_alignment(self, waves: List[Wave]) -> float:
        """
                       (0-1)
        
        1.0 =         (        )
        0.0 =        
        """
        if len(waves) < 2:
            return 1.0
        
        #        
        x_sum = sum(math.cos(wave.phase) for wave in waves)
        y_sum = sum(math.sin(wave.phase) for wave in waves)
        
        #       /      
        resultant_length = math.sqrt(x_sum**2 + y_sum**2)
        max_length = len(waves)
        
        return resultant_length / max_length
    
    def process_multiple_matches(
        self, 
        concept_names: List[str], 
        coordinate_map: Dict[str, Any]
    ) -> List[str]:
        """
                                
        
        Args:
            concept_names:            
            coordinate_map: InternalUniverse      
            
        Returns:
                          
        """
        if len(concept_names) <= 1:
            return concept_names
        
        #                
        waves = []
        for name in concept_names:
            if name in coordinate_map:
                coord = coordinate_map[name]
                wave = Wave(
                    frequency=coord.frequency,
                    amplitude=coord.depth if hasattr(coord, 'depth') else 0.5,
                    phase=(coord.frequency % 1000) / 1000 * 2 * math.pi,  #          
                    source=name,
                    confidence=coord.depth if hasattr(coord, 'depth') else 0.5
                )
                waves.append(wave)
        
        if not waves:
            return concept_names
        
        #      
        result = self.calculate_interference(waves)
        
        #              
        #      :            
        #      :            /  
        
        if result.interference_type == InterferenceType.DESTRUCTIVE:
            #        ,             
            strongest = max(result.original_waves, key=lambda w: w.amplitude)
            logger.warning(f"  Destructive interference detected. Dominant: {strongest.source}")
            return [strongest.source]
        
        elif result.interference_type == InterferenceType.CONSTRUCTIVE:
            #        ,         
            sorted_waves = sorted(result.original_waves, key=lambda w: w.confidence, reverse=True)
            logger.info(f"  Constructive interference. Enhanced resonance.")
            return [w.source for w in sorted_waves]
        
        else:
            #        ,         
            return concept_names
    
    @staticmethod
    def analyze_field_interference(nodes: Dict[str, Any]) -> Dict[str, Any]:
        """
                        
        
        Args:
            nodes: ResonanceField     
            
        Returns:
                    
        """
        if not nodes:
            return {"type": "void", "coherence": 0.0, "hotspots": []}
        
        #                  
        active_nodes = [n for n in nodes.values() if getattr(n, 'energy', 0) > 0.5]
        
        if not active_nodes:
            return {"type": "dormant", "coherence": 0.0, "hotspots": []}
        
        #          
        frequencies = [n.frequency for n in active_nodes]
        freq_variance = sum((f - sum(frequencies)/len(frequencies))**2 for f in frequencies) / len(frequencies)
        
        #          
        energies = [n.energy for n in active_nodes]
        total_energy = sum(energies)
        
        #     (       )   
        avg_energy = total_energy / len(energies)
        hotspots = [n.id for n in active_nodes if n.energy > avg_energy * 1.5]
        
        #        (      =       )
        coherence = 1.0 / (1.0 + freq_variance / 1000)
        
        #         
        if coherence > 0.8:
            interference_type = InterferenceType.CONSTRUCTIVE.value
        elif coherence < 0.3:
            interference_type = InterferenceType.DESTRUCTIVE.value
        else:
            interference_type = InterferenceType.MIXED.value
        
        return {
            "type": interference_type,
            "coherence": coherence,
            "hotspots": hotspots,
            "active_count": len(active_nodes),
            "total_energy": total_energy,
            "frequency_variance": freq_variance
        }


# =============          =============

def demo_interference():
    """         """
    print("=" * 60)
    print("  Wave Interference Engine Demo")
    print("=" * 60)
    
    engine = WaveInterference()
    
    # 1.          
    print("\n[1] Constructive Interference (     )")
    print("-" * 40)
    wave1 = Wave(frequency=440.0, amplitude=0.6, phase=0.0, source="A")
    wave2 = Wave(frequency=442.0, amplitude=0.5, phase=0.1, source="B")  #         
    
    result = engine.calculate_interference([wave1, wave2])
    print(f"   Input: Wave A (440Hz, amp=0.6) + Wave B (442Hz, amp=0.5)")
    print(f"   Result: {result.interference_type.value}")
    print(f"   Resultant: freq={result.resultant_wave.frequency:.1f}Hz, amp={result.resultant_wave.amplitude:.2f}")
    print(f"   Confidence: {result.confidence:.2f}")
    
    # 2.          
    print("\n[2] Destructive Interference (     )")
    print("-" * 40)
    wave3 = Wave(frequency=440.0, amplitude=0.6, phase=0.0, source="C")
    wave4 = Wave(frequency=440.0, amplitude=0.5, phase=math.pi, source="D")  #      
    
    result2 = engine.calculate_interference([wave3, wave4])
    print(f"   Input: Wave C (440Hz, phase=0) + Wave D (440Hz, phase= )")
    print(f"   Result: {result2.interference_type.value}")
    print(f"   Resultant: amp={result2.resultant_wave.amplitude:.2f}")
    print(f"   Uncertainty: {result2.uncertainty:.2f}")
    
    # 3.       
    print("\n[3] Convergence (  )")
    print("-" * 40)
    waves = [
        Wave(frequency=440.0, amplitude=0.8, phase=0.0, source="Note1"),
        Wave(frequency=550.0, amplitude=0.6, phase=0.5, source="Note2"),
        Wave(frequency=660.0, amplitude=0.4, phase=1.0, source="Note3"),
    ]
    
    converged = engine.converge(waves)
    print(f"   Input: 3 waves (440Hz, 550Hz, 660Hz)")
    print(f"   Converged: freq={converged.frequency:.1f}Hz, amp={converged.amplitude:.2f}")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        demo_interference()
    else:
        print("Usage: python wave_interference.py --demo")
        print("\nTo run demo, use: python wave_interference.py --demo")