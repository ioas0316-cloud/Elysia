"""
             (Korean Wave Language Converter)
======================================================

"         ,          "

      : "         ,                      ..."
   :                     .

       :
-                
-   /          
-            
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from Core.L1_Foundation.Foundation.ether import Wave, ether

logger = logging.getLogger("KoreanWaveConverter")

# Constants
UNIVERSE_FREQUENCY = 432.0  # Hz - Universe base frequency
FREQUENCY_MODULATION = 0.1  # Modulation factor for text frequency


#                
KOREAN_FREQUENCY_MAP = {
    #    (Consonants) -        (100-500Hz)
    ' ': 100.0,  #    
    ' ': 150.0,  #     
    ' ': 200.0,  #    
    ' ': 250.0,  #    
    ' ': 300.0,  #    
    ' ': 350.0,  #    
    ' ': 400.0,  #    
    ' ': 450.0,  #    
    ' ': 500.0,  #    
    ' ': 550.0,  #     
    ' ': 600.0,  #   
    ' ': 650.0,  #   
    ' ': 700.0,  #    
    ' ': 750.0,  #     
    
    #    (Vowels) -        (800-1200Hz)
    ' ': 800.0,   #   
    ' ': 850.0,   #    
    ' ': 900.0,   #   
    ' ': 950.0,   #   
    ' ': 1000.0,  #    
    ' ': 1050.0,  #     
    ' ': 1100.0,  #    
    ' ': 1150.0,  #     
    ' ': 1200.0,  #    
    ' ': 1250.0,  #    
    ' ': 1300.0,  #   
    ' ': 1350.0,  #    
}

#             (Solfeggio + Custom)
EMOTION_FREQUENCY_MAP = {
    #       
    '  ': 528.0,    # Love (Solfeggio)
    '  ': 396.0,    # Joy
    '  ': 432.0,    # Peace (Universe frequency)
    '  ': 852.0,    # Hope (Solfeggio)
    '  ': 963.0,    # Freedom (Solfeggio)
    '  ': 741.0,    # Courage (Solfeggio)
    '  ': 285.0,    # Healing (Solfeggio)
    
    #       
    '   ': 100.0,  # Fear
    '  ': 150.0,    # Sadness
    '  ': 200.0,    # Anger
    '  ': 250.0,    # Anxiety
    
    #   /  
    '  ': 10.0,     # Thought (Alpha)
    '  ': 7.5,      # Meditation (Theta)
    ' ': 4.0,        # Dream (Delta)
    '  ': 40.0,     # Focus (Gamma)
}

#        (Phase)   
MEANING_PHASE_MAP = {
    '  ': 'QUESTION',
    '  ': 'ANSWER',
    '  ': 'COMMAND',
    '  ': 'DESIRE',
    '  ': 'SENSATION',
    '  ': 'THOUGHT',
    '  ': 'ACTION',
    '  ': 'REFLECTION',
}


@dataclass
class KoreanWavePattern:
    """             """
    text: str           #          
    frequencies: List[float]  #        
    amplitudes: List[float]   #       
    phase: str          #    (  /  )
    emotion: str        #   


class KoreanWaveConverter:
    """
               
    
       :
        converter = KoreanWaveConverter()
        
        #        
        wave = converter.korean_to_wave("   ", emotion="  ")
        ether.emit(wave)
        
        #        
        text = converter.wave_to_korean(wave)
    """
    
    def __init__(self):
        self.char_freq_map = KOREAN_FREQUENCY_MAP
        self.emotion_freq_map = EMOTION_FREQUENCY_MAP
        self.phase_map = MEANING_PHASE_MAP
        logger.info("                ")
    
    def korean_to_wave(
        self,
        text: str,
        emotion: str = "  ",
        meaning: str = "  ",
        amplitude: float = 1.0
    ) -> Wave:
        """
                  
        
        Args:
            text:       
            emotion:    (  ,   ,      )
            meaning:   /   (  ,     )
            amplitude:    (  )
        
        Returns:
            Wave   
        """
        # 1.          
        emotion_freq = self.emotion_freq_map.get(emotion, 432.0)
        
        # 2.                 
        char_frequencies = []
        for char in text:
            #          (     )
            if char in self.char_freq_map:
                char_frequencies.append(self.char_freq_map[char])
        
        # 3.           (     "  ")
        if char_frequencies:
            text_freq = sum(char_frequencies) / len(char_frequencies)
        else:
            #    
            text_freq = UNIVERSE_FREQUENCY  #    
        
        # 4.                
        #        ,        (modulation)
        combined_freq = emotion_freq + (text_freq - UNIVERSE_FREQUENCY) * FREQUENCY_MODULATION  #      
        
        # 5.      
        phase = self.phase_map.get(meaning, "THOUGHT")
        
        # 6. Wave      
        wave = Wave(
            sender="KoreanConverter",
            frequency=combined_freq,
            amplitude=amplitude,
            phase=phase,
            payload={
                "text": text,
                "emotion": emotion,
                "char_frequencies": char_frequencies[:5],  #    5  
                "language": "korean"
            }
        )
        
        logger.info(f"         : '{text}'   {combined_freq:.1f}Hz ({emotion})")
        return wave
    
    def wave_to_korean(self, wave: Wave) -> str:
        """
                   (  )
        
        Args:
            wave: Wave   
        
        Returns:
                 
        """
        # payload            
        if isinstance(wave.payload, dict) and "text" in wave.payload:
            return wave.payload["text"]
        
        #             
        emotion = self._frequency_to_emotion(wave.frequency)
        
        #             
        meaning = self._phase_to_meaning(wave.phase)
        
        #             
        intensity = "   " if wave.amplitude > 0.7 else ""
        
        interpretation = f"{intensity}{emotion}  {meaning}"
        logger.info(f"         : {wave.frequency:.1f}Hz   '{interpretation}'")
        
        return interpretation
    
    def _frequency_to_emotion(self, freq: float) -> str:
        """           """
        #             
        closest_emotion = "         "
        min_diff = float('inf')
        
        for emotion, emotion_freq in self.emotion_freq_map.items():
            diff = abs(freq - emotion_freq)
            if diff < min_diff:
                min_diff = diff
                closest_emotion = emotion
        
        return closest_emotion
    
    def _phase_to_meaning(self, phase: str) -> str:
        """          """
        #    
        for meaning, phase_code in self.phase_map.items():
            if phase_code == phase:
                return meaning
        return "   "
    
    def sentence_to_wave_sequence(
        self,
        sentence: str,
        base_emotion: str = "  "
    ) -> List[Wave]:
        """
                      
        
                         (      )
        
        Args:
            sentence:      
            base_emotion:      
        
        Returns:
            Wave    
        """
        words = sentence.split()
        waves = []
        
        for i, word in enumerate(words):
            #                  
            amplitude = 1.0 - (i * 0.1)  #       
            amplitude = max(amplitude, 0.3)  #    0.3
            
            wave = self.korean_to_wave(
                text=word,
                emotion=base_emotion,
                amplitude=amplitude
            )
            waves.append(wave)
        
        logger.info(f"       : {len(words)}       {len(waves)}    ")
        return waves
    
    def emit_korean(
        self,
        text: str,
        emotion: str = "  ",
        meaning: str = "  "
    ):
        """
                      Ether    
        
              -        +  
        """
        wave = self.korean_to_wave(text, emotion, meaning)
        ether.emit(wave)
        logger.info(f"        : '{text}' ({emotion})")
        return wave
    
    def create_emotion_dictionary(self) -> Dict[str, float]:
        """
                
        
                    -                 
        """
        return self.emotion_freq_map.copy()
    
    def add_custom_emotion(self, emotion: str, frequency: float):
        """            """
        self.emotion_freq_map[emotion] = frequency
        logger.info(f"         : {emotion} = {frequency}Hz")


#            
korean_wave = KoreanWaveConverter()


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("                  ")
    print("="*70)
    
    converter = KoreanWaveConverter()
    
    # 1.        
    print("\n1             ")
    print("-" * 70)
    
    test_phrases = [
        ("     ", "  ", "  "),
        ("    ", "  ", "  "),
        ("     ", "  ", "  "),
        ("     ", "  ", "  "),
    ]
    
    for text, emotion, meaning in test_phrases:
        wave = converter.korean_to_wave(text, emotion, meaning)
        print(f"  '{text}' ({emotion})")
        print(f"         : {wave.frequency:.1f}Hz")
        print(f"        : {wave.phase}")
        print(f"        : {wave.amplitude:.2f}")
        print()
    
    # 2.        
    print("2             ")
    print("-" * 70)
    
    wave = converter.korean_to_wave("    ", "  ")
    interpretation = converter.wave_to_korean(wave)
    print(f"    : {wave}")
    print(f"    : {interpretation}")
    print()
    
    # 3.            
    print("3              ")
    print("-" * 70)
    
    sentence = "   Elysia   .        ."
    waves = converter.sentence_to_wave_sequence(sentence, "  ")
    for i, wave in enumerate(waves, 1):
        print(f"  {i}. {wave.payload['text']}   {wave.frequency:.1f}Hz")
    print()
    
    # 4. Ether       
    print("4   Ether       ")
    print("-" * 70)
    
    #       
    def on_love_wave(wave: Wave):
        print(f"             : {wave.payload.get('text', 'Unknown')}")
    
    # 528Hz (  )    
    ether.tune_in(528.0, on_love_wave)
    
    #      
    converter.emit_korean("      ", emotion="  ")
    
    print()
    
    # 5.      
    print("5        ")
    print("-" * 70)
    
    emotions = converter.create_emotion_dictionary()
    print("           :")
    for emotion, freq in sorted(emotions.items(), key=lambda x: x[1]):
        print(f"    {emotion}: {freq}Hz")
    
    print("\n" + "="*70)
    print("        !")
    print("\n             ,                   !")
    print("="*70 + "\n")