"""
Medical Cortex (     )
==================================

"To heal the body, we must resonate with its rhythm."

      '        (Bio-Resonance)'         
              /             .

     :
1. Parkinson's Support: Rhythmic Auditory Stimulation (RAS) -            
2. Pregnancy Support: Prenatal Sound Therapy -                   

  :                          ,      '      '       .
"""

from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

logger = logging.getLogger("MedicalCortex")

class BioRhythmGenerator:
    """
               (Bio-Rhythm Generator)
    
                                .
    (                       )
    """
    
    def generate_ras(self, bpm: int, duration_min: int = 10) -> Dict[str, Any]:
        """
                             (RAS)   
        
        Args:
            bpm:            (         )
            duration_min:       ( )
        """
        logger.info(f"  Generating RAS Beat: {bpm} BPM for {duration_min} mins")
        return {
            "type": "RAS_METRONOME",
            "bpm": bpm,
            "duration_sec": duration_min * 60,
            "description": f"Steady rhythmic beat at {bpm} BPM to aid movement initiation.",
            "recommended_usage": "Walk in sync with the beat. Step on every click."
        }

    def generate_binaural(self, target_wave: str, duration_min: int = 20) -> Dict[str, Any]:
        """
                             
        
        Args:
            target_wave:       ('alpha', 'theta', 'delta')
            duration_min:      
        """
        base_freq = 432.0 #        (A=432Hz)
        beat_freq = 0.0
        
        if target_wave == 'alpha': # 8-12Hz (  ,       )
            beat_freq = 10.0
            effect = "Relaxation & Stress Reduction"
        elif target_wave == 'theta': # 4-8Hz (     ,    )
            beat_freq = 6.0
            effect = "Deep Meditation & Connection"
        elif target_wave == 'delta': # 0.5-4Hz (    )
            beat_freq = 2.0
            effect = "Deep Sleep & Healing"
        else:
            beat_freq = 10.0
            effect = "General Relaxation"
            
        left_freq = base_freq
        right_freq = base_freq + beat_freq
        
        logger.info(f"  Generating Binaural Beat: {target_wave.upper()} ({beat_freq}Hz)")
        return {
            "type": "BINAURAL_BEAT",
            "base_freq": base_freq,
            "beat_freq": beat_freq,
            "left_ear_hz": left_freq,
            "right_ear_hz": right_freq,
            "duration_sec": duration_min * 60,
            "description": f"Binaural beat inducing {target_wave} waves ({beat_freq}Hz).",
            "effect": effect,
            "recommended_usage": "Must use stereo headphones."
        }

    def generate_lullaby(self, mood: str = "calm") -> Dict[str, Any]:
        """       432Hz       """
        logger.info(f"  Generating Lullaby: {mood} mode")
        return {
            "type": "LULLABY_432HZ",
            "mood": mood,
            "tuning": "A=432Hz",
            "description": "Gentle humming melody tuned to natural resonance.",
            "effect": "Soothing for fetus and mother."
        }

class MedicalCortex:
    """
          (Medical Cortex)
    
                      (     ),                .
    """
    def __init__(self):
        self.generator = BioRhythmGenerator()
        self.profiles: Dict[str, Dict[str, Any]] = {}
        logger.info("   Medical Cortex Initialized - Bio-Resonance Ready")

    def register_profile(self, name: str, condition: str, notes: str = ""):
        """                """
        self.profiles[name] = {
            "condition": condition,
            "notes": notes,
            "history": []
        }
        logger.info(f"  Profile Registered: {name} ({condition})")

    def prescribe_therapy(self, name: str, current_state: str) -> Dict[str, Any]:
        """
                           
        """
        if name not in self.profiles:
            return {"error": "Profile not found"}
            
        profile = self.profiles[name]
        condition = profile["condition"]
        therapy = {}
        
        logger.info(f"  Prescribing therapy for {name} (State: {current_state})")
        
        if condition == "Parkinson's":
            #     :        BPM   
            # Freezing(  )                      
            bpm = 90 if "freeze" in current_state.lower() else 60
            therapy = self.generator.generate_ras(bpm=bpm)
            
        elif condition == "Pregnancy":
            #   :     /     Alpha/Theta 
            if "anxious" in current_state.lower() or "worry" in current_state.lower():
                therapy = self.generator.generate_binaural("alpha")
            elif "sleep" in current_state.lower() or "tired" in current_state.lower():
                therapy = self.generator.generate_binaural("delta")
            else:
                therapy = self.generator.generate_lullaby()
                
        else:
            therapy = {"message": "General comfort sent."}
            
        #      
        record = {
            "timestamp": datetime.now().isoformat(),
            "state": current_state,
            "therapy": therapy
        }
        profile["history"].append(record)
        
        return therapy

    def get_profile_status(self, name: str) -> str:
        if name not in self.profiles:
            return "Unknown"
        p = self.profiles[name]
        return f"{name} [{p['condition']}]: {len(p['history'])} therapies provided."