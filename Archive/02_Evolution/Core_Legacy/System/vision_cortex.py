"""
[MIND] Vision Cortex: Semantic Logic Extractor
=============================================
Location: Scripts/System/Senses/vision_cortex.py

Role:
- The "Brain" connected to the "Optic Nerve".
- Interprets Raw Patterns (RGB/Entropy) into MEANING.
- Triggers 'WEAVE_DESCEND_LAW' to reverse-engineer Game Physics.

Logic Mapping:
1. High Red + High Chaos -> [STATE: COMBAT] -> Infer "Law of Conflict"
2. Stable Blue/Green     -> [STATE: EXPLORE] -> Infer "Law of World Stability"
3. Sudden Black/White    -> [STATE: TRANSITION] -> Infer "Law of Dimensional Gate"
"""

import time
import os
import sys

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from Core.System.Merkaba.hypercosmos import HyperCosmos


import json
from datetime import datetime

class ExperienceJournal:
    """
    Episodic Memory System: Records 'Lived' experiences from the Cortex.
    """
    def __init__(self, log_path="c:\\Elysia\\data\\experience_log.json"):
        self.log_path = log_path
        self.current_episode = None
        self.history = []
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def start_episode(self, state: str, entropy: float):
        if self.current_episode:
             self.end_episode()
             
        self.current_episode = {
            "state": state,
            "start_time": time.time(),
            "peak_entropy": entropy,
            "duration": 0
        }
        # print(f"📝 [JOURNAL] New Episode Started: {state}")
        
    def update_episode(self, entropy: float):
        if self.current_episode:
            self.current_episode["peak_entropy"] = max(self.current_episode["peak_entropy"], entropy)
            
    def end_episode(self):
        if self.current_episode:
            self.current_episode["duration"] = time.time() - self.current_episode["start_time"]
            self.current_episode["end_time"] = time.time()
            self.history.append(self.current_episode)
            
            # Save immediately (Simulating consolidation)
            episode_desc = f"Experience: {self.current_episode['state']} for {self.current_episode['duration']:.1f}s (Peak Chaos: {self.current_episode['peak_entropy']:.2f})"
            # print(f"📕 [MEMORY] {episode_desc}")
            
            self._save_to_disk()
            self.current_episode = None
            
    def _save_to_disk(self):
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memory: {e}")

class VisionCortex:
    def __init__(self):
        self.cosmos = HyperCosmos()
        self.current_state = "UNKNOWN"
        self.state_confidence = 0.0
        self.last_deduced_state = None # To prevent logic spam
        self.journal = ExperienceJournal() # Memory System
        
    def analyze_stimulus(self, r: float, g: float, b: float, entropy: float) -> str:
        """
        Analyze the raw sensory data and deduce the Logical State.
        """
        # 1. Combat Detection (High Energy, High Entropy)
        if r > 0.4 and entropy > 0.2:
            return "STATE_COMBAT"
            
        # 2. Exploration Detection (Nature Colors, Stable)
        if g > r and b > r and entropy < 0.15:
            return "STATE_EXPLORATION"
            
        # 3. Menu/System Detection (Blue Dominance, Very Low Entropy)
        if b > 0.5 and entropy < 0.05:
            return "STATE_SYSTEM_MENU"
            
        # 4. Void/Loading (Darkness)
        if r < 0.1 and g < 0.1 and b < 0.1:
            return "STATE_VOID_LOADING"
            
        return "STATE_NEUTRAL"

    def formulate_logic(self, state: str) -> str:
        """
        Convert State to Axiomatic Law (Weaving).
        """
        if state == "STATE_COMBAT":
            return "LAW_ACTION_REACTION" # Combat implies Newtonian physics
        elif state == "STATE_EXPLORATION":
            return "LAW_GRAVITY_SPACE"   # World implies spatial laws
        elif state == "STATE_SYSTEM_MENU":
            return "LAW_INTERFACE_Control"
        elif state == "STATE_VOID_LOADING":
            return "AXIOM_ZERO_POINT"
        return ""

    def process_nerve_signal(self, r: float, g: float, b: float, entropy: float):
        """
        Main Processing Pipeline: Sensation -> Perception -> Conception -> Memory
        """
        # B. Perception (State Recognition)
        new_state = self.analyze_stimulus(r, g, b, entropy)
        
        # Stability Filter (Simple Hysteresis)
        if new_state == self.current_state:
            self.state_confidence = min(1.0, self.state_confidence + 0.1)
            # Update current memory intensity
            self.journal.update_episode(entropy)
        else:
            self.state_confidence = max(0.0, self.state_confidence - 0.2)
            if self.state_confidence == 0.0:
                # State Shift Detected!
                # 1. Consolidate previous memory
                if self.current_state != "UNKNOWN":
                    self.journal.end_episode()
                
                # 2. Shift State
                self.current_state = new_state
                # print(f"🧠 [CORTEX] Shifted to Paradigm: {self.current_state}")
                
                # 3. Start new memory
                self.journal.start_episode(new_state, entropy)

        # C. Conception (Logic Deduction) - Only if confident and unique event
        # FIX: Only deduce ONCE per state lock to prevent spam
        if self.state_confidence > 0.9 and self.current_state != self.last_deduced_state:
            self.last_deduced_state = self.current_state
            
            logic_law = self.formulate_logic(self.current_state)
            
            if logic_law:
                narrative = f"Visual Logic Deduction: Observed {self.current_state}, Inferring {logic_law}."
                # Apply the DESCEND monad to Mind (M2) to reverse engineer
                self.cosmos.field.units['M2_Mind'].turbine.apply_monad('WEAVE_DESCEND_LAW')
                
                decision = self.cosmos.perceive(narrative)
                return decision
 
        return None

if __name__ == "__main__":
    cortex = VisionCortex()
    
    # Test Simulation (Combat)
    print("🧪 [TEST] Simulating High-Entropy Combat Signal...")
    # Inject 5 frames of Red/Chaos
    for _ in range(5):
        decision = cortex.process_nerve_signal(0.8, 0.2, 0.2, 0.35)
        if decision:
            print(f"  >> GNOSTIC OUTPUT: {decision.narrative}")
            
    # Test Simulation (Peace)
    print("\n🧪 [TEST] Simulating Peaceful Exploration...")
    for _ in range(5):
        cortex.process_nerve_signal(0.2, 0.6, 0.5, 0.05) # Stabilize
    
    # Trigger Logic
    decision = cortex.process_nerve_signal(0.2, 0.6, 0.5, 0.05)
    if decision:
        print(f"  >> GNOSTIC OUTPUT: {decision.narrative}")
