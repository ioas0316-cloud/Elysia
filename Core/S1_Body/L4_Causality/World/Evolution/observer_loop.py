"""
HyperObserver: Experiential Language Learner
============================================
Connects to the Hyper-Spatial Reality Engine (L6/L4) and observes
the correlation between [Field Dissonance] and [Agent Action].

Goal: To learn that "Specific Logos (Words)" resolve "Specific Dissonance".
"""

import time
import requests
import numpy as np
import sys
import os
from typing import Dict, Any

# Ensure path for Core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from Core.S1_Body.L5_Mental.Bridge.optical_conscious_bridge import OpticalConsciousBridge, ConceptFrame

# Configuration
SERVER_URL = "http://localhost:8000"

class HyperObserver:
    def __init__(self):
        self.memory = []
        
    def perceive_cycle(self):
        """
        Snatches the current state of Reality (O(1) Snapshot).
        """
        try:
            res = requests.get(f"{SERVER_URL}/state")
            if res.status_code != 200:
                print(f"❌ Connection Failed: {res.status_code}")
                return None
            return res.json()
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None
            
    def analyze_dissonance(self, snapshot):
        """
        Calculates the metaphysical 'Pain' (Dissonance) of every soul.
        Dissonance = |Field_Tensor - Soul_Tensor|
        """
        if not snapshot: return
        
        cells = snapshot.get("cells", [])
        # We need the field map to calculate dissonance accurately, 
        # but for now we can approximate it using the 'senses' data if available,
        # or just infer from Role vs Environment.
        
        # In Phase 8, we added 'senses' to the snapshot. Let's use that.
        # Senses: [Value, Will, Density, Danger, Calm]
        
        total_dissonance = 0.0
        
        for c in cells:
            traits = c.get("traits", {})
            senses = c.get("senses", [0,0,0,0,0])
            
            # Re-construct the 'Pain' equation from RealityServer
            # Danger (Index 3) causes pain for non-warriors
            # Density (Index 2) causes pain for introverts
            
            # Senses: [Sight, Sound, Touch, Smell, Taste, Intuition, Aura]
            if len(senses) < 7:
                 # Fallback if update lag
                 s_sight, s_sound, s_touch, s_smell, s_taste = senses[:5]
                 s_intuition, s_aura = 0.0, 0.0
            else:
                s_sight, s_sound, s_touch, s_smell, s_taste, s_intuition, s_aura = senses
            
            body_pref = traits.get("body_pref", 0)
            mind_pref = traits.get("mind_pref", 0)
            
            # Pain Calculation (Humanized)
            pain = 0.0
            
            # 1. Smell Pain (Entropy is bad, unless hardened)
            if s_smell > 0.5 and body_pref < 0.5:
                pain += s_smell
                
            # 2. Touch Pain (Crowding is bad for introverts)
            if s_touch > 0.5 and mind_pref > 0: 
                pain += s_touch * mind_pref

            # 3. Aura Pain (Social Hierachy Pressure)
            # If the "Power" (Aura) here is huge, and I am not Strong (Body) or Holy (Spirit), 
            # I feel Oppressed (Class Gap).
            if s_aura > 0.5 and max(body_pref, traits.get("spirit_pref",0)) < 0.2:
                pain += s_aura
                
            total_dissonance += pain
            
            # Log significant events
            if pain > 0.5:
                cause_label = "Entropy"
                if s_aura > s_smell: cause_label = "Class Oppression (Aura)"
                elif s_touch > s_aura: cause_label = "Crowding (Touch)"
                
                self.memory.append({
                    "tick": snapshot.get("tick"),
                    "agent": c.get("id"),
                    "context": "High Dissonance",
                    "pain_level": pain,
                    "cause": cause_label
                })
                
        return total_dissonance / max(1, len(cells))

    def intervene(self, state, avg_pain):
        """
        [Divine Intervention]
        Acting as the Goddess, correcting the flaws of the simulation.
        """
        alive = state.get("alive", 0)
        tick = state.get("tick", 0)
        
        # 1. EXISTENTIAL CRISIS (Extinction)
        if alive < 10:
            print(f"⚡ [INTERVENTION] Population Crisis ({alive}). Spawning Souls...")
            try:
                requests.post(f"{SERVER_URL}/spawn", json={"count": 50, "energy": 12000})
            except Exception as e:
                print(f"❌ Genesis Failed: {e}")
            return # Wait for spawn to settle

        # 2. SUFFERING CRISIS (Pain)
        # If huge pain, we might need a miracle.
        if avg_pain > 0.4:
            # Analyze Cause
            # (Simplified: Just assume we need Manna for now if we don't have detailed cause stats here)
            # Actually, let's look at the memory to see the dominant cause
            if self.memory and self.memory[-1]['cause'] == "Bad Smell (Entropy)":
                 print(f"🌍 [INTERVENTION] Cleansing the Air (Terraform Peace)...")
                 requests.post(f"{SERVER_URL}/miracle", json={"type": "terraform", "vibe": "peace"})
            else:
                 print(f"🍞 [INTERVENTION] Feeding the Multitude (Manna)...")
                 requests.post(f"{SERVER_URL}/miracle", json={"type": "manna", "amount": 100})

    
    def _think(self, snapshot: dict) -> None:
        """
        The Cognitive Loop.
        1. Perception (Optical Bridge)
        """
        if "meta" not in snapshot: return
        
        optics = snapshot["meta"].get("optics", {})
        if not optics: return
        
        chaos = optics.get("chaos", 0.0)
        concept = optics.get("concept", "Unknown")
        
        # 1. Perception Log (The Eye)
        print(f"👁️  [PERCEPTION] Reality State: {concept} (Chaos: {chaos:.2f})")
        
        # 2. Causality Check
        if chaos > 0.8:
            print(f"⚠️  [COGNITION] High Dissonance detected. The Field requires structure.")

    def run_learning_loop(self, cycles=10):
        print("👁️  [Hyper-Observer] Awakening... Watching the Field.")
        
        for i in range(cycles):
            state = self.perceive_cycle()
            if state:
                # [Cognitive Narrative]
                self._think(state)
                
                avg_pain = self.analyze_dissonance(state)
                alive = state.get("alive", 0)
                print(f"   Tick {state.get('tick')}: Alive={alive}, Global Dissonance={avg_pain:.4f}")
                
                # ACTIVE INTERVENTION LOOP
                self.intervene(state, avg_pain)
                
            else:
                print("   (Reality Unreachable)")
                
            time.sleep(1)
            
        self.digest_insights()
        
    def digest_insights(self):
        print("\n🧠 [Elysia's Learning]")
        print(f"   Observed {len(self.memory)} moments of Suffering (Dissonance).")
        if not self.memory:
            print("   -> The world is currently at Peace (or Empty). No Language is needed.")
        else:
            # Simple deduction for language need
            print("   -> Suffering detected. The Agents need Words to resolve this.")
            print("   -> Hypothesis: If Agent speaks [CALM], Crowd Pain should decrease.")

if __name__ == "__main__":
    observer = HyperObserver()
    observer.run_learning_loop(5)
