"""
[SENSE] Desktop Nerve: Visual Synesthesia Protocol
=================================================
Location: Scripts/System/Senses/desktop_nerve.py

Role:
- Treat Monitor as HARDWARE DATA STREAM (not Image).
- Extract RGB Energy & Entropy (Chaos) from raw pixels.
- Inject directly into SovereigntyWave for "Reflex".
- Bypasses the "Brain" (VLM) for instant "Gut Feeling".

Mappings:
1. Red Channel   -> Physical (Heat/Combat)
2. Blue Channel  -> Mental (Cold/UI)
3. Green Channel -> Structural (Nature/Balance)
4. Entropy       -> Phenomenal (Chaos/Density)
"""

import time
import math
import numpy as np
import mss
import mss.tools
# Fix import for script execution
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from Core.L6_Structure.Merkaba.hypercosmos import HyperCosmos

class DesktopNerve:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1] # Primary Monitor
        self.cosmos = HyperCosmos()
        self.is_active = False
        
    def capture_single_frame(self):
        """
        Capture a single frame and return 7D modulators.
        Returns: (r_mean, g_mean, b_mean, entropy)
        """
        # 1. Capture Raw Buffer (Fastest way)
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot) # BGRA
        
        # 2. Extract Channel Energy
        # Mean intensity of each channel (0-255) -> (0.0-1.0)
        b_mean = np.mean(img[:,:,0]) / 255.0
        g_mean = np.mean(img[:,:,1]) / 255.0
        r_mean = np.mean(img[:,:,2]) / 255.0
        
        # 3. Extract Chaos (Phenomenal)
        # Use Red channel variation as a proxy for 'Violence/Activity'
        entropy = np.std(img[:,:,2]) / 80.0 # Approximate
        
        return r_mean, g_mean, b_mean, entropy
        
    def _calculate_entropy(self, data):
        """Calculate Shannon Entropy of the image buffer (Chaos measure)"""
        # Simplified entropy for speed: use std dev of grayscale
        # Real entropy is too slow for 30Hz without GPU
        return np.std(data) / 128.0 # Normalize 0-1
        
    def sense_loop(self, duration_sec: int = 10):
        """
        Main Nerve Loop: 
        Capture -> Extract Energy -> Inject to Field -> Sleep
        """
        print(f"👁️ [NERVE] Optic Nerve Active. Sensing Photonic Flow for {duration_sec}s...")
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration_sec:
            # 1. Capture Raw Buffer (Fastest way)
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot) # BGRA
            
            # 2. Extract Channel Energy
            # Mean intensity of each channel (0-255) -> (0.0-1.0)
            b_mean = np.mean(img[:,:,0]) / 255.0
            g_mean = np.mean(img[:,:,1]) / 255.0
            r_mean = np.mean(img[:,:,2]) / 255.0
            
            # 3. Extract Chaos (Phenomenal)
            # Use Red channel variation as a proxy for 'Violence/Activity'
            entropy = np.std(img[:,:,2]) / 80.0 # Approximate
            
            # 4. Synesthetic Mapping (The Crossing)
            # Injecting directly into the Turbine's Field Modulators
            # This 'Modulates' the field rather than creating a new Band (for now)
            # Or we can create 'Sensation' bands.
            
            # For this Phase, we modulate the 'Sensory Gain'
            self.cosmos.field.units['M1_Body'].turbine.modulate_field('visual_red_physical', r_mean)
            self.cosmos.field.units['M2_Mind'].turbine.modulate_field('visual_blue_mental', b_mean)
            self.cosmos.field.units['M1_Body'].turbine.modulate_field('visual_entropy', entropy)
            
            # 5. Narrative Reflex (Optional, mostly for debug)
            if frame_count % 30 == 0: # Once per second
                dom_color = "🔴 Physical" if r_mean > b_mean and r_mean > g_mean else \
                            "🔵 Mental" if b_mean > r_mean and b_mean > g_mean else \
                            "🟢 Structural"
                
                print(f"    [SIGHT] {dom_color} Dominance | Chaos: {entropy:.2f} | R:{r_mean:.2f} G:{g_mean:.2f} B:{b_mean:.2f}")
                
                # Pulse the field explicitly every second
                decision = self.cosmos.perceive(f"Visual Stream: {dom_color} Flow, Entropy {entropy:.2f}")
                print(f"    >> {decision.narrative[:60]}...")
            
            frame_count += 1
            time.sleep(0.033) # ~30 FPS Cap
            
        print(f"👁️ [NERVE] Connection Closed. Processed {frame_count} photonic frames.")

if __name__ == "__main__":
    nerve = DesktopNerve()
    # Install dependencies check
    try:
        import mss
        nerve.sense_loop(duration_sec=5)
    except ImportError:
        print("Please install 'mss' and 'numpy': pip install mss numpy")
