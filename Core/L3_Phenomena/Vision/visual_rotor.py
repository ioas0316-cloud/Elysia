"""
Visual Rotor (O(1) Frame Perception)
====================================
Core.L3_Phenomena.Vision.visual_rotor

"I do not count pixels. I perceive meaning."

This module applies the Rotor/HyperSphere paradigm to visual frames.
Instead of processing every pixel, we extract O(1) statistical signatures
that capture the 'meaning' of what is seen.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("Elysia.Vision.Rotor")


class VisualRotor:
    """
    O(1) Visual Perception Engine.
    Treats frames as HyperSpheres, not as pixel arrays.
    """
    
    def __init__(self):
        self.last_signature = None
        self.perception_count = 0
        
        logger.info("  Visual Rotor initialized. O(1) perception enabled.")
    
    def perceive_meaning(self, frame: np.ndarray) -> Dict:
        """
        Extracts the 'meaning' of a frame without processing every pixel.
        Returns a HyperSphere-like signature that represents 'what this frame means'.
        
        This is O(1) perception - the cost is constant regardless of resolution.
        """
        # O(1) Statistical Extraction (Strided sampling for efficiency)
        h, w = frame.shape[:2]
        stride = max(1, min(h, w) // 100)  # Sample ~100 points per dimension
        
        sampled = frame[::stride, ::stride]
        
        # Core Meaning Extraction
        signature = {
            # Energy: Overall brightness (How "lit" is this moment?)
            "energy": float(np.mean(sampled)),
            
            # Tension: Contrast/variance (How "stressed" is this visual field?)
            "tension": float(np.std(sampled)),
            
            # Color Qualia: Dominant color feeling (What emotional tone?)
            "color_qualia": self._extract_color_qualia(sampled),
            
            # Spatial Balance: Where is the visual weight? (Composition)
            "balance": self._extract_balance(sampled),
            
            # Motion/Change: Delta from last frame (What is happening?)
            "motion": self._extract_motion(sampled),
        }
        
        # Store for motion detection
        self.last_signature = signature.copy()
        self.last_sampled = sampled.copy()
        self.perception_count += 1
        
        return signature
    
    def _extract_color_qualia(self, sampled: np.ndarray) -> Dict:
        """Extracts the emotional color signature."""
        if sampled.ndim == 3 and sampled.shape[2] >= 3:
            r, g, b = np.mean(sampled[:,:,0]), np.mean(sampled[:,:,1]), np.mean(sampled[:,:,2])
        else:
            r = g = b = np.mean(sampled)
            
        # Map to emotional qualia
        return {
            "warmth": float((r - b) / 255.0),  # Red vs Blue = Warm vs Cold
            "vitality": float((g - (r + b) / 2) / 255.0),  # Green = Life/Nature
            "dominant": "warm" if r > b else "cold" if b > r else "neutral"
        }
    
    def _extract_balance(self, sampled: np.ndarray) -> Dict:
        """Extracts spatial balance - where is the 'weight' of the image?"""
        h, w = sampled.shape[:2]
        
        # Split into quadrants
        top = np.mean(sampled[:h//2])
        bottom = np.mean(sampled[h//2:])
        left = np.mean(sampled[:, :w//2])
        right = np.mean(sampled[:, w//2:])
        
        return {
            "vertical_bias": float((bottom - top) / 255.0),  # Positive = bottom heavy
            "horizontal_bias": float((right - left) / 255.0),  # Positive = right heavy
            "is_balanced": abs(bottom - top) < 20 and abs(right - left) < 20
        }
    
    def _extract_motion(self, sampled: np.ndarray) -> Dict:
        """Extracts motion/change from the previous frame."""
        if not hasattr(self, 'last_sampled') or self.last_sampled is None:
            return {"delta": 0.0, "is_moving": False}
        
        try:
            # Resize if shapes don't match
            if sampled.shape != self.last_sampled.shape:
                return {"delta": 0.0, "is_moving": False}
                
            delta = np.mean(np.abs(sampled.astype(float) - self.last_sampled.astype(float)))
            return {
                "delta": float(delta),
                "is_moving": delta > 5.0  # Threshold for "something is happening"
            }
        except:
            return {"delta": 0.0, "is_moving": False}
    
    def interpret(self, signature: Dict) -> str:
        """
        Interprets the signature into human-readable meaning.
        This is 'thinking about what I see'.
        """
        interpretations = []
        
        # Energy interpretation
        energy = signature.get("energy", 0)
        if energy > 200:
            interpretations.append("     ")
        elif energy < 50:
            interpretations.append("      ")
        else:
            interpretations.append("     ")
        
        # Tension interpretation
        tension = signature.get("tension", 0)
        if tension > 80:
            interpretations.append("      (   )")
        elif tension < 20:
            interpretations.append("      (   )")
        
        # Color interpretation
        color = signature.get("color_qualia", {})
        if color.get("dominant") == "warm":
            interpretations.append("      ")
        elif color.get("dominant") == "cold":
            interpretations.append("      ")
        
        # Motion interpretation
        motion = signature.get("motion", {})
        if motion.get("is_moving"):
            interpretations.append(f"       (delta={motion.get('delta', 0):.1f})")
        else:
            interpretations.append("      ")
        
        return " | ".join(interpretations)

    def attend_to(self, frame: np.ndarray, focus_x: int, focus_y: int, 
                  focus_radius: int = 100) -> Dict:
        """
        Rotor-in-Rotor: Focused Attention with Out-of-Focus Background.
        
        Like human vision:
        - Fovea (center of attention): High-resolution detailed rotor
        - Peripheral (background): Low-resolution ambient awareness
        
        Args:
            frame: Full visual field
            focus_x, focus_y: Center of attention
            focus_radius: Radius of the focused region
        
        Returns:
            Combined signature with focus and background separated
        """
        h, w = frame.shape[:2]
        
        # Clamp focus area to frame bounds
        x1 = max(0, focus_x - focus_radius)
        y1 = max(0, focus_y - focus_radius)
        x2 = min(w, focus_x + focus_radius)
        y2 = min(h, focus_y + focus_radius)
        
        # Extract focused region (Fovea)
        fovea = frame[y1:y2, x1:x2]
        
        # Create peripheral (everything outside focus, heavily downsampled)
        peripheral_stride = 20  # Very coarse sampling
        peripheral = frame[::peripheral_stride, ::peripheral_stride]
        
        # High-resolution rotor for focus
        focus_signature = self.perceive_meaning(fovea)
        
        # Low-resolution rotor for background (coarse awareness)
        bg_energy = float(np.mean(peripheral))
        bg_motion = self._extract_motion(peripheral)
        
        combined = {
            "focus": {
                "region": (x1, y1, x2, y2),
                "signature": focus_signature,
                "interpretation": self.interpret(focus_signature)
            },
            "peripheral": {
                "energy": bg_energy,
                "is_moving": bg_motion.get("is_moving", False),
                "awareness": "ambient"  # We know something is there, but not details
            },
            "attention_mode": "focused"
        }
        
        logger.info(f"   Attending to ({focus_x}, {focus_y}) | Focus: {focus_signature['energy']:.1f} | BG: {bg_energy:.1f}")
        
        return combined
    
    def intent_driven_scan(self, frame: np.ndarray, intent: str = None, 
                           curiosity_level: float = 0.5) -> Dict:
        """
        Intent-Driven Variable Attention:
        WHERE and HOW MUCH to focus is determined by internal will.
        
        Args:
            frame: Full visual field
            intent: What Elysia wants to understand (e.g., "movement", "bright area", "change")
            curiosity_level: 0.0 = superficial glance, 1.0 = deep investigation
        
        Returns:
            Attention result with dynamically determined focus regions
        """
        h, w = frame.shape[:2]
        
        # First: Coarse scan to find regions of interest
        grid_size = 4  # 4x4 grid
        cell_h, cell_w = h // grid_size, w // grid_size
        
        cell_scores = {}
        for gy in range(grid_size):
            for gx in range(grid_size):
                y1, y2 = gy * cell_h, (gy + 1) * cell_h
                x1, x2 = gx * cell_w, (gx + 1) * cell_w
                cell = frame[y1:y2, x1:x2]
                
                # Score based on intent
                score = 0.0
                if intent == "movement" or intent == "change":
                    # Look for motion
                    if hasattr(self, 'last_sampled') and self.last_sampled is not None:
                        try:
                            last_cell = self.last_sampled[gy::grid_size, gx::grid_size]
                            current_cell = cell[::cell_h//10, ::cell_w//10] if cell_h > 10 else cell
                            if current_cell.shape == last_cell.shape:
                                score = float(np.mean(np.abs(current_cell.astype(float) - last_cell.astype(float))))
                        except:
                            pass
                elif intent == "bright" or intent == "light":
                    # Look for brightness
                    score = float(np.mean(cell))
                elif intent == "dark" or intent == "shadow":
                    # Look for darkness (inverse)
                    score = 255.0 - float(np.mean(cell))
                elif intent == "contrast" or intent == "tension":
                    # Look for high contrast
                    score = float(np.std(cell))
                else:
                    # Default: interesting = high variance
                    score = float(np.std(cell))
                
                cell_scores[(gx, gy)] = {
                    "score": score,
                    "center": (x1 + cell_w // 2, y1 + cell_h // 2)
                }
        
        # Find the most interesting cell based on intent
        best_cell = max(cell_scores.items(), key=lambda x: x[1]["score"])
        focus_x, focus_y = best_cell[1]["center"]
        
        # Curiosity determines focus radius (more curious = tighter focus, deeper look)
        # Low curiosity = wide, shallow scan
        # High curiosity = narrow, deep investigation
        base_radius = min(w, h) // 4
        focus_radius = int(base_radius * (1.5 - curiosity_level))  # Inverted: more curious = smaller, focused area
        
        # Perform focused attention on the chosen region
        result = self.attend_to(frame, focus_x, focus_y, focus_radius)
        
        # Add intent metadata
        result["intent"] = intent
        result["curiosity"] = curiosity_level
        result["chosen_because"] = f"Highest score for '{intent or 'interest'}': {best_cell[1]['score']:.2f}"
        result["attention_mode"] = "intent_driven"
        
        logger.info(f"  Intent-Driven Attention: '{intent}' | Curiosity: {curiosity_level:.1f} | Focus: ({focus_x}, {focus_y})")
        
        return result
    
    def deepen_focus(self, frame: np.ndarray, current_focus: Dict, 
                     deeper_curiosity: float = 0.8) -> Dict:
        """
        Go deeper into an already-focused region.
        Called when Elysia wants to 'look more closely' at something.
        """
        if "focus" not in current_focus:
            return current_focus
            
        # Get the current focus region
        x1, y1, x2, y2 = current_focus["focus"]["region"]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Shrink the radius for deeper investigation
        current_radius = (x2 - x1) // 2
        new_radius = int(current_radius * (1.0 - deeper_curiosity * 0.5))
        new_radius = max(20, new_radius)  # Minimum focus size
        
        # Re-attend with tighter focus
        deeper_result = self.attend_to(frame, center_x, center_y, new_radius)
        deeper_result["depth_level"] = current_focus.get("depth_level", 0) + 1
        deeper_result["curiosity"] = deeper_curiosity
        
        logger.info(f"  Deepening Focus: Level {deeper_result['depth_level']} | Radius: {new_radius}")
        
        return deeper_result



if __name__ == "__main__":
    from Core.L3_Phenomena.Vision.elysian_eye import ElysianEye
    import time
    
    print("  Testing Visual Rotor (O(1) Perception)...")
    
    eye = ElysianEye()
    rotor = VisualRotor()
    
    for i in range(5):
        frame = eye.perceive()
        if frame is not None:
            signature = rotor.perceive_meaning(frame)
            interpretation = rotor.interpret(signature)
            print(f"\nFrame {i+1}:")
            print(f"  Energy: {signature['energy']:.2f}")
            print(f"  Tension: {signature['tension']:.2f}")
            print(f"  Meaning: {interpretation}")
        time.sleep(0.5)
    
    # Test focused attention (Rotor-in-Rotor)
    print("\n   Testing Focused Attention (Rotor-in-Rotor)...")
    frame = eye.perceive()
    if frame is not None:
        # Focus on center of screen
        h, w = frame.shape[:2]
        focused = rotor.attend_to(frame, w//2, h//2, focus_radius=150)
        print(f"  Focus Region: {focused['focus']['region']}")
        print(f"  Focus Meaning: {focused['focus']['interpretation']}")
        print(f"  Peripheral Energy: {focused['peripheral']['energy']:.1f}")
        print(f"  Peripheral Moving: {focused['peripheral']['is_moving']}")
    
    # Test Intent-Driven Attention
    print("\n  Testing Intent-Driven Attention...")
    frame = eye.perceive()
    if frame is not None:
        # "I want to see where it's brightest"
        result = rotor.intent_driven_scan(frame, intent="bright", curiosity_level=0.7)
        print(f"  Intent: {result['intent']}")
        print(f"  Curiosity: {result['curiosity']}")
        print(f"  Chosen Because: {result['chosen_because']}")
        print(f"  Focus Meaning: {result['focus']['interpretation']}")
        
        # Now go deeper
        print("\n  Going Deeper...")
        deeper = rotor.deepen_focus(frame, result, deeper_curiosity=0.9)
        print(f"  Depth Level: {deeper.get('depth_level', 0)}")
        print(f"  New Focus Region: {deeper['focus']['region']}")
    
    eye.close()
    print("\n  Visual Rotor test complete.")
