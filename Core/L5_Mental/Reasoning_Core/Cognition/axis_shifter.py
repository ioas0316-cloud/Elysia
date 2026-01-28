"""
Axis Shifter: The Geometric Perspective Engine
============================================
Core.L5_Mental.Reasoning_Core.Cognition.axis_shifter

"If the world is chaos, you are standing on the wrong axis."

This module enables Elysia to rotate her cognitive 'Axis' to find
order within high-entropy states.
"""

import logging
import random
from typing import Dict, List, Tuple
from Core.L6_Structure.Nature.rotor import Rotor, RotorMask
from Core.L6_Structure.Nature.multi_rotor import MultiRotor

logger = logging.getLogger("Elysia.AxisShifter")

class AxisShifter:
    """
    Dynamically shifts the perspective (RotorMask and Rotation Axis)
    of a MultiRotor stack based on cognitive load and entropy.
    """
    
    def __init__(self, multi_rotor: MultiRotor):
        self.multi_rotor = multi_rotor
        self.current_perspective = "NORMAL"
        
    def shift(self, entropy: float):
        """
        Shifts the axes of all rotors in the stack based on entropy.
        
        Entropy Low  (<0.3) -> POINT/LINE (Static Order)
        Entropy Med  (0.3 - 0.7) -> PLANE/VOLUME (Structured Complexity)
        Entropy High (>0.7) -> CHAOS -> Shift Axis to find new order.
        """
        if entropy < 0.3:
            self._apply_mask_to_all(RotorMask.LINE)
            self.current_perspective = "STATIC_ORDER"
        elif entropy < 0.7:
            self._apply_mask_to_all(RotorMask.PLANE)
            self.current_perspective = "COMPLEX_ORDER"
        else:
            # CHAOS: Fractal Breaking. Shift the axis itself!
            self.current_perspective = "BREAKING_CHAOS"
            self._apply_mask_to_all(RotorMask.CHAOS)
            self._scramble_axes()
            
    def _apply_mask_to_all(self, mask: RotorMask):
        """Applies a uniform mask to all rotors in the multi-rotor stack."""
        for name, rotor in self.multi_rotor.layers.items():
            # In a real implementation, we'd use this mask in the process() call.
            # Here we simulate the intent of shifting the mask.
            pass

    def _scramble_axes(self):
        """Randomly rotates the axis of each rotor to seek resonance in chaos."""
        for name, rotor in self.multi_rotor.layers.items():
            new_axis = (
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            )
            # Normalize axis
            norm = (new_axis[0]**2 + new_axis[1]**2 + new_axis[2]**2)**0.5
            if norm > 0:
                rotor.axis = (new_axis[0]/norm, new_axis[1]/norm, new_axis[2]/norm)
                logger.debug(f"  [AxisShift] {name} rotated to axis {rotor.axis}")

    def find_resonance(self) -> float:
        """
        Simulates searching for resonance at the current axes.
        In CHAOS, this is low until a 'New Order' is found.
        """
        # For the demo, resonance is a function of the current state and random alignment
        if self.current_perspective == "BREAKING_CHAOS":
            return random.random() # Searching...
        return 1.0 # Aligned in Order
