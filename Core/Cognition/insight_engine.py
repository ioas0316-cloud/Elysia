"""
[PHASE 86] Insight Engine: The Sovereign Eye
============================================
Core.Cognition.insight_engine

"Why calculate what can simply reside in rotation?"

This module analyzes data streams for patterns (cycles, constants)
and proposes structural optimizations (Rotors, Axioms) to replace calculation.

Principle: "Insight is the collapse of complexity into simplicity."
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import math

logger = logging.getLogger("InsightEngine")

class InsightEngine:
    def __init__(self):
        self.observation_window = []
        self.min_samples_for_insight = 10
        
    def observe(self, data_point: float):
        """feeds data into the observation window."""
        self.observation_window.append(data_point)
        # Keep window size manageable
        if len(self.observation_window) > 100:
            self.observation_window.pop(0)
            
    def analyze_stream(self) -> Optional[Dict[str, Any]]:
        """
        Analyzes the current window for patterns AND necessity.
        Returns a proposal dict if a pattern is found or a preservation order is needed.
        """
        if len(self.observation_window) < self.min_samples_for_insight:
            return None
            
        data = self.observation_window
        
        # 0. Measure Life (Variance & Entropy)
        val_range = max(data) - min(data)
        
        # 1. Check for Constant (Dead Stillness)
        if val_range < 0.001:
            return {
                "type": "Constant",
                "value": sum(data) / len(data),
                "confidence": 1.0,
                "proposal": "Replace calculation with fixed Axiom (Eternal Stasis)."
            }
            
        # 2. Check for Cyclic Pattern (Sine Wave) -> Rotor Candidate
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i-1] < data[i] > data[i+1]:
                peaks.append(i)
                
        if len(peaks) >= 2:
            intervals = [peaks[j] - peaks[j-1] for j in range(1, len(peaks))]
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            
            if variance < 1.0: # Highly regular peaks
                period = avg_interval
                amplitude = val_range / 2
                return {
                    "type": "Cyclic",
                    "period": period,
                    "confidence": 0.9,
                    "proposal": f"Replace calculation with Rotor (Period={period:.1f}). Reduce redundant Strain."
                }
                
        # 3. Check for Living Chaos (High Entropy / Non-Cyclic) -> Preserve O(N)
        # If it's not constant and not cyclic, but has high variance, it's LIFE.
        if val_range > 0.1:
            return {
                "type": "Living_Chaos",
                "confidence": 0.8,
                "proposal": "PRESERVE O(N). This is a creative process. Do not fossilize into a Rotor."
            }
            
        return None

    def propose_optimization(self, insight: Dict[str, Any]) -> str:
        """From Insight to Actionable Proposal."""
        if insight['type'] == 'Cyclic':
            return f"PROPOSAL: Construct EnvironmentRotor(period={insight['period']:.1f})."
        elif insight['type'] == 'Constant':
             return f"PROPOSAL: Define Axiom(Value={insight['value']:.2f})."
        elif insight['type'] == 'Living_Chaos':
             return f"DECISION: Respect the Calculation. This is growth, not repetition."
        return "No optimization found."
