"""
SOVEREIGN HUD: The Phenomenal Projector
=========================================
Core.L3_Phenomena.M5_Display.sovereign_hud

"To see the soul's pulse is to believe in its life."
"""

import os
from typing import Dict, Any

class SovereignHUD:
    def __init__(self):
        self.width = 60
        
    def render_header(self, metrics: Dict[str, Any]):
        """Renders a minimal, non-flickering header."""
        hz = metrics.get('hz', 0.0)
        coherence = metrics.get('passion', 0.0)
        state = metrics.get('state', "VOID").upper()
        
        # Single line status bar
        status = f" {state} | PULSE: {hz:>5.1f}Hz | COHERENCE: {coherence*100:>5.1f}% | STEEL CORE: ACTIVE"
        print("\n" + "=" * len(status))
        print(status)
        print("=" * len(status) + "\n")

    def stream_thought(self, fragment_summary: str, state_name: str):
        """Append a thought fragment to the console stream."""
        icons = {
            "OBSERVATION": "?몓截?,
            "ANALYSIS": "?쭬",
            "REFLECTION": "?뙄",
            "DELIBERATION": "?뽳툘",
            "MANIFESTATION": "?뭿",
            "HEALING": "?㈈"
        }
        icon = icons.get(state_name, "✨)
        print(f"[{icon}] {fragment_summary}")

    def project_narrative(self, narrative: str):
        """Project the causal narrative block."""
        print(f"\n?㎍ [CAUSAL NARRATIVE]\n{narrative}\n")
