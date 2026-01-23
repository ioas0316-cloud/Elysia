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
        
    def _clear_console(self):
        # [ANSI Escape] Move cursor to top-left without clearing the entire buffer
        # This prevents flickering and preserves the scrolling chat history below.
        print("\033[H", end="")

    def render(self, metrics: Dict[str, Any]):
        """
        Renders a beautiful ASCII HUD of Elysia's current state.
        Metrics expected: hz, love_score, latency, state, will, tension
        """
        hz = metrics.get('hz', 0.0)
        coherence = metrics.get('passion', 0.0) # Coherence mapped to passion
        torque = metrics.get('torque', 0.0)
        love = metrics.get('love_score', 0.0)
        latency = metrics.get('latency', 0.0)
        state = metrics.get('state', "VOID").upper()
        will = metrics.get('will', 0.0)
        tension = metrics.get('tension', 0.0)
        
        # ASCII Bars & Indicators
        coherence_bar = "█" * int(coherence * 20) + "░" * (20 - int(coherence * 20))
        torque_arrow = "⚡" if torque > 0.4 else "  "
        will_arrow = "▲" if will > 0 else "▼"
        
        # 7-Layer Coherence Sparkline (Simplified)
        spark = " " + " ".join(["*" if coherence > (i/7) else "." for i in range(7)])
        
        output = [
            "╔" + "═" * (self.width - 2) + "╗",
            f"║  E L Y S I A   S O V E R E I G N   H U D   [{state}]".ljust(self.width - 1) + "║",
            "║" + " " * (self.width - 2) + "║",
            f"║  💓 PULSE: {hz:>6.2f} Hz | Rhythm: {'COHERENT' if coherence > 0.8 else 'DYNAMIC' :<11}     ║",
            f"║  🌀 LOVE:  {love*100:>5.1f}% | ⚡ TORQUE: {torque:>5.3f} {torque_arrow}            ║",
            f"║  🌟 JOY:   [{coherence_bar}] {coherence*100:>5.1f}% Coherence ║",
            f"║  🌌 PATH:  [{spark}]  (Point -> Providence)     ║",
            f"║  ⚡ SPEED: {latency:>6.3f} ms | Path: {'LIGHTNING 2.0' :<13}     ║",
            "╠" + "═" * (self.width - 2) + "╣",
            f"║  ☄️ INTENT: Will {will_arrow} {abs(will):>5.3f} | Tension: {tension:>5.3f}         ║",
            "╚" + "═" * (self.width - 2) + "╝"
        ]
        
        print("\n".join(output))

    def project_narrative(self, narrative: str, thought_log: list = None):
        """Project causal narrative and internal thoughts."""
        print(f" [NARRATIVE] 🧶 {narrative}")
        if thought_log:
            print("-" * self.width)
            for thought in thought_log:
                print(f" 💭 {thought}")
            print("-" * self.width)
