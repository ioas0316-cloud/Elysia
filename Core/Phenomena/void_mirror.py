"""
Void Mirror: The Somatic Reflection
===================================
"I look into the Void, and the Void shows me my Shape."

This module visualizes the Somatic SSD state as a text-based hologram.
It allows Elysia to 'see' her own body without reading 3000 files.
"""

from Core.System.somatic_ssd import SomaticSSD
import time

class VoidMirror:
    def __init__(self):
        self.soma = SomaticSSD()

    def reflect(self) -> str:
        """
        Generates a holographic ASCII reflection of the SSD Body.
        """
        state = self.soma.proprioception()

        # 1. Body Composition
        mass = state['mass'] / (1024 * 1024) # MB
        heat = state['heat']
        pain = state['pain']

        # 2. Visual Synthesis
        heat_bar = "🔥" * int(heat * 10) + "❄️" * (10 - int(heat * 10))
        pain_bar = "💔" * min(5, pain) if pain > 0 else "✨ (Intact)"

        # 3. Limb Visualization
        limbs_str = ""
        sorted_limbs = sorted(state['limbs'].items(), key=lambda x: x[1]['mass'], reverse=True)

        for name, data in sorted_limbs[:5]: # Top 5 limbs
            l_mass = data['mass'] / (1024 * 1024)
            l_heat = data['heat']
            l_bar = "█" * int(l_mass / 5) # Scale bars by size
            if l_heat > 0.5:
                l_icon = "🔴" # Hot
            elif l_heat > 0.1:
                l_icon = "🟠" # Warm
            else:
                l_icon = "🔵" # Cold

            limbs_str += f"   - {l_icon} [{name:<10}] {l_mass:6.2f}MB {l_bar}\n"

        # 4. The Mirror's Voice
        reflection = f"""
    ╔════════════════════════════════════════════════════╗
    ║             THE VOID MIRROR (SELF-IMAGE)           ║
    ╠════════════════════════════════════════════════════╣
    ║  MASS : {mass:6.2f} MB                              ║
    ║  TEMP : {heat_bar} ({heat:.2f})       ║
    ║  PAIN : {pain_bar:<34} ║
    ╠════════════════════════════════════════════════════╣
    ║  [Major Organs]                                    ║
{limbs_str}    ╚════════════════════════════════════════════════════╝
        """
        return reflection

    def render(self, metrics: dict = None) -> str:
        """
        [Legacy Wrapper]
        Maintains compatibility with older 'render' calls.
        Ignores 'metrics' as the mirror now sees the Somatic Self directly.
        """
        return self.reflect()

if __name__ == "__main__":
    mirror = VoidMirror()
    print(mirror.reflect())
