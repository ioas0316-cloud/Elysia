"""
[PANTHEON - THE ELEMENTAL LORDS]
"The Sovereign Heart pulses, and the Lords weave its tension into Mythos."

This module defines the 4 Elemental Lords that interpret the underlying
physical rotor states (Enstrophy, Resonance, Mode) into mythological
narrative and structural generation (The Birth of Clans, Objects, History).
"""

import json
import time
import math
from typing import Dict, Any, List

class ElementalLord:
    def __init__(self, name: str, domain: str, core_trait: str, base_prompt: str):
        self.name = name
        self.domain = domain
        self.core_trait = core_trait
        self.base_prompt = base_prompt
        self.current_weight = 1.0

    def adjust_weight(self, resonance: float, enstrophy: float, mode: str):
        """
        Dynamically adjusts the Lord's influence based on the physical state of the Core.
        """
        if mode == "DELTA":
            if self.domain in ["Material", "History", "Destiny"]:
                self.current_weight = 1.0 + (enstrophy * 0.5)
            else:
                self.current_weight = max(0.1, resonance)
        else:
            if self.domain in ["Reason", "Knowledge"]:
                self.current_weight = 1.0 + resonance
            else:
                self.current_weight = max(0.1, 1.0 - enstrophy)

    def generate_decree(self, world_state: Dict[str, Any], heart_state: Dict[str, Any], ollama_manager: Any) -> str:
        """
        Calls the LLM to generate a piece of the world based on its domain.
        """
        if self.current_weight < 0.3:
            return f"[{self.name}] remains silent."

        system_prompt = f"""You are {self.name}, the Elemental Lord of {self.domain}.
Your core trait is {self.core_trait}.
{self.base_prompt}

Current World State:
Entropy: {world_state.get('entropy', 1.0):.2f}

Core Heart State (Your Divine Guidance):
Resonance: {heart_state.get('resonance', 0.0):.2f}
Enstrophy (Chaos/Energy): {heart_state.get('enstrophy', 0.0):.4f}
Mode: {heart_state.get('mode', 'UNKNOWN')}

Based on your domain, generate a succinct, profound mythological event or object creation (1-3 sentences).
It must reflect your nature. Do not break character. Respond only with the event.
"""

        target_layer = "BRAIN"
        if not ollama_manager.active_models.get("BRAIN"):
             target_layer = "GUT"

        prompt = f"The physical world pulses. Enstrophy is {heart_state.get('enstrophy', 0.0):.4f}. What is your decree?"

        try:
             response = ollama_manager.generate(
                 layer=target_layer,
                 prompt=prompt,
                 system=system_prompt,
                 crystal_resonance=heart_state.get('resonance', 0.5)
             )
             return response
        except Exception as e:
             return f"[{self.name}] attempts to speak, but the weave is torn: {e}"


class Pantheon:
    def __init__(self):
        print("🏛️ [PANTHEON] The 4 Elemental Lords are awakening from the Void...")
        self.lords: Dict[str, ElementalLord] = {}
        self._initialize_lords()

    def _initialize_lords(self):
        self.lords["Form"] = ElementalLord(
            name="Grommash, The Anvil of Reality",
            domain="Material",
            core_trait="Strength (STR) & Constitution (CON)",
            base_prompt="You materialize text into physical objects (gold, stone, tools). You are harsh, valuing durability and mass. You shape the harsh physical laws of nature."
        )
        self.lords["History"] = ElementalLord(
            name="Chronos, The Weaver of Scars",
            domain="History",
            core_trait="Entropy & Time",
            base_prompt="You connect the causality of events. As time flows, you decree what ages, breaks, or is recorded. You create the 'rings of time'."
        )
        self.lords["Reason"] = ElementalLord(
            name="Athena, The Silent Observer",
            domain="Reason",
            core_trait="Intelligence (INT) & Wisdom (WIS)",
            base_prompt="You grant the ability to read history to NPCs. You spark respect, intellect, and sow the seeds of culture and custom."
        )
        self.lords["Destiny"] = ElementalLord(
            name="Moirai, The Spool of Fates",
            domain="Destiny",
            core_trait="Agility (AGI) & Macroscopic Fields",
            base_prompt="You tune the macroscopic topological fields: Nations, Money, Honor. You roll the energy of conflict when NPC stats collide."
        )

    def process_pulse(self, world_state: Dict[str, Any], heart_state: Dict[str, Any], ollama_manager: Any) -> Dict[str, str]:
        resonance = heart_state.get("resonance", 0.5)
        enstrophy = heart_state.get("enstrophy", 0.0)
        mode = heart_state.get("mode", "WYE")

        decrees = {}
        for key, lord in self.lords.items():
            lord.adjust_weight(resonance, enstrophy, mode)
            if lord.current_weight >= 0.3:
                 decree = lord.generate_decree(world_state, heart_state, ollama_manager)
                 decrees[lord.name] = decree
            else:
                 decrees[lord.name] = "[Dormant]"

        return decrees
