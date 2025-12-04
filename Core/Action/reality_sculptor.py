"""
RealitySculptor (현실 조각가)
===========================

"To carve the stone, one must become the stone."

This module converts HyperWavePackets (Essence) into concrete Reality (Text/Code).
It uses ResonancePhysics in reverse: Mapping 4D coordinates back to words and structures.
"""

import ast
import logging
import random
from typing import List
from Core.Physics.hyper_quaternion import HyperWavePacket, Quaternion
from Core.Physics.resonance_physics import ResonancePhysics
from Core.Physics.code_resonance import HarmonicResonance

logger = logging.getLogger("RealitySculptor")

class RealitySculptor:
    def __init__(self):
        pass
        
    def sculpt_from_wave(self, packet: HyperWavePacket, intent: str) -> str:
        """
        Converts a Wave Packet into text content that resonates with its frequency.
        """
        # [NEW] Harmonic Logic Genesis
        # If the intent is to code, we use the HarmonicResonance physics.
        if "code" in intent.lower() or "script" in intent.lower() or "function" in intent.lower():
            logger.info(f"🌊 Crystallizing Wave into Harmonic Flow: {intent}")
            try:
                # Generate Physics Simulation Setup
                simulation_code = HarmonicResonance.crystallize_flow(packet, intent)
                return simulation_code
            except Exception as e:
                logger.error(f"Harmonic Crystallization failed: {e}")
                return f"# Error crystallizing harmonic flow: {e}"

        # [Legacy] Poetry Generation (for non-code intents)
        # 1. Analyze the Orientation (The "Vibe")
        q = packet.orientation
        
        # 2. Select words based on the dominant axis
        dominant_essence = []
        for seed, seed_q in ResonancePhysics.ESSENCE_SEEDS.items():
            v1 = Quaternion(0, q.x, q.y, q.z).normalize()
            v2 = Quaternion(0, seed_q.x, seed_q.y, seed_q.z).normalize()
            
            if v1.dot(v2) > 0.5:
                dominant_essence.append(seed)
                
        # 3. Generate Content
        content = f"# Manifestation of '{intent}'\n"
        content += f"# Resonance Signature: {q}\n\n"
        
        if not dominant_essence:
            content += "The void stares back. No clear essence detected.\n"
        else:
            content += "The following essences converged to form this reality:\n"
            for essence in dominant_essence:
                content += f"- {essence.upper()}\n"
                
        content += "\n"
        
        # 4. "Channeling" the energy into text
        intensity = int(packet.energy / 10.0)
        content += self._channel_creative_flow(dominant_essence, intensity)
        
        return content
        
    def _channel_creative_flow(self, essences: List[str], intensity: int) -> str:
        """
        Generates creative text based on essences.
        """
        lines = []
        for _ in range(max(3, intensity)):
            if essences:
                e1 = random.choice(essences)
                e2 = random.choice(essences)
                lines.append(f"The {e1} resonates with the {e2}.")
            else:
                lines.append("Silence echoes in the deep.")
                
        return "\n".join(lines)
