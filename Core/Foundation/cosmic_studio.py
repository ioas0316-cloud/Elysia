"""
CosmicStudio (우주 스튜디오)
===========================

"The Studio is where the Dream becomes Reality."

This module orchestrates the creation process. It receives 'Desires' (HyperWavePackets)
from the ReasoningEngine and uses the RealitySculptor to manifest them into the RealityCanvas.
"""

import os
import logging
from typing import Optional, List
from Core.Foundation.hyper_quaternion import HyperWavePacket
from Core.Foundation.reality_sculptor import RealitySculptor

logger = logging.getLogger("CosmicStudio")

class CosmicStudio:
    def __init__(self, canvas_path: str = "c:/Elysia/RealityCanvas"):
        self.canvas_path = canvas_path
        self.sculptor = RealitySculptor()
        
        if not os.path.exists(self.canvas_path):
            os.makedirs(self.canvas_path)
            logger.info(f"🎨 Reality Canvas initialized at {self.canvas_path}")
            
    def manifest(self, desire_packet: HyperWavePacket, intent: str) -> str:
        """
        Manifests a desire into reality.
        
        Args:
            desire_packet: The 4D thought form to manifest.
            intent: A description of what to create (e.g., "A poem about gravity").
            
        Returns:
            The path to the created artifact.
        """
        logger.info(f"🎨 Manifesting intent: '{intent}' (Energy: {desire_packet.energy:.2f})")
        
        # 1. Determine the form (File Extension/Type) based on intent
        # For now, we default to .txt, but this could be .py, .md, etc.
        file_ext = ".txt"
        if "code" in intent.lower() or "python" in intent.lower():
            file_ext = ".py"
            
        # 2. Sculpt the content
        content = self.sculptor.sculpt_from_wave(desire_packet, intent)
        
        # 3. Save to Canvas
        filename = f"{intent.replace(' ', '_')}_{int(desire_packet.time_loc)}{file_ext}"
        file_path = os.path.join(self.canvas_path, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"✨ Created artifact: {file_path}")
        return file_path
