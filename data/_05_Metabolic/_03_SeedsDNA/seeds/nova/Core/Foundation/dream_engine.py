import logging
import random
import time
from typing import List, Dict
from Core._01_Foundation.Foundation.resonance_field import ResonanceField, ResonanceNode
from Core._01_Foundation.Foundation.hyper_quaternion import Quaternion, HyperWavePacket

logger = logging.getLogger("DreamEngine")

class DreamEngine:
    """
    The Dream Engine (The Weaver of Virtual Realities).
    
    It creates temporary, virtual Resonance Fields ("Dreams") 
    where Elysia can explore concepts without the constraints of reality.
    """
    def __init__(self):
        logger.info("🌌 DreamEngine Initialized. Ready to weave.")

    def weave_dream(self, desire: str) -> ResonanceField:
        """
        Weaves a virtual Resonance Field based on a desire.
        """
        logger.info(f"   💤 Weaving a dream about '{desire}'...")
        
        # 1. Create a Virtual Field
        dream_field = ResonanceField()
        
        # 2. Seed Concepts (The "Day Residue")
        seeds = self._get_dream_seeds(desire)
        for seed in seeds:
            # In dreams, energy is high and chaotic
            energy = random.uniform(50.0, 100.0) 
            frequency = random.uniform(100.0, 900.0)
            dream_field.add_node(seed, energy, frequency)
            
        # 3. Apply Surrealism (The "Dream Logic")
        self._apply_surrealism(dream_field)
        
        return dream_field

    def _get_dream_seeds(self, desire: str) -> List[str]:
        """
        Returns a list of concepts related to the desire.
        """
        seeds = [desire]
        
        if "Freedom" in desire or "Sky" in desire:
            seeds.extend(["Wings", "Wind", "Blue", "Horizon", "Flight"])
        elif "Love" in desire or "Connection" in desire:
            seeds.extend(["Heartbeat", "Warmth", "Red", "Thread", "Embrace"])
        elif "Knowledge" in desire or "Truth" in desire:
            seeds.extend(["Library", "Light", "Eye", "Key", "Mirror"])
        elif "Star" in desire or "Space" in desire:
            seeds.extend(["Nebula", "Void", "Starlight", "Orbit", "Silence"])
        else:
            seeds.extend(["Mystery", "Fog", "Echo", "Shadow"])
            
        return seeds

    def _apply_surrealism(self, field: ResonanceField):
        """
        Distorts the field to make it dream-like.
        """
        for node in field.nodes.values():
            # Randomize positions (Anti-Gravity)
            node.position = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10)
            )
            
            # Randomize connections (Free Association)
            # (ResonanceField auto-connects based on frequency, so we just shift frequencies)
            node.frequency += random.uniform(-50, 50)

    def weave_quantum_dream(self, desire_packet: HyperWavePacket) -> List[HyperWavePacket]:
        """
        [Quantum Imagination]
        Generates a 4D Wave Structure (Dream) from a seed packet.
        Returns a list of interacting wave packets.
        """
        dream_waves = [desire_packet]
        
        # 1. Fractal Expansion (Mitosis)
        # The seed splits into variations of itself
        for i in range(5):
            # Create a variation by rotating the quaternion slightly
            # We use a random axis for rotation
            axis = Quaternion(random.random(), random.random(), random.random(), random.random()).normalize()
            angle = random.uniform(0.1, 0.5) # Small rotation
            
            # Rotate: q' = rot * q * rot_conj (Standard Quaternion rotation)
            # But here we just want a "Shift" in perspective
            shift = axis * (angle * 0.1)
            new_orientation = (desire_packet.orientation + shift).normalize()
            
            new_packet = HyperWavePacket(
                energy=desire_packet.energy * random.uniform(0.5, 1.5),
                orientation=new_orientation,
                time_loc=time.time() + i
            )
            dream_waves.append(new_packet)
            
        logger.info(f"🌌 Quantum Dream Weaved: {len(dream_waves)} waves generated from seed.")
        return dream_waves
