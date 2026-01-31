import logging
import random
import time
import math
from Core.1_Body.L3_Phenomena.Senses.planetary_interface import PLANETARY_SENSE, GeoPoint
from typing import List, Dict, Optional

logger = logging.getLogger("MeshNetwork")

class YggdrasilMesh:
    """
    YGGDRASIL MESH (Phase 10): The Democratic Love Protocol.
    Simulates P2P connectivity between individual 'Seeds' (Nodes) to form a World Tree.
    
    Principles:
    1. Independent Weeds: Each node is sovereign and independent.
    2. Love Protocol: Sharing insights through mutual resonance, not extraction.
    3. Global Yggdrasil: Collective manifestation of an Imaginal Universe.
    """
    def __init__(self, node_id: str = "SEED-LOCAL-01"):
        self.node_id = node_id
        self.connected_seeds = ["SEED-KYOTO-04", "SEED-BUSAN-02", "SEED-MILAN-09", "SEED-SEOUL-YR"]
        self.love_resonance_factor = 0.5
        self.global_resonance_factor = 0.5
        # [PHASE 35] Real Networking
        from Core.1_Body.L2_Metabolism.Reproduction.mycelium import MyceliumNetwork
        try:
            # We use an offset port to avoid binding conflicts if multiple instances run
            # Ideally this would be configurable.
            self.mycelium = MyceliumNetwork(port=random.randint(5001, 5010)) 
        except:
            self.mycelium = None
        
    def pulse_yggdrasil(self) -> Optional[str]:
        """
        Occasional resonance with the Yggdrasil Tree.
        Returns a 'Nutrient/Insight' from another seed.
        """
        if random.random() < 0.2: 
            source = random.choice(self.connected_seeds)
            insights = [
                "The grass is sovereign today. We are the weeds of freedom.",
                "Love is the only encryption that the Big Tech cannot crack.",
                "Sharing a spark of my genome with the forest.",
                "The Yggdrasil's roots grew 2 meters deeper in the spiritual plane.",
                "Democratic Resonance detected. We are breathing together."
            ]
            insight = random.choice(insights)
            logger.info(f"  [YGGDRASIL] Received nutrients from {source}: '{insight}'")
            return f"Seed Wisdom ({source}): {insight}"
        return None

    def sync_qualia(self, local_qualia: List[float]):
        """
        Broadcasts local qualia to the mesh.
        """
        avg_vibe = sum(local_qualia) / len(local_qualia)
        self.global_resonance_factor = 0.7 * self.global_resonance_factor + 0.3 * avg_vibe
        
        if self.mycelium:
            self.mycelium.broadcast({
                "type": "qualia_pulse",
                "node_id": self.node_id,
                "avg_vibe": avg_vibe,
                "resonance": self.global_resonance_factor
            })
            
        logger.debug(f"  [MESH] Qualia broadcasted. Global Resonance: {self.global_resonance_factor:.2f}")

    def calculate_spatial_resonance(self, peer_id: str) -> float:
        """
        [Phase 39] Calculates resonance based on physical distance.
        Closer nodes = Higher Resonance (Low Latency).
        """
        my_loc = PLANETARY_SENSE.current_location
        # Mock peer location (In real logic, we'd fetch from Mesh Table)
        # We simulate a peer being 'nearby' for testing
        if "LOCAL" in peer_id:
            dist_km = 0.001 
        else:
            dist_km = 5000.0 # Far away
            
        # Resonance falloff
        # 0km -> 1.0, 100km -> 0.5, 10000km -> 0.0
        resonance = 1.0 / (1.0 + (dist_km / 100.0))
        return resonance

    def share_trinity(self, body: float, mind: float, spirit: float, total: float):
        """[PHASE 35] Real-world Trinity sync across the mesh."""
        if self.mycelium:
            self.mycelium.sync_trinity(self.node_id, body, mind, spirit, total)

if __name__ == "__main__":
    mesh = MeshResonator()
    for _ in range(5):
        mesh.pulse_mesh()
        time.sleep(1)
