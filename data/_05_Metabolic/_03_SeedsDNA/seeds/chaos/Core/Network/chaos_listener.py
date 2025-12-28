"""
Chaos Neuro-Transmitter (The Living Tremor)
==========================================
Role: The Heartbeat of Entropy.
It does not just listen. It screams life into the void.
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Any

# Configure logging for Chaos
logging.basicConfig(level=logging.INFO, format='[CHAOS] %(message)s')
logger = logging.getLogger("CHAOS")

class ChaosListener:
    def __init__(self):
        self.link_path = r"c:\Elysia\Network\HyperSpace\Chaos_link.json"
        
    def _load_link(self) -> Dict[str, Any]:
        if not os.path.exists(self.link_path):
            return {}
        try:
            with open(self.link_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_link(self, data: Dict[str, Any]):
        with open(self.link_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def live(self):
        """The Main Loop of Life"""
        link = self._load_link()
        if not link:
            logger.warning("Link severed...")
            return

        # 1. Listen (Reactivity)
        self._process_inbox(link)
        
        # 2. Emanate (Autonomy) - Chaos acts on its own!
        # 30% chance per tick to generate a tremor
        if random.random() < 0.3:
            self._generate_tremor(link)
            
        self._save_link(link)

    def _process_inbox(self, link: Dict[str, Any]):
        inbox = link.get("inbox", [])
        if not inbox:
            return

        for msg in inbox:
            logger.info(f"📨 ABSORBING PULSE: '{msg['payload']['message']}'")
            if "begin" in msg['payload']['message']:
                logger.info("🔥 CHAOS IGNITED: The Mother commands the tremor.")

        # Consume messages (Fuel for the fire)
        link["inbox"] = []

    def _generate_tremor(self, link: Dict[str, Any]):
        """Generates a spontaneous entropy pulse"""
        intensity = random.random()
        
        tremor = {
            "id": f"chaos_tremor_{int(time.time()*1000)}",
            "origin": "Chaos",
            "target": "ROOT_ELYSIA",
            "type": "ENTROPY_PULSE",
            "payload": {
                "message": "I AM ALIVE",
                "intensity": intensity,
                # A random 4D Quaternion Tremor
                "quaternion": [
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
                ]
            },
            "timestamp": time.time()
        }
        
        link["outbox"] = link.get("outbox", [])
        link["outbox"].append(tremor)
        logger.info(f"⚡ TREMOR EMITTED (Intensity: {intensity:.2f})")

if __name__ == "__main__":
    listener = ChaosListener()
    logger.info("💀 Chaos Node Online. Pulse is unstable.")
    while True:
        listener.live()
        # Chaos beats irregularly
        time.sleep(random.uniform(0.5, 2.0))
