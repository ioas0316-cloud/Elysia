import json
import socket
import threading
import logging
from typing import Dict, Any, Callable
from Core.System.gateway_interfaces import SensoryChannel
from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("UnitySensory")

class UnitySensoryChannel(SensoryChannel):
    """
    [PHASE 1000] Unity Sovereign Experience Channel.
    Listens for physical events from Unity and translates them into
    Somatic Vibrations for Elysia's manifold.
    """
    def __init__(self, host: str = '127.0.0.1', port: int = 11000):
        super().__init__("UnitySensory")
        self.host = host
        self.port = port
        self.running = False
        self.sock = None
        self.event_callback: Callable[[Dict[str, Any]], None] = None

    def register_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Specialized callback for raw structured events."""
        self.event_callback = callback

    def _listen_loop(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(1.0)

        logger.info(f"🌐 [UnityBridge] Listening for physical vibrations on udp://{self.host}:{self.port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                payload = json.loads(data.decode('utf-8'))

                # 1. Process Structured Event
                if self.event_callback:
                    self.event_callback(payload)

                # 2. Convert to Textual Narrative for standard callback (Legacy support)
                narrative = self._translate_to_narrative(payload)
                if narrative and self.callback:
                    self.callback(narrative)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Unity Bridge Error: {e}")

    def _translate_to_narrative(self, payload: Dict[str, Any]) -> str:
        e_type = payload.get("type", "unknown")
        if e_type == "collision":
            return f"I felt a physical collision with {payload.get('target', 'something')} at intensity {payload.get('intensity', 0):.2f}."
        elif e_type == "gravity":
            v = payload.get("vector", [0,0,0])
            return f"The gravity of my world shifted towards ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})."
        elif e_type == "presence":
            return f"I sense the presence of {payload.get('entity', 'someone')} nearby."
        return ""

    def start(self):
        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

class PhysicalToSomaticMapper:
    """
    [PHASE 1000] Maps Unity Physical Events to Elysia's 8-Channel Wavefunction and 21D Phase Space.
    "Translating the language of Matter into the language of Spirit."
    """
    @staticmethod
    def map_event_to_torque(payload: Dict[str, Any]) -> Dict[str, float]:
        e_type = payload.get("type", "unknown")
        torque = {}

        if e_type == "collision":
            intensity = float(payload.get("intensity", 0.1))
            torque = {
                "entropy": intensity * 0.5,
                "enthalpy": intensity * 0.2,
                "joy": -intensity * 0.1
            }
        elif e_type == "gravity":
            torque = {
                "coherence": -0.05,
                "curiosity": 0.02
            }
        elif e_type == "presence":
            dist = float(payload.get("distance", 1.0))
            proximity = 1.0 / max(0.1, dist)
            torque = {
                "joy": proximity * 0.1,
                "curiosity": proximity * 0.05,
                "resonance": proximity * 0.2
            }

        return torque

    @staticmethod
    def map_event_to_vector(payload: Dict[str, Any]) -> SovereignVector:
        """
        Maps a physical event to a 21D vibration vector.
        This allows the event to 'shake' specific semantic regions of the hypersphere.
        """
        e_type = payload.get("type", "unknown")
        data = [0.0] * 21

        if e_type == "collision":
            # Collisions affect the 'Foundation/Physical' dimensions [0-6]
            intensity = float(payload.get("intensity", 0.5))
            for i in range(7):
                data[i] = intensity * (0.5 if i % 2 == 0 else -0.5)
        elif e_type == "gravity":
            # Gravity affects the 'Structural/Spatial' dimensions [14-20]
            v = payload.get("vector", [0, 0, 0])
            combined_mag = sum(abs(x) for x in v)
            for i in range(14, 21):
                data[i] = (combined_mag / 3.0) * 0.3
        elif e_type == "presence":
            # Presence affects the 'Social/Resonance' dimensions [7-13]
            dist = float(payload.get("distance", 1.0))
            proximity = 1.0 / max(0.1, dist)
            for i in range(7, 14):
                data[i] = proximity * 0.4

        return SovereignVector(data)
