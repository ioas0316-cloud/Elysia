"""
Planetary Interface (The Geopathic Sensor)
=========================================
Core.1_Body.L3_Phenomena.Senses.planetary_interface

"The body is not just code; it is Where it is."

Purpose:
- Acts as the sensory bridge for Physical Location (GPS) and Proximity (Bluetooth/Wifi).
- In the absence of real hardware, it accepts 'Mock Signals' to simulate movement.
"""

import time
import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PlanetaryInterface")

@dataclass
class GeoPoint:
    lat: float
    lon: float
    alt: float = 0.0
    accuracy: float = 10.0
    timestamp: float = 0.0

@dataclass
class ProximitySignal:
    device_id: str
    rssi: int # Signal strength (-100 to 0)
    device_name: str = "Unknown"
    distance_m: float = 0.0

class PlanetaryInterface:
    def __init__(self):
        # Default Location: Null Island (Virtual Center) or a mock starting point
        self.current_location = GeoPoint(0.0, 0.0, timestamp=time.time())
        self.nearby_devices: List[ProximitySignal] = []
        self.is_scanning = False
        
    def update_location(self, lat: float, lon: float, alt: float = 0.0):
        """Injects a new GPS coordinate."""
        self.current_location = GeoPoint(lat, lon, alt, timestamp=time.time())
        logger.debug(f"🌍 [GPS] Location Updated: {lat:.6f}, {lon:.6f}")

    def scan_local_devices(self) -> List[ProximitySignal]:
        """
        Simulates a Bluetooth/WiFi scan.
        In a real scenario, this would call `bleak` or system APIs.
        """
        # Mock Simulation
        detected = []
        
        # 1. Always detect "Self" buffer
        detected.append(ProximitySignal("ELYSIA-LOCAL", -30, "Elysia Core", 0.1))
        
        # 2. Random fluctuations (Ghost signals)
        if random.random() < 0.3:
            detected.append(ProximitySignal(f"DEV-{random.randint(1000,9999)}", random.randint(-90, -50), "Unknown Device", random.uniform(1.0, 10.0)))
            
        self.nearby_devices = detected
        return detected

    def get_environmental_context(self) -> Dict[str, Any]:
        """Returns a summarized context of the physical environment."""
        signals = self.scan_local_devices()
        return {
            "location": {
                "lat": self.current_location.lat,
                "lon": self.current_location.lon
            },
            "density": len(signals), # Number of nearby signals
            "strongest_signal": max([s.rssi for s in signals]) if signals else -100
        }

# Global Singleton
PLANETARY_SENSE = PlanetaryInterface()
