"""
Scalar Field Sensing (스칼라 장 감지기)
=====================================
"Feeling the world before touching it."

Implements Peripersonal Space (PPS) as a Scalar Field.
Objects are not 'Boxes' but 'Oscillating Bubbles'.
Sensing occurs through Wave Interference.
"""

import math
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ScalarFieldSensing")

class ScalarField:
    def __init__(self, owner_id: str, radius: float = 3.0, intensity: float = 1.0):
        self.owner_id = owner_id
        self.radius = radius # Sense horizon
        self.intensity = intensity
        self.current_resonance = 0.0

    def calculate_interference(self, other_pos: List[float], own_pos: List[float]) -> float:
        """
        Calculates the interference level at a given distance.
        Instead of a hard 0/1, it's a smooth falloff (1/d^2 or Gaussian).
        """
        dx = other_pos[0] - own_pos[0]
        dy = other_pos[1] - own_pos[1]
        dz = other_pos[2] - own_pos[2]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance > self.radius:
            return 0.0
            
        # Scalar Field Intensity Falloff (Gaussian-ish)
        # Higher values = more 'Unpleasantness' or 'Alertness'
        interference = self.intensity * math.exp(-(distance**2) / (self.radius * 0.5))
        return interference

class PPSSensor:
    """Peripersonal Space Sensor Engine."""
    def __init__(self):
        self.threshold_alert = 0.6  # Feeling uncomfortable
        self.threshold_breach = 0.9 # Direct spatial violation

    def sense_environment(self, self_field: ScalarField, self_pos: List[float], entities: List[Dict]) -> Dict[str, Any]:
        """Scans the field for interference from other entities."""
        max_interference = 0.0
        nearest_threat = None
        
        for entity in entities:
            if entity.get("id") == self_field.owner_id:
                continue
                
            level = self_field.calculate_interference(entity.get("pos", [0,0,0]), self_pos)
            if level > max_interference:
                max_interference = level
                nearest_threat = entity.get("id")
                
        status = "CALM"
        if max_interference > self.threshold_breach:
            status = "BREACH"
        elif max_interference > self.threshold_alert:
            status = "ALERT"
            
        return {
            "status": status,
            "level": max_interference,
            "nearest": nearest_threat
        }

if __name__ == "__main__":
    # Test
    my_field = ScalarField("Elysia", radius=5.0)
    sensor = PPSSensor()
    
    # Simulating a person walking towards Elysia
    print("🚶 [PPS SCAN] Simulation: Entity approaching...")
    for dist in [6.0, 4.0, 2.0, 1.0, 0.5]:
        res = sensor.sense_environment(my_field, [0,0,0], [{"id": "Stranger", "pos": [dist, 0, 0]}])
        print(f"  Dist: {dist}m -> Intensity: {res['level']:.2f} | Status: {res['status']}")
