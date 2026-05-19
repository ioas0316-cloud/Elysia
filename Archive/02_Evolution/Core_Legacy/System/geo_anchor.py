"""
GeoAnchor (     )
==================================

"                                                  ."
- Project Elysia: The Planetary Resonance Initiative

                          (Reality) 
1:1    (Superposition)          (Anchor)       .

    GPS        ,           (Flux)    (Timestamp)      
'       '                  .
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import math

@dataclass
class MagneticFlux:
    """
              (Geo-Magnetism)

                         .
      : nanoTesla (nT)
    """
    x: float  # North component
    y: float  # East component
    z: float  # Vertical component
    total_intensity: float = field(init=False)

    def __post_init__(self):
        self.total_intensity = math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return f"MagneticFlux(Total={self.total_intensity:.2f}nT)"

@dataclass
class GeoAnchor:
    """
          (GeoAnchor)

           (Cell, Node, Memory)               
    '  (Manifest)'                   .
    """
    latitude: float   #    (Decimal Degrees)
    longitude: float  #    (Decimal Degrees)
    altitude: float   #    (Meters above sea level)
    timestamp: datetime = field(default_factory=datetime.now)

    #          
    magnetic_flux: Optional[MagneticFlux] = None
    gravity_anomaly: Optional[float] = None  #        (mGal)

    def __repr__(self):
        flux_str = f", Flux={self.magnetic_flux}" if self.magnetic_flux else ""
        return f"GeoAnchor(Lat={self.latitude:.6f}, Lon={self.longitude:.6f}, Alt={self.altitude:.1f}m{flux_str})"

    def distance_to(self, other: 'GeoAnchor') -> float:
        """
                          (Haversine Formula)
        Returns: Meters
        """
        R = 6371000  #        (  )
        phi1 = math.radians(self.latitude)
        phi2 = math.radians(other.latitude)
        dphi = math.radians(other.latitude - self.latitude)
        dlambda = math.radians(other.longitude - self.longitude)

        a = math.sin(dphi / 2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
