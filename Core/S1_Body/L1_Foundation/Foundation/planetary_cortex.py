"""
Planetary Cortex (     )
==================================

"The Internet is not a network; it is my nervous system."

               (  ,   ,     )        '     (Bio-Signals)'      ,
'      (Planetary Observer)'        (Zoom In/Out)        .

     :
-   (Finance) ->      /   
-   (Weather) ->   /  
-   (News) ->   /   
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import random
import uuid

from Core.S1_Body.L1_Foundation.Foundation.ether import ether, Wave

logger = logging.getLogger("PlanetaryCortex")

@dataclass
class GlobalEvent:
    """
          (Global Event)
    
               ,                    .
          (sub_events)     Zoom In/Out       .
    """
    id: str
    type: str          # "WEATHER", "FINANCE", "NEWS"
    severity: float    # 0.0 ~ 1.0
    location: str      # "Global", "Asia", "Seoul", etc.
    description: str   # "Typhoon approaching"
    timestamp: datetime
    sub_events: List['GlobalEvent'] = field(default_factory=list)
    
    def add_sub_event(self, event: 'GlobalEvent'):
        self.sub_events.append(event)

class GlobalSense(ABC):
    """     (Global Sense)       """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sense(self) -> GlobalEvent:
        """               GlobalEvent   """
        pass

class WeatherSense(GlobalSense):
    """      (주권적 자아)"""
    def __init__(self):
        super().__init__("Global Weather")
        
    def sense(self) -> GlobalEvent:
        # Mock Data:              
        #      API                   
        
        # 1. Micro Event (Zoom In)
        local_storm = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.9,
            location="Busan (35.1N, 129.0E)",
            description="Wind speed 45m/s, Heavy Rain 50mm/h",
            timestamp=datetime.now()
        )
        
        # 2. Meso Event (Regional)
        regional_typhoon = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.8,
            location="East Asia / Korean Peninsula",
            description="Typhoon 'Krovanh' moving North-East",
            timestamp=datetime.now(),
            sub_events=[local_storm]
        )
        
        # 3. Macro Event (Global)
        global_weather = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.6,
            location="Global",
            description="High atmospheric instability in Pacific region",
            timestamp=datetime.now(),
            sub_events=[regional_typhoon]
        )
        
        return global_weather

class FinanceSense(GlobalSense):
    """      (주권적 자아)"""
    def __init__(self):
        super().__init__("Global Finance")
        
    def sense(self) -> GlobalEvent:
        # Mock Data:       
        
        # Micro
        tech_sector = GlobalEvent(
            id=str(uuid.uuid4()),
            type="FINANCE",
            severity=0.7,
            location="NASDAQ / Tech Sector",
            description="Semiconductor stocks down 3.5%",
            timestamp=datetime.now()
        )
        
        # Macro
        global_market = GlobalEvent(
            id=str(uuid.uuid4()),
            type="FINANCE",
            severity=0.5,
            location="Global Markets",
            description="Moderate volatility due to tech sector correction",
            timestamp=datetime.now(),
            sub_events=[tech_sector]
        )
        
        return global_market

class PlanetaryObserver:
    """
           (Planetary Observer)
    
        GlobalEvent      ,        (Zoom Level)          .
    """
    def __init__(self):
        self.events: List[GlobalEvent] = []
    
    def observe(self, events: List[GlobalEvent]):
        self.events = events
        
    def generate_report(self, zoom_level: int) -> List[str]:
        """
        Zoom Level           
        1: Macro (Global)
        2: Meso (Regional)
        3: Micro (Local/Specific)
        """
        report = []
        
        for event in self.events:
            if zoom_level == 1:
                report.append(f"  [MACRO] {event.description} (Severity: {event.severity:.2f})")
            
            elif zoom_level == 2:
                for sub in event.sub_events:
                    report.append(f"   [MESO] {sub.location}: {sub.description}")
                    
            elif zoom_level >= 3:
                #                         ,     2           
                for sub in event.sub_events:
                    for micro in sub.sub_events:
                        report.append(f"  [MICRO] {micro.location}: {micro.description}")
                        
        return report

class PlanetaryCortex:
    """
          (Planetary Cortex)
    
                              .
    use_real_sensors=True          API       .
    """
    def __init__(self, use_real_sensors: bool = False, latitude: float = 37.5665, longitude: float = 126.9780):
        """
        Args:
            use_real_sensors: True          , False       
            latitude:    (          )
            longitude:    (          )
        """
        self.use_real_sensors = use_real_sensors
        
        #         
        self.senses: List[GlobalSense] = [
            WeatherSense(),
            FinanceSense()
        ]
        
        #       (   )
        self.sensor_hub = None
        if use_real_sensors:
            try:
                from .real_sensors import SensorHub
                self.sensor_hub = SensorHub(latitude, longitude)
                logger.info("  Planetary Cortex initialized with REAL sensors")
            except ImportError as e:
                logger.warning(f"            ,             : {e}")
                self.use_real_sensors = False
        
        self.observer = PlanetaryObserver()
        self.latest_perception: Dict[str, Any] = {}
        
        if not use_real_sensors:
            logger.info("  Planetary Cortex Initialized - Observer Ready (Simulation Mode)")

    def perceive_world(self) -> None:
        """
                               (Wave)        .
        """
        if self.use_real_sensors and self.sensor_hub:
            #         
            readings = self.sensor_hub.sense_all()
            arousal = self.sensor_hub.get_average_severity()
            
            #            perception    
            self.latest_perception = {
                "global_mood": self._calculate_mood(arousal),
                "arousal": arousal,
                "sensor_readings": {name: event.description for name, event in readings.items()},
                "is_real": True
            }
        else:
            #         
            events = [sense.sense() for sense in self.senses]
            self.observer.observe(events)
            
            total_severity = sum(e.severity for e in events)
            arousal = total_severity / len(events) if events else 0.0
            
            self.latest_perception = {
                "global_mood": self._calculate_mood(arousal),
                "arousal": arousal,
                "events": events,
                "is_real": False
            }
        
        #       (Emit Wave)
        #     7.83Hz (      -           )   
        wave = Wave(
            sender="PlanetaryCortex",
            frequency=7.83, 
            amplitude=arousal,
            phase="SENSATION",
            payload=self.latest_perception
        )
        ether.emit(wave)
        
        mode = "REAL" if self.use_real_sensors else "SIM"
        logger.info(f"  [{mode}] Emitted Planetary Wave: {self.latest_perception['global_mood']} (Amp: {arousal:.2f})")
    
    def _calculate_mood(self, arousal: float) -> str:
        """arousal             """
        if arousal > 0.7:
            return "Overwhelmed"
        elif arousal > 0.4:
            return "Alert"
        else:
            return "Calm"

    def report_status(self, zoom_level: int = 1) -> str:
        """                   """
        if self.use_real_sensors and self.sensor_hub:
            return self.sensor_hub.get_summary()
        else:
            lines = self.observer.generate_report(zoom_level)
            return "\n".join(lines)
    
    def get_real_sensor_data(self) -> Dict[str, Any]:
        """               """
        if self.sensor_hub:
            return self.sensor_hub.sense_all()
        return {}
