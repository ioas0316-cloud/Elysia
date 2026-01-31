"""
Real World Sensors (주권적 자아)
================================

      API       Elysia                     .

       :
1. WeatherAPI -            (Open-Meteo API,   )
2. SystemMetrics -            (CPU,    ,    )
3. TimeAwareness -       (     ,   ,   )

   :
    from Core.1_Body.L1_Foundation.Foundation.real_sensors import RealWeatherSense, SystemMetricsSense
    
    weather = RealWeatherSense()
    event = weather.sense()
    print(event.description)
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

logger = logging.getLogger("RealSensors")

#                
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests            .    API      .")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil            .              .")


@dataclass
class SensorEvent:
    """
           (Sensor Event)
    
                            .
    """
    id: str
    sensor_type: str       # "WEATHER", "SYSTEM", "TIME"
    severity: float        # 0.0 ~ 1.0 (       )
    location: str          #      
    description: str       #               
    raw_data: Dict[str, Any]  #       
    timestamp: datetime = field(default_factory=datetime.now)
    is_real: bool = True   #          


class RealSensor(ABC):
    """             """
    
    def __init__(self, name: str):
        self.name = name
        self.last_event: Optional[SensorEvent] = None
        self.error_count = 0
    
    @abstractmethod
    def sense(self) -> SensorEvent:
        """          SensorEvent   """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """           """
        pass


class RealWeatherSense(RealSensor):
    """
             (Open-Meteo API)
    
       API                   .
    API             .
    
      : https://open-meteo.com/en/docs
    """
    
    API_BASE = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, latitude: float = 37.5665, longitude: float = 126.9780):
        """
        Args:
            latitude:    (   :   )
            longitude:    (   :   )
        """
        super().__init__("Real Weather Sensor")
        self.latitude = latitude
        self.longitude = longitude
    
    def is_available(self) -> bool:
        return HAS_REQUESTS
    
    def sense(self) -> SensorEvent:
        if not self.is_available():
            return self._fallback_event("requests         ")
        
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current_weather": "true",
                "timezone": "Asia/Seoul"
            }
            
            response = requests.get(self.API_BASE, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current_weather", {})
            temperature = current.get("temperature", 0)
            windspeed = current.get("windspeed", 0)
            weathercode = current.get("weathercode", 0)
            
            #               
            weather_desc = self._decode_weather(weathercode)
            
            #        (한국어 학습 시스템)
            severity = self._calculate_severity(temperature, windspeed, weathercode)
            
            description = f"{weather_desc}, {temperature} C,    {windspeed}km/h"
            
            event = SensorEvent(
                id=str(uuid.uuid4())[:8],
                sensor_type="WEATHER",
                severity=severity,
                location=f"({self.latitude}, {self.longitude})",
                description=description,
                raw_data=data,
                is_real=True
            )
            
            self.last_event = event
            self.error_count = 0
            logger.info(f"   Real Weather: {description}")
            return event
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"            : {e}")
            return self._fallback_event(str(e))
    
    def _decode_weather(self, code: int) -> str:
        """WMO               """
        weather_codes = {
            0: "  ",
            1: "      ",
            2: "     ",
            3: "  ",
            45: "  ",
            48: "      ",
            51: "       ",
            53: "   ",
            55: "      ",
            61: "    ",
            63: " ",
            65: "  ",
            71: "    ",
            73: " ",
            75: "  ",
            80: "   ",
            95: "  "
        }
        return weather_codes.get(code, f"      {code}")
    
    def _calculate_severity(self, temp: float, wind: float, code: int) -> float:
        """         """
        severity = 0.0
        
        #       (주권적 자아)
        if temp < -10 or temp > 35:
            severity += 0.4
        elif temp < 0 or temp > 30:
            severity += 0.2
        
        #      
        if wind > 50:
            severity += 0.4
        elif wind > 30:
            severity += 0.2
        
        #         
        if code >= 95:  #   
            severity += 0.3
        elif code >= 65:  #   /  
            severity += 0.2
        
        return min(severity, 1.0)
    
    def _fallback_event(self, reason: str) -> SensorEvent:
        """       (API     )"""
        return SensorEvent(
            id=str(uuid.uuid4())[:8],
            sensor_type="WEATHER",
            severity=0.3,
            location="Unknown",
            description=f"            : {reason}",
            raw_data={"error": reason},
            is_real=False
        )


class SystemMetricsSense(RealSensor):
    """
              
    
                 Elysia  '     '       .
    - CPU               
    -              
    -                  
    """
    
    def __init__(self):
        super().__init__("System Metrics Sensor")
    
    def is_available(self) -> bool:
        return HAS_PSUTIL
    
    def sense(self) -> SensorEvent:
        if not self.is_available():
            return self._fallback_event("psutil         ")
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            #                  
            import platform
            if platform.system() == 'Windows':
                disk = psutil.disk_usage('C:\\')
            else:
                disk = psutil.disk_usage('/')
            
            #        (코드 베이스 구조 로터)
            severity = 0.0
            if cpu_percent > 90:
                severity += 0.4
            elif cpu_percent > 70:
                severity += 0.2
            
            if memory.percent > 90:
                severity += 0.4
            elif memory.percent > 80:
                severity += 0.2
            
            if disk.percent > 95:
                severity += 0.2
            
            severity = min(severity, 1.0)
            
            description = (
                f"CPU: {cpu_percent:.1f}%, "
                f"   : {memory.percent:.1f}%, "
                f"   : {disk.percent:.1f}%"
            )
            
            raw_data = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            event = SensorEvent(
                id=str(uuid.uuid4())[:8],
                sensor_type="SYSTEM",
                severity=severity,
                location="Host System",
                description=description,
                raw_data=raw_data,
                is_real=True
            )
            
            self.last_event = event
            logger.info(f"  System Metrics: {description}")
            return event
            
        except Exception as e:
            logger.error(f"             : {e}")
            return self._fallback_event(str(e))
    
    def _fallback_event(self, reason: str) -> SensorEvent:
        return SensorEvent(
            id=str(uuid.uuid4())[:8],
            sensor_type="SYSTEM",
            severity=0.5,
            location="Unknown",
            description=f"            : {reason}",
            raw_data={"error": reason},
            is_real=False
        )


class TimeAwarenessSense(RealSensor):
    """
            
    
         ,   ,          .
                   .
    """
    
    def __init__(self, timezone_offset: int = 9):
        """
        Args:
            timezone_offset: UTC            (   : KST +9)
        """
        super().__init__("Time Awareness Sensor")
        self.timezone_offset = timezone_offset
    
    def is_available(self) -> bool:
        return True  #         
    
    def sense(self) -> SensorEvent:
        from datetime import timedelta, timezone
        
        # UTC                
        utc_now = datetime.now(timezone.utc)
        local_tz = timezone(timedelta(hours=self.timezone_offset))
        now = utc_now.astimezone(local_tz)
        hour = now.hour
        weekday = now.strftime("%A")
        month = now.month
        
        #       
        if 5 <= hour < 12:
            time_of_day = "  "
            time_mood = "      "
        elif 12 <= hour < 18:
            time_of_day = "  "
            time_mood = "   "
        elif 18 <= hour < 22:
            time_of_day = "  "
            time_mood = "  "
        else:
            time_of_day = " "
            time_mood = "   "
        
        #      
        if month in [3, 4, 5]:
            season = " "
        elif month in [6, 7, 8]:
            season = "  "
        elif month in [9, 10, 11]:
            season = "  "
        else:
            season = "  "
        
        #     (코드 베이스 구조 로터)
        severity = 0.5
        if hour < 6 or hour > 22:
            severity = 0.3
        if weekday in ["Saturday", "Sunday"]:
            severity *= 0.8
        
        description = f"{season} {time_of_day}, {weekday}, {time_mood}"
        
        event = SensorEvent(
            id=str(uuid.uuid4())[:8],
            sensor_type="TIME",
            severity=severity,
            location="Temporal Space",
            description=description,
            raw_data={
                "hour": hour,
                "weekday": weekday,
                "month": month,
                "season": season,
                "time_of_day": time_of_day,
                "time_mood": time_mood
            },
            is_real=True
        )
        
        self.last_event = event
        logger.info(f"  Time Awareness: {description}")
        return event


class SensorHub:
    """
          (Sensor Hub)
    
                       .
    """
    
    def __init__(self, latitude: float = 37.5665, longitude: float = 126.9780):
        self.sensors: List[RealSensor] = [
            RealWeatherSense(latitude, longitude),
            SystemMetricsSense(),
            TimeAwarenessSense()
        ]
        self.last_readings: Dict[str, SensorEvent] = {}
        logger.info(f"   Sensor Hub initialized with {len(self.sensors)} sensors")
    
    def sense_all(self) -> Dict[str, SensorEvent]:
        """              """
        readings = {}
        for sensor in self.sensors:
            if sensor.is_available():
                event = sensor.sense()
                readings[sensor.name] = event
        
        self.last_readings = readings
        return readings
    
    def get_summary(self) -> str:
        """        """
        if not self.last_readings:
            self.sense_all()
        
        lines = ["===           ==="]
        for name, event in self.last_readings.items():
            real_marker = " " if event.is_real else " "
            lines.append(f"{real_marker} [{event.sensor_type}] {event.description}")
        
        return "\n".join(lines)
    
    def get_average_severity(self) -> float:
        """         """
        if not self.last_readings:
            return 0.0
        
        severities = [e.severity for e in self.last_readings.values()]
        return sum(severities) / len(severities)


#      
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== Real World Sensors Demo ===\n")
    
    hub = SensorHub()
    readings = hub.sense_all()
    
    print("\n" + hub.get_summary())
    print(f"\n      : {hub.get_average_severity():.2f}")
