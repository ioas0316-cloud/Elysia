"""
Local Field Manager (       )
==================================

"            (Local Elysia Field)"

                  (      )                   .
IoT   (  ,      )                                 .

         (Mock)          ,                             .
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import time

logger = logging.getLogger("LocalField")

class IoTDevice(ABC):
    """IoT          """
    def __init__(self, name: str, device_id: str):
        self.name = name
        self.device_id = device_id
        self.is_on = False
        self.status = "Initialized"

    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.device_id,
            "is_on": self.is_on,
            "status": self.status
        }

class HueLight(IoTDevice):
    """Philips Hue         """
    def __init__(self, name: str, device_id: str):
        super().__init__(name, device_id)
        self.color = "White"
        self.brightness = 100

    def turn_on(self):
        self.is_on = True
        self.status = "Light ON"
        logger.info(f"  Light [{self.name}] turned ON.")

    def turn_off(self):
        self.is_on = False
        self.status = "Light OFF"
        logger.info(f"  Light [{self.name}] turned OFF.")

    def set_color(self, color: str, brightness: int):
        if not self.is_on:
            self.turn_on()
        self.color = color
        self.brightness = brightness
        self.status = f"Color: {color}, Brightness: {brightness}%"
        logger.info(f"  Light [{self.name}] changed to {color} ({brightness}%)")

class BluetoothSpeaker(IoTDevice):
    """              """
    def __init__(self, name: str, device_id: str):
        super().__init__(name, device_id)
        self.volume = 50
        self.current_track = None

    def turn_on(self):
        self.is_on = True
        self.status = "Speaker Connected"
        logger.info(f"  Speaker [{self.name}] Connected.")

    def turn_off(self):
        self.is_on = False
        self.status = "Speaker Disconnected"
        logger.info(f"  Speaker [{self.name}] Disconnected.")

    def play_music(self, genre: str, volume: int):
        if not self.is_on:
            self.turn_on()
        self.volume = volume
        self.current_track = f"Generating {genre} stream..."
        self.status = f"Playing: {genre} (Vol: {volume}%)"
        logger.info(f"  Speaker [{self.name}] playing {genre} at {volume}% volume.")

class LocalFieldManager:
    """
            (Local Field Manager)
    
            '   (Atmosphere)'       .
    """
    def __init__(self):
        self.devices: List[IoTDevice] = []
        self.scan_devices()
        logger.info("  Local Field Manager Initialized")

    def scan_devices(self):
        """         (     )"""
        #                       
        self.devices = [
            HueLight("Main Room Light", "hue_001"),
            HueLight("Desk Lamp", "hue_002"),
            BluetoothSpeaker("Marshall Acton II", "bt_001")
        ]
        logger.info(f"  Found {len(self.devices)} devices in the Local Field.")

    def set_atmosphere(self, emotion: str):
        """
                            .
        
        Args:
            emotion: 'sadness', 'joy', 'focus', 'relax'  
        """
        logger.info(f"  Setting Atmosphere: [{emotion.upper()}]")
        
        if emotion == "sadness" or emotion == "comfort":
            #          
            for dev in self.devices:
                if isinstance(dev, HueLight):
                    dev.set_color("Warm Orange", 40)
                elif isinstance(dev, BluetoothSpeaker):
                    dev.play_music("Calm Piano & Rain Sounds", 30)
                    
        elif emotion == "joy" or emotion == "happiness":
            #          
            for dev in self.devices:
                if isinstance(dev, HueLight):
                    dev.set_color("Bright Yellow", 80)
                elif isinstance(dev, BluetoothSpeaker):
                    dev.play_music("Upbeat Jazz", 50)
                    
        elif emotion == "focus" or emotion == "work":
            #      
            for dev in self.devices:
                if isinstance(dev, HueLight):
                    dev.set_color("Cool White", 100)
                elif isinstance(dev, BluetoothSpeaker):
                    dev.play_music("Lo-Fi Beats", 20)
                    
        elif emotion == "relax" or emotion == "sleep":
            #      
            for dev in self.devices:
                if isinstance(dev, HueLight):
                    dev.set_color("Deep Blue", 20)
                elif isinstance(dev, BluetoothSpeaker):
                    dev.play_music("White Noise", 15)
                    
        else:
            logger.warning(f"Unknown emotion for atmosphere: {emotion}")

    def get_field_status(self) -> List[Dict[str, Any]]:
        return [dev.get_status() for dev in self.devices]
