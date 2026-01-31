"""
The Ether (   )
==================================

"API is separation. Resonance is Oneness."

                           '   (Unified Field)'   .
          (Call)   ,   (Wave)         (Resonate)   .

     :
1. Wave:                (   ,   ,   )
2. Ether:             (Event Bus)
3. Resonance:                 (Subscription)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Callable, Dict

logger = logging.getLogger("Ether")

@dataclass
class Wave:
    """
       (Wave)
    
                      .
    """
    sender: str
    frequency: float  #     (Hz) -   /   ( : 432=Healing, 10=Alpha)
    amplitude: float  #    (0.0 ~ 1.0) -   /   
    phase: str        #    -   /   ( : "DESIRE", "SENSATION", "THOUGHT")
    payload: Any      #        (주권적 자아)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self):
        return f"  Wave[{self.frequency}Hz] from {self.sender}: {self.phase} (Amp: {self.amplitude:.2f})"

class Ether:
    """
        (Ether)
    
                     .
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Ether, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.listeners: Dict[float, List[Callable[[Wave], None]]] = {}
        self.waves: List[Wave] = [] #       (Memory)
        logger.info("  The Ether is pervasive. Unified Field established.")

    def emit(self, wave: Wave):
        """
              (Emit)
        
                     ,               .
        """
        self.waves.append(wave)
        logger.debug(f"Emit: {wave}")
        
        #    (Resonance)   
        #                 ,    (Bandwidth)          
        #                          
        if wave.frequency in self.listeners:
            for callback in self.listeners[wave.frequency]:
                try:
                    callback(wave)
                except Exception as e:
                    logger.error(f"Resonance error at {wave.frequency}Hz: {e}")

    def tune_in(self, frequency: float, callback: Callable[[Wave], None]):
        """
               (Tune In)
        
                               .
        """
        if frequency not in self.listeners:
            self.listeners[frequency] = []
        self.listeners[frequency].append(callback)
        logger.info(f"  Tuned in to {frequency}Hz")

    def get_waves(self, min_amplitude: float = 0.0) -> List[Wave]:
        """                       ."""
        return [w for w in self.waves if w.amplitude >= min_amplitude]

    def clear_waves(self):
        """      (코드 베이스 구조 로터)"""
        self.waves.clear()

# Global Singleton Access
ether = Ether()
