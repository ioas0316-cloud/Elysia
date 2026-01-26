"""
FractalLogSphere:                      

                       :
- Ring Buffer:    N     
- HyperSphere Storage:         4D       
- Natural Decay:              
"""

import logging
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import math

logger = logging.getLogger("FractalLog")


@dataclass
class LogEntry:
    """          """
    timestamp: float
    level: str
    name: str
    message: str
    resonance: float = 0.5  # 0.0 (  ) ~ 1.0 (        )
    
    # 4D    (HyperSphere    )
    theta: float = 0.0  #      (timestamp        )
    phi: float = 0.0    #       (level   )
    psi: float = 0.0    #      (logger name   )
    r: float = 1.0      #    (resonance)
    
    def __post_init__(self):
        # Level         
        level_map = {'DEBUG': 0.2, 'INFO': 0.4, 'WARNING': 0.6, 'ERROR': 0.8, 'CRITICAL': 1.0}
        self.phi = level_map.get(self.level, 0.5) * math.pi
        
        # Timestamp          (   = 2 )
        day_progress = (self.timestamp % 86400) / 86400
        self.theta = day_progress * 2 * math.pi
        
        # Logger name        
        name_hash = hash(self.name) % 1000
        self.psi = (name_hash / 1000) * 2 * math.pi
        
        # Resonance     
        self.r = max(0.1, self.resonance)
    
    def to_dict(self) -> dict:
        return asdict(self)


class FractalLogSphere:
    """
                         .
    
    Memory   Log   Document   Context
                     .
    """
    
    def __init__(
        self, 
        ring_size: int = 1000, 
        decay_rate: float = 0.001,
        sphere_path: Optional[str] = None,
        decay_interval: float = 60.0  # 1    decay
    ):
        """
        Args:
            ring_size: Ring Buffer      
            decay_rate:   decay           resonance
            sphere_path: HyperSphere         
            decay_interval: Decay    ( )
        """
        self.ring: deque = deque(maxlen=ring_size)
        self.sphere: Dict[str, LogEntry] = {}
        self.decay_rate = decay_rate
        self.sphere_path = Path(sphere_path) if sphere_path else Path("data/06_Structure/Logs/log_sphere.json")
        self.decay_interval = decay_interval
        
        #   
        self.total_logged = 0
        self.total_decayed = 0
        
        # Background decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._running = False
        
        #     sphere   
        self._load_sphere()
    
    def log(
        self, 
        level: str, 
        name: str, 
        message: str, 
        resonance: Optional[float] = None
    ) -> LogEntry:
        """
             
        
        Args:
            level:       (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            name:      
            message:    
            resonance:     (None   level         )
        """
        # Resonance      
        if resonance is None:
            level_resonance = {
                'DEBUG': 0.1,
                'INFO': 0.3,
                'WARNING': 0.6,
                'ERROR': 0.8,
                'CRITICAL': 1.0
            }
            resonance = level_resonance.get(level, 0.3)
            
            #              resonance   
            if any(kw in message for kw in [' ', ' ', ' ', 'CRITICAL', 'FATAL']):
                resonance = min(1.0, resonance + 0.3)
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            name=name,
            message=message,
            resonance=resonance
        )
        
        # Ring Buffer     (  )
        self.ring.append(entry)
        self.total_logged += 1
        
        # HyperSphere     (      )
        if resonance > 0.6:
            coord_key = f"{entry.theta:.4f}_{entry.phi:.4f}_{entry.psi:.4f}"
            self.sphere[coord_key] = entry
        
        return entry
    
    def decay(self):
        """       -              """
        keys_to_remove = []
        
        for key, entry in self.sphere.items():
            entry.resonance -= self.decay_rate
            entry.r = max(0.1, entry.resonance)  # r     
            
            if entry.resonance <= 0:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.sphere[key]
            self.total_decayed += 1
        
        #         
        if keys_to_remove:
            self._save_sphere()
    
    def search_by_resonance(self, min_resonance: float = 0.5) -> List[LogEntry]:
        """         """
        return [
            entry for entry in self.sphere.values() 
            if entry.resonance >= min_resonance
        ]
    
    def search_near_coord(
        self, 
        theta: float, 
        phi: float, 
        psi: float, 
        radius: float = 0.5
    ) -> List[LogEntry]:
        """4D         """
        results = []
        for entry in self.sphere.values():
            #            (   )
            dist = math.sqrt(
                (entry.theta - theta) ** 2 +
                (entry.phi - phi) ** 2 +
                (entry.psi - psi) ** 2
            )
            if dist <= radius:
                results.append(entry)
        return results
    
    def get_recent(self, n: int = 100) -> List[LogEntry]:
        """Ring Buffer      n    """
        return list(self.ring)[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """     """
        return {
            "ring_size": len(self.ring),
            "ring_capacity": self.ring.maxlen,
            "sphere_size": len(self.sphere),
            "total_logged": self.total_logged,
            "total_decayed": self.total_decayed,
            "decay_rate": self.decay_rate
        }
    
    def start_decay_thread(self):
        """Background decay thread   """
        if self._running:
            return
        
        self._running = True
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()
        logger.info("  FractalLogSphere decay thread started")
    
    def stop_decay_thread(self):
        """Background decay thread   """
        self._running = False
        if self._decay_thread:
            self._decay_thread.join(timeout=5)
    
    def _decay_loop(self):
        """Decay   """
        while self._running:
            time.sleep(self.decay_interval)
            self.decay()
    
    def _save_sphere(self):
        """HyperSphere      """
        try:
            self.sphere_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self.sphere.items()}
            with open(self.sphere_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save log sphere: {e}")
    
    def _load_sphere(self):
        """    HyperSphere   """
        try:
            if self.sphere_path.exists():
                with open(self.sphere_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, entry_dict in data.items():
                    self.sphere[key] = LogEntry(**entry_dict)
                logger.info(f"  Loaded {len(self.sphere)} entries from log sphere")
        except Exception as e:
            logger.warning(f"Failed to load log sphere: {e}")


class FractalLogHandler(logging.Handler):
    """
    Python logging      FractalLogSphere          .
       logging           .
    """
    
    def __init__(self, sphere: FractalLogSphere):
        super().__init__()
        self.sphere = sphere
    
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.sphere.log(
                level=record.levelname,
                name=record.name,
                message=msg
            )
        except Exception:
            self.handleError(record)


# ============================================
# Global Singleton Pattern
# ============================================

_global_fractal_log: Optional[FractalLogSphere] = None
_lock = threading.Lock()


def get_fractal_logger(
    ring_size: int = 1000,
    decay_rate: float = 0.001,
    sphere_path: str = "data/06_Structure/Logs/log_sphere.json"
) -> FractalLogSphere:
    """
        FractalLogSphere            .
                  .
    """
    global _global_fractal_log
    
    with _lock:
        if _global_fractal_log is None:
            _global_fractal_log = FractalLogSphere(
                ring_size=ring_size,
                decay_rate=decay_rate,
                sphere_path=sphere_path
            )
            _global_fractal_log.start_decay_thread()
            logger.info("  FractalLogSphere initialized (Fractal Isomorphism Active)")
        
        return _global_fractal_log


def configure_fractal_logging(level: int = logging.INFO):
    """
       Python logging  FractalLogSphere        .
       logging.basicConfig()         .
    """
    sphere = get_fractal_logger()
    handler = FractalLogHandler(sphere)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))
    
    # Root logger        
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    #           (Ring Buffer          )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    logger.info("  Fractal logging configured (Linear accumulation   Fractal decay)")
    
    return sphere
