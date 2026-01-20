"""
FractalLogSphere: 프랙탈 동형성 원칙에 따른 로그 시스템

모든 엘리시아 시스템은 같은 구조를 따른다:
- Ring Buffer: 최근 N개만 유지
- HyperSphere Storage: 중요 이벤트만 4D 좌표로 저장
- Natural Decay: 시간에 따른 자연적 망각
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
    """프랙탈 로그 엔트리"""
    timestamp: float
    level: str
    name: str
    message: str
    resonance: float = 0.5  # 0.0 (무시) ~ 1.0 (절대 잊지 않음)
    
    # 4D 좌표 (HyperSphere 저장용)
    theta: float = 0.0  # 시간 축 (timestamp의 주기적 맵핑)
    phi: float = 0.0    # 중요도 축 (level 기반)
    psi: float = 0.0    # 맥락 축 (logger name 해시)
    r: float = 1.0      # 깊이 (resonance)
    
    def __post_init__(self):
        # Level → 중요도 맵핑
        level_map = {'DEBUG': 0.2, 'INFO': 0.4, 'WARNING': 0.6, 'ERROR': 0.8, 'CRITICAL': 1.0}
        self.phi = level_map.get(self.level, 0.5) * math.pi
        
        # Timestamp → 주기적 각도 (하루 = 2π)
        day_progress = (self.timestamp % 86400) / 86400
        self.theta = day_progress * 2 * math.pi
        
        # Logger name → 맥락 각도
        name_hash = hash(self.name) % 1000
        self.psi = (name_hash / 1000) * 2 * math.pi
        
        # Resonance → 깊이
        self.r = max(0.1, self.resonance)
    
    def to_dict(self) -> dict:
        return asdict(self)


class FractalLogSphere:
    """
    프랙탈 동형성 원칙에 따른 로그 시스템.
    
    Memory ≅ Log ≅ Document ≅ Context
    모든 것이 같은 원리로 움직인다.
    """
    
    def __init__(
        self, 
        ring_size: int = 1000, 
        decay_rate: float = 0.001,
        sphere_path: Optional[str] = None,
        decay_interval: float = 60.0  # 1분마다 decay
    ):
        """
        Args:
            ring_size: Ring Buffer 최대 크기
            decay_rate: 매 decay 주기마다 감소하는 resonance
            sphere_path: HyperSphere 영구 저장 경로
            decay_interval: Decay 주기 (초)
        """
        self.ring: deque = deque(maxlen=ring_size)
        self.sphere: Dict[str, LogEntry] = {}
        self.decay_rate = decay_rate
        self.sphere_path = Path(sphere_path) if sphere_path else Path("data/06_Structure/Logs/log_sphere.json")
        self.decay_interval = decay_interval
        
        # 통계
        self.total_logged = 0
        self.total_decayed = 0
        
        # Background decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 저장된 sphere 로드
        self._load_sphere()
    
    def log(
        self, 
        level: str, 
        name: str, 
        message: str, 
        resonance: Optional[float] = None
    ) -> LogEntry:
        """
        로그 기록
        
        Args:
            level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            name: 로거 이름
            message: 메시지
            resonance: 중요도 (None이면 level 기반 자동 계산)
        """
        # Resonance 자동 계산
        if resonance is None:
            level_resonance = {
                'DEBUG': 0.1,
                'INFO': 0.3,
                'WARNING': 0.6,
                'ERROR': 0.8,
                'CRITICAL': 1.0
            }
            resonance = level_resonance.get(level, 0.3)
            
            # 특별한 키워드가 있으면 resonance 증가
            if any(kw in message for kw in ['✨', '🚀', '💡', 'CRITICAL', 'FATAL']):
                resonance = min(1.0, resonance + 0.3)
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            name=name,
            message=message,
            resonance=resonance
        )
        
        # Ring Buffer에 추가 (항상)
        self.ring.append(entry)
        self.total_logged += 1
        
        # HyperSphere에 저장 (중요한 것만)
        if resonance > 0.6:
            coord_key = f"{entry.theta:.4f}_{entry.phi:.4f}_{entry.psi:.4f}"
            self.sphere[coord_key] = entry
        
        return entry
    
    def decay(self):
        """자연적 망각 - 공명도 낮은 것부터 제거"""
        keys_to_remove = []
        
        for key, entry in self.sphere.items():
            entry.resonance -= self.decay_rate
            entry.r = max(0.1, entry.resonance)  # r도 동기화
            
            if entry.resonance <= 0:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.sphere[key]
            self.total_decayed += 1
        
        # 변경 사항 저장
        if keys_to_remove:
            self._save_sphere()
    
    def search_by_resonance(self, min_resonance: float = 0.5) -> List[LogEntry]:
        """공명도 기반 검색"""
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
        """4D 좌표 근처 검색"""
        results = []
        for entry in self.sphere.values():
            # 유클리드 거리 계산 (단순화)
            dist = math.sqrt(
                (entry.theta - theta) ** 2 +
                (entry.phi - phi) ** 2 +
                (entry.psi - psi) ** 2
            )
            if dist <= radius:
                results.append(entry)
        return results
    
    def get_recent(self, n: int = 100) -> List[LogEntry]:
        """Ring Buffer에서 최근 n개 조회"""
        return list(self.ring)[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            "ring_size": len(self.ring),
            "ring_capacity": self.ring.maxlen,
            "sphere_size": len(self.sphere),
            "total_logged": self.total_logged,
            "total_decayed": self.total_decayed,
            "decay_rate": self.decay_rate
        }
    
    def start_decay_thread(self):
        """Background decay thread 시작"""
        if self._running:
            return
        
        self._running = True
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()
        logger.info("🔄 FractalLogSphere decay thread started")
    
    def stop_decay_thread(self):
        """Background decay thread 중지"""
        self._running = False
        if self._decay_thread:
            self._decay_thread.join(timeout=5)
    
    def _decay_loop(self):
        """Decay 루프"""
        while self._running:
            time.sleep(self.decay_interval)
            self.decay()
    
    def _save_sphere(self):
        """HyperSphere 영구 저장"""
        try:
            self.sphere_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self.sphere.items()}
            with open(self.sphere_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save log sphere: {e}")
    
    def _load_sphere(self):
        """저장된 HyperSphere 로드"""
        try:
            if self.sphere_path.exists():
                with open(self.sphere_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, entry_dict in data.items():
                    self.sphere[key] = LogEntry(**entry_dict)
                logger.info(f"📂 Loaded {len(self.sphere)} entries from log sphere")
        except Exception as e:
            logger.warning(f"Failed to load log sphere: {e}")


class FractalLogHandler(logging.Handler):
    """
    Python logging 시스템과 FractalLogSphere를 연결하는 핸들러.
    기존 logging 인프라와 호환됩니다.
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
    글로벌 FractalLogSphere 인스턴스를 반환합니다.
    최초 호출 시 초기화됩니다.
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
            logger.info("🔮 FractalLogSphere initialized (Fractal Isomorphism Active)")
        
        return _global_fractal_log


def configure_fractal_logging(level: int = logging.INFO):
    """
    표준 Python logging을 FractalLogSphere로 라우팅합니다.
    기존 logging.basicConfig() 대신 사용합니다.
    """
    sphere = get_fractal_logger()
    handler = FractalLogHandler(sphere)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))
    
    # Root logger에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # 콘솔 출력도 유지 (Ring Buffer의 최근 것만 출력)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    logger.info("✅ Fractal logging configured (Linear accumulation → Fractal decay)")
    
    return sphere
