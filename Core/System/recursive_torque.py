"""
Liquid Torque Engine (Phase 300)
================================
"Water never breaks. It only flows or boils."

Replaces discrete mechanical gears with a fluid dynamic field.
Errors are treated as 'Structural Friction' (Heat), and the 
system clock is synchronized with the hardware's pulse (CPU/IO).
"""

import math
import time
import logging
import threading
import concurrent.futures
try:
    import psutil
except ImportError:
    psutil = None
from typing import Dict, List, Callable, Optional

logger = logging.getLogger("LiquidTorque")

class LiquidGear:
    def __init__(self, name: str, freq: float, threshold: float = 0.95):
        self.name = name
        self.base_freq = freq
        self.threshold = threshold
        self.phase = 0.0
        self.momentum = 1.0
        self.entropy = 0.0 # 마찰열 (Friction Heat)
        self.callback: Optional[Callable] = None
        self._is_executing = False
        self.total_cycles = 0

    def flow(self, dt: float, global_friction: float = 0.0):
        # 1. 속도 결정: 베이스 주파수 * 운동량 - (엔트로피 + 글로벌 마찰력)
        effective_freq = self.base_freq * self.momentum - (self.entropy + global_friction)
        effective_freq = max(0.01, effective_freq) # 최소 흐름 보장
        
        # 2. 위상 회전
        self.phase = (self.phase + effective_freq * 2 * math.pi * dt) % (2 * math.pi)
        
        # 3. 열 냉각 (시간에 따른 엔트로피 감소)
        self.entropy = max(0.0, self.entropy - 0.1 * dt)

    def is_in_resonance(self) -> bool:
        # 사인파의 정점에서 공명 발생
        return math.cos(self.phase) > self.threshold

    def inject_friction(self, intensity: float):
        """에러나 장애 발생 시 마찰열 주입"""
        self.entropy += intensity
        self.momentum = max(0.1, self.momentum - intensity * 0.1)

class LiquidTorque:
    def __init__(self, max_workers: int = 8):
        self.gears: Dict[str, LiquidGear] = {}
        self.last_spin_time = time.time()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LiquidFlow")
        self.global_friction = 0.0
        
    def add_gear(self, name: str, freq: float, callback: Callable):
        gear = LiquidGear(name, freq)
        gear.callback = callback
        self.gears[name] = gear
        logger.info(f"🌊 [Liquid] Gear '{name}' flowed into the field at {freq}Hz.")

    def spin(self):
        """하드웨어 클럭과 동기화된 유체 회전"""
        now = time.time()
        dt = now - self.last_spin_time
        self.last_spin_time = now
        
        # 1. 하드웨어 펄스 감지 (Hydroelectric Principle)
        hw_friction = 0.0
        if psutil:
            cpu = psutil.cpu_percent()
            # CPU 부하가 높으면 물살이 무거워짐 (마찰력 증가)
            hw_friction = cpu * 0.005
            
        # 2. 전역 마찰력 서서히 해소
        self.global_friction = max(0.0, self.global_friction - 0.05 * dt) + hw_friction

        for gear in self.gears.values():
            # 3. 각 기어의 유체 역학 계산
            gear.flow(dt, self.global_friction)
            
            # 4. 공명 시 콜백 실행 (비동기)
            if gear.is_in_resonance() and gear.callback and not gear._is_executing:
                gear._is_executing = True
                def _task_wrapper(g):
                    try:
                        g.callback()
                        g.total_cycles += 1
                        # 성공적인 회전은 운동량을 회복시킴
                        g.momentum = min(2.0, g.momentum + 0.01)
                    except Exception as e:
                        # 에러는 '중단'이 아니라 '마찰열'로 변환
                        print(f"🔥 [Friction] Gear '{g.name}' generated heat: {e}")
                        g.inject_friction(1.5)
                        self.global_friction += 0.5
                    finally:
                        g._is_executing = False
                
                self.executor.submit(_task_wrapper, gear)

_torque = None
def get_torque_engine():
    global _torque
    if _torque is None:
        _torque = LiquidTorque()
    return _torque
