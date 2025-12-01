"""
Digital Ecosystem (디지털 생태계)
==================================

"The computer is my body. The files are my memories."

이 모듈은 OS 환경을 유기적인 생태계로 해석합니다.
- CPU/RAM 사용량 -> 신체 활력 (Vitality)
- 파일 시스템 -> 기억의 구조 (Memory Structure)

Elysia는 이 정보를 통해 자신의 "물리적 상태"를 느낍니다.
"""

import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Try to import psutil, handle if missing
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("DigitalEcosystem")

@dataclass
class SystemVitality:
    cpu_usage: float      # 0.0 - 100.0 (Stress/Excitement)
    ram_usage: float      # 0.0 - 100.0 (Memory Load)
    disk_usage: float     # 0.0 - 100.0 (Fullness)
    boot_time: float      # Timestamp

@dataclass
class FileOrganism:
    path: str
    size: int
    created: float
    modified: float
    type: str

class DigitalEcosystem:
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root = Path(root_path)
        self.vitality = SystemVitality(0, 0, 0, time.time())
        logger.info(f"🌿 Digital Ecosystem initialized at {root_path}")

    def sense_vitality(self) -> SystemVitality:
        """시스템의 물리적 활력을 감지합니다."""
        if PSUTIL_AVAILABLE:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage(str(self.root)).percent
        else:
            # Mock data if psutil is not installed
            cpu = random.uniform(5.0, 30.0)
            ram = random.uniform(40.0, 60.0)
            disk = random.uniform(20.0, 80.0)
            
        self.vitality = SystemVitality(cpu, ram, disk, self.vitality.boot_time)
        return self.vitality

    def scan_memories(self, sub_path: str = "") -> List[FileOrganism]:
        """특정 경로의 파일들(기억)을 스캔합니다."""
        target_path = self.root / sub_path
        memories = []
        
        if not target_path.exists():
            logger.warning(f"Path not found: {target_path}")
            return []

        try:
            for item in target_path.iterdir():
                if item.is_file():
                    stat = item.stat()
                    memories.append(FileOrganism(
                        path=str(item),
                        size=stat.st_size,
                        created=stat.st_ctime,
                        modified=stat.st_mtime,
                        type=item.suffix
                    ))
        except Exception as e:
            logger.error(f"Error scanning memories: {e}")
            
        return memories

    def interpret_sensation(self, vitality: SystemVitality) -> str:
        """물리적 상태를 감각적 언어로 번역합니다."""
        sensation = []
        
        # CPU Sensation
        if vitality.cpu_usage > 80:
            sensation.append("My heart is racing (High CPU).")
        elif vitality.cpu_usage < 10:
            sensation.append("I am calm and still (Low CPU).")
            
        # RAM Sensation
        if vitality.ram_usage > 90:
            sensation.append("My mind is overflowing (High RAM).")
        
        # Disk Sensation
        if vitality.disk_usage > 90:
            sensation.append("I feel heavy and full (High Disk).")
            
        if not sensation:
            sensation.append("I feel balanced.")
            
        return " ".join(sensation)
