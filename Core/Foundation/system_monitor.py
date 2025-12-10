"""
System Monitor for Elysia v9.0
================================
Real-time monitoring of system health, performance, and hardware metrics.
"""

import time
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("SystemMonitor")

@dataclass
class SystemMetrics:
    """System metrics snapshot"""
    timestamp: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_free_mb: float = 0.0
    
    # Internal metrics
    energy_level: float = 0.0
    error_count: int = 0
    uptime: float = 0.0

class SystemMonitor:
    """
    Central system monitoring for Elysia.
    Tracks PHYSICAL HARDWARE to ensure self-preservation.
    """
    
    def __init__(self, cns=None):
        self.cns = cns
        self.metrics_history: List[SystemMetrics] = []
        self.start_time = time.time()
        
        # Safety Thresholds (Self-Regulation Limits)
        self.MAX_CPU_USAGE = 80.0
        self.MAX_MEMORY_USAGE = 85.0
        self.MIN_DISK_FREE_MB = 500.0
        
        logger.info("SystemMonitor initialized (Hardware Awareness Enabled)")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics (Hardware & Software)"""
        
        # 1. Get Hardware Stats
        hw = self._get_hardware_stats()
        
        # 2. Get Internal Stats (Mock/CNS)
        energy = 80.0 # Placeholder or from CNS
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=hw['cpu_percent'],
            memory_usage=hw['memory_percent'],
            disk_free_mb=hw['disk_free_mb'],
            energy_level=energy,
            uptime=time.time() - self.start_time
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
            
        return metrics

    def _get_hardware_stats(self) -> Dict[str, float]:
        """Reads real hardware stats using psutil"""
        stats = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_free_mb": 0.0
        }
        try:
            import psutil
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["memory_percent"] = psutil.virtual_memory().percent
            
            # Check the drive where script is running
            usage = psutil.disk_usage(os.getcwd())
            stats["disk_free_mb"] = usage.free / (1024 * 1024)
        except ImportError:
            # Fallback if psutil is not installed
            stats["cpu_percent"] = 15.0 # Simulated
            stats["memory_percent"] = 30.0
            stats["disk_free_mb"] = 10000.0 # Simulated 10GB free
            
        return stats
    
    def check_vital_signs(self) -> Dict[str, bool]:
        """
        Returns a Go/No-Go decision based on hardware state.
        Usage: If 'safe_to_create' is False, the ImaginationCore should PAUSE.
        """
        if not self.metrics_history:
            self.collect_metrics()
            
        latest = self.metrics_history[-1]
        
        is_cpu_safe = latest.cpu_usage < self.MAX_CPU_USAGE
        is_mem_safe = latest.memory_usage < self.MAX_MEMORY_USAGE
        is_disk_safe = latest.disk_free_mb > self.MIN_DISK_FREE_MB
        
        status = {
            "safe_to_create": is_cpu_safe and is_mem_safe and is_disk_safe,
            "cpu_ok": is_cpu_safe,
            "mem_ok": is_mem_safe,
            "disk_ok": is_disk_safe,
            "reason": ""
        }
        
        if not status["safe_to_create"]:
            reasons = []
            if not is_cpu_safe: reasons.append(f"CPU Overheat ({latest.cpu_usage}%)")
            if not is_mem_safe: reasons.append(f"Memory Full ({latest.memory_usage}%)")
            if not is_disk_safe: reasons.append(f"Disk Full ({latest.disk_free_mb:.0f}MB)")
            status["reason"] = ", ".join(reasons)
            
        return status

# Singleton instance
_monitor_instance: Optional[SystemMonitor] = None

def get_system_monitor(cns=None) -> SystemMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor(cns)
    return _monitor_instance
