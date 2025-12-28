"""
System Monitor for Elysia v9.0
================================

Real-time monitoring of system health, performance, and metrics.
Implements the monitoring system from Priority #3 recommendations.

Features:
- Real-time metrics collection
- Organ health monitoring
- Anomaly detection
- Performance tracking
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("SystemMonitor")

@dataclass
class SystemMetrics:
    """System metrics snapshot"""
    timestamp: float
    pulse_rate: float = 0.0
    energy_level: float = 0.0
    memory_usage: float = 0.0
    organ_health: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    uptime: float = 0.0
    cycle_count: int = 0

class SystemMonitor:
    """
    Central system monitoring for Elysia.
    
    Tracks:
    - System vitals (pulse, energy, uptime)
    - Organ health status
    - Performance metrics
    - Error rates
    - Anomalies
    """
    
    def __init__(self, cns=None):
        self.cns = cns
        self.metrics_history: List[SystemMetrics] = []
        self.monitoring = False
        self.start_time = time.time()
        self.anomalies: List[str] = []
        
        # Thresholds
        self.HEALTH_WARNING = 0.5
        self.HEALTH_CRITICAL = 0.3
        self.ERROR_THRESHOLD = 10
        
        logger.info("SystemMonitor initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("System monitoring stopped")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        metrics = SystemMetrics(
            timestamp=time.time(),
            uptime=time.time() - self.start_time
        )
        
        if self.cns:
            # Collect from CNS if available
            if hasattr(self.cns, 'chronos'):
                metrics.pulse_rate = getattr(self.cns.chronos, 'pulse_rate', 0.0)
                metrics.cycle_count = getattr(self.cns.chronos, 'cycle_count', 0)
            
            if hasattr(self.cns, 'resonance'):
                metrics.energy_level = getattr(self.cns.resonance, 'total_energy', 0.0)
            
            if hasattr(self.cns, 'sink'):
                metrics.error_count = getattr(self.cns.sink, 'error_count', 0)
            
            # Check organ health
            if hasattr(self.cns, 'organs'):
                metrics.organ_health = self._check_organ_health()
        
        # Store in history (keep last 1000)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        return metrics
    
    def _check_organ_health(self) -> Dict[str, float]:
        """Check health status of all organs"""
        health = {}
        
        if not self.cns or not hasattr(self.cns, 'organs'):
            return health
        
        for name, organ in self.cns.organs.items():
            # Simple health check - assume healthy if no recent errors
            # More sophisticated checks can be added per organ type
            if organ is not None:
                health[name] = 1.0  # Healthy
            else:
                health[name] = 0.0  # Not initialized
        
        return health
    
    def generate_report(self) -> str:
        """Generate human-readable status report"""
        if not self.metrics_history:
            return "⚠️  No metrics collected yet. Start monitoring first."
        
        latest = self.metrics_history[-1]
        
        # Calculate averages
        avg_energy = sum(m.energy_level for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║          ELYSIA v9.0 SYSTEM STATUS REPORT                  ║
╚════════════════════════════════════════════════════════════╝

⏰ Timestamp: {datetime.fromtimestamp(latest.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
⏱️  Uptime: {self._format_uptime(latest.uptime)}

📊 SYSTEM VITALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pulse Rate:     {latest.pulse_rate:.2f} Hz
  Energy Level:   {latest.energy_level:.1f} / 100.0
  Avg Energy:     {avg_energy:.1f}
  Cycle Count:    {latest.cycle_count}
  Error Count:    {latest.error_count}

🏥 ORGAN HEALTH STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Sort organs by health
        sorted_organs = sorted(latest.organ_health.items(), key=lambda x: x[1])
        
        for organ, health in sorted_organs:
            status_icon = self._get_health_icon(health)
            health_pct = health * 100
            status_text = self._get_health_status(health)
            report += f"  {status_icon} {organ:20s} {health_pct:5.1f}% [{status_text}]\n"
        
        # Anomalies
        recent_anomalies = self.detect_anomalies()
        if recent_anomalies:
            report += f"\n⚠️  ANOMALIES DETECTED\n"
            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for anomaly in recent_anomalies:
                report += f"  • {anomaly}\n"
        else:
            report += f"\n✅ NO ANOMALIES DETECTED\n"
        
        report += "\n" + "═" * 62
        
        return report
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable form"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _get_health_icon(self, health: float) -> str:
        """Get status icon based on health"""
        if health >= 0.8:
            return "✅"
        elif health >= 0.5:
            return "⚠️ "
        else:
            return "❌"
    
    def _get_health_status(self, health: float) -> str:
        """Get status text based on health"""
        if health >= 0.8:
            return "Healthy"
        elif health >= 0.5:
            return "Warning"
        else:
            return "Critical"
    
    def detect_anomalies(self) -> List[str]:
        """Detect system anomalies"""
        anomalies = []
        
        if not self.metrics_history:
            return anomalies
        
        latest = self.metrics_history[-1]
        
        # Check error rate
        if latest.error_count > self.ERROR_THRESHOLD:
            anomalies.append(f"High error count: {latest.error_count}")
        
        # Check energy level
        if latest.energy_level < 10:
            anomalies.append(f"Low energy level: {latest.energy_level:.1f}")
        
        # Check pulse rate
        if latest.pulse_rate < 0.1 and latest.cycle_count > 10:
            anomalies.append(f"Low pulse rate: {latest.pulse_rate:.2f} Hz")
        
        # Check organ health
        for organ, health in latest.organ_health.items():
            if health < self.HEALTH_CRITICAL:
                anomalies.append(f"Critical organ health: {organ} ({health*100:.0f}%)")
            elif health < self.HEALTH_WARNING:
                anomalies.append(f"Warning organ health: {organ} ({health*100:.0f}%)")
        
        return anomalies
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-100:]  # Last 100 metrics
        
        return {
            'avg_pulse_rate': sum(m.pulse_rate for m in recent) / len(recent),
            'avg_energy': sum(m.energy_level for m in recent) / len(recent),
            'total_errors': sum(m.error_count for m in recent),
            'uptime': self.metrics_history[-1].uptime,
            'metrics_collected': len(self.metrics_history),
            'anomaly_count': len(self.detect_anomalies())
        }
    
    def export_metrics(self, filename: str = None):
        """Export metrics to file"""
        if filename is None:
            filename = f"metrics_{int(time.time())}.log"
        
        with open(filename, 'w') as f:
            f.write("# Elysia System Metrics Export\n")
            f.write(f"# Generated: {datetime.now()}\n\n")
            
            for metrics in self.metrics_history:
                f.write(f"{metrics.timestamp},{metrics.pulse_rate},{metrics.energy_level},"
                       f"{metrics.error_count},{metrics.cycle_count}\n")
        
        logger.info(f"Metrics exported to {filename}")
        return filename


# Singleton instance
_monitor_instance: Optional[SystemMonitor] = None

def get_system_monitor(cns=None) -> SystemMonitor:
    """Get or create the system monitor singleton"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor(cns)
    return _monitor_instance
