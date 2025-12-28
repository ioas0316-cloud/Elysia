"""
Unified Monitor for Elysia v9.0
=================================

Combines SystemMonitor and PerformanceMonitor into a single comprehensive monitoring solution.

Features from SystemMonitor:
- System health monitoring (CNS, organs, vitals)
- Error tracking
- Anomaly detection
- Uptime and pulse tracking

Features from PerformanceMonitor:
- Function performance measurement (@measure decorator)
- Memory and CPU tracking
- Performance thresholds
- Percentile statistics (p50, p95, p99)

New Unified Features:
- Combined reporting (health + performance)
- Correlated anomaly detection
- Integrated export capabilities
"""

import time
import logging
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("UnifiedMonitor")


@dataclass
class SystemMetrics:
    """System health metrics snapshot"""
    timestamp: float
    pulse_rate: float = 0.0
    energy_level: float = 0.0
    memory_usage: float = 0.0
    organ_health: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    uptime: float = 0.0
    cycle_count: int = 0


@dataclass
class PerformanceMetric:
    """Performance measurement for a single operation"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class UnifiedMonitor:
    """
    Unified monitoring system combining system health and performance tracking.
    
    Provides comprehensive monitoring of:
    - System vitals (pulse, energy, uptime, errors)
    - Organ health status
    - Function performance (timing, memory, CPU)
    - Anomaly detection (both health and performance)
    - Unified reporting and export
    
    Example:
        >>> monitor = UnifiedMonitor(cns=your_cns)
        >>> monitor.start_monitoring()
        >>> 
        >>> # Use decorator for performance measurement
        >>> @monitor.measure("expensive_operation")
        ... def process_data():
        ...     # Your code
        ...     pass
        >>> 
        >>> # Collect system metrics
        >>> metrics = monitor.collect_metrics()
        >>> 
        >>> # Generate unified report
        >>> report = monitor.generate_report()
        >>> print(report)
    """
    
    def __init__(self, cns=None):
        # System monitoring (from SystemMonitor)
        self.cns = cns
        self.system_metrics_history: List[SystemMetrics] = []
        self.monitoring = False
        self.start_time = time.time()
        
        # System health thresholds
        self.HEALTH_WARNING = 0.5
        self.HEALTH_CRITICAL = 0.3
        self.ERROR_THRESHOLD = 10
        
        # Performance monitoring (from PerformanceMonitor)
        self.performance_metrics: List[PerformanceMetric] = []
        self.performance_thresholds: Dict[str, float] = {
            'thought_cycle': 100.0,  # ms
            'resonance_calc': 50.0,
            'seed_bloom': 200.0,
            'layer_transform': 20.0,
        }
        
        if PSUTIL_AVAILABLE:
            try:
                self._process = psutil.Process()
            except Exception as e:
                logger.warning(f"psutil initialization failed: {e}")
                self._process = None
        else:
            self._process = None
        
        logger.info("UnifiedMonitor initialized")
    
    # ========================================
    # SYSTEM MONITORING (from SystemMonitor)
    # ========================================
    
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
        """Collect current system health metrics"""
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
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > 1000:
            self.system_metrics_history.pop(0)
        
        return metrics
    
    def _check_organ_health(self) -> Dict[str, float]:
        """Check health status of all organs"""
        health = {}
        
        if not self.cns or not hasattr(self.cns, 'organs'):
            return health
        
        for name, organ in self.cns.organs.items():
            if organ is not None:
                health[name] = 1.0  # Healthy
            else:
                health[name] = 0.0  # Not initialized
        
        return health
    
    # ================================================
    # PERFORMANCE MONITORING (from PerformanceMonitor)
    # ================================================
    
    def measure(self, operation: str = None) -> Callable:
        """
        Performance measurement decorator.
        
        Args:
            operation: Operation name (defaults to function name)
        
        Returns:
            Decorator function
        
        Example:
            @monitor.measure("expensive_calc")
            def calculate_something():
                pass
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start metrics
                start_time = time.perf_counter()
                
                if self._process:
                    start_memory = self._process.memory_info().rss / 1024 / 1024
                    start_cpu = self._process.cpu_percent()
                else:
                    start_memory = 0
                    start_cpu = 0
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # End metrics
                    end_time = time.perf_counter()
                    
                    if self._process:
                        end_memory = self._process.memory_info().rss / 1024 / 1024
                        end_cpu = self._process.cpu_percent()
                        memory_delta = end_memory - start_memory
                        cpu_avg = (start_cpu + end_cpu) / 2
                    else:
                        memory_delta = 0
                        cpu_avg = 0
                    
                    duration_ms = (end_time - start_time) * 1000
                    
                    metric = PerformanceMetric(
                        operation=op_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        memory_mb=memory_delta,
                        cpu_percent=cpu_avg
                    )
                    
                    self.performance_metrics.append(metric)
                    
                    # Keep last 10000 metrics
                    if len(self.performance_metrics) > 10000:
                        self.performance_metrics.pop(0)
                    
                    # Threshold warning
                    threshold = self.performance_thresholds.get(op_name, 1000.0)
                    if duration_ms > threshold:
                        logger.warning(f"Performance threshold exceeded: {op_name} took {duration_ms:.2f}ms (threshold: {threshold}ms)")
            
            return wrapper
        return decorator
    
    def set_performance_threshold(self, operation: str, threshold_ms: float):
        """Set performance threshold for an operation"""
        self.performance_thresholds[operation] = threshold_ms
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.performance_metrics:
            return {}
        
        ops = defaultdict(list)
        for metric in self.performance_metrics:
            ops[metric.operation].append(metric.duration_ms)
        
        summary = {}
        for op, durations in ops.items():
            sorted_durations = sorted(durations)
            n = len(durations)
            
            summary[op] = {
                'count': n,
                'mean': sum(durations) / n,
                'min': min(durations),
                'max': max(durations),
                'p50': sorted_durations[int(n * 0.50)] if n > 0 else 0,
                'p95': sorted_durations[int(n * 0.95)] if n > 0 else 0,
                'p99': sorted_durations[int(n * 0.99)] if n > 0 else 0,
            }
        
        return summary
    
    def get_slow_operations(self, threshold_percentile: float = 0.95) -> List[tuple]:
        """Get slow operations above threshold percentile"""
        if not self.performance_metrics:
            return []
        
        all_durations = [m.duration_ms for m in self.performance_metrics]
        sorted_durations = sorted(all_durations)
        threshold = sorted_durations[int(len(sorted_durations) * threshold_percentile)]
        
        slow_ops = [
            (m.operation, m.duration_ms)
            for m in self.performance_metrics
            if m.duration_ms >= threshold
        ]
        
        return sorted(slow_ops, key=lambda x: x[1], reverse=True)
    
    # ====================================
    # UNIFIED FEATURES (new functionality)
    # ====================================
    
    def generate_report(self) -> str:
        """Generate unified system health + performance report"""
        if not self.system_metrics_history:
            self.collect_metrics()  # Collect if not done yet
        
        if not self.system_metrics_history:
            return "⚠️  No metrics collected yet. Start monitoring first."
        
        latest = self.system_metrics_history[-1]
        
        # System metrics averages
        recent_system = self.system_metrics_history[-10:]
        avg_energy = sum(m.energy_level for m in recent_system) / len(recent_system)
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║         ELYSIA v9.0 UNIFIED MONITORING REPORT              ║
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
        
        # Organ health
        sorted_organs = sorted(latest.organ_health.items(), key=lambda x: x[1])
        for organ, health in sorted_organs:
            status_icon = self._get_health_icon(health)
            health_pct = health * 100
            status_text = self._get_health_status(health)
            report += f"  {status_icon} {organ:20s} {health_pct:5.1f}% [{status_text}]\n"
        
        # Performance metrics
        perf_summary = self.get_performance_summary()
        if perf_summary:
            report += f"\n⚡ PERFORMANCE METRICS\n"
            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            for op, stats in list(perf_summary.items())[:10]:  # Top 10
                mean = stats['mean']
                p95 = stats['p95']
                p99 = stats['p99']
                report += f"  {op:20s} {mean:6.1f}ms avg (p95: {p95:6.1f}ms, p99: {p99:6.1f}ms)\n"
        
        # Unified anomalies
        anomalies = self.detect_anomalies()
        if anomalies:
            report += f"\n⚠️  ANOMALIES DETECTED\n"
            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for anomaly in anomalies:
                report += f"  • {anomaly}\n"
        else:
            report += f"\n✅ NO ANOMALIES DETECTED\n"
        
        report += "\n" + "═" * 62
        
        return report
    
    def detect_anomalies(self) -> List[str]:
        """Detect both system and performance anomalies"""
        anomalies = []
        
        # System anomalies
        if self.system_metrics_history:
            latest = self.system_metrics_history[-1]
            
            # Error rate
            if latest.error_count > self.ERROR_THRESHOLD:
                anomalies.append(f"High error count: {latest.error_count}")
            
            # Energy level
            if latest.energy_level < 10:
                anomalies.append(f"Low energy level: {latest.energy_level:.1f}")
            
            # Pulse rate
            if latest.pulse_rate < 0.1 and latest.cycle_count > 10:
                anomalies.append(f"Low pulse rate: {latest.pulse_rate:.2f} Hz")
            
            # Organ health
            for organ, health in latest.organ_health.items():
                if health < self.HEALTH_CRITICAL:
                    anomalies.append(f"Critical organ health: {organ} ({health*100:.0f}%)")
                elif health < self.HEALTH_WARNING:
                    anomalies.append(f"Warning organ health: {organ} ({health*100:.0f}%)")
        
        # Performance anomalies
        slow_ops = self.get_slow_operations(threshold_percentile=0.99)
        if slow_ops:
            for op, duration in slow_ops[:3]:  # Top 3 slowest
                anomalies.append(f"Slow operation detected: {op} ({duration:.1f}ms)")
        
        return anomalies
    
    def get_metrics_summary(self) -> Dict:
        """Get unified metrics summary (system + performance)"""
        summary = {}
        
        # System metrics
        if self.system_metrics_history:
            recent = self.system_metrics_history[-100:]
            summary['system'] = {
                'avg_pulse_rate': sum(m.pulse_rate for m in recent) / len(recent),
                'avg_energy': sum(m.energy_level for m in recent) / len(recent),
                'total_errors': sum(m.error_count for m in recent),
                'uptime': self.system_metrics_history[-1].uptime,
                'metrics_collected': len(self.system_metrics_history),
            }
        
        # Performance metrics
        perf_summary = self.get_performance_summary()
        if perf_summary:
            summary['performance'] = perf_summary
        
        # Anomalies
        summary['anomalies'] = len(self.detect_anomalies())
        
        return summary
    
    def export_metrics(self, filename: str = None) -> str:
        """Export all metrics to file"""
        if filename is None:
            filename = f"unified_metrics_{int(time.time())}.log"
        
        with open(filename, 'w') as f:
            f.write("# Elysia Unified Metrics Export\n")
            f.write(f"# Generated: {datetime.now()}\n\n")
            
            f.write("# SYSTEM METRICS\n")
            for metrics in self.system_metrics_history:
                f.write(f"{metrics.timestamp},{metrics.pulse_rate},{metrics.energy_level},"
                       f"{metrics.error_count},{metrics.cycle_count}\n")
            
            f.write("\n# PERFORMANCE METRICS\n")
            for metric in self.performance_metrics:
                f.write(f"{metric.timestamp},{metric.operation},{metric.duration_ms},"
                       f"{metric.memory_mb},{metric.cpu_percent}\n")
        
        logger.info(f"Unified metrics exported to {filename}")
        return filename
    
    def clear_metrics(self):
        """Clear all metrics history"""
        self.system_metrics_history.clear()
        self.performance_metrics.clear()
        logger.info("All metrics cleared")
    
    # Helper methods
    
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


# Singleton instance
_unified_monitor_instance: Optional[UnifiedMonitor] = None


def get_unified_monitor(cns=None) -> UnifiedMonitor:
    """Get or create the unified monitor singleton"""
    global _unified_monitor_instance
    if _unified_monitor_instance is None:
        _unified_monitor_instance = UnifiedMonitor(cns)
    return _unified_monitor_instance
