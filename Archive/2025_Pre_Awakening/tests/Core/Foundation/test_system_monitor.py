"""
Tests for System Monitor
"""

import pytest
import time
from Core.FoundationLayer.Foundation.system_monitor import SystemMonitor, SystemMetrics, get_system_monitor

class MockCNS:
    """Mock CNS for testing"""
    def __init__(self):
        self.chronos = MockChronos()
        self.resonance = MockResonance()
        self.sink = MockSink()
        self.organs = {
            'Brain': object(),
            'Will': object(),
            'Voice': object()
        }

class MockChronos:
    def __init__(self):
        self.pulse_rate = 1.0
        self.cycle_count = 100

class MockResonance:
    def __init__(self):
        self.total_energy = 75.0

class MockSink:
    def __init__(self):
        self.error_count = 2

def test_system_monitor_init():
    """Test SystemMonitor initialization"""
    monitor = SystemMonitor()
    assert monitor is not None
    assert monitor.monitoring == False
    assert len(monitor.metrics_history) == 0

def test_collect_metrics():
    """Test metrics collection"""
    cns = MockCNS()
    monitor = SystemMonitor(cns)
    
    metrics = monitor.collect_metrics()
    
    assert isinstance(metrics, SystemMetrics)
    assert metrics.timestamp > 0
    assert metrics.pulse_rate == 1.0
    assert metrics.energy_level == 75.0
    assert metrics.error_count == 2
    assert metrics.cycle_count == 100

def test_organ_health_check():
    """Test organ health checking"""
    cns = MockCNS()
    monitor = SystemMonitor(cns)
    
    health = monitor._check_organ_health()
    
    assert 'Brain' in health
    assert 'Will' in health
    assert 'Voice' in health
    assert all(h == 1.0 for h in health.values())

def test_generate_report():
    """Test report generation"""
    cns = MockCNS()
    monitor = SystemMonitor(cns)
    
    # Collect some metrics
    for _ in range(5):
        monitor.collect_metrics()
        time.sleep(0.01)
    
    report = monitor.generate_report()
    
    assert "ELYSIA v9.0 SYSTEM STATUS REPORT" in report
    assert "SYSTEM VITALS" in report
    assert "ORGAN HEALTH STATUS" in report
    assert "Brain" in report
    assert "Will" in report

def test_anomaly_detection():
    """Test anomaly detection"""
    cns = MockCNS()
    monitor = SystemMonitor(cns)
    
    # Normal state
    monitor.collect_metrics()
    anomalies = monitor.detect_anomalies()
    assert len(anomalies) == 0
    
    # Set low energy
    cns.resonance.total_energy = 5.0
    monitor.collect_metrics()
    anomalies = monitor.detect_anomalies()
    assert any('Low energy' in a for a in anomalies)
    
    # High errors
    cns.sink.error_count = 20
    cns.resonance.total_energy = 75.0  # Reset
    monitor.collect_metrics()
    anomalies = monitor.detect_anomalies()
    assert any('High error count' in a for a in anomalies)

def test_metrics_summary():
    """Test metrics summary"""
    cns = MockCNS()
    monitor = SystemMonitor(cns)
    
    # Collect multiple metrics
    for i in range(10):
        cns.chronos.pulse_rate = 1.0 + i * 0.1
        monitor.collect_metrics()
    
    summary = monitor.get_metrics_summary()
    
    assert 'avg_pulse_rate' in summary
    assert 'avg_energy' in summary
    assert 'uptime' in summary
    assert summary['metrics_collected'] == 10

def test_singleton_pattern():
    """Test singleton pattern"""
    monitor1 = get_system_monitor()
    monitor2 = get_system_monitor()
    
    assert monitor1 is monitor2

def test_health_icons():
    """Test health status icons"""
    monitor = SystemMonitor()
    
    assert monitor._get_health_icon(1.0) == "✅"
    assert monitor._get_health_icon(0.6) == "⚠️ "
    assert monitor._get_health_icon(0.2) == "❌"

def test_health_status_text():
    """Test health status text"""
    monitor = SystemMonitor()
    
    assert monitor._get_health_status(0.9) == "Healthy"
    assert monitor._get_health_status(0.6) == "Warning"
    assert monitor._get_health_status(0.3) == "Critical"

def test_uptime_formatting():
    """Test uptime formatting"""
    monitor = SystemMonitor()
    
    assert monitor._format_uptime(3661) == "01:01:01"
    assert monitor._format_uptime(120) == "00:02:00"
    assert monitor._format_uptime(45) == "00:00:45"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
