"""
Tests for Unified Monitor
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from Core.Foundation.unified_monitor import UnifiedMonitor, get_unified_monitor, SystemMetrics


class TestUnifiedMonitor:
    """Test suite for unified monitoring"""
    
    def test_initialization(self):
        """Test monitor can be initialized"""
        monitor = UnifiedMonitor()
        assert monitor is not None
        assert monitor.system_metrics_history == []
        assert monitor.performance_metrics == []
        assert not monitor.monitoring
    
    def test_start_stop_monitoring(self):
        """Test monitoring can be started and stopped"""
        monitor = UnifiedMonitor()
        
        assert not monitor.monitoring
        monitor.start_monitoring()
        assert monitor.monitoring
        
        monitor.stop_monitoring()
        assert not monitor.monitoring
    
    def test_collect_metrics_without_cns(self):
        """Test metrics collection without CNS"""
        monitor = UnifiedMonitor()
        monitor.start_monitoring()
        
        metrics = monitor.collect_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp > 0
        assert metrics.uptime >= 0
        assert len(monitor.system_metrics_history) == 1
    
    def test_collect_metrics_with_cns(self):
        """Test metrics collection with CNS"""
        # Mock CNS
        cns = Mock()
        cns.chronos = Mock(pulse_rate=1.5, cycle_count=100)
        cns.resonance = Mock(total_energy=85.0)
        cns.sink = Mock(error_count=2)
        cns.organs = {'Voice': Mock(), 'Brain': Mock()}
        
        monitor = UnifiedMonitor(cns=cns)
        metrics = monitor.collect_metrics()
        
        assert metrics.pulse_rate == 1.5
        assert metrics.cycle_count == 100
        assert metrics.energy_level == 85.0
        assert metrics.error_count == 2
        assert 'Voice' in metrics.organ_health
        assert 'Brain' in metrics.organ_health
    
    def test_performance_measure_decorator(self):
        """Test performance measurement decorator"""
        monitor = UnifiedMonitor()
        
        @monitor.measure("test_operation")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert len(monitor.performance_metrics) == 1
        assert monitor.performance_metrics[0].operation == "test_operation"
        assert monitor.performance_metrics[0].duration_ms >= 10.0
    
    def test_measure_default_name(self):
        """Test decorator uses function name if operation not specified"""
        monitor = UnifiedMonitor()
        
        @monitor.measure()
        def my_function():
            return "ok"
        
        my_function()
        
        assert len(monitor.performance_metrics) == 1
        assert monitor.performance_metrics[0].operation == "my_function"
    
    def test_performance_threshold(self):
        """Test performance threshold detection"""
        monitor = UnifiedMonitor()
        monitor.set_performance_threshold("slow_op", 5.0)  # 5ms threshold
        
        @monitor.measure("slow_op")
        def slow_function():
            time.sleep(0.01)  # 10ms
        
        slow_function()
        
        # Metric should be collected despite exceeding threshold
        assert len(monitor.performance_metrics) == 1
    
    def test_get_performance_summary(self):
        """Test performance summary statistics"""
        monitor = UnifiedMonitor()
        
        @monitor.measure("op1")
        def operation1():
            time.sleep(0.01)
        
        @monitor.measure("op2")
        def operation2():
            time.sleep(0.02)
        
        # Run operations
        operation1()
        operation1()
        operation2()
        
        summary = monitor.get_performance_summary()
        
        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 2
        assert summary["op2"]["count"] == 1
        assert "mean" in summary["op1"]
        assert "p95" in summary["op1"]
    
    def test_detect_anomalies_high_errors(self):
        """Test anomaly detection for high error count"""
        cns = Mock()
        cns.chronos = Mock(pulse_rate=1.0, cycle_count=50)
        cns.resonance = Mock(total_energy=50.0)
        cns.sink = Mock(error_count=20)  # High error count
        cns.organs = {}
        
        monitor = UnifiedMonitor(cns=cns)
        monitor.collect_metrics()
        
        anomalies = monitor.detect_anomalies()
        
        assert len(anomalies) > 0
        assert any("error" in a.lower() for a in anomalies)
    
    def test_detect_anomalies_low_energy(self):
        """Test anomaly detection for low energy"""
        cns = Mock()
        cns.chronos = Mock(pulse_rate=1.0, cycle_count=50)
        cns.resonance = Mock(total_energy=5.0)  # Low energy
        cns.sink = Mock(error_count=0)
        cns.organs = {}
        
        monitor = UnifiedMonitor(cns=cns)
        monitor.collect_metrics()
        
        anomalies = monitor.detect_anomalies()
        
        assert len(anomalies) > 0
        assert any("energy" in a.lower() for a in anomalies)
    
    def test_generate_report(self):
        """Test unified report generation"""
        cns = Mock()
        cns.chronos = Mock(pulse_rate=1.2, cycle_count=100)
        cns.resonance = Mock(total_energy=80.0)
        cns.sink = Mock(error_count=3)
        cns.organs = {'Voice': Mock(), 'Brain': Mock()}
        
        monitor = UnifiedMonitor(cns=cns)
        monitor.collect_metrics()
        
        # Add some performance metrics
        @monitor.measure("test_op")
        def test_func():
            time.sleep(0.01)
        
        test_func()
        
        report = monitor.generate_report()
        
        assert "UNIFIED MONITORING REPORT" in report
        assert "SYSTEM VITALS" in report
        assert "ORGAN HEALTH" in report
        assert "1.20 Hz" in report or "1.2 Hz" in report  # Pulse rate
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        monitor = UnifiedMonitor()
        monitor.collect_metrics()
        
        @monitor.measure("test")
        def test_func():
            pass
        
        test_func()
        
        summary = monitor.get_metrics_summary()
        
        assert 'system' in summary
        assert 'performance' in summary
        assert 'anomalies' in summary
    
    def test_clear_metrics(self):
        """Test metrics can be cleared"""
        monitor = UnifiedMonitor()
        monitor.collect_metrics()
        
        @monitor.measure("test")
        def test_func():
            pass
        
        test_func()
        
        assert len(monitor.system_metrics_history) > 0
        assert len(monitor.performance_metrics) > 0
        
        monitor.clear_metrics()
        
        assert len(monitor.system_metrics_history) == 0
        assert len(monitor.performance_metrics) == 0
    
    def test_get_slow_operations(self):
        """Test identifying slow operations"""
        monitor = UnifiedMonitor()
        
        @monitor.measure("fast")
        def fast_op():
            time.sleep(0.001)
        
        @monitor.measure("slow")
        def slow_op():
            time.sleep(0.05)
        
        # Run operations
        for _ in range(3):
            fast_op()
        
        slow_op()
        
        slow_ops = monitor.get_slow_operations(threshold_percentile=0.75)
        
        assert len(slow_ops) > 0
        # Slow operation should be in the list
        assert any(op == "slow" for op, _ in slow_ops)
    
    def test_singleton_pattern(self):
        """Test singleton instance works"""
        monitor1 = get_unified_monitor()
        monitor2 = get_unified_monitor()
        
        assert monitor1 is monitor2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
