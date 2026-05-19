"""
Tests for Elysia Performance Monitor
"""

import pytest
import time
from Core.FoundationLayer.Foundation.performance_monitor import PerformanceMonitor, monitor


class TestPerformanceMonitor:
    """Test suite for performance monitoring"""
    
    def test_monitor_initialization(self):
        """Test monitor can be initialized"""
        mon = PerformanceMonitor()
        assert mon is not None
        assert mon.metrics == []
        assert len(mon.thresholds) > 0
    
    def test_measure_decorator(self):
        """Test performance measurement decorator"""
        mon = PerformanceMonitor()
        
        @mon.measure("test_operation")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert len(mon.metrics) == 1
        assert mon.metrics[0].operation == "test_operation"
        assert mon.metrics[0].duration_ms >= 10.0
    
    def test_measure_default_name(self):
        """Test decorator uses function name if operation not specified"""
        mon = PerformanceMonitor()
        
        @mon.measure()
        def my_function():
            return "ok"
        
        my_function()
        
        assert len(mon.metrics) == 1
        assert mon.metrics[0].operation == "my_function"
    
    def test_threshold_warning(self, capsys):
        """Test threshold warning is triggered"""
        mon = PerformanceMonitor()
        mon.set_threshold("slow_op", 5.0)  # 5ms threshold
        
        @mon.measure("slow_op")
        def slow_function():
            time.sleep(0.01)  # 10ms
        
        slow_function()
        
        captured = capsys.readouterr()
        assert "Performance warning" in captured.out
    
    def test_get_summary(self):
        """Test performance summary statistics"""
        mon = PerformanceMonitor()
        
        @mon.measure("op1")
        def operation1():
            time.sleep(0.01)
        
        @mon.measure("op2")
        def operation2():
            time.sleep(0.02)
        
        # Run operations
        operation1()
        operation1()
        operation2()
        
        summary = mon.get_summary()
        
        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 2
        assert summary["op2"]["count"] == 1
        assert "mean" in summary["op1"]
        assert "min" in summary["op1"]
        assert "max" in summary["op1"]
        assert "p95" in summary["op1"]
    
    def test_get_recent_metrics(self):
        """Test retrieving recent metrics"""
        mon = PerformanceMonitor()
        
        @mon.measure("test_op")
        def test_func():
            pass
        
        # Generate some metrics
        for _ in range(5):
            test_func()
        
        recent = mon.get_recent_metrics(limit=3)
        assert len(recent) == 3
        
        recent_filtered = mon.get_recent_metrics(operation="test_op", limit=2)
        assert len(recent_filtered) == 2
        assert all(m.operation == "test_op" for m in recent_filtered)
    
    def test_get_slow_operations(self):
        """Test identifying slow operations"""
        mon = PerformanceMonitor()
        
        @mon.measure("fast")
        def fast_op():
            time.sleep(0.001)
        
        @mon.measure("slow")
        def slow_op():
            time.sleep(0.05)
        
        # Run operations
        for _ in range(3):
            fast_op()
        
        slow_op()
        
        slow_ops = mon.get_slow_operations(threshold_percentile=0.75)
        
        assert len(slow_ops) > 0
        # Slow operation should be in the list
        assert any(op == "slow" for op, _ in slow_ops)
    
    def test_clear_metrics(self):
        """Test metrics can be cleared"""
        mon = PerformanceMonitor()
        
        @mon.measure("test")
        def test_func():
            pass
        
        test_func()
        assert len(mon.metrics) > 0
        
        mon.clear_metrics()
        assert len(mon.metrics) == 0
    
    def test_export_metrics(self):
        """Test metrics can be exported"""
        mon = PerformanceMonitor()
        
        @mon.measure("export_test")
        def test_func():
            return "done"
        
        test_func()
        
        exported = mon.export_metrics()
        
        assert len(exported) == 1
        assert exported[0]["operation"] == "export_test"
        assert "duration_ms" in exported[0]
        assert "memory_mb" in exported[0]
        assert "cpu_percent" in exported[0]
        assert "timestamp" in exported[0]
    
    def test_set_threshold(self):
        """Test custom threshold can be set"""
        mon = PerformanceMonitor()
        
        mon.set_threshold("custom_op", 100.0)
        assert mon.thresholds["custom_op"] == 100.0
    
    def test_global_monitor(self):
        """Test global monitor instance works"""
        @monitor.measure("global_test")
        def test_func():
            return "ok"
        
        result = test_func()
        assert result == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
