#!/usr/bin/env python3
"""
Quick validation for UnifiedMonitor
"""

import time
from Core._01_Foundation._02_Logic.unified_monitor import UnifiedMonitor

print("🧪 Testing Unified Monitor\n")
print("=" * 60)

# Create monitor
monitor = UnifiedMonitor()
monitor.start_monitoring()

print("✅ UnifiedMonitor initialized")

# Test system metrics
metrics = monitor.collect_metrics()
print(f"✅ System metrics collected: uptime={metrics.uptime:.2f}s")

# Test performance decorator
@monitor.measure("test_operation")
def test_function():
    time.sleep(0.01)
    return "success"

result = test_function()
print(f"✅ Performance decorator works: {result}")

# Test another operation
@monitor.measure("fast_operation")
def fast_function():
    time.sleep(0.001)
    return "fast"

fast_function()
fast_function()
print("✅ Multiple operations measured")

# Generate report
print("\n" + "=" * 60)
print("📊 UNIFIED REPORT")
print("=" * 60)
report = monitor.generate_report()
print(report)

# Test summary
summary = monitor.get_metrics_summary()
print("\n" + "=" * 60)
print("📈 METRICS SUMMARY")
print("=" * 60)
print(f"System metrics collected: {summary.get('system', {}).get('metrics_collected', 0)}")
if 'performance' in summary:
    print(f"Performance operations tracked: {len(summary['performance'])}")
    for op, stats in summary['performance'].items():
        print(f"  {op}: {stats['count']} calls, avg {stats['mean']:.2f}ms")
print(f"Anomalies detected: {summary.get('anomalies', 0)}")

print("\n" + "=" * 60)
print("🎉 Unified Monitor validation complete!")
print("=" * 60)
