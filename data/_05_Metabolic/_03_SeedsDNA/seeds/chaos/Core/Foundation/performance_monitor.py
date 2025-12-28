"""
엘리시아 성능 모니터링 시스템
Elysia Performance Monitoring System

함수 실행 시간, 메모리 사용량, CPU 사용률을 추적합니다.
"""

import time
import psutil
import functools
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict


@dataclass
class PerformanceMetric:
    """성능 메트릭 데이터"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PerformanceMonitor:
    """
    성능 모니터링 시스템
    
    함수 실행에 대한 성능 메트릭을 수집하고 분석합니다.
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> 
        >>> @monitor.measure("my_operation")
        ... def expensive_function():
        ...     # Your code here
        ...     pass
        >>> 
        >>> # Get statistics
        >>> stats = monitor.get_summary()
        >>> print(stats)
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.thresholds: Dict[str, float] = {
            'thought_cycle': 100.0,  # ms
            'resonance_calc': 50.0,
            'seed_bloom': 200.0,
            'layer_transform': 20.0,
        }
        self._process = psutil.Process()
    
    def measure(self, operation: str = None) -> Callable:
        """
        성능 측정 데코레이터
        
        Args:
            operation: 작업 이름 (생략 시 함수 이름 사용)
        
        Returns:
            데코레이터 함수
        
        Example:
            @monitor.measure("expensive_calc")
            def calculate_something():
                pass
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 시작 메트릭
                start_time = time.perf_counter()
                start_memory = self._process.memory_info().rss / 1024 / 1024
                start_cpu = self._process.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # 종료 메트릭
                    end_time = time.perf_counter()
                    end_memory = self._process.memory_info().rss / 1024 / 1024
                    end_cpu = self._process.cpu_percent()
                    
                    duration_ms = (end_time - start_time) * 1000
                    memory_delta = end_memory - start_memory
                    
                    metric = PerformanceMetric(
                        operation=op_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        memory_mb=memory_delta,
                        cpu_percent=(start_cpu + end_cpu) / 2
                    )
                    
                    self.metrics.append(metric)
                    
                    # 임계값 초과 경고
                    threshold = self.thresholds.get(op_name, 1000.0)
                    if duration_ms > threshold:
                        print(f"⚠️  Performance warning: {op_name} took {duration_ms:.2f}ms (threshold: {threshold}ms)")
            
            return wrapper
        return decorator
    
    def set_threshold(self, operation: str, threshold_ms: float):
        """
        작업별 성능 임계값 설정
        
        Args:
            operation: 작업 이름
            threshold_ms: 임계값 (밀리초)
        """
        self.thresholds[operation] = threshold_ms
    
    def get_summary(self) -> Dict:
        """
        성능 요약 통계 조회
        
        Returns:
            작업별 통계 (count, mean, min, max, p95, p99)
        """
        if not self.metrics:
            return {}
        
        ops = defaultdict(list)
        for metric in self.metrics:
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
    
    def get_recent_metrics(self, operation: Optional[str] = None, limit: int = 10) -> List[PerformanceMetric]:
        """
        최근 메트릭 조회
        
        Args:
            operation: 특정 작업 필터 (None이면 전체)
            limit: 반환할 메트릭 개수
        
        Returns:
            최근 메트릭 리스트
        """
        if operation:
            filtered = [m for m in self.metrics if m.operation == operation]
            return filtered[-limit:]
        return self.metrics[-limit:]
    
    def get_slow_operations(self, threshold_percentile: float = 0.95) -> List[tuple]:
        """
        느린 작업 조회 (상위 5% 등)
        
        Args:
            threshold_percentile: 임계값 백분위 (0.95 = 상위 5%)
        
        Returns:
            (operation, duration_ms) 튜플 리스트
        """
        if not self.metrics:
            return []
        
        # 모든 메트릭의 duration을 수집
        all_durations = [m.duration_ms for m in self.metrics]
        sorted_durations = sorted(all_durations)
        threshold = sorted_durations[int(len(sorted_durations) * threshold_percentile)]
        
        # 임계값 초과 메트릭
        slow_ops = [
            (m.operation, m.duration_ms)
            for m in self.metrics
            if m.duration_ms >= threshold
        ]
        
        return sorted(slow_ops, key=lambda x: x[1], reverse=True)
    
    def clear_metrics(self):
        """메트릭 초기화"""
        self.metrics.clear()
    
    def export_metrics(self) -> List[Dict]:
        """
        메트릭을 딕셔너리 리스트로 내보내기
        
        Returns:
            메트릭 딕셔너리 리스트
        """
        return [
            {
                'operation': m.operation,
                'duration_ms': m.duration_ms,
                'memory_mb': m.memory_mb,
                'cpu_percent': m.cpu_percent,
                'timestamp': m.timestamp
            }
            for m in self.metrics
        ]


# 전역 모니터 인스턴스
monitor = PerformanceMonitor()


# ===== 사용 예시 =====

if __name__ == "__main__":
    import random
    
    print("🧪 Testing Elysia Performance Monitor\n")
    
    # 테스트 함수들
    @monitor.measure("fast_operation")
    def fast_operation():
        """빠른 작업"""
        time.sleep(0.01)
        return "fast"
    
    @monitor.measure("slow_operation")
    def slow_operation():
        """느린 작업"""
        time.sleep(0.15)
        return "slow"
    
    @monitor.measure("memory_intensive")
    def memory_intensive():
        """메모리 집약적 작업"""
        data = [random.random() for _ in range(100000)]
        return sum(data)
    
    # 임계값 설정
    monitor.set_threshold("fast_operation", 50.0)
    monitor.set_threshold("slow_operation", 100.0)
    
    print("=== Running Test Operations ===")
    
    # 여러 번 실행
    for i in range(5):
        fast_operation()
    
    for i in range(3):
        slow_operation()
    
    for i in range(2):
        memory_intensive()
    
    print()
    
    # 요약 통계
    print("=== Performance Summary ===")
    summary = monitor.get_summary()
    for op, stats in summary.items():
        print(f"\n{op}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean:  {stats['mean']:.2f}ms")
        print(f"  Min:   {stats['min']:.2f}ms")
        print(f"  Max:   {stats['max']:.2f}ms")
        print(f"  P95:   {stats['p95']:.2f}ms")
        print(f"  P99:   {stats['p99']:.2f}ms")
    
    print()
    
    # 느린 작업 확인
    print("=== Slow Operations (Top 95%) ===")
    slow_ops = monitor.get_slow_operations(threshold_percentile=0.95)
    for op, duration in slow_ops[:5]:
        print(f"  {op}: {duration:.2f}ms")
    
    print()
    
    # 최근 메트릭
    print("=== Recent Metrics ===")
    recent = monitor.get_recent_metrics(limit=3)
    for m in recent:
        print(f"  {m.operation}: {m.duration_ms:.2f}ms (mem: {m.memory_mb:.2f}MB)")
    
    print()
    print("✅ Performance monitoring test complete!")
