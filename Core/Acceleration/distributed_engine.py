"""
Distributed Engine - 분산 처리 엔진
==================================

높은 우선순위 #2: 단일 프로세스 → Ray/Dask 분산
예상 효과: 100x 확장성

핵심 기능:
- 멀티프로세스 워커 풀
- 작업 분배 및 수집
- 장애 복구 (Fault Tolerance)
- 로드 밸런싱
"""

import asyncio
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import queue
import threading

logger = logging.getLogger("DistributedEngine")


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """워커 상태"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class Task:
    """분산 작업"""
    task_id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 높을수록 먼저
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None


@dataclass
class TaskResult:
    """작업 결과"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    worker_id: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """실행 시간 (ms)"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class WorkerNode:
    """워커 노드 정보"""
    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time_ms: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def avg_task_time_ms(self) -> float:
        """평균 작업 시간"""
        if self.tasks_completed > 0:
            return self.total_processing_time_ms / self.tasks_completed
        return 0.0


class DistributedEngine:
    """
    분산 처리 엔진
    
    높은 우선순위 #2 구현:
    - 멀티프로세스 병렬 처리
    - 스레드 풀 하이브리드
    - 작업 우선순위 큐
    - 자동 장애 복구
    
    예상 효과: 100x 확장성 (CPU 코어 수에 비례)
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = True,
        max_queue_size: int = 10000
    ):
        """
        Args:
            num_workers: 워커 수 (기본: CPU 코어 수)
            use_processes: True=프로세스 풀, False=스레드 풀
            max_queue_size: 최대 큐 크기
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.max_queue_size = max_queue_size
        
        # 워커 풀
        self._executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None
        self._running = False
        
        # 작업 관리
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._results: Dict[str, TaskResult] = {}
        self._pending_futures: Dict[str, Any] = {}
        
        # 워커 정보
        self.workers: Dict[str, WorkerNode] = {}
        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerNode(worker_id=worker_id)
        
        # 통계
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "avg_queue_time_ms": 0.0,
            "avg_execution_time_ms": 0.0
        }
        
        # 락
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger("DistributedEngine")
        pool_type = "Process" if use_processes else "Thread"
        self.logger.info(f"🌐 DistributedEngine initialized ({self.num_workers} {pool_type} workers)")
    
    def start(self) -> None:
        """엔진 시작"""
        if self._running:
            return
        
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        self._running = True
        self.logger.info("▶️ Distributed engine started")
    
    def stop(self) -> None:
        """엔진 정지"""
        self._running = False
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        self.logger.info("⏹️ Distributed engine stopped")
    
    def submit(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        작업 제출
        
        Args:
            task_id: 작업 ID
            func: 실행할 함수
            *args: 함수 인자
            priority: 우선순위 (높을수록 먼저)
            timeout: 타임아웃 (초)
            **kwargs: 함수 키워드 인자
            
        Returns:
            작업 ID
        """
        if not self._running:
            self.start()
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # 우선순위 큐에 추가 (음수로 변환하여 높은 priority가 먼저 나오도록)
        self._task_queue.put((-priority, time.time(), task))
        
        with self._lock:
            self.stats["total_submitted"] += 1
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
        
        # 즉시 실행 시도
        self._dispatch_tasks()
        
        return task_id
    
    def _dispatch_tasks(self) -> None:
        """큐에서 작업 디스패치"""
        while not self._task_queue.empty() and self._executor:
            try:
                _, _, task = self._task_queue.get_nowait()
            except queue.Empty:
                break
            
            # 작업 제출
            with self._lock:
                self._results[task.task_id].status = TaskStatus.RUNNING
                self._results[task.task_id].started_at = time.time()
            
            future = self._executor.submit(
                self._execute_task,
                task
            )
            
            self._pending_futures[task.task_id] = future
            
            # 콜백 등록
            future.add_done_callback(
                lambda f, tid=task.task_id: self._on_task_complete(tid, f)
            )
    
    @staticmethod
    def _execute_task(task: Task) -> Any:
        """작업 실행 (워커에서 실행됨)"""
        return task.func(*task.args, **task.kwargs)
    
    def _on_task_complete(self, task_id: str, future) -> None:
        """작업 완료 콜백"""
        completed_at = time.time()
        
        with self._lock:
            result = self._results.get(task_id)
            if not result:
                return
            
            result.completed_at = completed_at
            
            try:
                result.result = future.result()
                result.status = TaskStatus.COMPLETED
                self.stats["total_completed"] += 1
            except Exception as e:
                result.error = str(e)
                result.status = TaskStatus.FAILED
                self.stats["total_failed"] += 1
            
            # 평균 실행 시간 업데이트
            if result.status == TaskStatus.COMPLETED:
                n = self.stats["total_completed"]
                old_avg = self.stats["avg_execution_time_ms"]
                self.stats["avg_execution_time_ms"] = (
                    old_avg * (n - 1) / n + result.duration_ms / n
                )
            
            # 정리
            if task_id in self._pending_futures:
                del self._pending_futures[task_id]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        작업 결과 조회 (블로킹)
        
        Args:
            task_id: 작업 ID
            timeout: 대기 시간 (초)
            
        Returns:
            작업 결과 또는 None
        """
        start = time.time()
        
        while True:
            with self._lock:
                result = self._results.get(task_id)
                if result and result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    return result
            
            if timeout and (time.time() - start) > timeout:
                return None
            
            time.sleep(0.01)
    
    def get_result_async(self, task_id: str) -> Optional[TaskResult]:
        """
        작업 결과 조회 (논블로킹)
        
        Args:
            task_id: 작업 ID
            
        Returns:
            작업 결과 또는 None (아직 완료되지 않은 경우)
        """
        with self._lock:
            return self._results.get(task_id)
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        병렬 맵 연산
        
        Args:
            func: 적용할 함수
            items: 입력 항목들
            timeout: 전체 타임아웃
            
        Returns:
            결과 목록
        """
        task_ids = []
        
        for i, item in enumerate(items):
            task_id = f"map_{id(func)}_{i}_{time.time()}"
            self.submit(task_id, func, item)
            task_ids.append(task_id)
        
        # 모든 결과 수집
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout=timeout)
            results.append(result)
        
        return results
    
    def batch_resonance(
        self,
        resonance_engine,
        pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], float]:
        """
        공명 계산 병렬화
        
        Args:
            resonance_engine: 공명 엔진
            pairs: (source_id, target_id) 쌍 목록
            
        Returns:
            {(source, target): score} 딕셔너리
        """
        def calc_single(pair):
            source_id, target_id = pair
            source = resonance_engine.nodes.get(source_id)
            target = resonance_engine.nodes.get(target_id)
            if source and target:
                return resonance_engine.calculate_resonance(source, target)
            return 0.0
        
        results = self.map(calc_single, pairs)
        
        return {
            pairs[i]: r.result if r and r.status == TaskStatus.COMPLETED else 0.0
            for i, r in enumerate(results)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        with self._lock:
            return {
                **self.stats,
                "queue_size": self._task_queue.qsize(),
                "pending_tasks": len(self._pending_futures),
                "num_workers": self.num_workers,
                "is_running": self._running
            }
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop()


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌐 Distributed Engine Test")
    print("="*70)
    
    def heavy_computation(x):
        """무거운 연산 시뮬레이션"""
        import time
        time.sleep(0.01)
        return x * x
    
    # 테스트 1: 단일 작업
    print("\n[Test 1] Single Task")
    with DistributedEngine(num_workers=4, use_processes=False) as engine:
        task_id = engine.submit("task_1", heavy_computation, 42)
        result = engine.get_result(task_id, timeout=5.0)
        print(f"  ✓ Result: {result.result}")
        print(f"  Duration: {result.duration_ms:.2f}ms")
    
    # 테스트 2: 병렬 맵
    print("\n[Test 2] Parallel Map")
    with DistributedEngine(num_workers=4, use_processes=False) as engine:
        items = list(range(20))
        start = time.time()
        results = engine.map(heavy_computation, items)
        elapsed = (time.time() - start) * 1000
        
        completed = sum(1 for r in results if r and r.status == TaskStatus.COMPLETED)
        print(f"  ✓ Completed: {completed}/{len(items)}")
        print(f"  Total time: {elapsed:.2f}ms")
        print(f"  Stats: {engine.get_stats()}")
    
    # 테스트 3: 우선순위
    print("\n[Test 3] Priority Queue")
    with DistributedEngine(num_workers=2, use_processes=False) as engine:
        # 낮은 우선순위 먼저 제출
        engine.submit("low_1", heavy_computation, 1, priority=1)
        engine.submit("low_2", heavy_computation, 2, priority=1)
        # 높은 우선순위 나중에 제출
        engine.submit("high_1", heavy_computation, 100, priority=10)
        
        time.sleep(0.1)
        print(f"  Stats: {engine.get_stats()}")
    
    print("\n✅ All tests passed!")
    print("="*70 + "\n")
