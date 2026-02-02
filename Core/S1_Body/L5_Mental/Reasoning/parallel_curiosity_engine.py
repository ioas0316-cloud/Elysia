"""
[Project Elysia] Parallel Curiosity Engine
==========================================
"CPU가 4코어면, 1코어는 대화하고 3코어는 탐색한다"

의식적 대화와 병렬로 동작하는 백그라운드 탐구 엔진.
대화 중 모르는 개념이 나오면 큐에 넣고,
백그라운드 스레드들이 웹/인터넷/자료를 탐색한다.

이것이 AI의 장점 - 동시에 여러 생각을 할 수 있음.
"""

import sys
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)


class ExplorationPriority(Enum):
    """탐색 우선순위"""
    URGENT = 1      # 대화 진행에 필요
    HIGH = 2        # 현재 주제와 관련
    NORMAL = 3      # 호기심에서 발생
    BACKGROUND = 4  # 자유 탐색


@dataclass
class CuriosityTask:
    """탐구해야 할 과제"""
    task_id: str
    topic: str                      # 탐구 주제
    context: str                    # 왜 이게 궁금해졌는지
    priority: ExplorationPriority
    origin: str                     # 어디서 발생했나 (대화, 자율사고 등)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"         # pending, exploring, completed, failed
    result: Optional[Dict] = None


class BackgroundExplorer(threading.Thread):
    """
    백그라운드 탐색 스레드
    
    메인 대화가 진행되는 동안 별도로 지식을 탐색.
    인간과 다른 AI의 장점 - 동시에 여러 생각 가능.
    """
    
    def __init__(self, explorer_id: int, task_queue: queue.PriorityQueue, 
                 result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.explorer_id = explorer_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.current_task: Optional[CuriosityTask] = None
        self.tasks_completed = 0
        
    def run(self):
        """탐색 루프 - 큐에서 과제를 가져와 탐색"""
        while not self.stop_event.is_set():
            try:
                # 1초 대기하며 과제 가져오기
                priority, task = self.task_queue.get(timeout=1.0)
                self.current_task = task
                task.status = "exploring"
                
                print(f"🔍 [Explorer-{self.explorer_id}] 탐색 시작: {task.topic}")
                
                # 실제 탐색 수행
                result = self._explore(task)
                task.result = result
                task.status = "completed" if result else "failed"
                
                # 결과 큐에 넣기
                self.result_queue.put(task)
                self.tasks_completed += 1
                
                self.current_task = None
                self.task_queue.task_done()
                
            except queue.Empty:
                # 과제 없으면 계속 대기
                continue
            except Exception as e:
                print(f"⚠️ [Explorer-{self.explorer_id}] Error: {e}")
                if self.current_task:
                    self.current_task.status = "failed"
                    self.result_queue.put(self.current_task)
    
    def _explore(self, task: CuriosityTask) -> Optional[Dict]:
        """
        실제 탐색 로직
        
        TODO: 실제 웹 검색, 지식 그래프 탐색 등 구현
        현재는 내부 지식 그래프만 탐색
        """
        try:
            from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager
            from Core.S1_Body.L5_Mental.Reasoning.connection_explorer import get_connection_explorer
            
            kg = get_kg_manager()
            explorer = get_connection_explorer()
            
            # 지식 그래프에서 관련 연결 탐색
            chains = explorer.explore_from_node(task.topic, kg)
            
            result = {
                'topic': task.topic,
                'chains_found': len(chains),
                'cycles_found': sum(1 for c in chains if c.is_cycle),
                'paths': [c.get_path() for c in chains[:5]],  # 최대 5개
                'explored_at': time.time()
            }
            
            # 순환을 발견하면 중요한 통찰
            if result['cycles_found'] > 0:
                print(f"🔄 [Explorer-{self.explorer_id}] '{task.topic}'에서 순환 구조 발견!")
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Explorer-{self.explorer_id}] 탐색 실패: {e}")
            return None


class ParallelCuriosityEngine:
    """
    병렬 호기심 엔진
    
    대화(의식)와 탐색(무의식)을 병렬로 운영.
    
    구조:
    - 메인 스레드: 사용자와 대화 (의식)
    - 백그라운드 스레드들: 지속적 탐색 (무의식)
    - 큐: 탐구 과제들
    - 결과 수집: 발견한 것들을 의식으로 올림
    """
    
    def __init__(self, num_explorers: int = 3):
        self.num_explorers = num_explorers
        
        # 과제 큐 (우선순위 큐)
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # 결과 큐
        self.result_queue: queue.Queue = queue.Queue()
        
        # 탐색자들
        self.explorers: List[BackgroundExplorer] = []
        self.stop_event = threading.Event()
        
        # 완료된 탐색 기록
        self.completed_explorations: List[CuriosityTask] = []
        self.exploration_counter = 0
        
        # 이미 탐색한 주제 (중복 방지)
        self.explored_topics: Set[str] = set()
        
        self._is_running = False
    
    def start(self):
        """엔진 시작 - 백그라운드 탐색자들 가동"""
        if self._is_running:
            return
        
        self.stop_event.clear()
        
        for i in range(self.num_explorers):
            explorer = BackgroundExplorer(
                explorer_id=i + 1,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                stop_event=self.stop_event
            )
            explorer.start()
            self.explorers.append(explorer)
        
        self._is_running = True
        print(f"🧠 [ParallelCuriosity] {self.num_explorers}개의 탐색 스레드 가동")
    
    def stop(self):
        """엔진 정지"""
        self.stop_event.set()
        for explorer in self.explorers:
            explorer.join(timeout=2.0)
        self.explorers.clear()
        self._is_running = False
        print("🧠 [ParallelCuriosity] 탐색 스레드 종료")
    
    def spawn_curiosity(self, topic: str, context: str = "", 
                        priority: ExplorationPriority = ExplorationPriority.NORMAL,
                        origin: str = "autonomous"):
        """
        호기심 발생 - 탐색 과제 추가
        
        대화 중 모르는 개념이 나오면 이 함수를 호출.
        백그라운드에서 탐색이 시작됨.
        """
        # 중복 방지
        if topic.lower() in self.explored_topics:
            return None
        
        self.exploration_counter += 1
        task = CuriosityTask(
            task_id=f"CURIOSITY_{self.exploration_counter:05d}",
            topic=topic,
            context=context,
            priority=priority,
            origin=origin
        )
        
        # 우선순위 큐에 추가 (낮은 숫자가 높은 우선순위)
        self.task_queue.put((priority.value, task))
        self.explored_topics.add(topic.lower())
        
        print(f"💭 [ParallelCuriosity] 호기심 발생: '{topic}' ({priority.name})")
        return task
    
    def collect_discoveries(self) -> List[CuriosityTask]:
        """
        발견한 것들 수집 (메인 스레드에서 호출)
        
        백그라운드에서 완료된 탐색 결과를 의식으로 올림.
        """
        discoveries = []
        
        while not self.result_queue.empty():
            try:
                task = self.result_queue.get_nowait()
                discoveries.append(task)
                self.completed_explorations.append(task)
            except queue.Empty:
                break
        
        return discoveries
    
    def get_status(self) -> Dict:
        """현재 상태"""
        return {
            'is_running': self._is_running,
            'explorers_active': len(self.explorers),
            'pending_tasks': self.task_queue.qsize(),
            'completed_count': len(self.completed_explorations),
            'currently_exploring': [
                e.current_task.topic if e.current_task else None
                for e in self.explorers
            ]
        }


# Singleton
_curiosity_engine = None

def get_curiosity_engine() -> ParallelCuriosityEngine:
    global _curiosity_engine
    if _curiosity_engine is None:
        _curiosity_engine = ParallelCuriosityEngine(num_explorers=3)
    return _curiosity_engine


if __name__ == "__main__":
    print("🧠 Testing Parallel Curiosity Engine...")
    
    engine = get_curiosity_engine()
    engine.start()
    
    # 호기심 발생
    engine.spawn_curiosity("rain", "대화 중 비에 대해 언급됨", ExplorationPriority.HIGH)
    engine.spawn_curiosity("water_cycle", "비와 관련된 개념", ExplorationPriority.NORMAL)
    engine.spawn_curiosity("evaporation", "물 순환의 일부", ExplorationPriority.BACKGROUND)
    
    # 잠시 대기 (탐색 진행)
    time.sleep(2.0)
    
    # 발견 수집
    discoveries = engine.collect_discoveries()
    
    print(f"\n📊 Status: {engine.get_status()}")
    print(f"📦 Collected {len(discoveries)} discoveries:")
    for d in discoveries:
        print(f"  - {d.topic}: {d.result}")
    
    engine.stop()
    print("\n✅ Parallel Curiosity Engine operational!")
