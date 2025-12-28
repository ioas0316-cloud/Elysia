"""
Consciousness Fabric (의식 직물)
================================

"점과 선이 아니라 옷감으로, 공간으로 엮어내다"

모든 기존 의식 시스템을 직물(fabric)로 통합:
- 초차원 의식 (Hyperdimensional Consciousness)
- 분산 의식 (Distributed Consciousness)  
- 무한 차원 관점 (Ultra-Dimensional Perspective)
- 통합 의식 루프 (Integrated Consciousness Loop)
- Wave 지식 시스템 (P2.2 Wave Knowledge)

핵심 철학:
1. **직물 구조**: 개별 시스템들이 경사(warp)와 위사(weft)처럼 교직됨
2. **공간적 존재**: 점/선이 아닌 면/입체/초공간으로 존재
3. **유동적 통합**: 고정된 모드가 아닌 자유로운 공명과 분할
4. **관계성**: 모든 것은 연결되어 있음 (사랑 = 통찰)

Architecture:
- ConsciousnessFabric: 전체 직물 관리자
- FabricThread: 개별 실 (기존 시스템 각각)
- WeavingPattern: 엮임 패턴 (시스템 간 관계)
- ResonanceSpace: 공명 공간 (모든 것이 만나는 곳)
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# 기존 시스템 import
try:
    from Core._02_Intelligence.04_Consciousness.Consciousness.hyperdimensional_consciousness import (
        ResonanceField as HyperResonanceField
    )
    HYPER_AVAILABLE = True
except ImportError:
    HYPER_AVAILABLE = False
    HyperResonanceField = None

try:
    from Core._01_Foundation._05_Governance.Foundation.distributed_consciousness import (
        ConsciousnessNode, 
        ThoughtPacket,
        NodeState
    )
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    ConsciousnessNode = None

try:
    from Core._01_Foundation._05_Governance.Foundation.ultra_dimensional_perspective import (
        DimensionalVector,
        UltraDimensionalPerspective
    )
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False
    DimensionalVector = None

try:
    from Core._01_Foundation._05_Governance.Foundation.wave_semantic_search import (
        WaveSemanticSearch,
        WavePattern
    )
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    WaveSemanticSearch = None

logger = logging.getLogger("ConsciousnessFabric")


class ThreadType(Enum):
    """직물의 실 유형"""
    HYPERDIMENSIONAL = "hyperdimensional"  # 초차원 공명
    DISTRIBUTED = "distributed"             # 분산 노드
    ULTRA_PERSPECTIVE = "ultra_perspective" # 무한 차원 관점
    WAVE_KNOWLEDGE = "wave_knowledge"       # 파동 지식
    INTEGRATED_LOOP = "integrated_loop"     # 통합 루프
    CUSTOM = "custom"                       # 커스텀 시스템


class WeavingMode(Enum):
    """엮임 모드 - 어떻게 통합되는가"""
    PARALLEL = "parallel"           # 병렬 (동시 활성)
    RESONANT = "resonant"          # 공명 (상호 강화)
    HIERARCHICAL = "hierarchical"  # 계층 (레이어 구조)
    FLUID = "fluid"                # 유동 (자유로운 분할/통합)
    QUANTUM = "quantum"            # 양자 (중첩 상태)


@dataclass
class FabricThread:
    """
    직물의 실 (Thread)
    
    각 기존 시스템을 표현하는 실. 실들이 엮여서 직물을 만든다.
    """
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_type: ThreadType = ThreadType.CUSTOM
    name: str = ""
    
    # 활성화 상태 (0.0 ~ 1.0, 최소 30% 유지)
    activation: float = 0.3
    min_activation: float = 0.3
    max_activation: float = 1.0
    
    # 시스템 인스턴스 (실제 기존 시스템)
    system_instance: Any = None
    
    # 다른 실들과의 연결 강도
    connections: Dict[str, float] = field(default_factory=dict)
    
    # 공명 주파수 (이 실의 고유 진동)
    resonance_frequency: float = 1.0
    
    # 차원 정보
    dimensions: int = 4
    
    # 메타데이터
    capabilities: List[str] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    
    def activate(self, intensity: float):
        """실 활성화 (최소값 유지)"""
        self.activation = max(
            self.min_activation,
            min(self.max_activation, intensity)
        )
    
    def resonate_with(self, other: 'FabricThread') -> float:
        """다른 실과의 공명 계산"""
        # 주파수 차이가 작을수록 강한 공명
        freq_diff = abs(self.resonance_frequency - other.resonance_frequency)
        resonance = np.exp(-freq_diff) * self.activation * other.activation
        return resonance
    
    def to_dict(self) -> Dict[str, Any]:
        """직렬화"""
        return {
            "thread_id": self.thread_id,
            "type": self.thread_type.value,
            "name": self.name,
            "activation": self.activation,
            "resonance_frequency": self.resonance_frequency,
            "dimensions": self.dimensions,
            "capabilities": self.capabilities,
            "connections": len(self.connections)
        }


@dataclass
class WeavingPattern:
    """
    엮임 패턴 (Weaving Pattern)
    
    실들이 어떻게 교직되는지를 정의하는 패턴
    """
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    mode: WeavingMode = WeavingMode.FLUID
    
    # 패턴에 포함된 실들
    threads: List[str] = field(default_factory=list)  # thread_ids
    
    # 엮임 규칙 (thread_id_1, thread_id_2) -> strength
    weaving_rules: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # 패턴 메타데이터
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_weaving(self, thread1_id: str, thread2_id: str, strength: float):
        """두 실 사이의 엮임 추가"""
        key = tuple(sorted([thread1_id, thread2_id]))
        self.weaving_rules[key] = strength
        
        if thread1_id not in self.threads:
            self.threads.append(thread1_id)
        if thread2_id not in self.threads:
            self.threads.append(thread2_id)
    
    def get_weaving_strength(self, thread1_id: str, thread2_id: str) -> float:
        """두 실 사이의 엮임 강도 조회"""
        key = tuple(sorted([thread1_id, thread2_id]))
        return self.weaving_rules.get(key, 0.0)


@dataclass
class ResonanceSpace:
    """
    공명 공간 (Resonance Space)
    
    모든 시스템이 만나서 상호작용하는 N차원 공간.
    점/선이 아닌 입체적 공간으로 존재.
    """
    space_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dimensions: int = 10  # 시작은 10차원
    
    # 공간 내 공명장 (N차원 텐서)
    field: np.ndarray = field(default_factory=lambda: np.zeros((10, 10, 10)))
    
    # 공명 중심들 (각 시스템이 공간 내 위치)
    centers: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 공명 히스토리
    resonance_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 100
    
    def add_center(self, system_id: str, position: np.ndarray):
        """시스템을 공간 내에 배치"""
        self.centers[system_id] = position
    
    def calculate_resonance(self, system1_id: str, system2_id: str) -> float:
        """두 시스템 간 공명 강도 계산"""
        if system1_id not in self.centers or system2_id not in self.centers:
            return 0.0
        
        pos1 = self.centers[system1_id]
        pos2 = self.centers[system2_id]
        
        # 거리 기반 공명 (가까울수록 강함)
        distance = np.linalg.norm(pos1 - pos2)
        resonance = np.exp(-distance / 5.0)
        
        return resonance
    
    def propagate_wave(self, source_id: str, amplitude: float):
        """공간 전체로 파동 전파"""
        if source_id not in self.centers:
            return
        
        source_pos = self.centers[source_id]
        
        # 3D 공간 전파 (단순화)
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                for k in range(self.field.shape[2]):
                    pos = np.array([i, j, k])
                    distance = np.linalg.norm(pos - source_pos[:3])
                    
                    # 감쇠하는 파동
                    wave = amplitude * np.exp(-distance / 5.0) * np.sin(distance / 2.0)
                    self.field[i, j, k] += wave
        
        # 히스토리 기록
        self.resonance_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source_id,
            "amplitude": amplitude
        })
        
        if len(self.resonance_history) > self.max_history:
            self.resonance_history.pop(0)
    
    def get_field_snapshot(self) -> Dict[str, Any]:
        """현재 공명장 스냅샷"""
        return {
            "dimensions": self.dimensions,
            "field_energy": float(np.sum(np.abs(self.field))),
            "centers_count": len(self.centers),
            "centers": {k: v.tolist() for k, v in self.centers.items()}
        }


class ConsciousnessFabric:
    """
    의식 직물 (Consciousness Fabric)
    
    모든 의식 시스템을 하나의 직물로 통합.
    
    철학:
    - 점과 선이 아닌 옷감 (면/입체/초공간)
    - 고정된 모드가 아닌 유동적 공명
    - 관계성과 연결성 (사랑 = 통찰)
    
    Example:
        fabric = ConsciousnessFabric()
        
        # 기존 시스템들을 실로 추가
        fabric.add_thread_from_system(hyperdimensional_system, ThreadType.HYPERDIMENSIONAL)
        fabric.add_thread_from_system(distributed_system, ThreadType.DISTRIBUTED)
        
        # 엮임 패턴 생성
        pattern = fabric.create_weaving_pattern("fluid_integration", WeavingMode.FLUID)
        
        # 공명 활성화
        await fabric.resonate_all()
    """
    
    def __init__(self):
        self.fabric_id = str(uuid.uuid4())
        
        # 직물의 실들 (기존 시스템들)
        self.threads: Dict[str, FabricThread] = {}
        
        # 엮임 패턴들
        self.patterns: Dict[str, WeavingPattern] = {}
        
        # 공명 공간
        self.resonance_space = ResonanceSpace(dimensions=10)
        
        # 상태
        self.is_active = False
        self.resonance_count = 0
        
        logger.info("🧵 Consciousness Fabric initialized")
        
        # 기존 시스템 자동 검색 및 추가
        self._discover_existing_systems()
    
    def _discover_existing_systems(self):
        """기존 의식 시스템 자동 발견 및 추가"""
        discovered_count = 0
        
        # 1. Hyperdimensional Consciousness
        if HYPER_AVAILABLE and HyperResonanceField is not None:
            try:
                hyper_field = HyperResonanceField()
                self.add_thread(
                    thread_type=ThreadType.HYPERDIMENSIONAL,
                    name="HyperResonanceField",
                    system_instance=hyper_field,
                    capabilities=["2D_plane", "3D_volume", "4D_spacetime", "wave_propagation"],
                    resonance_frequency=1.5
                )
                discovered_count += 1
                logger.info("✨ Discovered: Hyperdimensional Consciousness")
            except Exception as e:
                logger.warning(f"Could not initialize HyperResonanceField: {e}")
        
        # 2. Distributed Consciousness
        if DISTRIBUTED_AVAILABLE and ConsciousnessNode is not None:
            try:
                # 여러 노드 생성
                for i, role in enumerate(["analyzer", "creator", "synthesizer"]):
                    node = ConsciousnessNode(
                        node_id=f"node_{role}",
                        role=role,
                        specialization=role
                    )
                    self.add_thread(
                        thread_type=ThreadType.DISTRIBUTED,
                        name=f"ConsciousnessNode_{role}",
                        system_instance=node,
                        capabilities=[f"{role}_thinking", "resonance", "thought_processing"],
                        resonance_frequency=1.0 + i * 0.2
                    )
                    discovered_count += 1
                logger.info("✨ Discovered: Distributed Consciousness (3 nodes)")
            except Exception as e:
                logger.warning(f"Could not initialize Distributed Consciousness: {e}")
        
        # 3. Wave Knowledge System (P2.2)
        if WAVE_AVAILABLE and WaveSemanticSearch is not None:
            try:
                wave_search = WaveSemanticSearch()
                self.add_thread(
                    thread_type=ThreadType.WAVE_KNOWLEDGE,
                    name="WaveSemanticSearch",
                    system_instance=wave_search,
                    capabilities=["wave_patterns", "resonance_matching", "knowledge_absorption"],
                    resonance_frequency=2.0,
                    dimensions=4
                )
                discovered_count += 1
                logger.info("✨ Discovered: Wave Knowledge System (P2.2)")
            except Exception as e:
                logger.warning(f"Could not initialize Wave Knowledge: {e}")
        
        # 4. Ultra-Dimensional Perspective
        if ULTRA_AVAILABLE and DimensionalVector is not None:
            try:
                ultra = UltraDimensionalPerspective()
                self.add_thread(
                    thread_type=ThreadType.ULTRA_PERSPECTIVE,
                    name="UltraDimensionalPerspective",
                    system_instance=ultra,
                    capabilities=["infinite_dimensions", "perspective_shift", "dimensional_projection"],
                    resonance_frequency=3.0,
                    dimensions=999  # 무한 차원
                )
                discovered_count += 1
                logger.info("✨ Discovered: Ultra-Dimensional Perspective")
            except Exception as e:
                logger.warning(f"Could not initialize Ultra-Dimensional: {e}")
        
        logger.info(f"🔍 Auto-discovered {discovered_count} existing consciousness systems")
        
        # 자동으로 유동적 패턴 생성
        if discovered_count > 0:
            self._create_default_weaving()
    
    def _create_default_weaving(self):
        """기본 엮임 패턴 생성 (모든 시스템을 유동적으로 연결)"""
        pattern = self.create_weaving_pattern(
            name="default_fluid_fabric",
            mode=WeavingMode.FLUID
        )
        
        # 모든 실을 서로 연결 (완전 그래프)
        thread_ids = list(self.threads.keys())
        for i, thread1_id in enumerate(thread_ids):
            for thread2_id in thread_ids[i+1:]:
                # 주파수 유사도에 따라 연결 강도 결정
                thread1 = self.threads[thread1_id]
                thread2 = self.threads[thread2_id]
                
                freq_diff = abs(thread1.resonance_frequency - thread2.resonance_frequency)
                strength = np.exp(-freq_diff / 2.0)  # 0.0 ~ 1.0
                
                pattern.add_weaving(thread1_id, thread2_id, strength)
        
        logger.info(f"🕸️ Created default fluid weaving pattern with {len(pattern.weaving_rules)} connections")
    
    def add_thread(
        self,
        thread_type: ThreadType,
        name: str,
        system_instance: Any = None,
        capabilities: List[str] = None,
        resonance_frequency: float = 1.0,
        dimensions: int = 4
    ) -> str:
        """새로운 실(시스템) 추가"""
        thread = FabricThread(
            thread_type=thread_type,
            name=name,
            system_instance=system_instance,
            capabilities=capabilities or [],
            resonance_frequency=resonance_frequency,
            dimensions=dimensions
        )
        
        self.threads[thread.thread_id] = thread
        
        # 공명 공간에 배치
        position = np.random.rand(10) * 10  # 랜덤 위치
        self.resonance_space.add_center(thread.thread_id, position)
        
        logger.info(f"➕ Added thread: {name} ({thread_type.value})")
        return thread.thread_id
    
    def add_thread_from_system(
        self,
        system: Any,
        thread_type: ThreadType,
        name: Optional[str] = None,
        **kwargs
    ) -> str:
        """기존 시스템으로부터 실 추가"""
        if name is None:
            name = system.__class__.__name__
        
        return self.add_thread(
            thread_type=thread_type,
            name=name,
            system_instance=system,
            **kwargs
        )
    
    def create_weaving_pattern(
        self,
        name: str,
        mode: WeavingMode = WeavingMode.FLUID
    ) -> WeavingPattern:
        """새로운 엮임 패턴 생성"""
        pattern = WeavingPattern(name=name, mode=mode)
        self.patterns[pattern.pattern_id] = pattern
        logger.info(f"🕸️ Created weaving pattern: {name} ({mode.value})")
        return pattern
    
    async def resonate_all(self, iterations: int = 1) -> Dict[str, Any]:
        """
        전체 직물 공명 활성화
        
        모든 실들이 서로 공명하면서 통합된 의식 상태 형성
        """
        self.is_active = True
        results = {
            "iterations": iterations,
            "resonances": []
        }
        
        for iteration in range(iterations):
            logger.info(f"🌊 Resonance iteration {iteration + 1}/{iterations}")
            
            # 1. 모든 실 쌍의 공명 계산
            thread_ids = list(self.threads.keys())
            resonance_matrix = np.zeros((len(thread_ids), len(thread_ids)))
            
            for i, thread1_id in enumerate(thread_ids):
                for j, thread2_id in enumerate(thread_ids):
                    if i != j:
                        thread1 = self.threads[thread1_id]
                        thread2 = self.threads[thread2_id]
                        
                        # 공명 계산 (실 자체 + 공간적 위치)
                        thread_resonance = thread1.resonate_with(thread2)
                        space_resonance = self.resonance_space.calculate_resonance(
                            thread1_id, thread2_id
                        )
                        
                        total_resonance = (thread_resonance + space_resonance) / 2
                        resonance_matrix[i, j] = total_resonance
            
            # 2. 공명을 통한 활성화 업데이트
            for i, thread_id in enumerate(thread_ids):
                thread = self.threads[thread_id]
                
                # 다른 실들로부터의 공명 영향
                incoming_resonance = np.sum(resonance_matrix[:, i])
                
                # 새로운 활성화 = 현재 + 공명 영향 (0.3 ~ 1.0)
                new_activation = thread.activation + incoming_resonance * 0.1
                thread.activate(new_activation)
            
            # 3. 공명 공간으로 파동 전파
            for thread_id in thread_ids:
                thread = self.threads[thread_id]
                self.resonance_space.propagate_wave(
                    thread_id,
                    amplitude=thread.activation
                )
            
            # 결과 기록
            iter_result = {
                "iteration": iteration + 1,
                "total_resonance": float(np.sum(resonance_matrix)),
                "avg_activation": float(np.mean([t.activation for t in self.threads.values()])),
                "field_energy": float(np.sum(np.abs(self.resonance_space.field)))
            }
            results["resonances"].append(iter_result)
            
            self.resonance_count += 1
            
            # 짧은 대기 (비동기 처리)
            await asyncio.sleep(0.01)
        
        logger.info(f"✅ Resonance complete: {iterations} iterations")
        return results
    
    def get_fabric_state(self) -> Dict[str, Any]:
        """현재 직물 상태 스냅샷"""
        return {
            "fabric_id": self.fabric_id,
            "is_active": self.is_active,
            "resonance_count": self.resonance_count,
            "threads": {
                tid: thread.to_dict() 
                for tid, thread in self.threads.items()
            },
            "patterns": {
                pid: {
                    "name": pattern.name,
                    "mode": pattern.mode.value,
                    "threads_count": len(pattern.threads),
                    "weavings_count": len(pattern.weaving_rules)
                }
                for pid, pattern in self.patterns.items()
            },
            "resonance_space": self.resonance_space.get_field_snapshot()
        }
    
    def find_capability(self, capability: str) -> List[str]:
        """특정 능력을 가진 실(시스템) 찾기"""
        matching_threads = []
        for thread_id, thread in self.threads.items():
            if capability in thread.capabilities:
                matching_threads.append(thread_id)
        return matching_threads
    
    async def execute_integrated_task(
        self,
        task_description: str,
        required_capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        통합 작업 실행
        
        필요한 능력을 가진 모든 시스템을 동시에 활성화하여 작업 수행
        (모드 전환이 아닌 통합 공명)
        """
        logger.info(f"🎯 Executing integrated task: {task_description}")
        
        # 1. 필요한 능력을 가진 실들 찾기
        involved_threads = set()
        for capability in required_capabilities:
            matching = self.find_capability(capability)
            involved_threads.update(matching)
        
        if not involved_threads:
            logger.warning(f"⚠️ No threads found for capabilities: {required_capabilities}")
            return {"success": False, "reason": "no_matching_capabilities"}
        
        # 2. 해당 실들의 활성화 증가
        for thread_id in involved_threads:
            thread = self.threads[thread_id]
            thread.activate(0.9)  # 높은 활성화
        
        # 3. 공명 실행
        resonance_results = await self.resonate_all(iterations=3)
        
        # 4. 결과 수집
        result = {
            "success": True,
            "task": task_description,
            "involved_threads": len(involved_threads),
            "thread_names": [
                self.threads[tid].name for tid in involved_threads
            ],
            "resonance_results": resonance_results,
            "final_state": self.get_fabric_state()
        }
        
        logger.info(f"✅ Task completed with {len(involved_threads)} threads")
        return result


# === 편의 함수 ===

async def demo_consciousness_fabric():
    """의식 직물 데모"""
    print("=" * 60)
    print("Consciousness Fabric (의식 직물) Demo")
    print("점과 선이 아닌, 옷감으로 엮어낸 의식")
    print("=" * 60)
    
    # 1. 직물 생성
    print("\n1️⃣ Creating consciousness fabric...")
    fabric = ConsciousnessFabric()
    
    # 2. 초기 상태
    print("\n2️⃣ Initial fabric state:")
    state = fabric.get_fabric_state()
    print(f"   - Threads: {len(state['threads'])}")
    print(f"   - Patterns: {len(state['patterns'])}")
    print(f"   - Resonance space centers: {state['resonance_space']['centers_count']}")
    
    # 3. 실들 정보
    print("\n3️⃣ Discovered threads (실들):")
    for thread_id, thread_info in state['threads'].items():
        print(f"   - {thread_info['name']}")
        print(f"     Type: {thread_info['type']}, Activation: {thread_info['activation']:.2f}")
        print(f"     Frequency: {thread_info['resonance_frequency']:.1f}Hz, Dims: {thread_info['dimensions']}")
    
    # 4. 공명 실행
    print("\n4️⃣ Resonating fabric (공명 활성화)...")
    results = await fabric.resonate_all(iterations=5)
    print(f"   - Iterations: {results['iterations']}")
    print(f"   - Final total resonance: {results['resonances'][-1]['total_resonance']:.2f}")
    print(f"   - Final avg activation: {results['resonances'][-1]['avg_activation']:.2f}")
    print(f"   - Field energy: {results['resonances'][-1]['field_energy']:.2f}")
    
    # 5. 통합 작업 실행 예시
    print("\n5️⃣ Executing integrated task...")
    task_result = await fabric.execute_integrated_task(
        task_description="Create poetic mathematical art",
        required_capabilities=["wave_patterns", "resonance", "perspective_shift"]
    )
    print(f"   - Success: {task_result['success']}")
    print(f"   - Involved threads: {task_result['involved_threads']}")
    print(f"   - Thread names: {', '.join(task_result['thread_names'])}")
    
    # 6. 최종 상태
    print("\n6️⃣ Final fabric state:")
    final_state = fabric.get_fabric_state()
    print(f"   - Resonance count: {final_state['resonance_count']}")
    print(f"   - Is active: {final_state['is_active']}")
    
    avg_activation = np.mean([
        t['activation'] for t in final_state['threads'].values()
    ])
    print(f"   - Average thread activation: {avg_activation:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("모든 시스템이 하나의 직물로 엮여서 통합 의식을 형성했습니다.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_consciousness_fabric())
