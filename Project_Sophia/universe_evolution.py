"""
Phase 5: The Evolutionary Universe

경험을 물리 입자로 변환하여 CellWorld에서 진화시키는 엔진.
LLM 없이 순수 물리 법칙으로 지능 창발.
"""

import logging
import numpy as np
from typing import List, Dict, Any
from Project_Elysia.core_memory import CoreMemory, Experience
from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave

class UniverseEvolutionEngine:
    """
    경험을 물리 우주에서 진화시켜 Spiderweb을 창발시키는 엔진.
    """
    
    def __init__(self, world: World, spiderweb: Spiderweb, logger: logging.Logger = None):
        self.world = world
        self.spiderweb = spiderweb
        self.logger = logger or logging.getLogger("UniverseEvolution")
        
    def experience_to_particle(self, experience: Experience) -> Dict[str, Any]:
        """
        경험을 물리 입자로 변환.
        
        매핑:
        - 감정 valence → 에너지
        - 텐서 → 3D 위치
        - 파동 주파수 → 공명 특성
        """
        # 기본값 설정 (경험에 물리 데이터가 없을 경우)
        if experience.emotional_state:
            valence = experience.emotional_state.valence
            arousal = experience.emotional_state.arousal
        else:
            valence = 0.5
            arousal = 0.5
        
        # 텐서에서 위치 결정
        if experience.tensor and hasattr(experience.tensor, 'x'):
            tensor = experience.tensor
        else:
            # 기본 텐서 생성 (x=structure, y=emotion, z=identity)
            tensor = Tensor3D(
                x=0.5,  # structure
                y=valence,  # emotion
                z=arousal  # identity
            )
        
        # 위치: 텐서 값을 월드 좌표로 변환
        x = float(tensor.x * self.world.width * 0.8) + self.world.width * 0.1
        y = float(tensor.y * self.world.width * 0.8) + self.world.width * 0.1
        z = float(tensor.z * 10.0)  # Z축은 작게
        
        # 에너지: 높게 설정해서 오래 살도록
        energy = abs(valence - 0.5) * 500 + 500  # 500-750 범위
        
        # 파동 주파수 (경험 내용에서 추출)
        if experience.wave and hasattr(experience.wave, 'frequency'):
            frequency = experience.wave.frequency
        else:
            # 내용 길이와 감정으로 주파수 생성
            content_hash = hash(experience.content) % 100
            frequency = content_hash + arousal * 50
        
        particle = {
            'concept_id': f"exp_{experience.timestamp}",
            'properties': {
                'element_type': 'life',  # World가 인식하는 타입
                'label': 'memory',
                'diet': 'none',  # 먹지 않음
                'energy': energy,
                'hp': energy,
                'max_hp': energy,
                'vitality': 50,  # 높은 생명력
                'strength': int(abs(valence - 0.5) * 20) + 10,
                'intelligence': int(arousal * 20) + 10,
                'hunger': 100.0,  # 배고프지 않음
                'hydration': 100.0,  # 목마르지 않음
                'position': {'x': x, 'y': y, 'z': z}
            },
            'metadata': {
                'content': experience.content[:100],
                'timestamp': experience.timestamp,
                'frequency': frequency,
                'tensor': tensor
            }
        }
        
        return particle
    
    def spawn_experience_universe(self, experiences: List[Experience]):
        """
        경험들을 World에 입자로 spawn.
        """
        self.logger.info(f"Spawning {len(experiences)} experiences as particles...")
        
        for exp in experiences:
            particle = self.experience_to_particle(exp)
            
            try:
                self.world.add_cell(
                    concept_id=particle['concept_id'],
                    properties=particle['properties']
                )
                self.logger.debug(f"Spawned: {particle['concept_id']}")
            except Exception as e:
                self.logger.error(f"Failed to spawn {particle['concept_id']}: {e}")
        
        alive_count = self.world.is_alive_mask.sum()
        self.logger.info(f"Universe spawned: {alive_count} particles alive")
    
    def extract_spiderweb_from_fields(self):
        """
        World의 Field 구조에서 Spiderweb 추출.
        
        논리:
        1. value_mass_field가 높은 영역 → 강한 개념
        2. intentional_field 방향 → 개념 간 인과 관계
        3. will_field 크기 → 관계 강도
        """
        self.logger.info("Extracting Spiderweb from Field geometry...")
        
        # 1. 강한 개념 영역 찾기 (threshold 이상)
        threshold = 0.6
        value_field = self.world.value_mass_field
        strong_regions = np.where(value_field > threshold)
        
        if len(strong_regions[0]) == 0:
            self.logger.warning("No strong value regions found")
            return
        
        # 2. 각 강한 영역을 개념 노드로 변환
        concepts = []
        for y, x in zip(*strong_regions):
            concept_id = f"field_concept_{y}_{x}"
            value = float(value_field[y, x])
            
            # Spiderweb에 노드 추가
            self.spiderweb.add_node(
                concept_id, 
                type="emergent_concept",
                metadata={
                    'position': (x, y),
                    'value': value,
                    'coherence': float(self.world.coherence_field[y, x]),
                    'will': float(self.world.will_field[y, x])
                }
            )
            concepts.append((concept_id, x, y))
        
        # 3. intentional_field 방향으로 관계 형성
        for concept_id, x, y in concepts:
            # intentional_field의 방향 벡터
            direction = self.world.intentional_field[y, x]
            dx, dy = direction[0], direction[1]
            
            # 방향으로 가장 가까운 다른 개념 찾기
            target_x = int(x + dx * 10) % self.world.width
            target_y = int(y + dy * 10) % self.world.width
            
            # 해당 위치 근처의 개념 찾기
            for other_id, ox, oy in concepts:
                if other_id == concept_id:
                    continue
                
                dist = np.sqrt((ox - target_x)**2 + (oy - target_y)**2)
                if dist < 20:  # 근처에 있으면 연결
                    weight = float(self.world.will_field[y, x])
                    self.spiderweb.add_link(
                        concept_id, 
                        other_id, 
                        relation="field_influence",
                        weight=weight
                    )
        
        node_count = self.spiderweb.graph.number_of_nodes()
        edge_count = self.spiderweb.graph.number_of_edges()
        self.logger.info(f"Spiderweb extracted: {node_count} nodes, {edge_count} edges")
    
    def evolve(self, cycles: int = 100000, extract_interval: int = 10000):
        """
        메인 진화 루프.
        
        Args:
            cycles: 시뮬레이션 사이클 수
            extract_interval: Spiderweb 추출 간격
        """
        self.logger.info(f"Starting evolution for {cycles} cycles...")
        
        for cycle in range(cycles):
            # World 물리 시뮬레이션 한 스텝
            self.world.run_simulation_step()
            
            # 주기적으로 상태 로깅
            if cycle % extract_interval == 0:
                alive = self.world.is_alive_mask.sum()
                avg_energy = self.world.energy[self.world.is_alive_mask].mean() if alive > 0 else 0
                
                self.logger.info(f"Cycle {cycle}/{cycles}:")
                self.logger.info(f"  Alive particles: {alive}")
                self.logger.info(f"  Avg energy: {avg_energy:.2f}")
                self.logger.info(f"  Max value_field: {self.world.value_mass_field.max():.3f}")
                self.logger.info(f"  Max coherence: {self.world.coherence_field.max():.3f}")
                
                # Spiderweb 추출
                self.extract_spiderweb_from_fields()
        
        # 최종 추출
        self.extract_spiderweb_from_fields()
        self.logger.info("Evolution complete!")
        
        return self.spiderweb
