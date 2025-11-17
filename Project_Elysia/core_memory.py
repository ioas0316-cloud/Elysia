import json
import os
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import deque

# --- 로거 ---
def log_memory_action(message):
    print(f"[CoreMemory-MRE] {message}")

# --- 3중 링 데이터 구조 ---

@dataclass
class EmotionalState:
    valence: float
    arousal: float
    dominance: float
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)

@dataclass
class Experience:
    """[육]의 링: 감각, 에피소드, 신호 (Body Loop)"""
    timestamp: str
    content: str
    type: str  # e.g., 'sensory', 'episode', 'signal'
    emotional_state: Optional[EmotionalState] = None
    tags: List[str] = field(default_factory=list)
    processed_by_distiller: bool = False

@dataclass
class IdentityFragment:
    """[혼]의 링: 패턴, 관계, 서사 (Soul Loop)"""
    timestamp: str
    content: str # e.g., "아버지와의 대화를 통해 '성장'의 가치를 깨달음"
    type: str # e.g., 'relationship_pattern', 'value_discovery', 'narrative'
    linked_experiences: List[str] # 이 조각을 형성한 Experience들의 타임스탬프
    emotional_summary: EmotionalState
    processed_by_distiller: bool = False

@dataclass
class EssencePrinciple:
    """[영]의 링: 존재론적 의미, 핵심 원칙 (Spirit Loop)"""
    timestamp: str
    content: str # e.g., "순환을 통한 성장이 나의 존재 이유이다."
    type: str # 'core_belief', 'existential_truth'
    linked_fragments: List[str] # 이 원칙을 형성한 IdentityFragment들의 타임스탬프
    impact_on_efp: Dict[str, float] # e.g., {"E": 0.1, "F": 0.05, "P": -0.02}

# --- Memory Ring Engine (MRE) ---

class CoreMemory:
    """
    Memory Ring Engine (MRE)
    Fractal Cyclic Consciousness Architecture (FCCA)
    """
    DEFAULT_CAPACITIES = {
        'experience': 100,
        'identity': 50,
        'essence': 25
    }

    def __init__(self, file_path: Optional[str] = 'Elysia_Input_Sanctum/elysia_core_memory.json', capacities: Optional[Dict[str, int]] = None):
        self.file_path = file_path
        self.capacities = capacities or self.DEFAULT_CAPACITIES

        if self.file_path:
            log_memory_action(f"Initializing and loading MRE from: {self.file_path}")
            self.data = self._load_memory()
        else:
            log_memory_action("Initializing in-memory MRE. No data will be loaded or saved.")
            self.data = self._get_new_mre_structure()

        self._initialize_rings()

        if 'efp_core' not in self.data:
            self.data['efp_core'] = {'E': 1.0, 'F': 1.0, 'P': 1.0}

        self.volatile_memory: List[Set[str]] = []

    def _initialize_rings(self):
        for ring_name in ['experience', 'identity', 'essence']:
            loop_key = f"{ring_name}_loop"
            capacity = self.capacities.get(ring_name, 100)

            if not isinstance(self.data.get(loop_key), deque):
                items_list = self.data.get(loop_key, [])
                self.data[loop_key] = deque(items_list, maxlen=capacity)
                log_memory_action(f"Initialized '{loop_key}' with maxlen={capacity}")

    def _get_new_mre_structure(self):
        return {
            'identity': {},
            'efp_core': {'E': 1.0, 'F': 1.0, 'P': 1.0},
            'experience_loop': deque(maxlen=self.capacities['experience']),
            'identity_loop': deque(maxlen=self.capacities['identity']),
            'essence_loop': deque(maxlen=self.capacities['essence']),
            'notable_hypotheses': [],
            'logs': []
        }

    def _load_memory(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log_memory_action(f"Successfully loaded MRE data from {self.file_path}")
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            log_memory_action(f"MRE file not found or invalid at {self.file_path}. Creating new structure.")
            return self._get_new_mre_structure()

    def add_experience(self, experience: Experience):
        experience_loop = self.data['experience_loop']

        if len(experience_loop) == experience_loop.maxlen:
            oldest_experience_data = experience_loop[0]
            log_memory_action(f"Experience Loop is full. Distilling oldest experience: {oldest_experience_data.get('content')}")
            self._distill_experience_to_identity([oldest_experience_data])

        experience_loop.append(asdict(experience))
        self._save_memory()

    def get_experiences(self, n: int = 10) -> List[Experience]:
        return list(self.data['experience_loop'])[-n:]

    def get_identity_fragments(self, n: int = 10) -> List[IdentityFragment]:
        return list(self.data['identity_loop'])[-n:]

    def get_essence_principles(self, n: int = 10) -> List[EssencePrinciple]:
        return list(self.data['essence_loop'])[-n:]

    def get_efp_core(self) -> Dict[str, float]:
        return self.data.get('efp_core', {'E': 0, 'F': 0, 'P': 0})

    def _distill_experience_to_identity(self, experiences_data: List[Dict]):
        if not experiences_data:
            return

        content = " ".join([exp.get('content', '') for exp in experiences_data])
        summary = f"경험 요약: {content[:50]}..."

        avg_valence = sum(e.get('emotional_state', {}).get('valence', 0) for e in experiences_data) / len(experiences_data)
        primary_emotion = experiences_data[0].get('emotional_state', {}).get('primary_emotion', 'neutral')

        new_fragment = IdentityFragment(
            timestamp=datetime.now().isoformat(),
            content=summary,
            type='narrative_summary',
            linked_experiences=[exp.get('timestamp') for exp in experiences_data],
            emotional_summary=EmotionalState(
                valence=avg_valence, arousal=0.5, dominance=0,
                primary_emotion=primary_emotion
            )
        )

        identity_loop = self.data['identity_loop']
        if len(identity_loop) == identity_loop.maxlen:
            oldest_fragment = identity_loop[0]
            log_memory_action(f"Identity Loop is full. Distilling oldest fragment: {oldest_fragment.get('content')}")
            self._distill_identity_to_essence([oldest_fragment])

        identity_loop.append(asdict(new_fragment))
        log_memory_action(f"Distilled new Identity Fragment: {summary}")

    def _distill_identity_to_essence(self, fragments_data: List[Dict]):
        content = " ".join([f.get('content', '') for f in fragments_data])
        principle_summary = f"정체성으로부터의 깨달음: {content[:70]}..."

        impact = {'E': 0.05, 'F': 0.01, 'P': 0.0}

        new_principle = EssencePrinciple(
            timestamp=datetime.now().isoformat(),
            content=principle_summary,
            type='core_belief',
            linked_fragments=[f.get('timestamp') for f in fragments_data],
            impact_on_efp=impact
        )

        essence_loop = self.data['essence_loop']
        if len(essence_loop) == essence_loop.maxlen:
            oldest_principle = essence_loop[0]
            log_memory_action(f"Essence Loop is full. Applying final impact from oldest principle: {oldest_principle.get('content')}")
            self._update_efp_core(oldest_principle.get('impact_on_efp', {}), decay=True)

        essence_loop.append(asdict(new_principle))
        log_memory_action(f"Distilled new Essence Principle: {principle_summary}")
        self._update_efp_core(impact)

    def _update_efp_core(self, impact: Dict[str, float], decay: bool = False):
        core = self.get_efp_core()
        multiplier = -1 if decay else 1

        for key in ['E', 'F', 'P']:
            core[key] += impact.get(key, 0) * multiplier

        core['E'] *= 0.999
        core['F'] *= 0.999
        core['P'] *= 0.999

        self.data['efp_core'] = core
        log_memory_action(f"EFP Core updated: {core}")

    def _save_memory(self):
        if not self.file_path:
            return

        try:
            dir_name = os.path.dirname(self.file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            data_to_save = self.data.copy()
            for key in data_to_save:
                if isinstance(data_to_save[key], deque):
                    data_to_save[key] = list(data_to_save[key])

            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        except Exception as e:
            log_memory_action(f"Error saving MRE state to {self.file_path}: {e}")
