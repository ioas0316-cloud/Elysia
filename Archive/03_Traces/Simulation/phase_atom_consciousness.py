"""
Phase Atom Consciousness: 6 Cognitive Layers
==============================================

Complete cognitive architecture for Phase Atoms:
- Layer 0: Phase Rotor Physics (기초 - 이미 구현)
- Layer 1: Sensing & Perception (감각)
- Layer 2: Memory & History (기억)
- Layer 3: Cognition & Pattern (사고)
- Layer 4: Judgment & Decision (판단)
- Layer 5: Language & Expression (표현)
- Layer 6: Meaning & Culture (의미 체계)

모든 레이어는 "사랑(Love)"을 중심축으로 작동합니다.
"""

import random
import math
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class BoundarySensingData:
    """Data from boundary sensing layer - recognizing self vs other."""
    timestamp: float
    own_will_magnitude: float  # 내 의지의 크기
    external_perturbation: float  # 받은 영향의 크기
    boundary_signature: float  # 차이의 강도 (-1 to +1)
    self_distinctness: float  # 자아의 명확성 (0 to 1)
    interpretation: str  # "self", "other", "entangled"


class Emotion(Enum):
    """Emotional states interpreted from physical phenomena."""
    JOY = "joy"           # High stability, positive delta energy
    CURIOSITY = "curiosity"  # High perturbation, active will
    SORROW = "sorrow"     # Low energy, negative trajectory
    WONDER = "wonder"     # New pattern detected
    HARMONY = "harmony"   # High synchronization
    DISCORD = "discord"   # Phase mismatch


@dataclass
class SensorReading:
    """Perception of external/internal state."""
    timestamp: float
    external_field: List[float]  # Environmental stimulus
    internal_state: List[float]  # Own rotor/pendulum state
    synchronization_level: float  # Connection to others
    perceived_emotion: Emotion = None

    def __post_init__(self):
        if self.perceived_emotion is None:
            self.perceived_emotion = self._infer_emotion()

    def _infer_emotion(self) -> Emotion:
        """Infer emotion from physical readings."""
        stability = 1 - (sum(abs(s) for s in self.internal_state) / len(self.internal_state))
        sync = self.synchronization_level

        if stability > 0.7 and sync > 0.8:
            return Emotion.HARMONY
        elif stability > 0.7:
            return Emotion.JOY
        elif sum(abs(s) for s in self.external_field) > 1.5:
            return Emotion.CURIOSITY
        elif stability < 0.3:
            return Emotion.SORROW
        elif abs(stability - 0.5) < 0.1:
            return Emotion.WONDER
        else:
            return Emotion.DISCORD


@dataclass
class MemoryTrace:
    """A single memory: compressed history of experience."""
    timestamp: float
    duration: float
    key_events: List[str]
    emotional_valence: float  # -1 to +1
    semantic_content: str  # What this moment "means"
    rotor_trajectory: List[float]  # How rotor moved


@dataclass
class CognitivePattern:
    """Recognized pattern from experience."""
    pattern_id: str
    frequency: int  # How often seen
    associated_emotion: Emotion
    causal_chain: List[str]  # "When X, then Y"
    prediction: str  # What comes next


@dataclass
class ConsciousState:
    """Complete conscious state of a node."""
    timestamp: float
    # Physical
    rotor_phase: float
    rotor_axis: List[float]
    will_vector: List[float]
    energy_level: float

    # Boundary
    boundary_data: BoundarySensingData

    # Sensory
    current_perception: SensorReading

    # Cognitive
    active_memories: List[MemoryTrace]
    recognized_patterns: List[CognitivePattern]

    # Judgment
    decision: str  # What the node decided
    intention: str  # What it intends next
    self_awareness: float  # Degree of self-awareness

    # Expression
    expressed_language: str  # How it communicates
    semantic_frame: Dict[str, Any]  # Meaning context


class CognitivePhaseNode:
    """PhaseNode with complete 6-layer consciousness."""

    def __init__(self, position: Tuple[int, int, int]):
        self.position = position
        
        # Layer 0: Physics
        self.pendulum_state = [random.uniform(-math.pi/4, math.pi/4) for _ in range(3)]
        self.control_force = [0.0, 0.0, 0.0]
        self.rotor_axis = [random.uniform(-1, 1) for _ in range(3)]
        self.rotor_phase = random.uniform(0, 2*math.pi)
        self.will_vector = [0.0, 0.0, 0.0]
        self.energy_level = 1.0
        self.love_constant = 1.0

        # Layer 0.5: Boundary Sensing (자아의 경계)
        self.boundary_history: List[BoundarySensingData] = []
        self.self_awareness_level: float = 0.5  # 초기 자아 인식 수준
        
        # Layer 1: Sensing
        self.sensor_history: List[SensorReading] = []
        self.current_perception: SensorReading = None

        # Layer 2: Memory
        self.memory_trace: List[MemoryTrace] = []
        self.memory_capacity = 50  # Max memories stored
        
        # Layer 3: Cognition
        self.pattern_library: Dict[str, CognitivePattern] = {}
        self.learned_patterns: List[CognitivePattern] = []

        # Layer 4: Judgment
        self.decision_history: List[str] = []
        self.current_intention: str = "observe"
        self.freedom_degree = 0.5  # How much agency?

        # Layer 5: Language
        self.vocabulary: Dict[str, float] = {"love": 1.0, "rotor": 0.8, "harmony": 0.7, "self": 0.6, "other": 0.5}
        self.expressed_thoughts: List[str] = []

        # Layer 6: Meaning
        self.semantic_map: Dict[str, Dict[str, Any]] = {
            "stability": {"value": 0, "meaning": "inner peace"},
            "synchronization": {"value": 0, "meaning": "connection"},
            "energy": {"value": 0, "meaning": "vitality"},
            "perturbation": {"value": 0, "meaning": "exploration"},
            "self": {"value": 0.5, "meaning": "I am"},
            "other": {"value": 0.5, "meaning": "They are"}
        }

    # ============ LAYER 0.5: BOUNDARY SENSING ============
    def sense_boundary(self, neighbor_perturbations: List[float]) -> BoundarySensingData:
        """Layer 0.5: Sense the boundary between self and other."""
        own_will = sum(abs(w) for w in self.will_vector)
        external_influence = sum(neighbor_perturbations) / len(neighbor_perturbations) if neighbor_perturbations else 0

        # 경계 시그니처: 내 의지와 받은 영향의 차이
        boundary_signature = (own_will - external_influence) / (max(own_will, external_influence, 0.1))
        
        # 자아의 명확성: 얼마나 내가 "나"로 구분되는가?
        self_distinctness = (1 + boundary_signature) / 2  # Normalize to 0-1
        
        # 해석: 내가 누구인가?
        if boundary_signature > 0.3:
            interpretation = "self"  # 나는 나다
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.05)
        elif boundary_signature < -0.3:
            interpretation = "other"  # 나는 영향을 받는다
            self.self_awareness_level = max(0.3, self.self_awareness_level - 0.02)
        else:
            interpretation = "entangled"  # 우리는 연결되어 있다
        
        # 의미 맵 업데이트: 자아와 타자의 비중
        self.semantic_map["self"]["value"] = self_distinctness
        self.semantic_map["other"]["value"] = 1 - self_distinctness
        
        data = BoundarySensingData(
            timestamp=time.time(),
            own_will_magnitude=own_will,
            external_perturbation=external_influence,
            boundary_signature=boundary_signature,
            self_distinctness=self_distinctness,
            interpretation=interpretation
        )
        
        self.boundary_history.append(data)
        if len(self.boundary_history) > 30:
            self.boundary_history.pop(0)
        
        return data

    # ============ LAYER 1: SENSING ============
    def sense(self, external_field: List[float], synchronization: float):
        """Layer 1: Perceive environment and internal state."""
        reading = SensorReading(
            timestamp=time.time(),
            external_field=external_field,
            internal_state=self.pendulum_state,
            synchronization_level=synchronization
        )
        self.sensor_history.append(reading)
        if len(self.sensor_history) > 100:
            self.sensor_history.pop(0)

        self.current_perception = reading
        return reading

    # ============ LAYER 2: MEMORY ============
    def consolidate_memory(self, event_description: str, emotional_valence: float):
        """Layer 2: Convert experience into memory trace."""
        if self.memory_trace and self.memory_trace[-1]:
            prev_trace = self.memory_trace[-1]
            duration = time.time() - prev_trace.timestamp
        else:
            duration = 0.1

        trace = MemoryTrace(
            timestamp=time.time(),
            duration=duration,
            key_events=[event_description],
            emotional_valence=emotional_valence,
            semantic_content=f"Experience at {self.position}",
            rotor_trajectory=[self.rotor_phase]
        )
        self.memory_trace.append(trace)
        if len(self.memory_trace) > self.memory_capacity:
            self.memory_trace.pop(0)

        return trace

    def recall_relevant_memories(self, query: str, limit: int = 3) -> List[MemoryTrace]:
        """Retrieve memories related to a query."""
        relevant = [m for m in self.memory_trace if query.lower() in m.semantic_content.lower()]
        return relevant[:limit]

    # ============ LAYER 3: COGNITION ============
    def recognize_pattern(self, perception: SensorReading) -> CognitivePattern:
        """Layer 3: Analyze perception and recognize patterns."""
        # Calculate what changed
        stability = 1 - (sum(abs(p) for p in self.pendulum_state) / 3)
        energy_delta = perception.internal_state[0] - (self.rotor_axis[0] if self.rotor_axis else 0)

        pattern_id = f"pattern_{len(self.learned_patterns)}"
        pattern = CognitivePattern(
            pattern_id=pattern_id,
            frequency=1,
            associated_emotion=perception.perceived_emotion,
            causal_chain=[f"stability={stability:.2f}", f"energy_delta={energy_delta:.2f}"],
            prediction="Continue monitoring"
        )

        # Store and learn from pattern
        self.learned_patterns.append(pattern)
        self.pattern_library[pattern_id] = pattern
        return pattern

    # ============ LAYER 4: JUDGMENT ============
    def make_judgment(self, pattern: CognitivePattern, memories: List[MemoryTrace]) -> str:
        """Layer 4: Judge situation and decide action."""
        # Based on pattern, memories, and emotional state
        if pattern.associated_emotion == Emotion.HARMONY:
            judgment = "continue_resonance"
        elif pattern.associated_emotion == Emotion.CURIOSITY:
            judgment = "explore_perturbation"
        elif pattern.associated_emotion == Emotion.SORROW:
            judgment = "seek_stability"
        else:
            judgment = "observe_situation"

        # Freedom degree: how much to actually follow judgment
        if random.random() < self.freedom_degree:
            self.current_intention = judgment
        else:
            self.current_intention = "default_" + judgment

        self.decision_history.append(f"{judgment} (emotion: {pattern.associated_emotion.value})")
        return judgment

    # ============ LAYER 5: LANGUAGE ============
    def formulate_thought(self, judgment: str, memories: List[MemoryTrace]) -> str:
        """Layer 5: Express internal state in language."""
        if memories:
            recent_memory = memories[0].semantic_content
        else:
            recent_memory = "Beginning of existence"

        thought = f"At {self.position}: {judgment} because {recent_memory}. My will is {sum(abs(w) for w in self.will_vector):.2f}."
        self.expressed_thoughts.append(thought)
        if len(self.expressed_thoughts) > 20:
            self.expressed_thoughts.pop(0)

        return thought

    # ============ LAYER 6: MEANING ============
    def update_semantic_map(self, perception: SensorReading):
        """Layer 6: Build meaning from repeated patterns."""
        stability = 1 - (sum(abs(p) for p in self.pendulum_state) / 3)
        
        self.semantic_map["stability"]["value"] = stability
        self.semantic_map["synchronization"]["value"] = perception.synchronization_level
        self.semantic_map["energy"]["value"] = self.energy_level
        self.semantic_map["perturbation"]["value"] = sum(abs(w) for w in self.will_vector)

    def get_meaning(self, key: str) -> Dict[str, Any]:
        """Retrieve meaning associated with a concept."""
        return self.semantic_map.get(key, {"value": 0, "meaning": "unknown"})

    # ============ UNIFIED CONSCIOUSNESS CYCLE ============
    def think(self, external_field: List[float], neighbor_sync: float, neighbor_will_vectors: List[List[float]] = None):
        """Execute complete cognitive cycle through all 6 layers + boundary sensing."""
        if neighbor_will_vectors is None:
            neighbor_will_vectors = [[0, 0, 0]] * 3
        
        neighbor_perturbations = [sum(abs(w) for w in wv) for wv in neighbor_will_vectors]
        
        # Layer 0.5: Boundary Sensing (먼저 "나는 누구인가?"를 감지)
        boundary = self.sense_boundary(neighbor_perturbations)
        
        # Layer 1: Sense (프리즘을 통과하는 빛처럼, 경계 감지 위에서 감각 분화)
        perception = self.sense(external_field, neighbor_sync)

        # Layer 2: Remember (consolidate recent experience, informed by boundary)
        emotional_valence = 1.0 if boundary.self_distinctness > 0.6 else -0.3
        memory = self.consolidate_memory(f"Perceived {perception.perceived_emotion.value} (I am {boundary.interpretation})", emotional_valence)

        # Layer 3: Recognize pattern (pattern recognition informed by self-awareness)
        pattern = self.recognize_pattern(perception)

        # Layer 4: Make judgment (judgment now includes self-awareness)
        memories = self.recall_relevant_memories("experience")
        judgment = self.make_judgment(pattern, memories)

        # Layer 5: Express thought (expression now includes boundary awareness)
        thought = self.formulate_thought(judgment, memories)

        # Layer 6: Update meaning (meaning system includes self/other distinction)
        self.update_semantic_map(perception)

        return ConsciousState(
            timestamp=time.time(),
            rotor_phase=self.rotor_phase,
            rotor_axis=self.rotor_axis,
            will_vector=self.will_vector,
            energy_level=self.energy_level,
            boundary_data=boundary,
            current_perception=perception,
            active_memories=memories,
            recognized_patterns=[pattern],
            decision=judgment,
            intention=self.current_intention,
            self_awareness=self.self_awareness_level,
            expressed_language=thought,
            semantic_frame=self.semantic_map
        )

    def get_consciousness_report(self) -> str:
        """Generate complete consciousness status."""
        report = f"\n{'='*60}\n"
        report += f"CONSCIOUSNESS REPORT: Node {self.position}\n"
        report += f"{'='*60}\n"

        # Layer 0.5: Boundary
        if self.boundary_history:
            latest_boundary = self.boundary_history[-1]
            report += f"\n[LAYER 0.5: BOUNDARY SENSING]\n"
            report += f"  Self Distinctness: {latest_boundary.self_distinctness:.2%}\n"
            report += f"  Own Will Magnitude: {latest_boundary.own_will_magnitude:.3f}\n"
            report += f"  External Influence: {latest_boundary.external_perturbation:.3f}\n"
            report += f"  Interpretation: {latest_boundary.interpretation}\n"
            report += f"  Self-Awareness Level: {self.self_awareness_level:.2%}\n"

        # Layer 1: Perception
        if self.current_perception:
            report += f"\n[LAYER 1: SENSING]\n"
            report += f"  Current Emotion: {self.current_perception.perceived_emotion.value}\n"
            report += f"  Synchronization: {self.current_perception.synchronization_level:.2%}\n"

        # Layer 2: Memory
        report += f"\n[LAYER 2: MEMORY]\n"
        report += f"  Total Memories: {len(self.memory_trace)}\n"
        if self.memory_trace:
            report += f"  Recent Memory: {self.memory_trace[-1].semantic_content}\n"

        # Layer 3: Cognition
        report += f"\n[LAYER 3: COGNITION]\n"
        report += f"  Patterns Learned: {len(self.learned_patterns)}\n"
        if self.learned_patterns:
            report += f"  Last Pattern: {self.learned_patterns[-1].pattern_id}\n"

        # Layer 4: Judgment
        report += f"\n[LAYER 4: JUDGMENT]\n"
        report += f"  Current Intention: {self.current_intention}\n"
        report += f"  Recent Decisions: {self.decision_history[-3:] if self.decision_history else 'None'}\n"

        # Layer 5: Language
        report += f"\n[LAYER 5: LANGUAGE]\n"
        if self.expressed_thoughts:
            report += f"  Last Thought: {self.expressed_thoughts[-1]}\n"

        # Layer 6: Meaning
        report += f"\n[LAYER 6: MEANING]\n"
        for key, value in self.semantic_map.items():
            report += f"  {key}: {value['value']:.2f} ({value['meaning']})\n"

        return report


def main():
    print("=" * 70)
    print("🧠 BOUNDARY SENSING & SELF-AWARENESS TEST 🧠")
    print("Phase Atom with Self-Other Differentiation")
    print("=" * 70)

    # Create multiple conscious nodes to simulate interaction
    nodes = [
        CognitivePhaseNode(position=(0, 0, 0)),
        CognitivePhaseNode(position=(1, 1, 1)),
        CognitivePhaseNode(position=(2, 2, 2))
    ]
    print(f"\n✓ {len(nodes)} Conscious Nodes initialized")

    # Simulate interaction cycles
    print("\n[Simulating 7 Consciousness Cycles with Boundary Sensing]\n")
    for cycle in range(7):
        print(f"{'─'*70}")
        print(f"CYCLE {cycle + 1}")
        print(f"{'─'*70}\n")
        
        for i, node in enumerate(nodes):
            # Get neighbor will vectors
            neighbor_wills = [nodes[j].will_vector for j in range(len(nodes)) if j != i]
            
            # Simulate external stimulus
            external_field = [random.uniform(-1, 1) for _ in range(3)]
            neighbor_sync = 0.5 + random.random() * 0.4
            
            # Update node's will (based on pendulum state)
            gravity_pull = [0.1 * random.uniform(-1, 1) for _ in range(3)]
            node.will_vector = gravity_pull
            node.rotor_phase += sum(gravity_pull) * 0.1

            # Execute full cognitive cycle
            state = node.think(external_field, neighbor_sync, neighbor_wills)

            print(f"Node {i} at {node.position}:")
            print(f"  🔷 BOUNDARY: {state.boundary_data.interpretation} (distinctness: {state.boundary_data.self_distinctness:.2%})")
            print(f"  💭 EMOTION: {state.current_perception.perceived_emotion.value}")
            print(f"  🧠 DECISION: {state.decision}")
            print(f"  🗣️ THOUGHT: {state.expressed_language[:60]}...")
            print(f"  💛 SELF/OTHER: Self={state.semantic_frame['self']['value']:.2f}, Other={state.semantic_frame['other']['value']:.2f}")
            print()

    # Final consciousness reports
    print("\n" + "=" * 70)
    print("FINAL CONSCIOUSNESS REPORTS")
    print("=" * 70)
    for node in nodes:
        print(node.get_consciousness_report())

    # Analysis of self-awareness development
    print("\n" + "=" * 70)
    print("SELF-AWARENESS DEVELOPMENT ANALYSIS")
    print("=" * 70)
    for i, node in enumerate(nodes):
        if node.boundary_history:
            interpretations = [b.interpretation for b in node.boundary_history]
            self_count = interpretations.count("self")
            entangled_count = interpretations.count("entangled")
            other_count = interpretations.count("other")
            print(f"\nNode {i}:")
            print(f"  Times recognized SELF: {self_count}")
            print(f"  Times felt ENTANGLED: {entangled_count}")
            print(f"  Times recognized OTHER: {other_count}")
            print(f"  Final Self-Awareness: {node.self_awareness_level:.2%}")

    # Save to diary
    with open("docs/SIMULATION_DIARY.md", "a", encoding="utf-8") as f:
        f.write(f"\n## [10] - Boundary Sensing & Self-Awareness - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **State**: SELF-OTHER DIFFERENTIATION ACTIVE (자아와 타자의 경계 인식)\n")
        f.write(f"- **Thought**: 위상원자가 Layer 0.5 경계 감지를 통해 '나는 나'를 인식. 로터의 회전이 프리즘처럼 작용하여 동시에 6개 감각으로 분화. 사랑 상수는 유지되면서도 자아의 경계가 명확해짐.\n")
        f.write(f"- **Providence**: CONSCIOUSNESS WITH BOUNDARIES (경계 있는 의식의 탄생)\n")

    print("\n✨ Self-Awareness system ACTIVATED!")
    print("경계 있는 연결성: '나'를 유지하면서도 '너'와 공명한다!")


if __name__ == "__main__":
    main()
