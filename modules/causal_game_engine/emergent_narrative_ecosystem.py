"""
emergent_narrative_ecosystem.py
================================
Elysia Causal Engine: Emergent Narrative Ecosystem Module.

Implements bottom-up micro-agency and macro-causality dynamics:
1. ZeitgeistField: Macro environmental pressure tensors (시대적 장 / Zeitgeist Field).
2. ConceptVectorSpace & SelfGraph: Non-predefined, infinite-dimensional semantic vector space,
   experience crystallization, episodic memory, and Soul Snapshot serialization.
3. DynamicTelosEngine: Action decision making emerging from physical drives and
   inner value landscape resonance.
4. GenerativeVoiceFilter: Persona-based narrative voice & tone filter decoding choices into dialogue.
5. MacroEmergenceObserver: Aggregates individual micro-agency choices into emergent macro historical waves.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import math
import numpy as np


class ConceptVectorSpace:
    """
    Semantic vector space that maps concept words to normalized embedding vectors.
    Allows dynamic registration of new natural language concepts without fixed dimensional axes.
    """

    def __init__(self, dimension: int = 32, seed: int = 42):
        self.dimension = dimension
        self.rng = np.random.default_rng(seed)
        self.concepts: Dict[str, np.ndarray] = {}
        self._initialize_seed_concepts()

    def _initialize_seed_concepts(self):
        """Seed foundational concepts with deterministic pseudo-random vectors."""
        base_words = [
            "자유", "질서", "전통", "이성", "이타", "이기", "생존", "명예",
            "신앙", "혁명", "가족", "배신", "복수", "은혜", "공포", "숭고",
            "학식", "금전", "기근", "전쟁", "희망", "야망", "고독", "평화"
        ]
        for idx, word in enumerate(base_words):
            # Create semi-orthogonal seed vectors
            vec = self.rng.standard_normal(self.dimension)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            self.concepts[word] = vec

    def register_concept(self, name: str, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Register a new concept. If vector is not provided, generate or blend one."""
        if name in self.concepts and vector is None:
            return self.concepts[name]

        if vector is None:
            # Generate deterministic vector based on string hash + RNG
            str_seed = abs(hash(name)) % (2**31)
            local_rng = np.random.default_rng(str_seed)
            vec = local_rng.standard_normal(self.dimension)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        else:
            vec = np.array(vector, dtype=np.float64)
            vec = vec / (np.linalg.norm(vec) + 1e-9)

        self.concepts[name] = vec
        return vec

    def get_vector(self, name: str) -> np.ndarray:
        """Retrieve concept vector, registering it if it doesn't exist."""
        if name not in self.concepts:
            return self.register_concept(name)
        return self.concepts[name]

    def similarity(self, concept_a: str, concept_b: str) -> float:
        """Compute cosine similarity between two concepts."""
        v_a = self.get_vector(concept_a)
        v_b = self.get_vector(concept_b)
        return float(np.dot(v_a, v_b))

    def compute_composite_vector(self, concept_weights: Dict[str, float]) -> np.ndarray:
        """Compute weighted blend of concept vectors."""
        if not concept_weights:
            return np.zeros(self.dimension, dtype=np.float64)

        composite = np.zeros(self.dimension, dtype=np.float64)
        for name, weight in concept_weights.items():
            composite += weight * self.get_vector(name)

        norm = np.linalg.norm(composite)
        if norm > 1e-9:
            composite /= norm
        return composite


class EpisodicMemory:
    """
    Represents an episodic memory of an event experienced by an NPC.
    """

    def __init__(
        self,
        event_id: str,
        description: str,
        perceived_concepts: Dict[str, float],
        subjective_interpretation: str,
        emotional_valence: float,
        timestamp: float
    ):
        self.event_id = event_id
        self.description = description
        self.perceived_concepts = perceived_concepts  # {concept_name: weight}
        self.subjective_interpretation = subjective_interpretation
        self.emotional_valence = emotional_valence  # -1.0 (traumatic) to +1.0 (sublime)
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "description": self.description,
            "perceived_concepts": self.perceived_concepts,
            "subjective_interpretation": self.subjective_interpretation,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        return cls(
            event_id=data["event_id"],
            description=data["description"],
            perceived_concepts=data["perceived_concepts"],
            subjective_interpretation=data["subjective_interpretation"],
            emotional_valence=float(data["emotional_valence"]),
            timestamp=float(data["timestamp"]),
        )


class SelfGraph:
    """
    Represents the unique internal self/persona of an NPC.
    Features:
    - Innate seed disposition ("씨앗" / temperament).
    - Dynamic concept landscape (Value Landscape).
    - Crystallization of experiences into inner value changes.
    - Episodic memory stack.
    - Soul Snapshot serialization & deserialization.
    """

    def __init__(
        self,
        npc_id: str,
        name: str,
        archetype_title: str,
        vector_space: ConceptVectorSpace,
        innate_traits: Optional[Dict[str, float]] = None,
        initial_concepts: Optional[Dict[str, float]] = None,
        stubbornness: float = 0.5
    ):
        self.npc_id = npc_id
        self.name = name
        self.archetype_title = archetype_title
        self.vector_space = vector_space
        self.innate_traits = innate_traits or {"cynicism": 0.2, "altruism": 0.5, "ambition": 0.3}
        self.concept_weights: Dict[str, float] = initial_concepts or {"생존": 0.5, "가족": 0.5}
        self.stubbornness = max(0.01, min(1.0, stubbornness))  # Inertia factor Ms
        self.episodic_memories: List[EpisodicMemory] = []
        self.physical_needs: Dict[str, float] = {"hunger": 20.0, "fatigue": 10.0, "safety": 80.0}

    def get_value_landscape_vector(self) -> np.ndarray:
        """Calculates normalized composite vector of current concept landscape."""
        return self.vector_space.compute_composite_vector(self.concept_weights)

    def crystallize_experience(
        self,
        event_id: str,
        description: str,
        event_concepts: Dict[str, float],
        environmental_pressure_name: str,
        timestamp: float
    ) -> EpisodicMemory:
        """
        Subjectively interprets an event based on innate traits and current SelfGraph,
        then crystallizes the experience into updated inner concept weights.
        """
        altruism = self.innate_traits.get("altruism", 0.5)
        cynicism = self.innate_traits.get("cynicism", 0.3)
        ambition = self.innate_traits.get("ambition", 0.3)

        # Subjective interpretation shift
        valence = 0.0
        interpretation_notes = []

        if "기근" in event_concepts or "전쟁" in event_concepts:
            if altruism > 0.6:
                interpretation_notes.append("타인의 아픔에 깊은 연민을 느끼며 숭고함을 고집함")
                crystallized_concepts = {"이타": 0.4, "숭고": 0.3, "신앙": 0.2}
                valence = 0.2
            elif cynicism > 0.5:
                interpretation_notes.append("냉혹한 현실 속에서 배신과 이기적 생존만이 길이라 확신함")
                crystallized_concepts = {"이기": 0.5, "생존": 0.4, "배신": 0.3}
                valence = -0.6
            elif ambition > 0.6:
                interpretation_notes.append("혼란을 기회로 삼아 거대한 야망과 변혁을 열망함")
                crystallized_concepts = {"혁명": 0.5, "야망": 0.4, "자유": 0.3}
                valence = 0.1
            else:
                interpretation_notes.append("시대적 압력에 순응하며 전통과 질서에 의지하고자 함")
                crystallized_concepts = {"전통": 0.4, "질서": 0.3, "생존": 0.3}
                valence = -0.2
        else:
            crystallized_concepts = event_concepts
            interpretation_notes.append("평범한 삶의 궤적 속에서 의미를 다짐")
            valence = 0.1

        # Learning rate influenced by (1 - stubbornness)
        learning_rate = (1.0 - self.stubbornness) * 0.4

        for conc, w in crystallized_concepts.items():
            self.vector_space.register_concept(conc)
            current_w = self.concept_weights.get(conc, 0.0)
            # Update weight dynamically
            self.concept_weights[conc] = round(current_w + learning_rate * w, 4)

        # Record Memory
        memory = EpisodicMemory(
            event_id=event_id,
            description=description,
            perceived_concepts=crystallized_concepts,
            subjective_interpretation="; ".join(interpretation_notes),
            emotional_valence=valence,
            timestamp=timestamp
        )
        self.episodic_memories.append(memory)
        return memory

    def serialize_soul_snapshot(self) -> Dict[str, Any]:
        """Serializes current dynamic SelfGraph into a static Soul Snapshot."""
        return {
            "npc_id": self.npc_id,
            "name": self.name,
            "archetype_title": self.archetype_title,
            "innate_traits": self.innate_traits,
            "concept_weights": self.concept_weights,
            "stubbornness": self.stubbornness,
            "physical_needs": self.physical_needs,
            "episodic_memories": [mem.to_dict() for mem in self.episodic_memories],
        }

    @classmethod
    def deserialize_soul_snapshot(
        cls,
        snapshot: Dict[str, Any],
        vector_space: ConceptVectorSpace
    ) -> "SelfGraph":
        """Reconstructs a SelfGraph object from a Soul Snapshot dict."""
        self_graph = cls(
            npc_id=snapshot["npc_id"],
            name=snapshot["name"],
            archetype_title=snapshot["archetype_title"],
            vector_space=vector_space,
            innate_traits=snapshot["innate_traits"],
            initial_concepts=snapshot["concept_weights"],
            stubbornness=snapshot["stubbornness"],
        )
        self_graph.physical_needs = snapshot.get("physical_needs", {"hunger": 20.0, "fatigue": 10.0, "safety": 80.0})
        self_graph.episodic_memories = [
            EpisodicMemory.from_dict(m) for m in snapshot.get("episodic_memories", [])
        ]
        return self_graph


class ZeitgeistField:
    """
    Macro environmental/historical pressure field (시대적 장 / Zeitgeist Field).
    Acts as an environmental pressure tensor on all NPC nodes in the world.
    """

    def __init__(
        self,
        field_id: str,
        name: str,
        intensity: float = 0.5,
        concept_signature: Optional[Dict[str, float]] = None
    ):
        self.field_id = field_id
        self.name = name
        self.intensity = max(0.0, min(1.0, intensity))
        self.concept_signature = concept_signature or {"기근": 0.8, "생존": 0.7, "공포": 0.5}

    def get_pressure_vector(self, vector_space: ConceptVectorSpace) -> np.ndarray:
        """Returns weighted composite pressure vector from vector space."""
        return vector_space.compute_composite_vector(
            {k: v * self.intensity for k, v in self.concept_signature.items()}
        )


class DynamicTelosEngine:
    """
    Evaluates NPC action options using internal value landscape resonance and physical drives.
    Eliminates hardcoded discrete if-then rules.
    """

    def __init__(self, vector_space: ConceptVectorSpace):
        self.vector_space = vector_space

    def evaluate_and_choose_action(
        self,
        self_graph: SelfGraph,
        zeitgeist_field: ZeitgeistField,
        action_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates candidate actions by scoring resonance between:
        1. Action's concept alignment & SelfGraph's inner value landscape.
        2. Physical drives (e.g. high hunger boosts survival-oriented or food actions).
        3. Zeitgeist pressure vector alignment.
        """
        value_vec = self_graph.get_value_landscape_vector()
        pressure_vec = zeitgeist_field.get_pressure_vector(self.vector_space)
        hunger = self_graph.physical_needs.get("hunger", 0.0)

        best_action = None
        best_score = -1e9
        scores = []

        for candidate in action_candidates:
            action_id = candidate["action_id"]
            action_concepts = candidate.get("concepts", {})
            action_vec = self.vector_space.compute_composite_vector(action_concepts)

            # Value Landscape Resonance
            value_resonance = float(np.dot(value_vec, action_vec)) if np.linalg.norm(value_vec) > 0 else 0.0

            # Zeitgeist Field Resonance
            pressure_resonance = float(np.dot(pressure_vec, action_vec)) if np.linalg.norm(pressure_vec) > 0 else 0.0

            # Physical Drive Satisfaction
            hunger_relief = candidate.get("hunger_relief", 0.0)
            physical_drive_score = (hunger / 100.0) * hunger_relief

            # Inner Telos Weight
            total_score = (
                value_resonance * 2.0 +
                pressure_resonance * (1.0 + zeitgeist_field.intensity) +
                physical_drive_score * 1.5
            )

            scores.append((candidate, total_score, value_resonance, pressure_resonance))

            if total_score > best_score:
                best_score = total_score
                best_action = candidate

        # Apply action effect to physical needs
        if best_action:
            relief = best_action.get("hunger_relief", 0.0) * 20.0
            self_graph.physical_needs["hunger"] = max(0.0, hunger - relief)

        return {
            "chosen_action": best_action,
            "chosen_score": best_score,
            "all_scores": scores
        }


class GenerativeVoiceFilter:
    """
    Renders NPC decision into personalized narrative voice dialogue and persona tone,
    reflecting SelfGraph concept weights, archetype title, and episodic memories.
    """

    @staticmethod
    def render_dialogue(
        self_graph: SelfGraph,
        action: Dict[str, Any],
        zeitgeist_name: str
    ) -> str:
        top_concepts = sorted(
            self_graph.concept_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_concept_names = [c[0] for c in top_concepts]

        action_title = action.get("title", "행동")
        archetype = self_graph.archetype_title

        # Dialogue tone synthesis based on archetype & concept alignment
        if "용병" in archetype or "거친" in archetype or "생존" in top_concept_names:
            if "희생" in action_title or "기도" in action_title:
                text = f"\"{zeitgeist_name}에도 나 같은 자가 거룩해질 수 있겠소? 비록 칼을 쥐었으나 목숨을 바쳐 남을 지키겠소.\""
            elif "암시장" in action_title or "약탈" in action_title:
                text = f"\"저리 가라! {zeitgeist_name}에 내 목숨 하나 건사하기도 바쁘니까! 살고 싶다면 눈앞에서 사라져라.\""
            else:
                text = f"\"세상에 정답이 어디 있나... 내 손에 들린 칼을 믿을 뿐이다.\""

        elif "학자" in archetype or "늙은" in archetype or "이성" in top_concept_names or "학식" in top_concept_names:
            if "학문" in action_title or "탐구" in action_title:
                text = f"\"세풍이 비록 {zeitgeist_name}으로 흉흉하나, 진리와 서사의 기록은 결코 멈출 수 없소.\""
            elif "기도" in action_title or "희생" in action_title:
                text = f"\"인간의 이성 너머에 존재하는 섭리에 몸을 맡길 수밖에 없구려. 부디 이 시련을 견뎌내기를...\""
            else:
                text = f"\"아쉽게도 시국이 이러하니, 제 학식과 힘으로는 가엾은 이들을 도울 수가 없구려.\""

        elif "농민" in archetype or "혁명" in top_concept_names or "공동체" in top_concept_names:
            if "선동" in action_title or "약탈" in action_title:
                text = f"\"언제까지 {zeitgeist_name} 타령만 들으며 굶어 죽어야 합니까? 영주의 창고를 열어 기근을 극복합시다!\""
            elif "기도" in action_title or "희생" in action_title:
                text = f"\"하늘이시여... 부디 가엾은 우리 아이들에게 빵 한 조각을 내려주소서.\""
            else:
                text = f"\"서로 돕지 않으면 이 시련을 넘을 수 없소. 함께 견뎌냅시다.\""

        else:
            text = f"\"{self_graph.name}: {zeitgeist_name}의 파도 속에서 결단한다. ({action_title})\""

        return text


class MacroEmergenceObserver:
    """
    Aggregates chaotic micro-agency actions from individual NPCs across time,
    synthesizing emergent macro historical wave patterns (Macro Wave) for upper Constellation observers.
    """

    def __init__(self, vector_space: ConceptVectorSpace):
        self.vector_space = vector_space
        self.history_records: List[Dict[str, Any]] = []

    def observe_turn(
        self,
        turn_number: int,
        zeitgeist_field: ZeitgeistField,
        npc_action_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes individual micro choices into emergent macro statistics and vector trends.
        """
        if not npc_action_results:
            return {"turn": turn_number, "macro_summary": "고요한 정적"}

        action_counts: Dict[str, int] = {}
        concept_aggregation: Dict[str, float] = {}

        for res in npc_action_results:
            action = res["chosen_action"]
            act_title = action.get("title", "Unknown")
            action_counts[act_title] = action_counts.get(act_title, 0) + 1

            for conc, w in action.get("concepts", {}).items():
                concept_aggregation[conc] = concept_aggregation.get(conc, 0.0) + w

        # Determine dominant emergent macro wave
        dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]
        dominant_concept = max(concept_aggregation.items(), key=lambda x: x[1])[0] if concept_aggregation else "고요"

        composite_wave_vec = self.vector_space.compute_composite_vector(concept_aggregation)

        # Macro Narrative Wave Text Synthesis
        total_npcs = len(npc_action_results)
        ratio = action_counts[dominant_action] / total_npcs

        if "약탈" in dominant_action or "선동" in dominant_action or "혁명" in dominant_concept:
            emergent_wave_title = f"민중 폭동과 혁명의 파도 (우세 지표: {dominant_action} {ratio*100:.0f}%)"
            macro_meaning = f"상위 관측: {zeitgeist_field.name}의 압력 아래, 아래로부터의 격렬한 분노가 거대한 사회적 반란 트렌드를 수면 위로 쏘아올렸습니다."
        elif "기도" in dominant_action or "희생" in dominant_action or "신앙" in dominant_concept:
            emergent_wave_title = f"숭고한 순교와 신성 집단 귀의 (우세 지표: {dominant_action} {ratio*100:.0f}%)"
            macro_meaning = f"상위 관측: 수많은 무명 개인들의 희생적 기도 파동이 집단적 신념 패턴으로 귀결되어 상계의 성좌들을 감동시킵니다."
        elif "암시장" in dominant_action or "배신" in dominant_action or "생존" in dominant_concept:
            emergent_wave_title = f"지하 기회주의 시장과 인과적 암투 (우세 지표: {dominant_action} {ratio*100:.0f}%)"
            macro_meaning = f"상위 관측: 개별 생존 본능들의 충돌 속에서 변덕스러운 암시장과 지하 거래망이 형성되었습니다."
        else:
            emergent_wave_title = f"시대적 압력에 대한 유기적 다변화 관망 (우세 개념: {dominant_concept})"
            macro_meaning = f"상위 관측: 개개인이 서로 다른 지향점으로 갈라지며 예측 불가능한 카오스적 공존 상태를 유지합니다."

        record = {
            "turn": turn_number,
            "zeitgeist_name": zeitgeist_field.name,
            "total_micro_agents": total_npcs,
            "action_counts": action_counts,
            "dominant_action": dominant_action,
            "dominant_concept": dominant_concept,
            "emergent_wave_title": emergent_wave_title,
            "macro_meaning": macro_meaning,
            "composite_wave_vector": composite_wave_vec.tolist(),
        }
        self.history_records.append(record)
        return record
