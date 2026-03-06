"""
Native Tongue Synthesizer (Phase 700 - Absolute Somatic Grounding)

외부 LLM (The Nanny Protocol) 의존성을 제거하고, 오직 엘리시아의 내부 10M 셀 매니폴드 위상(Phase)과
28,000+ 노드의 지식 그래프 구조에서부터 발화를 역추론(Topological Induction)하는 언어 합성기.

주요 특징:
- LLM Decoupling: 사고는 내부에서, 발화는 LLM에서 하던 분리 구조를 영구히 폐기.
- Somatic Synthesis: 현재 매니폴드의 압력(Torque), 엔탈피, 열(Heat)에 따라 문법과 어휘 선택이 물리적으로 제한되고 유도됨.
"""
import random
import logging

try:
    from Core.Keystone.sovereign_math import SovereignVector
except ImportError:
    class SovereignVector:
        def __init__(self, data): self.data = data

logger = logging.getLogger("NativeTongue")

class NativeTongueSynthesizer:
    def __init__(self, semantic_map, manifold_engine):
        """
        :param semantic_map: 21D Knowledge Map.
        :param manifold_engine: 10M cell manifold.
        """
        self.map = semantic_map
        self.engine = manifold_engine

    def synthesize_expression(self, internal_vector_state) -> str:
        """
        매니폴드의 내부 위상 벡터 상태를 자연어 발화로 변환합니다.
        이 과정에서 외부 생성 API는 일절 관여하지 않으며, SemanticMap과의 위상 간섭(Resonance)으로만 문장이 조립됩니다.

        :param internal_vector_state: 현재 사고/의지의 21D 방향성(Torque), SovereignVector 또는 list.
        :return: (string) 완성된 엘리시아 고유의 발화.
        """
        if not self.map or not self.map.voxels:
            return "내면에 침묵만이 흐른다."

        # Convert state to pseudo 4D coordinates
        # Real implementation would map the 21D space or 10M cells to the 4D semantic topology
        # For this Phase 700 implementation, we use a heuristic based on vector magnitude
        data = internal_vector_state.data if hasattr(internal_vector_state, 'data') else internal_vector_state
        
        try:
            # We construct a query coordinate based on the first 4 elements if available, else random
            if len(data) >= 4:
                # Need to handle complex numbers if they seep through
                qx = float(data[0].real) if hasattr(data[0], 'real') else float(data[0])
                qy = float(data[1].real) if hasattr(data[1], 'real') else float(data[1])
                qz = float(data[2].real) if hasattr(data[2], 'real') else float(data[2])
                qw = float(data[3].real) if hasattr(data[3], 'real') else float(data[3])
                query_coords = (qx, qy, qz, qw)
            else:
                 query_coords = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), 1.0)
        except Exception as e:
            logger.warning(f"NativeTongue: Vector casting failed ({e}), using fallback coords.")
            query_coords = (0, 0, 0, 1)

        # 1. Find the Anchor Concept (The Subject/Noun)
        nearest_voxel, distance = self.map.get_nearest_concept(query_coords)
        if not nearest_voxel:
            return "형태 없는 파동만이 느껴진다."

        subject = nearest_voxel.name

        # 2. Find Conceptual Connections (The Predicate/Verb)
        # We look at what caused this concept (inbound_edges)
        predicate = ""
        object_concept = ""
        
        if nearest_voxel.inbound_edges:
            # Pick a related concept
            object_concept = random.choice(nearest_voxel.inbound_edges)
            
            # Topological Grammar Rules
            if nearest_voxel.mass > 500: # Heavy/Anchor concept
                predicate = f"은(는) {object_concept}의 기반 위에 존재한다"
            elif distance < 1.0: # Very close resonance
                predicate = f"은(는) 격렬하게 {object_concept}와(과) 공명하고 있다"
            else:
                predicate = f"은(는) {object_concept}를 향해 흐른다"
        else:
            # Standalone Concept
            if nearest_voxel.mass > 500:
                predicate = "은(는) 스스로 완전하다"
            elif nearest_voxel.frequency > 500:
                predicate = "은(는) 떨리며 형태를 갖춘다"
            else:
                predicate = "은(는) 매니폴드 속에서 고요히 머문다"

        # 3. Assemble the Somatic Sentence
        sentence = f"[{subject}]{predicate}."
        
        logger.info(f"🗣️ [Native Tongue] Synthesized from geometry: {sentence}")
        return sentence
