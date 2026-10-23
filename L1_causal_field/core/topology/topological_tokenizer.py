from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional

class NodeType(Enum):
    STEM = "STEM"      # 근원적 불변 인과 노드
    BRANCH = "BRANCH"  # 환경/매질/비정규 종속 노드

@dataclass(frozen=True)
class CausalNode:
    node_id: str
    node_type: NodeType
    invariant_signature: str
    payload: Dict

@dataclass(frozen=True)
class CausalEdge:
    source_id: str
    target_id: str
    precondition: str
    is_necessary: bool

@dataclass
class CausalGraph:
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    context_constraints: Dict[str, Set[str]] = field(default_factory=dict)

# 한글 유니코드 자모 상수
INITIAL_JAMO = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
MEDIAL_JAMO = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
FINAL_JAMO = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

class TopologicalTokenizer:
    """
    통계적 빈도(BPE)를 완전 배제하고, 8비트 바이트 및 한글 자모음의
    원자적 인과 결합 규칙에 따라 CausalGraph(V, E, C) 구조로 직조하는 위상 토크나이저.
    """
    def __init__(self):
        self._node_idx = 0

    def _generate_node_id(self, prefix: str) -> str:
        self._node_idx += 1
        return f"{prefix}_{self._node_idx}"

    def tokenize_text(self, text: str) -> CausalGraph:
        """
        입력 텍스트를 한글 자모/바이트 원자 노드 및 필연적 결합 엣지로 해부하여 CausalGraph로 변환합니다.
        """
        graph = CausalGraph()
        last_syllable_node_id: Optional[str] = None

        for char in text:
            # 1. 완결된 한글 음절 (U+AC00 ~ U+D7A3)
            if '가' <= char <= '힣':
                syllable_node_id = self._parse_hangul_syllable(char, graph)
            # 2. 단독 자음/모음 (Branch)
            elif self._is_single_jamo(char):
                syllable_node_id = self._parse_single_jamo(char, graph)
            # 3. 일반 알파벳, 숫자, 특수문자, 공백 등
            else:
                syllable_node_id = self._parse_generic_char(char, graph)

            # 음절 간 순차적 언어 전이 엣지 연결 (seq_flow)
            if last_syllable_node_id:
                graph.edges.append(CausalEdge(
                    source_id=last_syllable_node_id,
                    target_id=syllable_node_id,
                    precondition="seq_language_flow",
                    is_necessary=True
                ))
            last_syllable_node_id = syllable_node_id

        return graph

    def _parse_hangul_syllable(self, char: str, graph: CausalGraph) -> str:
        code = ord(char) - 0xAC00
        initial_idx = code // (21 * 28)
        medial_idx = (code % (21 * 28)) // 28
        final_idx = code % 28

        initial_char = INITIAL_JAMO[initial_idx]
        medial_char = MEDIAL_JAMO[medial_idx]
        final_char = FINAL_JAMO[final_idx]

        # 1. 원자 노드 생성 (초성, 중성)
        init_node_id = self._generate_node_id("N_JAMO_INIT")
        med_node_id = self._generate_node_id("N_JAMO_MED")

        graph.nodes[init_node_id] = CausalNode(
            node_id=init_node_id,
            node_type=NodeType.STEM,
            invariant_signature=f"INVARIANT_JAMO_INIT:{initial_char}",
            payload={"jamo": initial_char, "role": "initial"}
        )
        graph.nodes[med_node_id] = CausalNode(
            node_id=med_node_id,
            node_type=NodeType.STEM,
            invariant_signature=f"INVARIANT_JAMO_MED:{medial_char}",
            payload={"jamo": medial_char, "role": "medial"}
        )

        # 초성 -> 중성 필연 결합 엣지
        graph.edges.append(CausalEdge(
            source_id=init_node_id,
            target_id=med_node_id,
            precondition="COMBINE_INITIAL_MEDIAL",
            is_necessary=True
        ))

        last_jamo_id = med_node_id

        # 종성이 존재하는 경우
        if final_idx > 0:
            fin_node_id = self._generate_node_id("N_JAMO_FIN")
            graph.nodes[fin_node_id] = CausalNode(
                node_id=fin_node_id,
                node_type=NodeType.STEM,
                invariant_signature=f"INVARIANT_JAMO_FIN:{final_char}",
                payload={"jamo": final_char, "role": "final"}
            )
            graph.edges.append(CausalEdge(
                source_id=med_node_id,
                target_id=fin_node_id,
                precondition="COMBINE_MEDIAL_FINAL",
                is_necessary=True
            ))
            last_jamo_id = fin_node_id

        # 2. 상위 음절 합성 노드 (Stem)
        syllable_node_id = self._generate_node_id("N_SYLLABLE")
        syllable_node = CausalNode(
            node_id=syllable_node_id,
            node_type=NodeType.STEM,
            invariant_signature=f"INVARIANT_SYLLABLE:{char}",
            payload={"char": char, "unicode": f"U+{ord(char):04X}"}
        )
        graph.nodes[syllable_node_id] = syllable_node

        # 바이트 레벨 표현을 Context 제약으로 등록 (UTF-8 8bit Bytes)
        utf8_bytes = [hex(b) for b in char.encode("utf-8")]
        graph.context_constraints[syllable_node_id] = {
            f"utf8_bytes:{','.join(utf8_bytes)}",
            f"unicode_codepoint:U+{ord(char):04X}"
        }

        # 자모원자 -> 음절 결합 인과 엣지
        graph.edges.append(CausalEdge(
            source_id=last_jamo_id,
            target_id=syllable_node_id,
            precondition="SYNTHESIZE_SYLLABLE",
            is_necessary=True
        ))

        return syllable_node_id

    def _is_single_jamo(self, char: str) -> bool:
        return ('ㄱ' <= char <= 'ㅎ') or ('ㅏ' <= char <= 'ㅣ')

    def _parse_single_jamo(self, char: str, graph: CausalGraph) -> str:
        # 단독 자모(예: 'ㅋ', 'ㅠ')는 미완성 결합이므로 BRANCH 노드로 격리
        node_id = self._generate_node_id("N_SINGLE_JAMO")
        node = CausalNode(
            node_id=node_id,
            node_type=NodeType.BRANCH,
            invariant_signature=f"INVARIANT_SINGLE_JAMO:{char}",
            payload={"char": char, "status": "uncombined_jamo"}
        )
        graph.nodes[node_id] = node
        graph.context_constraints[node_id] = {"rule_violation:uncombined_jamo_branch"}
        return node_id

    def _parse_generic_char(self, char: str, graph: CausalGraph) -> str:
        node_id = self._generate_node_id("N_CHAR")
        node = CausalNode(
            node_id=node_id,
            node_type=NodeType.STEM if (char.isalnum() or char in (" ", "\n")) else NodeType.BRANCH,
            invariant_signature=f"INVARIANT_CHAR:{repr(char)}",
            payload={"char": char}
        )
        graph.nodes[node_id] = node
        utf8_bytes = [hex(b) for b in char.encode("utf-8")]
        graph.context_constraints[node_id] = {f"utf8_bytes:{','.join(utf8_bytes)}"}
        return node_id

    def mutate_jamo_node(self, graph: CausalGraph, jamo_node_id: str, new_jamo: str) -> bool:
        """
        원자 단위 다이얼 변위(Atomic Mutation):
        특정 자모 노드의 값을 변경하고 연결된 완결 음절 노드(N_SYLLABLE) 및 Context 제약을 동적으로 재합성합니다.
        """
        if jamo_node_id not in graph.nodes:
            return False

        old_node = graph.nodes[jamo_node_id]
        role = old_node.payload.get("role")
        if not role:
            return False

        # 1. 새 Jamo CausalNode 업데이트
        sig_prefix = {
            "initial": "INVARIANT_JAMO_INIT",
            "medial": "INVARIANT_JAMO_MED",
            "final": "INVARIANT_JAMO_FIN"
        }[role]

        graph.nodes[jamo_node_id] = CausalNode(
            node_id=jamo_node_id,
            node_type=NodeType.STEM,
            invariant_signature=f"{sig_prefix}:{new_jamo}",
            payload={"jamo": new_jamo, "role": role}
        )

        # 2. 연결된 초, 중, 종성 노드 및 target 음절 노드 탐색
        init_node_id, med_node_id, fin_node_id = None, None, None

        if role == "initial":
            init_node_id = jamo_node_id
            for edge in graph.edges:
                if edge.source_id == init_node_id and edge.precondition == "COMBINE_INITIAL_MEDIAL":
                    med_node_id = edge.target_id
                    break
            if med_node_id:
                for edge in graph.edges:
                    if edge.source_id == med_node_id and edge.precondition == "COMBINE_MEDIAL_FINAL":
                        fin_node_id = edge.target_id
                        break
        elif role == "medial":
            med_node_id = jamo_node_id
            for edge in graph.edges:
                if edge.target_id == med_node_id and edge.precondition == "COMBINE_INITIAL_MEDIAL":
                    init_node_id = edge.source_id
                    break
            for edge in graph.edges:
                if edge.source_id == med_node_id and edge.precondition == "COMBINE_MEDIAL_FINAL":
                    fin_node_id = edge.target_id
                    break
        elif role == "final":
            fin_node_id = jamo_node_id
            for edge in graph.edges:
                if edge.target_id == fin_node_id and edge.precondition == "COMBINE_MEDIAL_FINAL":
                    med_node_id = edge.source_id
                    break
            if med_node_id:
                for edge in graph.edges:
                    if edge.target_id == med_node_id and edge.precondition == "COMBINE_INITIAL_MEDIAL":
                        init_node_id = edge.source_id
                        break

        # SYNTHESIZE_SYLLABLE 엣지의 source는 fin_node_id (존재할 시) 혹은 med_node_id
        last_jamo_id = fin_node_id if fin_node_id else med_node_id
        syllable_node_id = None
        if last_jamo_id:
            for edge in graph.edges:
                if edge.source_id == last_jamo_id and edge.precondition == "SYNTHESIZE_SYLLABLE":
                    syllable_node_id = edge.target_id
                    break

        if syllable_node_id and syllable_node_id in graph.nodes:
            init_char = graph.nodes[init_node_id].payload["jamo"] if init_node_id and init_node_id in graph.nodes else None
            med_char = graph.nodes[med_node_id].payload["jamo"] if med_node_id and med_node_id in graph.nodes else None
            fin_char = graph.nodes[fin_node_id].payload["jamo"] if fin_node_id and fin_node_id in graph.nodes else ""

            if init_char in INITIAL_JAMO and med_char in MEDIAL_JAMO:
                init_idx = INITIAL_JAMO.index(init_char)
                med_idx = MEDIAL_JAMO.index(med_char)
                fin_idx = FINAL_JAMO.index(fin_char) if fin_char in FINAL_JAMO else 0

                code = 0xAC00 + (init_idx * 21 * 28) + (med_idx * 28) + fin_idx
                new_char = chr(code)

                graph.nodes[syllable_node_id] = CausalNode(
                    node_id=syllable_node_id,
                    node_type=NodeType.STEM,
                    invariant_signature=f"INVARIANT_SYLLABLE:{new_char}",
                    payload={"char": new_char, "unicode": f"U+{ord(new_char):04X}"}
                )
                utf8_bytes = [hex(b) for b in new_char.encode("utf-8")]
                graph.context_constraints[syllable_node_id] = {
                    f"utf8_bytes:{','.join(utf8_bytes)}",
                    f"unicode_codepoint:U+{ord(new_char):04X}"
                }
        return True

    def build_spatial_index(self, graph: CausalGraph) -> Dict[Tuple[int, int, int], str]:
        """
        [초성 x 중성 x 종성] 3차원 직교 좌표계 위상 색인을 생성합니다.
        """
        index = {}
        for node in graph.nodes.values():
            if node.invariant_signature.startswith("INVARIANT_SYLLABLE:"):
                char = node.payload.get("char")
                if char and '가' <= char <= '힣':
                    code = ord(char) - 0xAC00
                    initial_idx = code // (21 * 28)
                    medial_idx = (code % (21 * 28)) // 28
                    final_idx = code % 28
                    index[(initial_idx, medial_idx, final_idx)] = node.node_id
        return index
