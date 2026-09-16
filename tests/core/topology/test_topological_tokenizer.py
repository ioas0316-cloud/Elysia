import pytest
from core.topology.topological_tokenizer import TopologicalTokenizer, NodeType, CausalGraph

def test_tokenize_hangul_syllable():
    tokenizer = TopologicalTokenizer()
    graph = tokenizer.tokenize_text("한")

    assert len(graph.nodes) == 4
    init_nodes = [n for n in graph.nodes.values() if "INVARIANT_JAMO_INIT:ㅎ" in n.invariant_signature]
    med_nodes = [n for n in graph.nodes.values() if "INVARIANT_JAMO_MED:ㅏ" in n.invariant_signature]
    fin_nodes = [n for n in graph.nodes.values() if "INVARIANT_JAMO_FIN:ㄴ" in n.invariant_signature]
    syllable_nodes = [n for n in graph.nodes.values() if "INVARIANT_SYLLABLE:한" in n.invariant_signature]

    assert len(init_nodes) == 1
    assert len(med_nodes) == 1
    assert len(fin_nodes) == 1
    assert len(syllable_nodes) == 1

    syllable_node = syllable_nodes[0]
    ctx = graph.context_constraints[syllable_node.node_id]
    assert "unicode_codepoint:U+D55C" in ctx
    assert "utf8_bytes:0xed,0x95,0x9c" in ctx

def test_tokenize_single_jamo_branch():
    tokenizer = TopologicalTokenizer()
    graph = tokenizer.tokenize_text("ㅋㅋㅋ")

    branch_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.BRANCH]
    assert len(branch_nodes) == 3
    for node in branch_nodes:
        assert node.invariant_signature == "INVARIANT_SINGLE_JAMO:ㅋ"
        assert "rule_violation:uncombined_jamo_branch" in graph.context_constraints[node.node_id]

def test_jamo_mutation():
    tokenizer = TopologicalTokenizer()
    graph = tokenizer.tokenize_text("한")

    init_node_id = [n.node_id for n in graph.nodes.values() if "INVARIANT_JAMO_INIT" in n.invariant_signature][0]

    # 초성 다이얼 변위: 'ㅎ' -> 'ㄴ' (한 -> 난)
    success = tokenizer.mutate_jamo_node(graph, init_node_id, 'ㄴ')
    assert success is True

    nan_nodes = [n for n in graph.nodes.values() if "INVARIANT_SYLLABLE:난" in n.invariant_signature]
    assert len(nan_nodes) == 1

    nan_node = nan_nodes[0]
    ctx = graph.context_constraints[nan_node.node_id]
    assert f"unicode_codepoint:U+{ord('난'):04X}" in ctx

def test_spatial_index():
    tokenizer = TopologicalTokenizer()
    graph = tokenizer.tokenize_text("한글")

    index = tokenizer.build_spatial_index(graph)
    assert len(index) == 2
    # '한' (ㅎ: 18, ㅏ: 0, ㄴ: 4)
    # INITIAL_JAMO: 'ㅎ' is index 18
    # MEDIAL_JAMO: 'ㅏ' is index 0
    # FINAL_JAMO: 'ㄴ' is index 4
    assert (18, 0, 4) in index
