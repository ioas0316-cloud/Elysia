"""
Unit tests for Causal DSL Compiler, Localized Rollback Engine, and Superposition Engine.
"""

import pytest
from core.engine.causal_dsl import CausalCompiler, CausalDSLParser, CausalSemanticAnalyzer
from core.engine.localized_rollback import LocalizedRollbackEngine, CausalNode, SignalCartridge
from core.engine.superposition_engine import SuperpositionEngine, ObserverRay, CausalVulkanCudaBridge

def test_causal_dsl_compilation():
    dsl_code = """
    signal GunshotSignal : id(0x01), size(16B) {
        uint16 source_id;
        uint16 intensity;
        half3  origin_pos;
    }

    node CrowdNPC {
        dormant {
            bounding_radius : 15.0m;
            entropy_factor  : High;
        }
        manifested {
            Matrix3x4 transform;
            uint16    anim_frame;
        }
    }

    rule OnGunshotAudible {
        trigger : GunshotSignal s;
        target  : CrowdNPC npc;
        when    : distance(s.origin_pos, npc.centroid) <= s.intensity;
        collapse {
            npc.state_bitmask |= 1;
        }
    }
    """
    compiled = CausalCompiler.compile(dsl_code)

    # Verify struct headers generated correctly
    assert "struct alignas(16) GunshotSignal" in compiled["header"]
    assert "struct alignas(32) CrowdNPC_Dormant" in compiled["header"]
    assert "struct alignas(64) CrowdNPC_Manifested" in compiled["header"]

    # Verify manifested struct contains members
    assert "float transform[12];" in compiled["header"]
    assert "uint16_t anim_frame;" in compiled["header"]

    # Verify CUDA kernel contains collapse statements
    assert "__global__ void OnGunshotAudible_kernel" in compiled["cuda"]
    assert "npc.state_bitmask |= 1;" in compiled["cuda"]

def test_causal_dsl_alignment_error():
    dsl_code = """
    signal BadSignal : id(0x01), size(4B) {
        Matrix4x4 heavy_data;
    }
    """
    parser = CausalDSLParser(dsl_code)
    prog = parser.parse()
    analyzer = CausalSemanticAnalyzer(prog)
    errors = analyzer.analyze()
    assert len(errors) > 0
    assert "declared size 4B" in errors[0]

def test_circular_causality_detection():
    dsl_code = """
    rule RuleA {
        trigger : SignalA a;
        target  : SignalB b;
        when    : true;
        collapse { b.state |= 1; }
    }
    rule RuleB {
        trigger : SignalB b;
        target  : SignalA a;
        when    : true;
        collapse { a.state |= 1; }
    }
    """
    parser = CausalDSLParser(dsl_code)
    prog = parser.parse()
    analyzer = CausalSemanticAnalyzer(prog)
    errors = analyzer.analyze()
    assert any("Circular causality" in err for err in errors)

def test_localized_rollback_engine():
    engine = LocalizedRollbackEngine()
    for i in range(10):
        engine.add_node(CausalNode(node_id=i))

    # Graph topology:
    # 0 -> 1 -> 2 -> 3
    # 0 -> 4
    # 5 -> 6 (Independent subtree)
    engine.add_edge(0, 1)
    engine.add_edge(1, 2)
    engine.add_edge(2, 3)
    engine.add_edge(0, 4)
    engine.add_edge(5, 6)

    missed_sig = SignalCartridge(signal_id=101, frame=100, target_node_id=0, payload={"value": 10})
    resimulated = engine.resimulate_dirty_subtrees(missed_sig, current_frame=105)

    assert set(resimulated) == {0, 1, 2, 3, 4}
    assert 5 not in resimulated and 6 not in resimulated

    # Topological order assertion: parent index must appear before child index
    pos_map = {node_id: idx for idx, node_id in enumerate(resimulated)}
    assert pos_map[0] < pos_map[1]
    assert pos_map[1] < pos_map[2]
    assert pos_map[2] < pos_map[3]
    assert pos_map[0] < pos_map[4]

    # Verify state updated on target node 0
    node0 = engine.nodes[0]
    assert node0.last_updated_frame == 105
    assert node0.active_signal_id == 101

def test_superposition_engine_lazy_collapse():
    engine = SuperpositionEngine(count=100)
    ray = ObserverRay(origin=[0.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0], max_distance=50.0)

    collapsed_ids = engine.collapse_superposition_nodes(observer_rays=[ray])
    assert len(collapsed_ids) > 0

    # Unobserved nodes remain in dormant state (is_collapsed == 0)
    uncollapsed_ids = [n.node_id for n in engine.superposition_nodes if n.is_collapsed == 0]
    assert len(uncollapsed_ids) > 0
    assert len(collapsed_ids) + len(uncollapsed_ids) == 100

    # Vulkan TLAS Bridge Verification
    bridge = CausalVulkanCudaBridge(instance_count=100)
    updated_instances = bridge.update_tlas_instances_zero_copy(engine.collapsed_nodes)
    assert updated_instances == len(collapsed_ids)
