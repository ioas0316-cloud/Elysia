"""
Tests for Gap 2-3 modules.
Tests CausalInterventionEngine, MultiModalPerceptionEngine.
"""

import pytest
import sys
import os
import importlib.util

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def load_module_directly(module_name, file_path):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly
reasoning_path = os.path.join(project_root, 'Core', 'Reasoning')
perception_path = os.path.join(project_root, 'Core', 'Perception')

causal_module = load_module_directly('causal', os.path.join(reasoning_path, 'causal_intervention.py'))
perception_module = load_module_directly('perception', os.path.join(perception_path, 'multi_modal.py'))

CausalInterventionEngine = causal_module.CausalInterventionEngine
CausalGraph = causal_module.CausalGraph
CausalNode = causal_module.CausalNode
CausalEdge = causal_module.CausalEdge
CausalRelationType = causal_module.CausalRelationType
CounterfactualQuery = causal_module.CounterfactualQuery

MultiModalPerceptionEngine = perception_module.MultiModalPerceptionEngine
ModalityType = perception_module.ModalityType
PerceptualInput = perception_module.PerceptualInput


# ============================================================
# Gap 2: Causal Intervention Tests
# ============================================================

class TestCausalInterventionEngine:
    """Tests for CausalInterventionEngine class (Gap 2)."""
    
    def test_creation(self):
        """Test engine creation."""
        engine = CausalInterventionEngine()
        assert engine is not None
        assert engine.epistemology is not None
    
    def test_explain_meaning(self):
        """Test epistemology explanation."""
        engine = CausalInterventionEngine()
        explanation = engine.explain_meaning()
        
        assert "point" in explanation
        assert "line" in explanation
        assert "인과" in explanation
    
    def test_create_graph(self):
        """Test graph creation."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test_graph")
        
        assert graph is not None
        assert graph.name == "test_graph"
        assert "test_graph" in engine.causal_graphs
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        node = CausalNode("x", "Variable X", value=0.5)
        graph.add_node(node)
        
        assert "x" in graph.nodes
        assert graph.nodes["x"].value == 0.5
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        graph.add_node(CausalNode("x", "X"))
        graph.add_node(CausalNode("y", "Y"))
        graph.add_edge(CausalEdge("x", "y", CausalRelationType.CAUSES))
        
        assert len(graph.edges) == 1
        assert "x" in graph.parents["y"]
        assert "y" in graph.children["x"]
    
    def test_do_intervention(self):
        """Test do-intervention."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        # X → Y
        graph.add_node(CausalNode("x", "X", value=0.3))
        graph.add_node(CausalNode("y", "Y", value=0.4))
        graph.add_edge(CausalEdge("x", "y", CausalRelationType.CAUSES, strength=1.0))
        
        result = engine.do_intervention(graph, "x", 1.0, "y")
        
        assert result is not None
        assert result.intervention_variable == "x"
        assert result.target_variable == "y"
        assert len(result.explanation) > 0
    
    def test_causal_chain(self):
        """Test causal chain: X → Y → Z."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("chain")
        
        graph.add_node(CausalNode("x", "X", value=0.2))
        graph.add_node(CausalNode("y", "Y", value=0.3))
        graph.add_node(CausalNode("z", "Z", value=0.4))
        
        graph.add_edge(CausalEdge("x", "y", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("y", "z", CausalRelationType.CAUSES))
        
        # X가 Z의 조상인지 확인
        ancestors = graph.get_ancestors("z")
        assert "x" in ancestors
        assert "y" in ancestors
    
    def test_get_descendants(self):
        """Test getting descendants."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        graph.add_node(CausalNode("a", "A"))
        graph.add_node(CausalNode("b", "B"))
        graph.add_node(CausalNode("c", "C"))
        
        graph.add_edge(CausalEdge("a", "b", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("b", "c", CausalRelationType.CAUSES))
        
        descendants = graph.get_descendants("a")
        assert "b" in descendants
        assert "c" in descendants
    
    def test_counterfactual_query(self):
        """Test counterfactual query."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        graph.add_node(CausalNode("rain", "Rain", value=0.0))
        graph.add_node(CausalNode("wet", "Wet", value=0.2))
        graph.add_edge(CausalEdge("rain", "wet", CausalRelationType.CAUSES))
        
        query = CounterfactualQuery(
            premise="rain=1.0",
            conclusion="wet=?",
            actual_x=0.0,
            counterfactual_x=1.0
        )
        
        result = engine.counterfactual_query(graph, query)
        assert result.result is not None
    
    def test_causal_path(self):
        """Test finding causal paths."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        graph.add_node(CausalNode("a", "A"))
        graph.add_node(CausalNode("b", "B"))
        graph.add_node(CausalNode("c", "C"))
        
        graph.add_edge(CausalEdge("a", "b", CausalRelationType.CAUSES))
        graph.add_edge(CausalEdge("b", "c", CausalRelationType.CAUSES))
        
        paths = engine.get_causal_path(graph, "a", "c")
        assert len(paths) > 0
        assert paths[0] == ["a", "b", "c"]
    
    def test_multi_scale_plan(self):
        """Test multi-scale planning."""
        engine = CausalInterventionEngine()
        graph = engine.create_graph("test")
        
        graph.add_node(CausalNode("action1", "Action1", value=0.0))
        graph.add_node(CausalNode("action2", "Action2", value=0.0))
        graph.add_node(CausalNode("goal", "Goal", value=0.2))
        
        graph.add_edge(CausalEdge("action1", "goal", CausalRelationType.CAUSES, strength=0.5))
        graph.add_edge(CausalEdge("action2", "goal", CausalRelationType.CAUSES, strength=0.3))
        
        plan = engine.multi_scale_plan(
            graph,
            "goal",
            1.0,
            ["action1", "action2"]
        )
        
        assert isinstance(plan, list)


class TestCausalNode:
    """Tests for CausalNode class."""
    
    def test_creation(self):
        """Test node creation."""
        node = CausalNode("test", "Test Node", value=0.5)
        
        assert node.id == "test"
        assert node.name == "Test Node"
        assert node.value == 0.5
    
    def test_epistemology(self):
        """Test node has epistemology."""
        node = CausalNode("test", "Test")
        
        assert "point" in node.epistemology
        assert "line" in node.epistemology
    
    def test_explain_meaning(self):
        """Test meaning explanation."""
        node = CausalNode("test", "Test")
        explanation = node.explain_meaning()
        
        assert "Test" in explanation


# ============================================================
# Gap 3: Multi-Modal Perception Tests
# ============================================================

class TestMultiModalPerceptionEngine:
    """Tests for MultiModalPerceptionEngine class (Gap 3)."""
    
    def test_creation(self):
        """Test engine creation."""
        engine = MultiModalPerceptionEngine()
        
        assert engine is not None
        assert engine.epistemology is not None
    
    def test_explain_meaning(self):
        """Test epistemology explanation."""
        engine = MultiModalPerceptionEngine()
        explanation = engine.explain_meaning()
        
        assert "point" in explanation
        assert "다중" in explanation or "모달" in explanation
    
    def test_default_processors(self):
        """Test default processors are registered."""
        engine = MultiModalPerceptionEngine()
        
        assert ModalityType.TEXT in engine.processors
        assert ModalityType.VISION in engine.processors
        assert ModalityType.AUDIO in engine.processors
        assert ModalityType.ACTION in engine.processors
    
    def test_perceive_text(self):
        """Test text perception."""
        engine = MultiModalPerceptionEngine()
        
        perception = engine.perceive(
            ModalityType.TEXT,
            "Hello, world!",
            confidence=0.9
        )
        
        assert perception is not None
        assert perception.modality == ModalityType.TEXT
        assert perception.confidence == 0.9
    
    def test_perceive_action(self):
        """Test action perception."""
        engine = MultiModalPerceptionEngine()
        
        perception = engine.perceive(
            ModalityType.ACTION,
            {"type": "move", "target": "forward"},
            confidence=0.85
        )
        
        assert perception.modality == ModalityType.ACTION
    
    def test_perceive_buffer(self):
        """Test perception buffer."""
        engine = MultiModalPerceptionEngine()
        
        engine.perceive(ModalityType.TEXT, "First")
        engine.perceive(ModalityType.TEXT, "Second")
        engine.perceive(ModalityType.TEXT, "Third")
        
        recent = engine.get_recent_perceptions(limit=2)
        assert len(recent) == 2
    
    def test_integrate(self):
        """Test multi-modal integration."""
        engine = MultiModalPerceptionEngine()
        
        text = engine.perceive(ModalityType.TEXT, "Hello")
        action = engine.perceive(ModalityType.ACTION, {"type": "speak"})
        
        representation = engine.integrate([text, action])
        
        assert representation is not None
        assert len(representation.modalities) == 2
        assert len(representation.unified_embedding) == 64
        assert representation.salience > 0
    
    def test_salience_calculation(self):
        """Test salience increases with modality diversity."""
        engine = MultiModalPerceptionEngine()
        
        text_only = engine.perceive(ModalityType.TEXT, "Test")
        single_rep = engine.integrate([text_only])
        
        action = engine.perceive(ModalityType.ACTION, {"type": "think"})
        multi_rep = engine.integrate([text_only, action])
        
        # More modalities should increase salience
        assert multi_rep.salience >= single_rep.salience
    
    def test_cross_modal_attention(self):
        """Test cross-modal attention."""
        engine = MultiModalPerceptionEngine()
        
        action = engine.perceive(ModalityType.ACTION, {"type": "speak"})
        
        attention = engine.cross_modal_attention(
            ModalityType.TEXT,
            "What are you doing?",
            [action]
        )
        
        assert isinstance(attention, dict)
    
    def test_enable_disable_modality(self):
        """Test enabling/disabling modalities."""
        engine = MultiModalPerceptionEngine()
        
        engine.disable_modality(ModalityType.VISION)
        assert not engine.processors[ModalityType.VISION].is_enabled
        
        engine.enable_modality(ModalityType.VISION)
        assert engine.processors[ModalityType.VISION].is_enabled
    
    def test_get_enabled_modalities(self):
        """Test getting enabled modalities."""
        engine = MultiModalPerceptionEngine()
        
        enabled = engine.get_enabled_modalities()
        assert ModalityType.TEXT in enabled
        
        engine.disable_modality(ModalityType.VISION)
        enabled = engine.get_enabled_modalities()
        assert ModalityType.VISION not in enabled
    
    def test_korean_text(self):
        """Test Korean text perception."""
        engine = MultiModalPerceptionEngine()
        
        perception = engine.perceive(
            ModalityType.TEXT,
            "안녕하세요, 엘리시아입니다!",
            confidence=0.95
        )
        
        representation = engine.integrate([perception])
        
        # Should detect Korean
        assert "text" in representation.interpretations
        assert representation.interpretations["text"]["language"] == "ko"


class TestPerceptualInput:
    """Tests for PerceptualInput class."""
    
    def test_creation(self):
        """Test input creation."""
        inp = PerceptualInput(
            modality=ModalityType.TEXT,
            data="Test data"
        )
        
        assert inp.modality == ModalityType.TEXT
        assert inp.data == "Test data"
        assert inp.confidence == 1.0
    
    def test_epistemology(self):
        """Test input has epistemology."""
        inp = PerceptualInput(
            modality=ModalityType.TEXT,
            data="Test"
        )
        
        assert "point" in inp.epistemology
        assert "line" in inp.epistemology
        assert "space" in inp.epistemology
        assert "god" in inp.epistemology


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
