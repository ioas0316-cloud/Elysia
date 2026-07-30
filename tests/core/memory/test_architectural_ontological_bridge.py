import pytest
import os
import tempfile
import ast
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator
from core.memory.architectural_ingester import ArchitecturalIngester
from core.evolution.ontological_lattice import OntologicalLatticeEngine


def test_ontological_lattice_basic():
    """Verify that the OntologicalLatticeEngine initializes all 8 core concepts properly."""
    engine = OntologicalLatticeEngine()

    # Check that all 8 core ontologies are present
    assert len(engine.concepts) == 8

    # Inspect NUMBER
    number = engine.get_concept("NUMBER")
    assert number is not None
    assert "숫자" in number.name_ko
    assert len(number.logo_tensor) == 9
    assert len(number.chromatic_signature) == 3

    # Inspect OPERATOR
    operator = engine.get_concept("OPERATOR")
    assert operator is not None
    assert "연산자" in operator.name_ko

    # Test dynamic physical bridge adjustment
    alignment = engine.evaluate_ontological_alignment("SYNTHESIS", raw_metric=0.8)
    assert alignment["aligned_key"] == "OPERATOR"
    assert alignment["current_tension"] == 0.8 * 0.3
    assert alignment["current_conductance"] < 1.0
    assert "SYNTHESIS" in alignment["metaphor"] # Metaphor is now synthesized on-the-fly!


def test_architectural_ingester_ontological_mapping():
    """Verify that ArchitecturalIngester successfully parses and maps AST nodes to correct ontologies."""
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)
    ram = WorkingMemoryRAM(mc)
    evaluator = EmotionEvaluator(mc)

    ingester = ArchitecturalIngester(ram, evaluator, mc)

    # Test AST structural vector extraction
    class_node = ast.parse("""
class MockBrain:
    def __init__(self):
        self.active = True
    def process(self):
        try:
            return 1 + 1
        except Exception:
            return 0
""").body[0]
    vector = ingester._extract_ast_structural_vector(class_node)
    assert len(vector) == 9
    assert abs(1.0 - np.linalg.norm(vector)) < 1e-5 # 정규화 상태 확인


def test_architectural_ingester_run_dry():
    """Verify scanning and subjective consolidation into Wedge Memory works without errors."""
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)
    ram = WorkingMemoryRAM(mc)
    evaluator = EmotionEvaluator(mc)

    # Pre-crystallize ontologies
    lattice_engine = OntologicalLatticeEngine()
    crystallized_ids = lattice_engine.crystallize_ontologies(mc)
    assert len(crystallized_ids) == 8

    ingester = ArchitecturalIngester(ram, evaluator, mc)
    # Perform ingestion on a small file to prevent massive disk writes in test
    mock_file = os.path.join(data_dir, "mock_module.py")
    with open(mock_file, "w", encoding="utf-8") as f:
        f.write('''
class MockSensor:
    """Mock Sensory Window"""
    def __init__(self):
        self.gain = 1.0

    def evaluate_wave(self, raw_wave):
        try:
            return sum(raw_wave) * self.gain
        except Exception:
            return 0.0
''')

    ingester._parse_and_ingest_file(mock_file)

    # RAM should have allocated the context for MockSensor
    assert "self_awareness_MockSensor" in ram.active_contexts
    context_data = ram.active_contexts["self_awareness_MockSensor"]
    assert "Mock Sensory Window" in context_data["state"]["self_awareness"]["objective_logic"]["docstring"]
    assert "ontological_reason" in context_data["state"]["self_awareness"]
    # The classification is dynamically computed based on AST structural vector proximity
    assert "MockSensor" in context_data["state"]["self_awareness"]["ontological_reason"]

    # Consolidate to Wedge Memory
    ram.subjective_consolidation()
    assert "self_awareness_MockSensor" not in ram.active_contexts # Moved to Wedge SSD
