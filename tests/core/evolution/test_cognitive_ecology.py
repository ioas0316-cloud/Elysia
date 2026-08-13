import pytest
import numpy as np
from core.evolution.cognitive_ecology import (
    EcologyAgent,
    MetaDisagreementProcessor,
    DisagreementPreservingMemoryNode,
    CognitiveEcologyEngine
)
from core.memory.causal_controller import CausalMemoryController

def test_ecology_agent_projection():
    # Test Causalist projection
    causalist = EcologyAgent(
        key="Causalist",
        name="Causalist Agent",
        chromatic_signature=np.array([0.9, 0.1, 0.0]),
        projection_focus="temporal_differential"
    )
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    projected = causalist.project(x)
    assert len(projected) == 5
    # Since temporal_differential scales even elements by 1.2 and odd elements by 0.2
    assert projected[0] == pytest.approx(0.6)
    assert projected[1] == pytest.approx(0.1)

def test_ecology_agent_belief_structure():
    skeptic = EcologyAgent(
        key="Skeptic",
        name="Skeptic Agent",
        chromatic_signature=np.array([0.2, 0.1, 0.7]),
        projection_focus="skeptic_outliers"
    )
    mat = skeptic.form_belief_structure("Information", length=5)
    assert mat.shape == (5, 5)
    # Ensure it's deterministic based on concept_key and key bytes
    mat_same = skeptic.form_belief_structure("Information", length=5)
    assert np.allclose(mat, mat_same)

def test_meta_disagreement_processor():
    processor = MetaDisagreementProcessor()

    # Form dummy agent belief graphs
    g1 = np.eye(5, dtype=np.float32)
    g2 = np.eye(5, dtype=np.float32) * 2.0

    belief_graphs = {
        "Causalist": g1,
        "Structuralist": g2
    }

    gaps = processor.compute_differential_gaps(belief_graphs)
    assert len(gaps) == 1
    assert gaps[("Causalist", "Structuralist")] == pytest.approx(np.sqrt(5.0))

    # Test meta reflection
    agents = {
        "Causalist": EcologyAgent("Causalist", "Causalist", np.array([1, 0, 0]), "temporal_differential"),
        "Structuralist": EcologyAgent("Structuralist", "Structuralist", np.array([0, 1, 0]), "topo_laplacian")
    }
    reflection = processor.process_meta_reflection("Information", gaps, belief_graphs, agents)
    assert reflection["active"] is True
    assert "왜 Causalist 모델과 Structuralist 모델이 서로 다르게 충돌하는가?" in reflection["meta_question"]
    assert reflection["tension_pair"] == ("Causalist", "Structuralist")
    assert reflection["candidate_principle_matrix"].shape == (5, 5)

def test_disagreement_preserving_memory_node():
    node = DisagreementPreservingMemoryNode("Information")
    gaps = {
        ("Causalist", "Structuralist"): 1.5,
        ("Causalist", "Skeptic"): 2.5,
        ("Structuralist", "Skeptic"): 3.0
    }
    node.record_contradictions(gaps)
    assert node.unresolved_contradiction_matrix is not None
    assert node.unresolved_contradiction_matrix.shape == (3, 3)
    # Total charge should be sum of gaps (symmetric matrix sum / 2.0)
    assert node.total_contradiction_charge == pytest.approx(1.5 + 2.5 + 3.0)

def test_cognitive_ecology_engine_cycle():
    # Instantiate engine with local memory controller (or dummy memory)
    mc = CausalMemoryController(data_dir="data")
    engine = CognitiveEcologyEngine(memory_controller=mc)

    raw_wave = b"Tesla Causal Vector Information Flow"
    concept = "Information"

    # We can pass simulated reality vector to falsification check
    simulated_reality = {
        "reality_vector": np.array([0.9, 0.8, 0.1, 0.5, 0.2], dtype=np.float32)
    }

    report = engine.process_ecology_breath(
        concept_key=concept,
        raw_wave=raw_wave,
        simulated_reality=simulated_reality
    )

    assert report["concept"] == concept
    assert report["unresolved_gaps_count"] > 0
    assert report["total_contradiction_charge"] > 0.0
    assert report["best_explaining_agent"] in engine.agents
    assert len(report["falsification_errors"]) == len(engine.agents)

    # Check that best explaining agent's resistance has decreased (conductance increased)
    best_agent_key = report["best_explaining_agent"]
    best_agent = engine.agents[best_agent_key]
    assert best_agent.resistance < 0.5
