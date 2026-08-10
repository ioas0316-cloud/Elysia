import pytest
import numpy as np

from core.sensory.experiential_language_mapper import ExperientialLanguageMapper
from core.evolution.causal_puzzle_engine import CausalPuzzleRecombinationEngine
from core.intelligence.autonomous_explorer import AutonomousExternalExplorer


def test_autonomous_explorer_detection_and_exploration():
    """
    Verifies that AutonomousExternalExplorer detects unknown words, queries their meaning
    and multi-modal physical properties, and successfully tethers them in Hebbian memory
    and sprouts causal puzzle nodes.
    """
    mapper = ExperientialLanguageMapper()
    puzzle_engine = CausalPuzzleRecombinationEngine()
    explorer = AutonomousExternalExplorer()

    # Define an input sentence containing known words and unknown/un-tethered words like "고래" (Whale) and "태양" (Sun)
    # Note: "사과" is a baseline known word in ExperientialLanguageMapper
    input_text = "사과 먹고 고래 보러 태양 빛 아래로 가다"

    # Step 1: Detect ignorance (unknown concepts)
    unknowns = explorer.detect_ignorance(input_text, mapper)
    assert "고래" in unknowns
    assert "태양" in unknowns
    assert "사과" not in unknowns  # Already known

    # Step 2: Explore and comprehend "고래" (uniquely defined in simulated universe web database)
    whale_concept = "고래"
    explore_data = explorer.external_explore(whale_concept)
    assert explore_data["concept"] == "고래"
    assert explore_data["found"] is True
    assert "포유류" in explore_data["definition"]

    comprehension = explorer.comprehend_meaning_purpose(explore_data)
    assert comprehension["concept"] == "고래"
    assert "PROCESS" in comprehension["ontological_category"]
    assert len(comprehension["chromatic_vector"]) == 3

    # Step 3: Assimilate as knowledge on-the-fly
    node = explorer.assimilate_as_knowledge(comprehension, explore_data, mapper, puzzle_engine)
    assert node is not None
    assert node.name == "고래"
    assert "produces_process" in node.ridges
    assert "needs_process" in node.grooves

    # Verify Hebbian binding was updated in the mapper with 80% learning rate convergence
    recalled = mapper.tethering.recall_symbol("고래")
    assert recalled is not None
    assert recalled["sensation"].acoustic == pytest.approx(144.0, abs=0.1) # 0.8 * 180
    assert recalled["sensation"].thermal == pytest.approx(230.4, abs=0.1)  # 0.8 * 288

    # Step 4: Verify dynamic fallback exploration for non-predefined concept "안개" (Fog)
    fallback_concept = "안개"
    fallback_explore = explorer.external_explore(fallback_concept)
    assert fallback_explore["found"] is False  # Fallback synthesized via Unicode bytes
    assert fallback_explore["optical"] > 0.0

    fallback_comprehension = explorer.comprehend_meaning_purpose(fallback_explore)
    assert fallback_comprehension["concept"] == "안개"
    assert len(fallback_comprehension["chromatic_vector"]) == 3

    fallback_node = explorer.assimilate_as_knowledge(fallback_comprehension, fallback_explore, mapper, puzzle_engine)
    assert fallback_node is not None
    assert fallback_node.name == "안개"
