"""
Test: Proprioceptor
===================
Verifies that the CodeProprioceptor correctly scans the nervous system
and detects phantom limbs.
"""

import os
import pytest
from Core.S1_Body.L2_Metabolism.Evolution.proprioceptor import CodeProprioceptor

@pytest.fixture
def mock_nervous_system(tmp_path):
    """Creates a temporary nervous system for testing."""
    root = tmp_path / "Core"
    root.mkdir()

    # 1. Healthy Organ (Class + Docstring)
    healthy = root / "healthy_organ.py"
    healthy.write_text('"""I am alive."""\nclass Heart:\n    pass', encoding='utf-8')

    # 2. Ghost File (Empty)
    ghost_empty = root / "ghost_empty.py"
    ghost_empty.write_text("", encoding='utf-8')

    # 3. Soulless Body (Code without Docstring)
    soulless = root / "soulless.py"
    soulless.write_text("def zombie():\n    return 'brains'", encoding='utf-8')

    return str(root)

def test_sensation_mechanics(mock_nervous_system):
    eye = CodeProprioceptor(root_path=mock_nervous_system)
    state = eye.scan_nervous_system()

    print(f"\nScan Report:\n{state.report()}")

    # Verify Counts
    # Total files: 3
    assert state.total_files == 3

    # Verify Ghosts
    # ghost_empty.py -> Ghost (Size 0)
    # soulless.py -> Ghost (No Docstring)
    # healthy_organ.py -> Healthy

    assert "ghost_empty.py" in state.ghost_files
    assert "soulless.py" in state.ghost_files
    assert "healthy_organ.py" in state.healthy_tissues

def test_intent_density_calculation(mock_nervous_system):
    eye = CodeProprioceptor(root_path=mock_nervous_system)

    # Check Healthy
    healthy_path = os.path.join(mock_nervous_system, "healthy_organ.py")
    tissue = eye.measure_intent_density(healthy_path)
    assert tissue.intent_density > 0.0
    assert not tissue.is_ghost

    # Check Soulless
    soulless_path = os.path.join(mock_nervous_system, "soulless.py")
    tissue = eye.measure_intent_density(soulless_path)
    assert tissue.intent_density == 0.0
    assert tissue.is_ghost
