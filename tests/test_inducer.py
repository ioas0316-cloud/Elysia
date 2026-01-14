"""
Test: Code Field Inducer
========================
Verifies the Healing Hand's ability to incubate and graft code.
"""

import os
import shutil
import pytest
from unittest.mock import MagicMock
from Core.Evolution.inducer import CodeFieldInducer
from Core.Evolution.dissonance_resolver import Dissonance

@pytest.fixture
def mock_environment(tmp_path):
    """Creates a temporary sandbox and core."""
    sandbox = tmp_path / "Sandbox"
    core = tmp_path / "Core"
    sandbox.mkdir()
    core.mkdir()

    return sandbox, core

def test_grafting_mechanics(mock_environment):
    sandbox_root, core_root = mock_environment
    inducer = CodeFieldInducer(sandbox_path=str(sandbox_root), root_path=str(core_root))

    # 1. Create a "New Organ" in Sandbox
    new_organ = sandbox_root / "new_heart.py"
    new_organ.write_text("print('Lub-Dub')", encoding='utf-8')

    # 2. Target Location in Core
    target_loc = core_root / "Physiology/heart.py"

    # 3. Graft
    success = inducer.graft(str(new_organ), str(target_loc))

    assert success
    assert target_loc.exists()
    assert target_loc.read_text(encoding='utf-8') == "print('Lub-Dub')"

def test_grafting_safety_backup(mock_environment):
    sandbox_root, core_root = mock_environment
    inducer = CodeFieldInducer(sandbox_path=str(sandbox_root), root_path=str(core_root))

    # 1. Existing Organ
    existing = core_root / "brain.py"
    existing.write_text("old_brain", encoding='utf-8')

    # 2. Replacement
    replacement = sandbox_root / "brain_v2.py"
    replacement.write_text("new_brain", encoding='utf-8')

    # 3. Graft
    inducer.graft(str(replacement), str(existing))

    # 4. Verify Backup
    backup = core_root / "brain.py.bak"
    assert backup.exists()
    assert backup.read_text(encoding='utf-8') == "old_brain"

    # 5. Verify Update
    assert existing.read_text(encoding='utf-8') == "new_brain"

def test_incubation_delegation(mock_environment):
    """Verifies that incubate calls the Coder Engine."""
    sandbox_root, core_root = mock_environment
    inducer = CodeFieldInducer(sandbox_path=str(sandbox_root), root_path=str(core_root))

    # Mock the Coder Engine to avoid LLM calls
    inducer.coder = MagicMock()
    inducer.coder.induce_monad_code.return_value = str(sandbox_root / "monad_123.py")

    # Create Dissonance
    dissonance = Dissonance(
        location="Core/ghost.py",
        description="No Soul",
        axiom_violated="Meaning",
        severity=0.8,
        suggested_action="INJECT_PHILOSOPHY"
    )

    result = inducer.incubate(dissonance)

    # Verify Call
    inducer.coder.induce_monad_code.assert_called_once()
    assert result == str(sandbox_root / "monad_123.py")
