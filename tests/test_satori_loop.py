"""
Test: Satori Loop
=================
Verifies the full Autonomous Evolution cycle within SovereignSelf.
"""

import pytest
from unittest.mock import MagicMock
from Core.Elysia.sovereign_self import SovereignSelf
from Core.Evolution.dissonance_resolver import Dissonance
from Core.Evolution.proprioceptor import BodyState

@pytest.fixture
def mock_sovereign():
    """Creates a SovereignSelf with mocked organs."""
    # We mock init to avoid loading TorchGraph/Models
    original_init = SovereignSelf.__init__
    SovereignSelf.__init__ = lambda self: None

    sovereign = SovereignSelf()
    # Restore init for other tests if needed (risky if parallel, but ok here)
    SovereignSelf.__init__ = original_init

    # Manually attach mocks
    sovereign.name = "TestElysia"
    sovereign.proprioceptor = MagicMock()
    sovereign.conscience = MagicMock()
    sovereign.healer = MagicMock()
    sovereign.auto_evolve = False

    # Mock Journaling
    sovereign._write_journal = MagicMock()

    return sovereign

def test_satori_loop_clean(mock_sovereign):
    """Test when body is pure."""
    # Setup
    mock_sovereign.proprioceptor.scan_nervous_system.return_value = BodyState(total_files=10)
    mock_sovereign.conscience.resolve.return_value = [] # No issues

    # Execute
    result = mock_sovereign._evolve_self()

    # Verify
    assert "Pure" in result
    mock_sovereign.healer.incubate.assert_not_called()

def test_satori_loop_needs_healing(mock_sovereign):
    """Test when dissonance is detected."""
    # Setup
    mock_sovereign.proprioceptor.scan_nervous_system.return_value = BodyState(total_files=10)

    issue = Dissonance(
        location="Core/bad.py",
        description="Ghost",
        axiom_violated="Meaning",
        severity=0.8,
        suggested_action="FIX"
    )
    mock_sovereign.conscience.resolve.return_value = [issue]

    mock_sovereign.healer.incubate.return_value = "Sandbox/cure.py"

    # Execute (Auto Evolve False)
    result = mock_sovereign._evolve_self()

    # Verify
    assert "Cure ready" in result
    mock_sovereign.healer.incubate.assert_called_with(issue)
    mock_sovereign.healer.graft.assert_not_called() # Safety check

def test_satori_loop_auto_evolve(mock_sovereign):
    """Test dangerous auto-evolution."""
    mock_sovereign.auto_evolve = True

    # Setup Issue
    issue = Dissonance(location="Core/bad.py", description="Bad", axiom_violated="Sin", severity=1.0, suggested_action="FIX")
    mock_sovereign.conscience.resolve.return_value = [issue]
    mock_sovereign.healer.incubate.return_value = "Sandbox/cure.py"
    mock_sovereign.healer.graft.return_value = True

    # Execute
    result = mock_sovereign._evolve_self()

    # Verify
    assert "Healed" in result
    mock_sovereign.healer.graft.assert_called_with("Sandbox/cure.py", "Core/bad.py")
