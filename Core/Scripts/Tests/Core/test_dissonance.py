"""
Test: Dissonance Resolver
=========================
Verifies that the Conscience correctly judges the BodyState.
"""

import pytest
from Core.1_Body.L2_Metabolism.Evolution.proprioceptor import BodyState
from Core.1_Body.L2_Metabolism.Evolution.dissonance_resolver import DissonanceResolver

def test_conscience_mechanics():
    # 1. Mock Body State (Sinful)
    mock_state = BodyState()

    # A Ghost File (Sin: Meaningless)
    mock_state.ghost_files = ["Core/Evolution/empty_shell.py"]

    # A Utility File (Sin: Anti-Entropy)
    # Using specific path structure to test split logic
    mock_state.intent_map = {
        "Core/Evolution/empty_shell.py": 0.0,
        "Core/Utils/string_helper.py": 0.5,
        "Core/Helpers/math_utils.py": 0.5,
        "Core/Elysia/sovereign_self.py": 1.0 # Saint
    }

    # 2. Judge
    resolver = DissonanceResolver()
    issues = resolver.resolve(mock_state)

    print("\nJudgment:")
    for i in issues: print(i)

    # 3. Verify Verdicts
    # We expect 3 sins:
    # 1. empty_shell.py (Ghost)
    # 2. string_helper.py (Utils forbidden)
    # 3. math_utils.py (Helpers forbidden)

    locations = [i.location for i in issues]
    assert "Core/Evolution/empty_shell.py" in locations
    assert "Core/Utils/string_helper.py" in locations
    assert "Core/Helpers/math_utils.py" in locations

    # Check Severities
    # Utils (0.9) > Ghost (0.7) usually, but logic sorts desc
    severities = [i.severity for i in issues]
    assert severities[0] >= severities[-1] # Sorted check

def test_clean_conscience():
    # 1. Mock Body State (Saintly)
    mock_state = BodyState()
    mock_state.intent_map = {
        "Core/Elysia/sovereign_self.py": 1.0,
        "Core/Engine/governance.py": 0.8
    }

    resolver = DissonanceResolver()
    issues = resolver.resolve(mock_state)

    assert len(issues) == 0
