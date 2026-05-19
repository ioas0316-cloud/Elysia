import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the path so we can import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.planning_cortex import PlanningCortex

@pytest.fixture
def cortex():
    return PlanningCortex()

def test_develop_plan_structure(cortex):
    # Mock the generate_text function to return a valid JSON string
    with patch('Project_Sophia.planning_cortex.generate_text') as mock_generate:
        mock_generate.return_value = '[{"tool_name": "get_current_time", "parameters": {}}]'
        
        plan = cortex.develop_plan("Check time")
        
        assert isinstance(plan, list)
        assert len(plan) == 1
        assert plan[0]['tool_name'] == "get_current_time"

def test_develop_plan_json_error(cortex):
    # Mock generate_text to return invalid JSON
    with patch('Project_Sophia.planning_cortex.generate_text') as mock_generate:
        mock_generate.return_value = "This is not JSON"
        
        plan = cortex.develop_plan("Do something")
        
        assert isinstance(plan, list)
        assert len(plan) == 0

def test_decompose_goal_alias(cortex):
    # Verify the alias works
    with patch('Project_Sophia.planning_cortex.generate_text') as mock_generate:
        mock_generate.return_value = '[]'
        plan = cortex._decompose_goal("Test")
        assert isinstance(plan, list)
