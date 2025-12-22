import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the path so we can import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.fractal_kernel import FractalKernel

@pytest.fixture
def kernel():
    return FractalKernel()

def test_process_recursion_depth_and_time(kernel):
    # Mock generate_text and get_current_time
    with patch('Project_Sophia.fractal_kernel.generate_text') as mock_generate, \
         patch('Project_Sophia.fractal_kernel.get_current_time') as mock_time:
        
        mock_generate.return_value = "Deepened thought"
        mock_time.return_value = "2025-12-01T12:00:00"
        
        # Call with max_depth=3
        result = kernel.process("Initial thought", depth=1, max_depth=3)
        
        # Should be called 3 times
        assert mock_generate.call_count == 3
        
        # Verify calls have correct perspectives
        # Call 1 (Depth 1): Present
        args1, _ = mock_generate.call_args_list[0]
        assert "Current Depth: 1" in args1[0]
        assert "Perspective: Present" in args1[0]
        assert "2025-12-01T12:00:00" in args1[0]
        
        # Call 2 (Depth 2): Past
        args2, _ = mock_generate.call_args_list[1]
        assert "Current Depth: 2" in args2[0]
        assert "Perspective: Past" in args2[0]
        
        # Call 3 (Depth 3): Future
        args3, _ = mock_generate.call_args_list[2]
        assert "Current Depth: 3" in args3[0]
        assert "Perspective: Future" in args3[0]

def test_resonate_error_handling(kernel):
    with patch('Project_Sophia.fractal_kernel.generate_text') as mock_generate, \
         patch('Project_Sophia.fractal_kernel.get_current_time') as mock_time:
        
        mock_generate.side_effect = Exception("API Error")
        mock_time.return_value = "2025-12-01T12:00:00"
        
        # Should return the original signal on error
        result = kernel._resonate("Original", depth=1)
        assert result == "Original"
