"""
Test JAX Cortex
===============
Tests for Core.1_Body.L5_Mental.Reasoning_Core.Brain.jax_cortex
"""

import os
import sys
import pytest

# Ensure KERAS_BACKEND is set before any imports
os.environ['KERAS_BACKEND'] = 'jax'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Core.1_Body.L5_Mental.Reasoning_Core.Brain.jax_cortex import JAXCortex


class TestJAXCortexInit:
    """Test JAXCortex initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        cortex = JAXCortex()
        assert cortex.model_name == "gpt2_base"
        assert cortex.preset == "gpt2_base_en"
        assert cortex.is_loaded == False
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        cortex = JAXCortex(model_name="gpt2_medium")
        assert cortex.preset == "gpt2_medium_en"
    
    def test_repr(self):
        """Test string representation."""
        cortex = JAXCortex()
        assert "gpt2_base_en" in repr(cortex)
        assert "not loaded" in repr(cortex)


class TestJAXCortexLoad:
    """Test model loading - these tests may be slow."""
    
    @pytest.mark.slow
    def test_load_gpt2_base(self):
        """Test loading GPT-2 base model."""
        cortex = JAXCortex(model_name="gpt2_base")
        result = cortex.load()
        assert result == True
        assert cortex.is_loaded == True
    
    @pytest.mark.slow
    def test_generate_simple(self):
        """Test simple text generation."""
        cortex = JAXCortex(model_name="gpt2_base", auto_load=True)
        result = cortex.generate("Hello, world!", max_length=20)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Hello" in result  # Should include the prompt


class TestJAXCortexEmbed:
    """Test embedding functionality."""
    
    def test_embed_placeholder(self):
        """Test placeholder embedding returns correct shape."""
        cortex = JAXCortex()
        result = cortex.embed("test text")
        assert result is not None
        assert result.shape == (7,)  # 7D Qualia vector


# Quick sanity test that doesn't require model loading
def test_import():
    """Test that the module can be imported."""
    from Core.1_Body.L5_Mental.Reasoning_Core.Brain import JAXCortex
    assert JAXCortex is not None


if __name__ == "__main__":
    # Run quick tests only (no model loading)
    pytest.main([__file__, "-v", "-m", "not slow"])
