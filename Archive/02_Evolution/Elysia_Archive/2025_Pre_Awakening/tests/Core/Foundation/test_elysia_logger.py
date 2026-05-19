"""
Tests for Elysia Logger
"""

import pytest
import logging
from pathlib import Path
from Core.FoundationLayer.Foundation.elysia_logger import ElysiaLogger


class TestElysiaLogger:
    """Test suite for Elysia logger"""
    
    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = ElysiaLogger("TestLogger", log_dir="logs/test")
        assert logger is not None
        assert logger.name == "TestLogger"
        assert logger.log_dir == Path("logs/test")
    
    def test_basic_logging(self):
        """Test basic logging methods"""
        logger = ElysiaLogger("TestBasic", log_dir="logs/test")
        
        # Should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    
    def test_contextual_logging(self):
        """Test logging with context"""
        logger = ElysiaLogger("TestContext", log_dir="logs/test")
        
        context = {'user': 'test_user', 'action': 'test_action'}
        logger.info("Test with context", context=context)
        
        # Should not raise exceptions
        assert True
    
    def test_thought_logging(self):
        """Test Elysia-specific thought logging"""
        logger = ElysiaLogger("TestThought", log_dir="logs/test")
        
        logger.log_thought(
            layer="2D",
            content="Processing concept",
            context={'emotion': 'calm'}
        )
        
        # Should not raise exceptions
        assert True
    
    def test_resonance_logging(self):
        """Test resonance logging"""
        logger = ElysiaLogger("TestResonance", log_dir="logs/test")
        
        logger.log_resonance(
            source="Love",
            target="Hope",
            score=0.847
        )
        
        assert True
    
    def test_evolution_logging(self):
        """Test evolution metric logging"""
        logger = ElysiaLogger("TestEvolution", log_dir="logs/test")
        
        logger.log_evolution(
            component="ResonanceField",
            metric="coherence",
            value=0.923
        )
        
        assert True
    
    def test_performance_logging(self):
        """Test performance logging"""
        logger = ElysiaLogger("TestPerformance", log_dir="logs/test")
        
        logger.log_performance(
            operation="test_operation",
            duration_ms=45.3
        )
        
        assert True
    
    def test_spirit_logging(self):
        """Test spirit activity logging"""
        logger = ElysiaLogger("TestSpirit", log_dir="logs/test")
        
        logger.log_spirit(
            spirit_name="Fire",
            frequency=450.0,
            amplitude=0.8
        )
        
        assert True
    
    def test_memory_logging(self):
        """Test memory operation logging"""
        logger = ElysiaLogger("TestMemory", log_dir="logs/test")
        
        logger.log_memory(
            operation="bloom",
            seed_name="concept_love",
            compression_ratio=1000.0
        )
        
        assert True
    
    def test_system_logging(self):
        """Test system event logging"""
        logger = ElysiaLogger("TestSystem", log_dir="logs/test")
        
        logger.log_system(
            event="startup",
            status="complete"
        )
        
        logger.log_system(
            event="error_detected",
            status="error"
        )
        
        assert True
    
    def test_exception_logging(self):
        """Test exception logging with traceback"""
        logger = ElysiaLogger("TestException", log_dir="logs/test")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.error("Exception occurred", exc_info=True)
        
        assert True
    
    def test_log_directory_creation(self):
        """Test log directory is created automatically"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            log_path = Path(temp_dir) / "test_logs"
            logger = ElysiaLogger("TestDir", log_dir=str(log_path))
            
            assert log_path.exists()
            assert log_path.is_dir()
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
