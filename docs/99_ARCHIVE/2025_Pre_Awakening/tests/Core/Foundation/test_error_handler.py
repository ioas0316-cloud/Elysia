"""
Tests for Elysia Error Handler
"""

import pytest
import time
from Core.FoundationLayer.Foundation.error_handler import ElysiaErrorHandler, error_handler


class TestErrorHandler:
    """Test suite for error handler functionality"""
    
    def test_initialization(self):
        """Test error handler initialization"""
        handler = ElysiaErrorHandler()
        assert handler is not None
        assert handler.error_count == {}
        assert handler.circuit_breakers == {}
    
    def test_retry_success_after_failures(self):
        """Test retry logic succeeds after temporary failures"""
        handler = ElysiaErrorHandler()
        attempt_count = [0]
        
        @handler.with_retry(max_retries=3, backoff_factor=0.1)
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise RuntimeError(f"Failure {attempt_count[0]}")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count[0] == 3
    
    def test_retry_exhaustion(self):
        """Test retry logic fails after max attempts"""
        handler = ElysiaErrorHandler()
        
        @handler.with_retry(max_retries=2, backoff_factor=0.1)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold"""
        handler = ElysiaErrorHandler()
        
        @handler.circuit_breaker(threshold=3, timeout=1.0)
        def failing_service():
            raise ConnectionError("Service unavailable")
        
        # Trigger circuit breaker
        for i in range(3):
            with pytest.raises(ConnectionError):
                failing_service()
        
        # Circuit should be open now
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            failing_service()
    
    def test_circuit_breaker_half_open(self):
        """Test circuit breaker enters half-open state after timeout"""
        handler = ElysiaErrorHandler()
        
        @handler.circuit_breaker(threshold=2, timeout=0.5)
        def intermittent_service():
            raise TimeoutError("Timeout")
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(TimeoutError):
                intermittent_service()
        
        # Wait for timeout
        time.sleep(0.6)
        
        # Should transition to half-open (one more attempt allowed)
        with pytest.raises(TimeoutError):
            intermittent_service()
    
    def test_safe_execute_success(self):
        """Test safe execute with successful function"""
        handler = ElysiaErrorHandler()
        
        def successful_function(x):
            return x * 2
        
        success, result = handler.safe_execute(successful_function, 5)
        assert success is True
        assert result == 10
    
    def test_safe_execute_failure(self):
        """Test safe execute with failing function"""
        handler = ElysiaErrorHandler()
        
        def failing_function():
            raise ValueError("Failed")
        
        success, result = handler.safe_execute(
            failing_function,
            default="fallback"
        )
        assert success is False
        assert result == "fallback"
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        handler = ElysiaErrorHandler()
        
        @handler.with_retry(max_retries=2, backoff_factor=0.1)
        def error_prone():
            raise RuntimeError("Error")
        
        # Generate some errors
        for _ in range(3):
            try:
                error_prone()
            except RuntimeError:
                pass
        
        stats = handler.get_error_stats()
        assert stats['total_errors'] > 0
        assert 'error_prone' in stats['errors_by_function']
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset"""
        handler = ElysiaErrorHandler()
        
        @handler.circuit_breaker(threshold=2, timeout=10.0)
        def service():
            raise ConnectionError()
        
        # Open circuit
        for _ in range(2):
            try:
                service()
            except ConnectionError:
                pass
        
        # Reset circuit
        handler.reset_circuit_breaker('service')
        
        # Should be able to call again (once before failing)
        with pytest.raises(ConnectionError):
            service()
    
    def test_global_error_handler(self):
        """Test global error handler instance"""
        @error_handler.with_retry(max_retries=2)
        def test_function():
            return "ok"
        
        result = test_function()
        assert result == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
