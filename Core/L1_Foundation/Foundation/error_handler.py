"""
                 
Elysia Unified Error Handling System

     ,       ,                  .
"""

import logging
import time
import functools
from typing import Optional, Callable, Any, Tuple, Type
from collections import defaultdict
from datetime import datetime, timezone


class ElysiaErrorHandler:
    """                 """
    
    def __init__(self):
        self.logger = logging.getLogger("Elysia.ErrorHandler")
        self.error_count = defaultdict(int)
        self.circuit_breakers = {}
        self.error_history = []
    
    def with_retry(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Callable:
        """
                        
        
        Args:
            max_retries:          
            backoff_factor:        (      )
            exceptions:            
        
        Returns:
                    
        
        Example:
            @error_handler.with_retry(max_retries=3)
            def fragile_operation():
                # may fail
                pass
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    
                    except exceptions as e:
                        last_exception = e
                        self.error_count[func.__name__] += 1
                        
                        if attempt < max_retries - 1:
                            wait_time = backoff_factor ** attempt
                            self.logger.warning(
                                f"    Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}. "
                                f"Waiting {wait_time:.2f}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(
                                f"  All {max_retries} retries failed for {func.__name__}: {e}"
                            )
                
                #          
                self._record_error(func.__name__, str(last_exception))
                raise last_exception
            
            return wrapper
        return decorator
    
    def circuit_breaker(
        self,
        threshold: int = 5,
        timeout: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Callable:
        """
                     
        
        Args:
            threshold:                 
            timeout:                    ( )
            exceptions:            
        
        Returns:
                    
        
        States:
            - closed:      
            - open:            
            - half_open:      
        
        Example:
            @error_handler.circuit_breaker(threshold=5, timeout=60)
            def external_api_call():
                # may fail frequently
                pass
        """
        def decorator(func: Callable) -> Callable:
            func_name = func.__name__
            
            #            
            if func_name not in self.circuit_breakers:
                self.circuit_breakers[func_name] = {
                    'failures': 0,
                    'last_failure': 0,
                    'state': 'closed',
                    'threshold': threshold,
                    'timeout': timeout
                }
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                breaker = self.circuit_breakers[func_name]
                current_time = time.time()
                
                #             
                if breaker['state'] == 'open':
                    time_since_failure = current_time - breaker['last_failure']
                    
                    if time_since_failure > timeout:
                        # Half-open       
                        breaker['state'] = 'half_open'
                        self.logger.info(
                            f"  Circuit breaker half-open for {func_name}. Testing..."
                        )
                    else:
                        #        
                        remaining = timeout - time_since_failure
                        error_msg = (
                            f"Circuit breaker open for {func_name}. "
                            f"Retry in {remaining:.1f}s"
                        )
                        self.logger.warning(f"  {error_msg}")
                        raise RuntimeError(error_msg)
                
                #         
                try:
                    result = func(*args, **kwargs)
                    
                    #        
                    if breaker['state'] == 'half_open':
                        # Half-open        Closed    
                        breaker['state'] = 'closed'
                        breaker['failures'] = 0
                        self.logger.info(
                            f"  Circuit breaker closed for {func_name}. Recovery complete!"
                        )
                    elif breaker['state'] == 'closed' and breaker['failures'] > 0:
                        #       -       
                        breaker['failures'] = max(0, breaker['failures'] - 1)
                    
                    return result
                
                except exceptions as e:
                    #      
                    breaker['failures'] += 1
                    breaker['last_failure'] = current_time
                    
                    self.logger.error(
                        f"    Function {func_name} failed: {e}. "
                        f"Failures: {breaker['failures']}/{threshold}"
                    )
                    
                    #       
                    if breaker['failures'] >= threshold:
                        breaker['state'] = 'open'
                        self.logger.error(
                            f"  Circuit breaker OPENED for {func_name}. "
                            f"Too many failures ({breaker['failures']})."
                        )
                    
                    self._record_error(func_name, str(e))
                    raise e
            
            return wrapper
        return decorator
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        default: Any = None,
        log_error: bool = True,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
                  (              )
        
        Args:
            func:       
            *args:      
            default:                
            log_error:         
            **kwargs:          
        
        Returns:
            (     ,          )
        
        Example:
            success, result = error_handler.safe_execute(
                risky_function,
                arg1, arg2,
                default="fallback_value"
            )
        """
        try:
            result = func(*args, **kwargs)
            return True, result
        
        except Exception as e:
            if log_error:
                self.logger.error(
                    f"  Safe execute failed for {func.__name__}: {e}"
                )
                self._record_error(func.__name__, str(e))
            
            return False, default
    
    def _record_error(self, function_name: str, error_message: str):
        """          """
        self.error_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'function': function_name,
            'error': error_message
        })
        
        #    1000     
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def get_error_stats(self) -> dict:
        """        """
        return {
            'total_errors': sum(self.error_count.values()),
            'errors_by_function': dict(self.error_count),
            'circuit_breakers': {
                name: {
                    'state': breaker['state'],
                    'failures': breaker['failures']
                }
                for name, breaker in self.circuit_breakers.items()
            },
            'recent_errors': self.error_history[-10:]  #    10 
        }
    
    def reset_circuit_breaker(self, function_name: str):
        """             """
        if function_name in self.circuit_breakers:
            self.circuit_breakers[function_name] = {
                'failures': 0,
                'last_failure': 0,
                'state': 'closed',
                'threshold': self.circuit_breakers[function_name]['threshold'],
                'timeout': self.circuit_breakers[function_name]['timeout']
            }
            self.logger.info(f"  Circuit breaker reset for {function_name}")


#               
error_handler = ElysiaErrorHandler()


# =====       =====

if __name__ == "__main__":
    #      
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("  Testing Elysia Error Handler\n")
    
    #     1:       
    print("=== Test 1: Retry Logic ===")
    
    attempt_count = [0]
    
    @error_handler.with_retry(max_retries=3, backoff_factor=1.5)
    def flaky_function():
        """            """
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise RuntimeError(f"Temporary failure (attempt {attempt_count[0]})")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"  Result: {result}\n")
    except Exception as e:
        print(f"  Failed: {e}\n")
    
    #     2:        
    print("=== Test 2: Circuit Breaker ===")
    
    call_count = [0]
    
    @error_handler.circuit_breaker(threshold=3, timeout=5.0)
    def unstable_api():
        """     API   """
        call_count[0] += 1
        raise RuntimeError(f"API Error (call {call_count[0]})")
    
    #            
    for i in range(5):
        try:
            unstable_api()
        except Exception as e:
            print(f"Call {i+1}: {e}")
    
    print()
    
    #     3:       
    print("=== Test 3: Safe Execute ===")
    
    def risky_function(x):
        if x < 0:
            raise ValueError("Negative value not allowed")
        return x * 2
    
    success, result = error_handler.safe_execute(risky_function, 5)
    print(f"Safe execute (5): success={success}, result={result}")
    
    success, result = error_handler.safe_execute(risky_function, -5, default=0)
    print(f"Safe execute (-5): success={success}, result={result}\n")
    
    #      
    print("=== Error Statistics ===")
    stats = error_handler.get_error_stats()
    print(f"Total errors: {stats['total_errors']}")
    print(f"Errors by function: {stats['errors_by_function']}")
    print(f"Circuit breakers: {stats['circuit_breakers']}")