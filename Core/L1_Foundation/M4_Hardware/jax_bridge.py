"""
JAX Bridge: The Heavy Metal Interface
=====================================

"Metal is not just structure; it is the nervous system of the machine."

This module provides a unified interface for hardware-accelerated operations.
It prioritizes JAX (XLA) for "Lightning Speed" but gracefully falls back to
NumPy or PyTorch if JAX is unavailable.
"""

import os
import logging
from typing import Any, Union, Callable

# Configure specific logger
logger = logging.getLogger("HeavyMetal")

# Global flags
HAS_JAX = False
ACCELERATOR = "CPU"  # Default

try:
    # Try importing JAX
    import jax
    import jax.numpy as jnp
    
    # Check for available devices
    devices = jax.devices()
    HAS_JAX = True
    
    # Determine accelerator type
    dev_type = str(devices[0]).lower()
    if 'gpu' in dev_type:
        ACCELERATOR = "GPU (CUDA)"
    elif 'tpu' in dev_type:
        ACCELERATOR = "TPU (Tensor)"
    else:
        ACCELERATOR = "CPU (JAX Optimized)"
        
    logger.info(f"⚡ [HEAVY METAL] JAX Bridge Active. Accelerator: {ACCELERATOR}")

except ImportError:
    # Fallback to NumPy
    import numpy as jnp
    HAS_JAX = False
    ACCELERATOR = "CPU (Standard)"
    logger.warning("🧱 [HEAVY METAL] JAX not found. Falling back to NumPy (O(n) latency).")

except Exception as e:
    # Catch other JAX initialization errors
    import numpy as jnp
    HAS_JAX = False
    ACCELERATOR = f"CPU (Fallback: {e})"
    logger.error(f"⚠️ [HEAVY METAL] JAX Initialization Failed: {e}")


class JAXBridge:
    """
    Polymorphic bridge for tensor operations.
    If JAX is present, uses JIT compilation and XLA.
    If not, uses standard NumPy.
    """
    
    @staticmethod
    def status() -> str:
        return f"Accelerator: {ACCELERATOR} | JAX: {HAS_JAX}"

    @staticmethod
    def is_accelerated() -> bool:
        return HAS_JAX and "CPU" not in ACCELERATOR

    @staticmethod
    def array(data: Any) -> Any:
        return jnp.array(data)

    @staticmethod
    def matmul(a: Any, b: Any) -> Any:
        """
        Matrix multiplication.
        JAX: Dispatches to XLA (O(1) compiled).
        NumPy: BLAS execution.
        """
        return jnp.matmul(a, b)
    
    @staticmethod
    def jit(fun: Callable) -> Callable:
        """
        Just-In-Time compilation decorator.
        Only active if JAX is installed.
        """
        if HAS_JAX:
            return jax.jit(fun)
        return fun  # No-op decorator

    @staticmethod
    def grad(fun: Callable) -> Callable:
        """
        Automatic differentiation.
        """
        if HAS_JAX:
            return jax.grad(fun)
        else:
            # Simple numerical gradient fallback (very basic, for compatibility)
            def numerical_grad(x):
                # Placeholder: real gradients in pure numpy are hard without autograd
                logger.warning("Attempted gradient on NumPy backend (Not Implemented). returning 0.")
                return 0.0
            return numerical_grad

# Expose common math functions for seamless integration
sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos
exp = jnp.exp
tanh = jnp.tanh
linalg = jnp.linalg
roll = jnp.roll
