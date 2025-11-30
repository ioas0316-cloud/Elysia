"""
Fractal Causality Integration
==============================

"원인과 과정과 결과가 무한히 순환되고 있습니다."

This module bridges the Legacy Fractal Causality Engine into Core.
"""

import sys
import os

# Add Legacy path
legacy_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Legacy', 'Language')
sys.path.insert(0, legacy_path)

try:
    from fractal_causality import (
        FractalCausalityEngine,
        FractalCausalNode,
        FractalCausalChain,
        CausalRole
    )
    
    __all__ = [
        'FractalCausalityEngine',
        'FractalCausalNode', 
        'FractalCausalChain',
        'CausalRole'
    ]
    
except ImportError as e:
    import logging
    logger = logging.getLogger("FractalCausality")
    logger.error(f"Failed to import Legacy Fractal Causality: {e}")
    
    # Fallback: define stubs
    class FractalCausalityEngine:
        def __init__(self, name="Elysia"):
            self.name = name
            self.nodes = {}
            
        def create_node(self, description, **kwargs):
            return None
            
        def create_chain(self, cause, process, effect, **kwargs):
            return None
    
    FractalCausalNode = None
    FractalCausalChain = None
    CausalRole = None
