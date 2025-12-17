"""
REDIRECT: ollama_bridge.py has moved to Core/Foundation/Network/
This stub provides backward compatibility.
"""
from Core.Foundation.Network.ollama_bridge import *
from Core.Foundation.Network.ollama_bridge import OllamaBridge

# Alias for common usage
try:
    from Core.Foundation.Network.ollama_bridge import ollama as get_ollama
except ImportError:
    get_ollama = None

import warnings
warnings.warn(
    "Core.Foundation.ollama_bridge is deprecated. Use Core.Foundation.Network.ollama_bridge instead.",
    DeprecationWarning,
    stacklevel=2
)
