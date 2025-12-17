"""
REDIRECT: ollama_bridge.py has moved to Core/Foundation/Network/
This stub provides backward compatibility.
"""
from Core.Foundation.Network.ollama_bridge import *
from Core.Foundation.Network.ollama_bridge import OllamaBridge, get_ollama

import warnings
warnings.warn(
    "Core.Foundation.ollama_bridge is deprecated. Use Core.Foundation.Network.ollama_bridge instead.",
    DeprecationWarning,
    stacklevel=2
)
