"""
REDIRECT: ollama_bridge.py has moved to Core/Foundation/Network/
This stub provides backward compatibility.
"""
import warnings
warnings.warn(
    "Core.1_Body.L1_Foundation.Foundation.ollama_bridge is deprecated. Use Core.1_Body.L1_Foundation.Foundation.Network.ollama_bridge instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from Core.1_Body.L1_Foundation.Foundation.Network.ollama_bridge import *

# Explicit imports for commonly used symbols
from Core.1_Body.L1_Foundation.Foundation.Network.ollama_bridge import (
    OllamaBridge,
    ollama,
    get_ollama_bridge,
)
