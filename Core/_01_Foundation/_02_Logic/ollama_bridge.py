"""
REDIRECT: ollama_bridge.py has moved to Core/Foundation/Network/
This stub provides backward compatibility.
"""
import warnings
warnings.warn(
    "Core._01_Foundation.ollama_bridge is deprecated. Use Core._01_Foundation.Network.ollama_bridge instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from Core._01_Foundation._02_Logic.Network.ollama_bridge import *

# Explicit imports for commonly used symbols
from Core._01_Foundation._02_Logic.Network.ollama_bridge import (
    OllamaBridge,
    ollama,
    get_ollama_bridge,
)

