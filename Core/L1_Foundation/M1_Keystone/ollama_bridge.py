"""
REDIRECT: ollama_bridge.py has moved to Core/Foundation/Network/
This stub provides backward compatibility.
"""
import warnings
warnings.warn(
    "Core.L1_Foundation.M1_Keystone.ollama_bridge is deprecated. Use Core.L1_Foundation.M1_Keystone.Network.ollama_bridge instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from Core.L1_Foundation.M1_Keystone.Network.ollama_bridge import *

# Explicit imports for commonly used symbols
from Core.L1_Foundation.M1_Keystone.Network.ollama_bridge import (
    OllamaBridge,
    ollama,
    get_ollama_bridge,
)
