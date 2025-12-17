"""
REDIRECT: self_reflector.py has moved to Core/Foundation/Autonomy/
This stub provides backward compatibility.
"""
from Core.Foundation.Autonomy.self_reflector import *

import warnings
warnings.warn(
    "Core.Foundation.self_reflector is deprecated. Use Core.Foundation.Autonomy.self_reflector instead.",
    DeprecationWarning,
    stacklevel=2
)
