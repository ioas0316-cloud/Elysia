"""
Internal Universe Facade (Redirect Module)
==========================================

                              .

     : Core/Memory/Vector/internal_universe.py

35                 InternalUniverse        ,
                           .

   :
    #     (            )
    from Core.System.internal_universe import InternalUniverse
    
    #    (   import)
    from Core.Cognition.Memory_Linguistics.Memory.Vector.internal_universe import InternalUniverse
"""

import warnings

#       (         )
warnings.warn(
    "Core.System.internal_universe is deprecated. "
    "Use Core.System.Vector.internal_universe instead.",
    DeprecationWarning,
    stacklevel=2
)

#                  
from Core.Cognition.internal_universe import *
from Core.Cognition.internal_universe import InternalUniverse, WorldCoordinate

#     export
__all__ = ['InternalUniverse', 'WorldCoordinate', 'get_internal_universe']

_internal_universe = None

def get_internal_universe() -> InternalUniverse:
    global _internal_universe
    if _internal_universe is None:
        _internal_universe = InternalUniverse()
    return _internal_universe
