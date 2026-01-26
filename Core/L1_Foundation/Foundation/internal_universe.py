"""
Internal Universe Facade (Redirect Module)
==========================================

                              .

     : Core/Memory/Vector/internal_universe.py

35                 InternalUniverse        ,
                           .

   :
    #     (            )
    from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
    
    #    (   import)
    from Core.L5_Mental.Intelligence.Memory_Linguistics.Memory.Vector.internal_universe import InternalUniverse
"""

import warnings

#       (         )
warnings.warn(
    "Core.L1_Foundation.Foundation.internal_universe is deprecated. "
    "Use Core.L2_Metabolism.Memory.Vector.internal_universe instead.",
    DeprecationWarning,
    stacklevel=2
)

#                  
from Core.L5_Mental.Intelligence.Memory.Vector.internal_universe import *
from Core.L5_Mental.Intelligence.Memory.Vector.internal_universe import InternalUniverse, WorldCoordinate

#     export
__all__ = ['InternalUniverse', 'WorldCoordinate', 'get_internal_universe']

_internal_universe = None

def get_internal_universe() -> InternalUniverse:
    global _internal_universe
    if _internal_universe is None:
        _internal_universe = InternalUniverse()
    return _internal_universe
