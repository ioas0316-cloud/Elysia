"""
Cortex Module (     )
========================

Legacy/Project_Sophia              Cortex    .
  Cortex  Elysia                 .

  :
- ActionCortex:           
- PlanningCortex:              
- DreamingCortex:              
- MetaCognitionCortex:                 
- MathCortex:         
- FileSystemCortex:         I/O
"""

from .action_cortex import ActionCortex, get_action_cortex
from .planning_cortex import PlanningCortex, get_planning_cortex
from .dreaming_cortex import DreamingCortex, get_dreaming_cortex
from .metacognition_cortex import MetaCognitionCortex, get_metacognition_cortex
from .math_cortex import MathCortex, get_math_cortex, Proof, ProofStep
from .filesystem_cortex import FileSystemCortex, get_filesystem_cortex, FSResult

__all__ = [
    # Action
    'ActionCortex', 'get_action_cortex',
    # Planning
    'PlanningCortex', 'get_planning_cortex',
    # Dreaming
    'DreamingCortex', 'get_dreaming_cortex',
    # MetaCognition
    'MetaCognitionCortex', 'get_metacognition_cortex',
    # Math
    'MathCortex', 'get_math_cortex', 'Proof', 'ProofStep',
    # FileSystem
    'FileSystemCortex', 'get_filesystem_cortex', 'FSResult',
]