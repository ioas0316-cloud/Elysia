"""
REDIRECT: torch_graph.py has moved to Core/Foundation/Graph/
This stub provides backward compatibility.
"""
from Core._01_Foundation._02_Logic.Graph.torch_graph import *
from Core._01_Foundation._02_Logic.Graph.torch_graph import get_torch_graph, TorchGraph

import warnings
warnings.warn(
    "Core._01_Foundation.torch_graph is deprecated. Use Core._01_Foundation.Graph.torch_graph instead.",
    DeprecationWarning,
    stacklevel=2
)
