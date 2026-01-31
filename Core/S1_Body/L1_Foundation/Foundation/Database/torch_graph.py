"""
REDIRECT: torch_graph.py has moved to Core/Foundation/Graph/
This stub provides backward compatibility.
"""
from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import *
from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph, TorchGraph

import warnings
warnings.warn(
    "Core.S1_Body.L1_Foundation.Foundation.torch_graph is deprecated. Use Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph instead.",
    DeprecationWarning,
    stacklevel=2
)
