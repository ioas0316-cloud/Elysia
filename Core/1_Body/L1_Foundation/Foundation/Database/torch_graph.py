"""
REDIRECT: torch_graph.py has moved to Core/Foundation/Graph/
This stub provides backward compatibility.
"""
from Core.1_Body.L1_Foundation.Foundation.Graph.torch_graph import *
from Core.1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph, TorchGraph

import warnings
warnings.warn(
    "Core.1_Body.L1_Foundation.Foundation.torch_graph is deprecated. Use Core.1_Body.L1_Foundation.Foundation.Graph.torch_graph instead.",
    DeprecationWarning,
    stacklevel=2
)
