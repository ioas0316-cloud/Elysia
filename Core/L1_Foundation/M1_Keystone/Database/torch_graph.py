"""
REDIRECT: torch_graph.py has moved to Core/Foundation/Graph/
This stub provides backward compatibility.
"""
from Core.L1_Foundation.M1_Keystone.Graph.torch_graph import *
from Core.L1_Foundation.M1_Keystone.Graph.torch_graph import get_torch_graph, TorchGraph

import warnings
warnings.warn(
    "Core.L1_Foundation.M1_Keystone.torch_graph is deprecated. Use Core.L1_Foundation.M1_Keystone.Graph.torch_graph instead.",
    DeprecationWarning,
    stacklevel=2
)
