"""
REDIRECT: omni_graph.py has moved to Core/Foundation/Graph/
This stub provides backward compatibility.
"""
from Core.L1_Foundation.Foundation.Graph.omni_graph import *
from Core.L1_Foundation.Foundation.Graph.omni_graph import OmniGraph, get_omni_graph

import warnings
warnings.warn(
    "Core.L1_Foundation.Foundation.omni_graph is deprecated. Use Core.L1_Foundation.Foundation.Graph.omni_graph instead.",
    DeprecationWarning,
    stacklevel=2
)