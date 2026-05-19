"""
City Topology Definitions
=========================

Maps the Elysia Codebase Structure to a "Cyber-City" Metaphor.
Used by the visualizer to place nodes and define districts.
"""

DISTRICTS = {
    "Foundation": {"color": "#FFFFFF", "pos": (0, 0, 0), "label": "The Capitol (Soul)"},
    "Sensory":    {"color": "#00FF00", "pos": (-2, 2, 0), "label": "The Port (Input)"},
    "Memory":     {"color": "#0000FF", "pos": (2, 2, 0),  "label": "The Archives (History)"},
    "Reasoning":  {"color": "#FFFF00", "pos": (0, 4, 1),  "label": "The Tower (Logic)"},
    "Ether":      {"color": "#8800FF", "pos": (0, -2, -1),"label": "The Grid (Substrate)"},
    "Orchestra":  {"color": "#FF0000", "pos": (0, 0, 2),  "label": "Opera House (Will)"},
}

# Define the physical layout (Roads)
# (Source, Target, Weight)
ROADS = [
    # Sensory feeds Foundation and Reasoning
    ("Sensory", "Foundation", 1.0),
    ("Sensory", "Reasoning", 0.8),

    # Foundation controls everything
    ("Foundation", "Orchestra", 1.0),
    ("Foundation", "Reasoning", 0.5),
    ("Foundation", "Ether", 1.0),

    # Orchestra directs
    ("Orchestra", "Sensory", 0.5),
    ("Orchestra", "Memory", 0.5),

    # Memory supports Reasoning
    ("Memory", "Reasoning", 0.9),
    ("Memory", "Foundation", 0.3),

    # Ether connects to all (Implicit, but drawn for structure)
    ("Ether", "Foundation", 0.2),
]

def get_node_position(district_name, jitter=0.0):
    """Returns base position for a district center."""
    if district_name in DISTRICTS:
        base = DISTRICTS[district_name]["pos"]
        return base
    return (0, 0, 0)
