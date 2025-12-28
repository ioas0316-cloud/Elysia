"""
Hierarchical Learning Module
============================
Manages knowledge domains and graph structures.
Restored during self-repair process.
"""

class Domain:
    """Represents a knowledge domain."""
    CORE = "Core"
    COGNITIVE = "Cognitive"
    CREATIVE = "Creative"
    
    def __init__(self, name: str):
        self.name = name

class HierarchicalKnowledgeGraph:
    """
    Manages the hierarchical structure of knowledge nodes.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        """Adds a knowledge node."""
        self.nodes[node.id] = node
        
    def connect(self, source_id, target_id, relation_type):
        """Connects two nodes."""
        self.edges.append((source_id, target_id, relation_type))

class KnowledgeNode:
    """Represents a single unit of knowledge."""
    def __init__(self, id, content, domain):
        self.id = id
        self.content = content
        self.domain = domain
