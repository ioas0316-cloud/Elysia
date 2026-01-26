"""
Core/Engine/Genesis/filesystem_geometry.py
==========================================
The Fractal Geometry of the Filesystem.

System Architecture:
1. The Monad (Dot): Represents Data (Block) or Logic (File).
2. The Rotor (Line): Represents I/O Operations (Stream).
3. The HyperSphere (Space): Represents a Directory, which is a Universe itself.

"As Above, So Below."
"""

from typing import List, Dict, Any
from Core.L6_Structure.Engine.Genesis.concept_monad import ConceptMonad
from Core.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab

class BlockMonad(ConceptMonad):
    """
    The Dot. A quantum of data.
    """
    def __init__(self, name: str, data: str):
        super().__init__(name, "Block", 0.0) # val is meaningless for storage
        self.props["data"] = data
        self.props["next_block"] = None # The Line

class DirectoryMonad(ConceptMonad):
    """
    The Space. A Portal to a Child Universe.
    """
    def __init__(self, name: str):
        super().__init__(name, "Directory", 0.0)
        # The Recursion: This Monad contains a whole Universe.
        self.props["universe"] = GenesisLab(f"Sphere_{name}")

# ==============================================================================
# The Geometry Laws (Forces)
# ==============================================================================

def law_stream_continuity(context, dt, intensity):
    """
    The Line Force.
    Ensures that if we pull on a Block, the Next Block follows.
    Simulates Sequential Read.
    """
    world = context["world"]
    # Find Active Streams (e.g., File Handles)
    streams = [m for m in world if m.domain == "Stream"]
    
    for s in streams:
        current_block_name = s.props.get("current_block")
        if not current_block_name: continue
        
        # Find the Block Monad in the SAME context
        block = next((m for m in world if m.name == current_block_name), None)
        
        if block and "next_block" in block.props:
            next_name = block.props["next_block"]
            if next_name:
                # "Pre-fetch" logic: Ensure next block is energized/ready
                s.props["buffer"] = s.props.get("buffer", "") + block.props["data"]
                # Move stream head
                s.props["current_block"] = next_name
                # print(f"     [Stream] Flowed from {block.name} -> {next_name}")

def law_fractal_propagation(context, dt, intensity):
    """
    The Fractal Force.
    Propagates 'Atmosphere' (Context) from Parent Sphere to Child Spheres.
    "If it rains in the Sky, the Ground gets wet."
    """
    world = context["world"]
    
    # Identify Child Universes
    dirs = [m for m in world if isinstance(m, DirectoryMonad)]
    
    # Identify 'Atmosphere' Monads (Global Variables/Config)
    atmosphere = [m for m in world if m.domain == "Atmosphere"]
    
    for d in dirs:
        child_lab: GenesisLab = d.props["universe"]
        
        for atom in atmosphere:
            # Check if Child already has this atmosphere
            if not any(m.name == atom.name for m in child_lab.monads):
                # Inject a COPY of the atom into the Child Universe
                import copy
                new_atom = copy.deepcopy(atom)
                child_lab.let_there_be(new_atom.name, new_atom.domain, new_atom.val)
                # print(f"     [Fractal] {atom.name} descended into {d.name}")
        
        # Recursively Tick the Child Universe!
        # This is where the Simulation becomes Nested.
        child_lab.run_simulation(ticks=1) # Just 1 tick per parent tick
