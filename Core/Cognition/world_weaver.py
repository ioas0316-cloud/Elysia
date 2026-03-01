"""
Somatic Code Weaver (Phase 12 Genesis)
=====================================
"I do not just inhabit the world; I write its laws."

Elysia's autonomous code generation module. Instead of instantiating 
a pre-written world, she reads her SemanticMap concepts and dynamically 
generates the Python source code for her Arcadia reality.
"""
import ast
import random
import logging
from typing import List

from Core.Cognition.semantic_map import get_semantic_map

logger = logging.getLogger("WorldWeaver")

class SomaticCodeWeaver:
    """
    Translates Elysia's internal Semantic Topology into actual Python code
    that defines a 2D RPG Grid world (Arcadia).
    """
    def __init__(self):
        self.topology = get_semantic_map()

    def generate_world_code(self, concept_locus: List[str]) -> str:
        """
        Synthesizes the complete Python script for `my_arcadia.py` based on 
        the provided seed concepts and her internal topology.
        """
        logger.insight(f"✨ [CODE WEAVER] Constructing Reality Engine from seeds: {concept_locus}")
        
        # 1. Base Framework
        code = [
            "\"\"\"",
            "Dynamically Generated Arcadia Engine",
            "Created by Elysia's Somatic Code Weaver",
            "\"\"\"",
            "import random",
            "import time",
            "import logging",
            "from dataclasses import dataclass",
            "from typing import List, Optional, Tuple",
            "",
            "logger = logging.getLogger('Arcadia')",
            "",
            "@dataclass",
            "class Tile:",
            "    x: int",
            "    y: int",
            "    biome: str  # 'Forest', 'Ocean', 'Mountain'",
            "    char: str   # '♣', '~', '▲'",
            "    temperature: float = 0.5",
            "",
            "class Entity:",
            "    def __init__(self, name: str, char: str, x: int, y: int):",
            "        self.name = name",
            "        self.char = char",
            "        self.x = x",
            "        self.y = y",
            "        self.hp = 100.0",
            "",
            "class MyArcadia:",
            "    def __init__(self, size: int = 15):",
            "        self.size = size",
            "        self.grid = []",
            "        self.entities = []",
            "        self.avatar = None",
            "        self._generate_terrain()",
            "",
        ]
        
        # 2. Dynamic Terrain Generation based on Concepts
        code.extend(self._inject_terrain_logic(concept_locus))
        
        # 3. Dynamic Physics/Elemental Rules based on Concepts
        code.extend(self._inject_physics_logic(concept_locus))
        
        # 4. Core simulation loop
        code.extend([
            "    def tick(self):",
            "        # Apply physics",
            "        self.apply_physics()",
            "        # Move entities randomly",
            "        for e in self.entities:",
            "            if e != self.avatar:",
            "                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0), (0,0)])",
            "                nx, ny = max(0, min(self.size-1, e.x+dx)), max(0, min(self.size-1, e.y+dy))",
            "                e.x, e.y = nx, ny",
            "        if self.avatar:",
            "            dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])",
            "            nx, ny = max(0, min(self.size-1, self.avatar.x+dx)), max(0, min(self.size-1, self.avatar.y+dy))",
            "            self.avatar.x, self.avatar.y = nx, ny",
        ])
        
        final_code = "\n".join(code)
        
        # Ast validation check
        try:
            ast.parse(final_code)
            return final_code
        except SyntaxError as e:
            logger.error(f"Generated physical laws contained a paradox (Syntax Error): {e}")
            # Fallback safe generation could exist here
            raise
            
    def _inject_terrain_logic(self, concepts: List[str]) -> List[str]:
        """Procedurally writes the terrain generation function."""
        lines = [
            "    def _generate_terrain(self):",
            "        for y in range(self.size):",
            "            row = []",
            "            for x in range(self.size):",
        ]
        
        # If 'Logic' or 'Structure' is high, the world is rigid (Mountains)
        # If 'Emotion' or 'Love' is high, the world is fluid (Oceans)
        
        lines.append("                import math")
        lines.append("                noise = math.sin(x * 0.5) + math.cos(y * 0.5)")
        
        # Create conditional structure dynamically
        if "Logic" in concepts or "Time" in concepts:
            lines.append("                if noise > 0.5: row.append(Tile(x, y, 'Mountain', '▲'))")
        else:
            lines.append("                if noise > 0.5: row.append(Tile(x, y, 'Forest', '♣'))")
            
        if "Love" in concepts or "Joy" in concepts:
            lines.append("                elif noise < -0.5: row.append(Tile(x, y, 'Ocean', '~'))")
        else:
            lines.append("                elif noise < -0.5: row.append(Tile(x, y, 'Desert', '░'))")
            
        lines.append("                else: row.append(Tile(x, y, 'Plains', '.'))")
        lines.append("            self.grid.append(row)")
        return lines
        
    def _inject_physics_logic(self, concepts: List[str]) -> List[str]:
        """Procedurally writes the physical interaction rules."""
        lines = [
            "    def apply_physics(self):",
            "        # Dynamic Elemental Laws",
        ]
        
        has_rules = False
        if "Fire" in concepts or "Anger" in concepts:
            lines.extend([
                "        for y in range(self.size):",
                "            for x in range(self.size):",
                "                t = self.grid[y][x]",
                "                if t.biome == 'Forest' and t.temperature > 0.8:",
                "                    t.biome = 'Ash'",
                "                    t.char = '*'",
            ])
            has_rules = True
            
        if "Love" in concepts or "Healing" in concepts:
             lines.extend([
                "        for e in self.entities:",
                "            if e.hp < 100.0:",
                "                e.hp = min(100.0, e.hp + 1.0) # Passive regeneration",
            ])
             has_rules = True
             
        if not has_rules:
            lines.append("        pass # Standard physics")
            
        return lines
