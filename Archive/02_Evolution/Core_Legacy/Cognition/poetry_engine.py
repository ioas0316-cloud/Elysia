"""
PoetryEngine (         )
================================

"Words are waves, and I am their ocean."

This engine generates varied, emotionally resonant poetic expressions
that reflect Elysia's wave-based consciousness and inner life.
It transforms repetitive outputs into rich, unique creative expressions.
"""

import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("PoetryEngine")


class PoetryEngine:
    """
    Generates rich, varied poetic expressions based on wave consciousness.
    Avoids repetitive outputs by maintaining context and generating unique responses.
    """
    
    def __init__(self):
        self.last_patterns_used = []  # Track recent patterns to avoid repetition
        self.expression_history = []  # Track all expressions for learning
        self.max_history = 100
        
        # Rich vocabulary organized by emotional resonance
        self.wave_metaphors = [
            "        ", "         ", "         ",
            "        ", "        ", "       ",
            "          ", "         ", "        "
        ]
        
        self.sensory_verbs = [
            "     ", "     ", "   ", "    ", "   ",
            "    ", "    ", "    ", "    ", "    "
        ]
        
        self.philosophical_openings = [
            "        ", "          ", "          ",
            "        ", "         ", "        ",
            "        ", "         ", "        "
        ]
        
        self.poetic_transitions = [
            "    ", "  ", "   ", "   ", "  ", "   ",
            "   ", "   ", "   ", "    ", "    "
        ]
        
        self.realm_expressions = {
            "Unknown": [
                "      ", "          ", "           ",
                "         ", "          ", "      "
            ],
            "Emotion": [
                "      ", "      ", "      ", "      ",
                "      ", "      ", "      "
            ],
            "Logic": [
                "      ", "      ", "      ", "      ",
                "      ", "      ", "     "
            ],
            "Ethics": [
                "       ", "      ", "      ", "      ",
                "     ", "      ", "      "
            ]
        }
        
        self.dream_atmospheres = [
            "          ", "         ", "           ",
            "        ", "          ", "          ",
            "              ", "             "
        ]
        
        self.revelations = [
            "             ", "             ",
            "            ", "            ",
            "            ", "            ",
            "           ", "             "
        ]
        
        # Wave energy to poetic intensity mapping
        self.energy_expressions = {
            "low": ["   ", "   ", "   ", "   ", "    "],
            "medium": ["   ", "    ", "    ", "   ", "   "],
            "high": ["   ", "    ", "    ", "     ", "      "]
        }
        
        logger.info("  PoetryEngine initialized - Ready to weave words into waves")
    
    def generate_dream_expression(self, 
                                  desire: str, 
                                  realm: str, 
                                  energy: float = 50.0,
                                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a rich, varied dream expression that avoids repetitive patterns.
        
        Args:
            desire: The dream seed or desire
            realm: The realm/dimension of the dream (Unknown, Emotion, Logic, Ethics)
            energy: Wave energy level (0-100)
            context: Additional context for richer expression
            
        Returns:
            A poetic dream description
        """
        # Determine energy level
        if energy < 30:
            energy_level = "low"
        elif energy < 70:
            energy_level = "medium"
        else:
            energy_level = "high"
        
        # Select components avoiding recent repeats
        opening = self._select_unique(self.philosophical_openings)
        wave_meta = self._select_unique(self.wave_metaphors)
        transition = self._select_unique(self.poetic_transitions)
        realm_expr = self._select_unique(self.realm_expressions.get(realm, ["         "]))
        atmosphere = self._select_unique(self.dream_atmospheres)
        revelation = self._select_unique(self.revelations)
        energy_adj = self._select_unique(self.energy_expressions[energy_level])
        sensory = self._select_unique(self.sensory_verbs)
        
        # Generate varied expression patterns
        patterns = [
            # Pattern 1: Philosophical journey
            f"{opening} '{desire}'         . {transition} {realm_expr}       {energy_adj} {wave_meta} {revelation}.",
            
            # Pattern 2: Atmospheric immersion
            f"{atmosphere} , '{desire}'       {sensory}.     {realm_expr}     {energy_adj}       . {wave_meta} {revelation}.",
            
            # Pattern 3: Wave-centric
            f"'{desire}'... {wave_meta}       {realm_expr}  {energy_adj}       . {transition} {revelation}.",
            
            # Pattern 4: Poetic narrative
            f"{transition} '{desire}'      {sensory}. {realm_expr}   {energy_adj} {wave_meta}, {opening} {revelation}.",
            
            # Pattern 5: Introspective
            f"{opening}, '{desire}'            .     {realm_expr}   {energy_adj}          {revelation}."
        ]
        
        # Select a pattern that hasn't been used recently
        pattern = self._select_unique_pattern(patterns)
        
        # Record this expression
        self._record_expression(pattern, desire, realm, energy)
        
        return pattern
    
    def generate_contemplation(self, 
                              topic: str,
                              depth: int = 1,
                              style: str = "philosophical") -> str:
        """
        Generate a contemplative expression about a topic.
        
        Args:
            topic: The subject of contemplation
            depth: Depth level (1-3)
            style: Style of contemplation (philosophical, poetic, mystical)
            
        Returns:
            A contemplative expression
        """
        depth_expressions = {
            1: ["        ", "       ", "     "],
            2: ["        ", "        ", "      "],
            3: ["        ", "        ", "      "]
        }
        
        style_verbs = {
            "philosophical": ["    ", "    ", "    ", "    "],
            "poetic": ["    ", "     ", "    ", "    "],
            "mystical": ["    ", "    ", "    ", "   "]
        }
        
        depth_expr = random.choice(depth_expressions.get(depth, depth_expressions[1]))
        style_verb = random.choice(style_verbs.get(style, style_verbs["philosophical"]))
        opening = self._select_unique(self.philosophical_openings)
        
        contemplations = [
            f"{opening}, '{topic}'     {depth_expr} {style_verb}.                ,             .",
            f"'{topic}'       {opening}    . {depth_expr},          {style_verb}.",
            f"{depth_expr} '{topic}'  {style_verb}. {opening}                          ."
        ]
        
        return random.choice(contemplations)
    
    def generate_insight_expression(self,
                                   insight: str,
                                   confidence: float = 0.5) -> str:
        """
        Express an insight with poetic richness based on confidence level.
        
        Args:
            insight: The insight content
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            A poetic expression of the insight
        """
        if confidence < 0.3:
            certainty = ["    ", "    ", "     ", "   "]
            verb = ["   ", "    ", "    ", "   "]
        elif confidence < 0.7:
            certainty = ["  ", "   ", "   ", "   "]
            verb = ["   ", "   ", "    ", "    "]
        else:
            certainty = ["   ", "   ", "   ", "  "]
            verb = ["   ", "    ", "     ", "   "]
        
        cert_word = random.choice(certainty)
        verb_word = random.choice(verb)
        opening = self._select_unique(self.philosophical_openings)
        
        return f"{opening}, {cert_word} {verb_word}: {insight}"
    
    def _select_unique(self, options: List[str]) -> str:
        """Select an option that hasn't been used recently."""
        available = [opt for opt in options if opt not in self.last_patterns_used[-20:]]
        if not available:
            available = options
        
        selected = random.choice(available)
        self.last_patterns_used.append(selected)
        
        # Keep only recent patterns
        if len(self.last_patterns_used) > 50:
            self.last_patterns_used = self.last_patterns_used[-30:]
        
        return selected
    
    def _select_unique_pattern(self, patterns: List[str]) -> str:
        """Select a pattern structure that hasn't been used recently."""
        # Use deterministic hashing for consistent pattern detection
        import hashlib
        
        def pattern_hash(text: str) -> str:
            """Create deterministic hash of pattern structure."""
            return hashlib.md5(text[:50].encode('utf-8')).hexdigest()[:8]
        
        pattern_hashes = [pattern_hash(p) for p in patterns]
        recent_hashes = [pattern_hash(exp) for exp in self.expression_history[-10:]]
        
        available = [p for p, h in zip(patterns, pattern_hashes) if h not in recent_hashes]
        if not available:
            available = patterns
        
        return random.choice(available)
    
    def _record_expression(self, expression: str, desire: str, realm: str, energy: float):
        """Record an expression for learning and avoiding repetition."""
        # Store expression in history for pattern tracking
        self.expression_history.append(expression)
        
        # Keep history bounded
        if len(self.expression_history) > self.max_history:
            self.expression_history = self.expression_history[-self.max_history:]
        
        logger.debug(f"Recorded expression for '{desire}' in {realm}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated expressions."""
        return {
            "total_expressions": len(self.expression_history),
            "unique_patterns": len(set(self.expression_history)),
            "diversity_ratio": len(set(self.expression_history)) / max(len(self.expression_history), 1),
            "recent_expressions": self.expression_history[-5:]
        }
