"""
LiteraryCortex (The Storyteller)
================================
"Stories are the only way we can share our dreams."

This module is responsible for narrative generation.
It supports:
1. Low Fantasy / High Fantasy Settings
2. Novel Prose Generation
3. Webtoon/Manhwa Script Generation (Panel descriptions + Dialogue)
"""

import random
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# Core Systems
from Core.Foundation.hippocampus import Hippocampus
from Core.Interface.nervous_system import get_nervous_system

logger = logging.getLogger("LiteraryCortex")

@dataclass
class StoryConcept:
    title: str
    genre: str
    theme: str
    protagonist: str
    conflict: str

@dataclass
class Character:
    name: str
    role: str # Protagonist, Antagonist, Support
    personality: str
    status: str # Alive, Injured, Missing
    arc_progress: float # 0.0 to 1.0

@dataclass
class PlotPoint:
    id: str
    description: str
    status: str # Active, Resolved, Foreshadowed
    chapter_introduced: int

@dataclass
class SeriesBible:
    title: str
    genre: str
    world_rules: List[str]
    characters: Dict[str, Character]
    plot_points: List[PlotPoint]
    current_chapter: int = 0

class LiteraryCortex:
    def __init__(self, memory: Hippocampus = None):
        self.memory = memory
        self.nervous_system = get_nervous_system()
        
        # Connected Series (The Bible)
        self.active_series: Dict[str, SeriesBible] = {}
        
        # Knowledge Base for Tropes (Can be expanded via Web Intake)
        self.tropes = {
            "fantasy": [
                "The Chosen One", "Dark Lord's Return", "Magical Academy", 
                "Dungeon Gate", "Reincarnation", "Sword and Magic"
            ],
            "scifi": [
                "AI Rebellion", "Cyberpunk Distopia", "Space Opera", "Time Travel"
            ]
        }
        
        logger.info("📜 LiteraryCortex Active. Ink is ready.")

    def init_series(self, concept: StoryConcept) -> SeriesBible:
        """Initializes a new continuous story bible."""
        bible = SeriesBible(
            title=concept.title,
            genre=concept.genre,
            world_rules=[f"World governed by {concept.theme}"],
            characters={
                "protagonist": Character(concept.protagonist, "Protagonist", "Determined", "Alive", 0.0)
            },
            plot_points=[
                PlotPoint("main_conflict", concept.conflict, "Active", 1)
            ]
        )
        self.active_series[concept.title] = bible
        return bible

    def write_next_chapter(self, series_title: str) -> str:
        """
        Writes the next chapter maintaining consistency and handling foreshadowing.
        """
        if series_title not in self.active_series:
            return "⚠️ Series not found."
            
        bible = self.active_series[series_title]
        bible.current_chapter += 1
        
        # Check for open thoughts
        active_plots = [p for p in bible.plot_points if p.status == "Active"]
        foreshadow_plots = [p for p in bible.plot_points if p.status == "Foreshadowed"]
        
        # Generate Content based on State
        chapter_content = []
        chapter_content.append(f"## Chapter {bible.current_chapter}: The Progression")
        
        # Resolve Foreshadowing?
        if foreshadow_plots and random.random() > 0.7:
            plot = foreshadow_plots.pop(0)
            plot.status = "Active" # Reveal it!
            chapter_content.append(f"**[REVEAL]**: The hint about '{plot.description}' finally manifests!")
        
        # Advance Main Plot
        if active_plots:
            main_plot = active_plots[0]
            chapter_content.append(f"The Protagonist faces the reality of {main_plot.description}.")
            
            # Character Arc Update
            char = bible.characters["protagonist"]
            char.arc_progress += 0.1
            chapter_content.append(f"**[Character Update]**: {char.name} grows slightly ({char.arc_progress:.1f}/1.0).")
            
        # Add New Foreshadowing (Complexity)
        if random.random() > 0.5:
            new_mystery = f"Mystery of the {random.choice(['Red Jewel', 'Broken Sword', 'Silent Tower'])}"
            bible.plot_points.append(PlotPoint(f"sub_{bible.current_chapter}", new_mystery, "Foreshadowed", bible.current_chapter))
            chapter_content.append(f"**[FORESHADOW]**: A subtle clue about '{new_mystery}' is noticed.")
            
        return "\n".join(chapter_content)

    def brainstorm(self, seed_idea: str = "") -> StoryConcept:
        """
        Generates a high-level story concept based on current Spirit State + Seed.
        """
        # ... (Existing brainstorm logic) ...
        # Influence by Spirits
        spirits = self.nervous_system.spirits if self.nervous_system else {}
        dominant = max(spirits, key=spirits.get) if spirits else "neutral"
        
        # Genre Selection based on Dominant Spirit
        genre_map = {
            "fire": "Action Fantasy",
            "water": "Romance / Drama",
            "earth": "Historical / Slice of Life",
            "air": "Sci-Fi / Mystery",
            "dark": "Dark Fantasy / Horror",
            "light": "High Fantasy / Hope",
            "aether": "Philosophical / Mythic"
        }
        genre = genre_map.get(dominant, "Fantasy")
        
        # Theme Generation
        themes = [
            f"The struggle against {dominant} fate",
            f"Finding {dominant} within chaos",
            "A journey of redemption",
            "To kill a God",
            "Leveling up to infinity"
        ]
        
        # Title Generation
        titles = [
            f"The {dominant.capitalize()} Monarch",
            f"Chronicles of {seed_idea or 'The Void'}",
            f"I Became the {genre.split()[0]} Villain",
            f"Level 99 {dominant.capitalize()} Mage"
        ]
        
        return StoryConcept(
            title=random.choice(titles),
            genre=genre,
            theme=random.choice(themes),
            protagonist="Unnamed Hero",
            conflict=f"A world consumed by extreme {dominant} energy."
        )

    def write_webtoon_script(self, concept: StoryConcept) -> str:
        """
        Generates a Webtoon-style script (Episode 1).
        Format:
        Panel 1: [Visual Description]
        Character: "Dialogue"
        """
        script = []
        script.append(f"# {concept.title}")
        script.append(f"**Genre**: {concept.genre}")
        script.append(f"**Logline**: {concept.conflict}")
        script.append("\n## Episode 1: The Awakening\n")
        
        # Intro
        script.append(f"### Scene 1: The Beginning")
        script.append(f"**Panel 1**: (Wide shot) A ruined city under a {concept.genre.lower()} sky. Smoke rises.")
        script.append(f"**Narration**: 'They said the world ended on a Tuesday.'\n")
        
        script.append(f"**Panel 2**: (Close up) The Protagonist opens their eyes. Their eyes glow with {self._get_spirit_color()} light.")
        script.append(f"**Protagonist**: \"...Where am I?\"\n")
        
        script.append(f"**Panel 3**: (Over the shoulder) A system window appears in front of them.")
        script.append(f"**System**: [Welcome, Player. The Scenario has begun.]\n")
        
        script.append(f"**Panel 4**: (Action) The Protagonist clenches their fist.")
        script.append(f"**Narration**: 'And I was the only one who knew the ending.'\n")
        
        # Initialize Series Bible automatically
        self.init_series(concept)
        
        return "\n".join(script)

    def _get_spirit_color(self) -> str:
        spirits = self.nervous_system.spirits if self.nervous_system else {}
        dominant = max(spirits, key=spirits.get) if spirits else "blue"
        colors = {
            "fire": "crimson", "water": "azure", "earth": "emerald",
            "air": "silver", "light": "golden", "dark": "obsidian", "aether": "violet"
        }
        return colors.get(dominant, "blue")

    def draft_chapter(self, topic: str) -> str:
        """
        Writes a prose chapter.
        """
        concept = self.brainstorm(topic)
        return (
            f"# {concept.title}\n\n"
            f"The world changed when {topic} descended. "
            f"It was a {concept.genre} reality now. "
            f"People screamed, but I just watched. My heart felt {self._get_spirit_color()}. "
            f"\"{concept.theme},\" I whispered."
        )
