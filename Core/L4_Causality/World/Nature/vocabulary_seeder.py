"""
Vocabulary Seeder
=================
"In the beginning was the Word, and the Word was manifold."

This module provides a rich, categorized lexicon to seed the Trinity Universe.
It expands the primitive "5 biomes" into a complex reality of thousands of potential concepts.
"""

from typing import List, Dict
import random

class VocabularySeeder:
    def __init__(self):
        self.categories = {
            "Nature": [
                "Forest", "Ocean", "Mountain", "River", "Desert", "Tundra", "Jungle", 
                "Volcano", "Canyon", "Cave", "Valley", "Marsh", "Reef", "Glacier", 
                "Storm", "Cloud", "Rain", "Mist", "Nebula", "Star", "Void", "Meteor",
                "Tree", "Flower", "Vine", "Moss", "Fern", "Bamboo", "Cactus", "Lily"
            ],
            "Material": [
                "Gold", "Iron", "Stone", "Wood", "Water", "Fire", "Ice", "Clay", "Sand",
                "Glass", "Steel", "Bronze", "Silver", "Diamond", "Ruby", "Emerald",
                "Obsidian", "Crystal", "Silk", "Cotton", "Leather", "Paper", "Ink",
                "Copper", "Mercury", "Sulfur", "Salt", "Ash", "Smoke", "Steam"
            ],
            "Emotion": [
                "Love", "Hate", "Joy", "Sorrow", "Anger", "Fear", "Hope", "Despair",
                "Courage", "Pride", "Shame", "Guilt", "Peace", "Anxiety", "Awe",
                "Envy", "Pity", "Greed", "Lust", "Compassion", "Gratitude", "Regret"
            ],
            "Abstract": [
                "Truth", "Lie", "Logic", "Chaos", "Order", "Time", "Space", "Life",
                "Death", "Soul", "Mind", "Spirit", "Dreams", "Memory", "Wisdom",
                "Knowledge", "Fate", "Luck", "Karma", "Honor", "Justice", "Freedom"
            ],
            "Civilization": [
                "City", "Village", "Tower", "Bridge", "Road", "Wall", "Market",
                "Temple", "School", "Library", "Forge", "Farm", "Garden", "Harbor",
                "Palace", "Ruins", "Tomb", "Statue", "Fountain", "Gate", "Throne"
            ],
            "Action": [
                "Create", "Destroy", "Heal", "Hurt", "Give", "Take", "Build", "Break",
                "Learn", "Teach", "Lead", "Follow", "Seek", "Hide", "Speak", "Listen",
                "Eat", "Starve", "Sleep", "Dream", "Wake", "Dance", "Sing", "Fight"
            ]
        }
        
    def get_all_words(self) -> List[str]:
        all_words = []
        for cat in self.categories.values():
            all_words.extend(cat)
        return all_words
        
    def get_random_word(self, category: str = None) -> str:
        if category and category in self.categories:
            return random.choice(self.categories[category])
        return random.choice(self.get_all_words())
        
    def get_random_batch(self, count: int) -> List[str]:
        return random.sample(self.get_all_words(), min(count, len(self.get_all_words())))

SEEDED_LEXICON = VocabularySeeder()