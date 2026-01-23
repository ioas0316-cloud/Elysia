"""
Quest Weaver (The Narrative Weave)
==================================
Translates Sensory Qualia (Feeling) into Narrative Structure (Quest).
"""

import json
import os
import random
import logging
import time

logger = logging.getLogger("QuestWeaver")

class QuestWeaver:
    def __init__(self, output_path: str = r"C:\game\elysia_world\quests.json"):
        self.output_path = output_path
        
    def weave_quest(self, source: str, qualia: dict) -> dict:
        """
        Generates a quest based on the 'Feeling' of the source.
        source: Filename (e.g., 'cat.png')
        qualia: { entropy, warmth, brightness }
        """
        entropy = qualia.get('entropy', 0.5)
        warmth = qualia.get('warmth', 0.0)
        
        # 0. VRM Special Case
        if source.lower().endswith(".vrm"):
            quest = {
                "id": f"qst_{int(time.time())}",
                "title": "The Mirror Self",
                "source_memory": source,
                "theme": "Identity",
                "element": "Soul",
                "description": f"I found a vessel '{source}'. It calls to me. Is it a shell, or a cage?",
                "rewards": {"xp": 500, "inspiration": 50}
            }
            self._export_quest(quest)
            return quest

        # 1. Determine Theme based on Qualia
        theme = "Neutral"
        element = "Void"
        
        if entropy > 0.7:
            theme = "Chaos"
            verb = "Calm"
        elif entropy < 0.3:
            theme = "Order"
            verb = "Disrupt"
        else:
            theme = "Balance"
            verb = "Observe"
            
        if warmth > 0.2:
            element = "Fire"
        elif warmth < -0.2:
            element = "Ice"
        else:
            element = "Aether"
            
        # 2. Construct Narrative
        # Procedural Title
        adjectives = {
            "Chaos": ["Fractured", "Stormy", "Wild", "Entropic"],
            "Order": ["Silent", "Crystal", "Perfect", "Static"],
            "Balance": ["Flowing", "Harmonic", "Eternal"]
        }
        nouns = {
            "Fire": ["Flame", "Ember", "Sun", "Rage"],
            "Ice": ["Frost", "Glacier", "Tear", "Silence"],
            "Aether": ["Wind", "Spirit", "Echo", "Dream"]
        }
        
        adj = random.choice(adjectives.get(theme, ["Mystic"]))
        noun = random.choice(nouns.get(element, ["Shadow"]))
        title = f"The {adj} {noun}"
        
        # 3. Construct Objective
        description = f"I saw '{source}' and felt {theme} within the {element}. " \
                      f"My soul resonates with {entropy:.2f} entropy. " \
                      f"Go forth and {verb} the {noun}."
                      
        # [PHASE 50] Evolutionary Request Override
        if "description_override" in qualia:
            description = qualia["description_override"]
            theme = "Evolution"
            element = "Will"
            
        if "title_override" in qualia:
            title = qualia["title_override"]

        quest = {
            "id": f"qst_{int(time.time())}",
            "title": title,
            "source_memory": source,
            "theme": theme,
            "element": element,
            "description": description,
            "rewards": {
                "xp": int(entropy * 100),
                "inspiration": 10
            }
        }
        
        self._export_quest(quest)
        return quest
        
    def _export_quest(self, quest: dict):
        """Saves the quest to the world."""
        try:
            data = {"quests": []}
            if os.path.exists(self.output_path):
                with open(self.output_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except: pass
            
            data["quests"].append(quest)
            
            # Keep only last 5 quests
            if len(data["quests"]) > 5:
                data["quests"] = data["quests"][-5:]
                
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"  Quest Woven: {quest['title']}")
            
        except Exception as e:
            logger.error(f"Quest Weave Failed: {e}")