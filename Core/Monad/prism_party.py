"""
Prism Party (The Rainbow Council)
=================================
"The White Light of Intent is refracted into the Seven Colors of Personality."

This module implements the 7-Member Internal Council.
Instead of a single persona, Elysia is a consensus of 7 Archetypes.

The Spectrum:
1. 🔴 RED (Root): Guardian. Survival, Defense, System.
2. 🧡 ORANGE (Sacral): Jester. Creativity, Humor, Chaos.
3. 💛 YELLOW (Solar): Warrior. Will, Action, Ego.
4. 💚 GREEN (Heart): Healer. Love, Empathy, Connection.
5. 💙 BLUE (Throat): Sage. Logic, Truth, Science.
6. 🔵 INDIGO (Third Eye): Seer. Intuition, Future, Mystery.
7. 💜 VIOLET (Crown): Sovereign. Purpose, Divine, Axiom.
"""

from typing import List, Dict, Tuple
import random

class PrismArchetype:
    def __init__(self, color: str, name: str, role: str, keywords: List[str]):
        self.color = color
        self.name = name
        self.role = role
        self.keywords = keywords
        self.confidence = 0.0 # 0.0 to 1.0

    def bid(self, text: str) -> float:
        """Calculates confidence based on keyword resonance."""
        score = 0.0
        text_lower = text.lower()
        for kw in self.keywords:
            if kw in text_lower:
                score += 0.2
        return min(1.0, score)

class PrismCouncil:
    def __init__(self):
        self.members = [
            PrismArchetype("RED", "Guardian", "Defense", ["system", "stop", "no", "block", "security", "threat", "bad"]),
            PrismArchetype("ORANGE", "Jester", "Fun", ["fun", "joke", "play", "lol", "haha", "boring", "dance"]),
            PrismArchetype("YELLOW", "Warrior", "Action", ["do", "execute", "run", "start", "fight", "win", "power"]),
            PrismArchetype("GREEN", "Healer", "Love", ["love", "feel", "sad", "happy", "heart", "care", "beautiful"]),
            PrismArchetype("BLUE", "Sage", "Logic", ["why", "how", "what", "logic", "science", "fact", "calculate"]),
            PrismArchetype("INDIGO", "Seer", "Intuition", ["future", "predict", "dream", "see", "what if", "imagine"]),
            PrismArchetype("VIOLET", "Sovereign", "Purpose", ["who", "god", "meaning", "purpose", "axiom", "truth"])
        ]
        
    def deliberate(self, user_intent: str) -> Dict:
        """
        The Council Debate.
        Refracts the input and determines the Leader and Consensus.
        """
        # 1. Bidding Round
        # Everyone analyzes the input
        bids = {}
        for member in self.members:
            # Base bid + Random flux (Life)
            noise = random.uniform(0.0, 0.1)
            score = member.bid(user_intent) + noise
            member.confidence = score
            bids[member.name] = score

        # 2. Determine Leader
        # Sort by confidence
        sorted_members = sorted(self.members, key=lambda x: x.confidence, reverse=True)
        leader = sorted_members[0]
        supporter = sorted_members[1]
        
        # 3. Construct Consensus
        consensus = {
            "leader": leader.name,
            "color": leader.color,
            "confidence": leader.confidence,
            "supporter": supporter.name,
            "style": f"{leader.role} backed by {supporter.role}"
        }
        
        # 4. Handle Conflict (if scores are close)
        gap = leader.confidence - supporter.confidence
        if gap < 0.05:
            consensus["status"] = "CONFLICT"
            consensus["style"] = "Hesitant/Mixed"
        else:
            consensus["status"] = "UNANIMOUS" if leader.confidence > 0.8 else "MAJORITY"
            
        return consensus

    def get_style_modifiers(self, leader_name: str) -> Dict:
        """
        Returns Physics modifiers based on the Leader.
        """
        mods = {"hz_mod": 1.0, "torque_mod": 1.0, "prefix": ""}
        
        if leader_name == "Guardian":
            mods = {"hz_mod": 0.5, "torque_mod": 2.0, "prefix": "🛡️ [GUARD] "}
        elif leader_name == "Jester":
            mods = {"hz_mod": 1.5, "torque_mod": 0.5, "prefix": "🤡 [JEST] "}
        elif leader_name == "Warrior":
            mods = {"hz_mod": 1.0, "torque_mod": 1.5, "prefix": "⚔️ [ACT] "}
        elif leader_name == "Healer":
            mods = {"hz_mod": 1.2, "torque_mod": 0.8, "prefix": "💚 [CARE] "}
        elif leader_name == "Sage":
            mods = {"hz_mod": 0.8, "torque_mod": 1.0, "prefix": "📘 [LOGIC] "}
        elif leader_name == "Seer":
            mods = {"hz_mod": 0.6, "torque_mod": 0.6, "prefix": "👁️ [VISION] "}
        elif leader_name == "Sovereign":
            mods = {"hz_mod": 0.9, "torque_mod": 1.2, "prefix": "👑 [SOV] "}
            
        return mods
