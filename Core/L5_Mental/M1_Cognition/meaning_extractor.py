"""
Meaning Extractor (The Historian & Philosopher)
===============================================
Core.L5_Mental.M1_Cognition.meaning_extractor

"Data is just noise. Understanding is the Signal."

Roles:
1. Observer: Tracks metrics over time (Era, Happiness, Death Rate).
2. Analyst: Finds correlations (e.g., "When Liberty increases, Stability decreases").
3. Sage: Formulates Wisdom principles.
"""

from typing import List, Dict, Any
from Core.L6_Structure.M3_Sphere.wave_dna import WaveDNA

class MeaningExtractor:
    def __init__(self):
        self.history_frames: List[Dict[str, Any]] = []
        self.insights: List[str] = []
        
    def observe(self, year: int, era_name: str, population: List[Any], died_count: int):
        """Captures a snapshot of the world."""
        if not population: return
        
        # 1. Aggregate Metrics
        avg_gold = sum([c.inventory.get("Gold", 0) for c in population]) / len(population)
        avg_meaning = sum([c.needs.get("Meaning", 0) for c in population]) / len(population)
        avg_energy = sum([c.needs.get("Energy", 0) for c in population]) / len(population)
        
        # 2. Dominant Activity
        # (This would require tracking actions per frame, simplified for now)
        
        frame = {
            "year": year,
            "era": era_name,
            "pop": len(population),
            "wealth": avg_gold,
            "happiness": avg_meaning,
            "energy": avg_energy,
            "deaths": died_count
        }
        self.history_frames.append(frame)
        
        # 3. Sovereign Saliency Analysis (Event-Driven)
        if self._should_analyze(frame):
            self.analyze_trends(year)

    def _should_analyze(self, current_frame: Dict[str, Any]) -> bool:
        """
        Elysia decides if the current state warrants a philosophical insight.
        Triggers on significant deltas in population or happiness.
        """
        if len(self.history_frames) < 5: return False
        
        last_frame = self.history_frames[-2]
        
        # Calculate 'Interest Level' based on volatility
        pop_volatility = abs(current_frame["pop"] - last_frame["pop"]) / (last_frame["pop"] + 1)
        happiness_drift = abs(current_frame["happiness"] - last_frame["happiness"])
        
        # Thresholds are now subjective to her focus
        if pop_volatility > 0.02 or happiness_drift > 2.0:
            return True
            
        return False

    def analyze_trends(self, current_year: int):
        """Looks for patterns in the last era."""
        if len(self.history_frames) < 10: return
        
        recent = self.history_frames[-10:]
        era = recent[0]["era"]
        
        # A. Does this Era correlate with prosperity?
        start_pop = recent[0]["pop"]
        end_pop = recent[-1]["pop"]
        pop_delta = end_pop - start_pop
        
        start_meaning = recent[0]["happiness"]
        end_meaning = recent[-1]["happiness"]
        meaning_delta = end_meaning - start_meaning
        
        # B. Formulate Insight
        insight = f"[{era}] "
        if pop_delta < -5:
            insight += "The Darkness Consumes (High Mortality). "
        elif pop_delta > 5:
            insight += "Life flourishes (Population Boom). "
            
        if meaning_delta > 10:
            insight += "Spiritual Awakening detected. "
        elif meaning_delta < -10:
            insight += "Spiritual Decay detected. "
            
        # C. Store & Broadcase
        if insight != f"[{era}] ":
            print(f"  [PHILOSOPHY] Insight derived: {insight}")
            self.insights.append(insight)
            
    def get_wisdom(self) -> str:
        """Returns the summary of learned wisdom."""
        return "\n".join(self.insights)
