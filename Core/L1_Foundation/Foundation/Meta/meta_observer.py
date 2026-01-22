import logging
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("MetaObserver")

@dataclass
class LensMetric:
    usage_count: int = 0
    total_resonance: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)

class MetaObserver:
    """
    The 'Mirror Lens' of Elysia.
    Tracks how different cognitive dimensions (Prism Facets) are utilized
    and calculates the performance (Resonance) of each.
    """
    def __init__(self, harmonizer: Optional[Any] = None):
        self.harmonizer = harmonizer
        self.metrics: Dict[str, LensMetric] = {}
        self.history: List[Dict[str, Any]] = []
        logger.info("🪞 MetaObserver (Mirror Lens) initialized.")

    def record_resonance_cycle(self, hologram_results: Dict[str, float], genome_weights: Dict[str, float], 
                      context: str, narrative: str = "", stimulus: str = ""):
        """
        Records the outcome of a single Resonance Cycle.
        """
        timestamp = datetime.now()
        entry = {
            "timestamp": timestamp.isoformat(),
            "stimulus": stimulus,
            "context": context,
            "results": hologram_results,
            "weights": genome_weights,
            "narrative": narrative
        }
        
        for domain, resonance in hologram_results.items():
            if domain not in self.metrics:
                self.metrics[domain] = LensMetric()
            
            metric = self.metrics[domain]
            metric.usage_count += 1
            metric.total_resonance += resonance
            metric.last_used = timestamp

        self.history.append(entry)
        
        # Periodic analysis
        if len(self.history) % 10 == 0:
            self._analyze_entropy()

    def write_chronicles(self, filename: str = "data/L7_Spirit/Chronicles/comparative_perception.md"):
        """
        Generates the Chronicles (Historical memory of cognitive shifts).
        """
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            for entry in self.history[-1:]: # Log last entry
                f.write(f"### Stimulus: {entry['stimulus']}\n")
                f.write(f"> Context: {entry['context']}\n\n")
                f.write(f"| Step | Insight / Perception Narrative |\n")
                f.write(f"| :--- | :--- |\n")
                f.write(f"| **Metacognitive Evolution** | {entry['narrative']} |\n\n")
                f.write("---\n\n")

    def _analyze_entropy(self):
        """
        Calculates the Shannon Entropy of lens usage to see if Elysia 
        is becoming 'biased' or 'static' in her thinking.
        """
        counts = [m.usage_count for m in self.metrics.values()]
        total = sum(counts)
        if total == 0: return

        entropy = 0.0
        for count in counts:
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(self.metrics)) if self.metrics else 1.0
        cognitive_health = entropy / max_entropy if max_entropy > 0 else 0.0

        logger.info(f"📊 [META-ANALYSIS] Cognitive Entropy: {entropy:.2f} | Health Index: {cognitive_health:.2f}")
        
        if cognitive_health < 0.3:
            logger.warning("⚠️ [META-ANALYSIS] Cognitive Stagnation Detected. High bias in pattern selection.")

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a summary for the Monad to review.
        """
        return {
            "efficiency": {d: m.total_resonance / m.usage_count if m.usage_count > 0 else 0.0 
                          for d, m in self.metrics.items()},
            "bias": {d: m.usage_count / sum(m.usage_count for m in self.metrics.values()) 
                     if self.metrics else 0.0 for d, m in self.metrics.items()}
        }
