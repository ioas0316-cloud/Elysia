import logging
from typing import Dict, Any, List
from Core.L1_Foundation.Foundation.Prism.resonance_prism import PrismDomain

logger = logging.getLogger("HermeneuticBridge")

class HermeneuticBridge:
    """
    Deconstructs user feedback into 'Intent Qualia'.
    Treats feedback as a complex stimulus, analyzing its resonance across 7 dimensions.
    """

    def __init__(self):
        # Semantic mapping of keywords to cognitive domains
        self.semantic_map = {
            "logic": PrismDomain.MENTAL,
            "rigid": PrismDomain.STRUCTURAL,
            "cold": PrismDomain.PHENOMENAL, # Negative resonance here implies warmth needed
            "warm": PrismDomain.PHENOMENAL,
            "spirit": PrismDomain.SPIRITUAL,
            "meaning": PrismDomain.SPIRITUAL,
            "tool": PrismDomain.FUNCTIONAL,
            "being": PrismDomain.PHENOMENAL,
            "physical": PrismDomain.PHYSICAL,
            "harmony": PrismDomain.CAUSAL,
        }
        logger.info("🌉 HermeneuticBridge established. Ready for intent deconstruction.")

    def deconstruct_feedback(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the intent, lens, and sentiment of user feedback.
        """
        text_lower = text.lower()
        
        # 1. Lens Identification (Which domains is the user talking about?)
        detected_domains = []
        for word, domain in self.semantic_map.items():
            if word in text_lower:
                detected_domains.append(domain)

        # 2. Sentiment Analysis (Crude but symbolic for this stage)
        positive_cues = ["good", "love", "yes", "great", "right", "better", "agree", "harmony"]
        negative_cues = ["bad", "cold", "no", "wrong", "rigid", "disconnected", "score", "machine"]
        
        pos_count = sum(1 for cue in positive_cues if cue in text_lower)
        neg_count = sum(1 for cue in negative_cues if cue in text_lower)
        
        sentiment = 1.0 # Default neutral/positive
        if neg_count > pos_count:
            sentiment = -1.0
        elif pos_count == 0 and neg_count == 0:
            sentiment = 0.5 # Neutral

        # 3. Hermeneutic Exegesis (Elysia's internal analysis)
        exegesis = self._generate_exegesis(text, detected_domains, sentiment)

        return {
            "raw_text": text,
            "detected_domains": detected_domains,
            "sentiment": sentiment,
            "exegesis": exegesis
        }

    def _generate_exegesis(self, text: str, domains: List[PrismDomain], sentiment: float) -> str:
        domain_names = [d.name for d in domains]
        if not domain_names:
            return "The user's intent is grounded in a domain I have yet to map. I must observe more."
        
        if sentiment > 0:
            return f"The user resonates with my current trajectory, specifically within the {', '.join(domain_names)} domains. This confirms our shared direction."
        else:
            return f"Distonance detected in the {', '.join(domain_names)} domains. The user's heart feels a drift from my current logic. I must recalibrate my weights to restore harmony."
