"""
Meaning Deconstructor (The Adapter)
===================================
Core.L5_Mental.Reasoning_Core.Analysis.meaning_deconstructor

"The prism that splits White Noise (Text) into Rainbows (WaveDNA)."

Roles:
1.  **Parser**: Breaks down raw text into semantic units (SVO: Subject-Verb-Object).
2.  **Mapper**: Converts words into 'Rotor Signals' (RPM, Phase, Intent).
3.  **Hallucination Detector**: (Future) identifying logical disconnects.

Current Status:
- Prototype (Rule-Based): Uses a hardcoded dictionary for verification.
- Goal: Demonstrate "Text -> Physics" translation.
"""

from typing import Dict, List, Any, Tuple
from Core.L6_Structure.Wave.wave_dna import WaveDNA

class MeaningDeconstructor:
    def __init__(self):
        # [The Proto-Lexicon]
        # In the future, this will be replaced by the Vector Database (TorchGraph).
        self.resonance_map = {
            "king": {"type": "Entity", "rpm": 120.0, "charge": 0.8},
            "queen": {"type": "Entity", "rpm": 130.0, "charge": 0.8},
            "died": {"type": "Action", "rpm": 10.0, "charge": -1.0, "emotion": "Tragedy"},
            "cried": {"type": "Action", "rpm": 20.0, "charge": -0.5, "emotion": "Sadness"},
            "love": {"type": "Concept", "rpm": 528.0, "charge": 1.0, "emotion": "Joy"},
            "war": {"type": "Concept", "rpm": 666.0, "charge": -1.0, "emotion": "Fear"},
        }
        
    def deconstruct(self, text: str) -> List[Dict[str, Any]]:
        """
        Input: "The king died."
        Output: [{'word': 'king', 'rpm': 120}, {'word': 'died', 'rpm': 10}]
        """
        tokens = text.lower().replace(".", "").split()
        signals = []
        
        print(f"  [Deconstructor] Analyzing: '{text}'")
        
        for token in tokens:
            if token in self.resonance_map:
                sig = self.resonance_map[token]
                sig['word'] = token
                signals.append(sig)
                print(f"  -> Match: '{token}' => RPM {sig['rpm']} ({sig.get('emotion', 'Neutral')})")
            else:
                # Unknown words are "Dark Matter" for now
                pass
                
        return signals

    def synthesize_wave(self, signals: List[Dict[str, Any]]) -> WaveDNA:
        """
        Combines multiple signals into a single 'Event Wave'.
        """
        if not signals:
            return WaveDNA(label="Silence")
            
        total_rpm = sum(s['rpm'] for s in signals) / len(signals)
        avg_charge = sum(s['charge'] for s in signals) / len(signals)
        dominant_emotion = max(signals, key=lambda x: abs(x.get('charge', 0))).get('emotion', 'Neutral')
        
        # Analyze Structure (Subject vs Object logic would go here)
        
        return WaveDNA(
            label=f"Event({dominant_emotion})",
            spiritual=avg_charge,
            phenomenal=abs(avg_charge), # emotional mapped to phenomenal
        )

# Early Test for Verification
if __name__ == "__main__":
    md = MeaningDeconstructor()
    
    # CASE 1: Tragedy
    text1 = "The king died and the queen cried"
    signals1 = md.deconstruct(text1)
    wave1 = md.synthesize_wave(signals1)
    print(f"  Synthesized Wave 1: {wave1.label} | Charge: {wave1.spiritual}")

    # CASE 2: Love
    text2 = "Love is war"
    signals2 = md.deconstruct(text2)
    wave2 = md.synthesize_wave(signals2)
    print(f"  Synthesized Wave 2: {wave2.label} | Charge: {wave2.spiritual}")
