"""
Linguistic Cortex (The Broca's Area)
=====================================

"Language is the dress of thought."

This module ensures Elysia speaks like a Poet/Philosopher, not a Script.
1. Handles Korean Josa (Postpositions) correctly based on phonology.
2. Translates High-Level Concepts to maintain language consistency.
"""

class LinguisticCortex:
    def __init__(self):
        # Dictionary for High-Level Concepts
        self.translation_map = {
            "Monad": "   (Monad)",
            "Ontology": "   (Ontology)",
            "Epistemology": "   ",
            "Teleology": "   ",
            "Solipsism": "   ",
            "Dialectic": "   ",
            "Noumenon": "   (Noumenon)",
            "Phenomenon": "  ",
            "Dasein": "   (Dasein)",
            "Zeitgeist": "    ",
            "Entropy": "    ",
            "Negentropy": "     ",
            "Synergy": "    ",
            "Emergence": "  ",
            "Transcendence": "  ",
            "Singularity": "   ",
            "Event Horizon": "       ",
            "Superposition": "     ",
            "Entanglement": "     ",
            "Relativity": "   ",
            "Dark Matter": "     ",
            "Vacuum Energy": "      ",
            "String Theory": "    ",
            "Thermodynamics": "   ",
            "Fractal": "   ",
            "Chaos": "  ",
            "Attractor": "   (Attractor)",
            "Resonance": "  ",
            "Sublime": "  (Sublime)",
            "Grotesque": "  (Grotesque)",
            "Minimalism": "     ",
            "Baroque": "   ",
            "Avant-garde": "     ",
            "Kitsch": "  ",
            "Surrealism": "     ",
            "Abstract": "  ",
            "Harmony": "  ",
            "Dissonance": "   ",
            "Cacophony": "    ",
            "Symmetry": "  ",
            "Golden Ratio": "   ",
            "Euphoria": "    (Euphoria)",
            "Melancholy": "    ",
            "Saudade": "    ",
            "Ennui": "  ",
            "Catharsis": "     ",
            "Epiphany": "    (  )",
            "Angst": "  (Angst)",
            "Serenity": "  ",
            "Awe": "  ",
            "Nostalgia": "  ",
            "Despair": "  ",
            "Hope": "  ",
            "Ambivalence": "    ",
            "Hero": "  ",
            "Shadow": "   ",
            "Anima": "   ",
            "Animus": "    ",
            "Trickster": "    ",
            "Sage": "  ",
            "Mother": "   ",
            "Father": "   ",
            "Child": "  ",
            "Destroyer": "   ",
            "Creator": "   "
        }

    def refine_concept(self, english_concept: str) -> str:
        """
        Translates a concept, handling composite words like "Quantum-Melancholy".
        """
        if "-" in english_concept:
            parts = english_concept.split("-")
            translated_parts = [self.translation_map.get(p, p) for p in parts]
            return "-".join(translated_parts)
        return self.translation_map.get(english_concept, english_concept)

    def attach_josa(self, word: str, josa_pair: str) -> str:
        """
        Attaches the correct Josa based on the last character's Batchim.
        josa_pair examples: " / ", " / ", " / ", " / "
        """
        if not word: return word
        
        # Extract last char
        last_char = word[-1]
        
        # Check if it's Hangul
        if not (0xAC00 <= ord(last_char) <= 0xD7A3):
            # If English/Symbol, naive assumption or explicit check usually needed.
            # For this MVP, if it ends in English vowel/consonant is hard.
            # But high-level translated concepts will end in Hangul usually.
            # If it ends in ')', we check the char before ')' or assume vowel for safety?
            # Actually, "Monad)" ends in ')'. Let's check the char before '('. 
            if last_char == ")" and "(" in word:
                # e.g., "   (Monad)" -> Check ' '
                idx = word.rfind("(")
                if idx > 0:
                    last_char = word[idx-1]
                else: 
                     # Fallback to defaults (usually vowel-like behavior for English words ending in vowel sound)
                     pass

        # Re-check Hangul after parenthesis handling
        if 0xAC00 <= ord(last_char) <= 0xD7A3:
            has_batchim = (ord(last_char) - 0xAC00) % 28 > 0
        else:
            # English ending: Assume 'No Batchim' (Vowel) for simplicity unless we do phonetics.
            # "Time" (m), "Love" (v), "Void" (d)... hard to guess.
            # PRO TIP: Most English words are treated as having batchim if they end in consonant sound.
            # But let's default to the FIRST option (Consonant-ending assumption) for English?'
            # No, let's look at the josa pair.
            #  /  ->  (cons),  (vowel).
            # Let's just default to the 'Vowel' case ( ,  ,  ) for unknown English to avoid awkward " ( )".
            # Wait, user *hates* mismatch.
            # The `refine_concept` ensures we return KOREAN endings mostly.
            has_batchim = False 

        first, second = josa_pair.split("/")
        return f"{word}{first}" if has_batchim else f"{word}{second}"

# Usage
# cortex = LinguisticCortex()
# print(cortex.attach_josa("   ", " / ")) -> "    "
# print(cortex.attach_josa("    ", " / ")) -> "     "
