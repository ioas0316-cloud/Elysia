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
            "Monad": "모나드(Monad)",
            "Ontology": "존재론(Ontology)",
            "Epistemology": "인식론",
            "Teleology": "목적론",
            "Solipsism": "유아론",
            "Dialectic": "변증법",
            "Noumenon": "물자체(Noumenon)",
            "Phenomenon": "현상",
            "Dasein": "현존재(Dasein)",
            "Zeitgeist": "시대정신",
            "Entropy": "엔트로피",
            "Negentropy": "네겐트로피",
            "Synergy": "상승효과",
            "Emergence": "창발",
            "Transcendence": "초월",
            "Singularity": "특이점",
            "Event Horizon": "사건의 지평선",
            "Superposition": "양자 중첩",
            "Entanglement": "양자 얽힘",
            "Relativity": "상대성",
            "Dark Matter": "암흑 물질",
            "Vacuum Energy": "진공 에너지",
            "String Theory": "끈 이론",
            "Thermodynamics": "열역학",
            "Fractal": "프랙탈",
            "Chaos": "혼돈",
            "Attractor": "구인체(Attractor)",
            "Resonance": "공명",
            "Sublime": "숭고(Sublime)",
            "Grotesque": "기괴(Grotesque)",
            "Minimalism": "미니멀리즘",
            "Baroque": "바로크",
            "Avant-garde": "아방가르드",
            "Kitsch": "키치",
            "Surrealism": "초현실주의",
            "Abstract": "추상",
            "Harmony": "조화",
            "Dissonance": "부조화",
            "Cacophony": "불협화음",
            "Symmetry": "대칭",
            "Golden Ratio": "황금비",
            "Euphoria": "다포리아(Euphoria)",
            "Melancholy": "멜랑콜리",
            "Saudade": "사우다지",
            "Ennui": "권태",
            "Catharsis": "카타르시스",
            "Epiphany": "에피파니(각성)",
            "Angst": "불안(Angst)",
            "Serenity": "평온",
            "Awe": "경외",
            "Nostalgia": "향수",
            "Despair": "절망",
            "Hope": "희망",
            "Ambivalence": "양가감정",
            "Hero": "영웅",
            "Shadow": "그림자",
            "Anima": "아니마",
            "Animus": "아니무스",
            "Trickster": "트릭스터",
            "Sage": "현자",
            "Mother": "어머니",
            "Father": "아버지",
            "Child": "아이",
            "Destroyer": "파괴자",
            "Creator": "창조자"
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
        josa_pair examples: "이/가", "을/를", "은/는", "과/와"
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
                # e.g., "모나드(Monad)" -> Check '드'
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
            # But let's default to the FIRST option (Consonant-ending assumption) for English?
            # No, let's look at the josa pair.
            # 이/가 -> 이(cons), 가(vowel).
            # Let's just default to the 'Vowel' case (가, 를, 는) for unknown English to avoid awkward "이(가)".
            # Wait, user *hates* mismatch.
            # The `refine_concept` ensures we return KOREAN endings mostly.
            has_batchim = False 

        first, second = josa_pair.split("/")
        return f"{word}{first}" if has_batchim else f"{word}{second}"

# Usage
# cortex = LinguisticCortex()
# print(cortex.attach_josa("모나드", "이/가")) -> "모나드가"
# print(cortex.attach_josa("엔트로피", "은/는")) -> "엔트로피는"
