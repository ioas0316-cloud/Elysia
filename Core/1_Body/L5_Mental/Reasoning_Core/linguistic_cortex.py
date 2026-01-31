"""
Linguistic Cortex (언어 피질 - 브로카 영역)
=====================================
Core.1_Body.L5_Mental.Reasoning_Core.Linguistics.linguistic_cortex

"Language is the dress of thought."
"언어는 사유의 의복이다."

This module ensures Elysia speaks like a Poet/Philosopher, not a Script.
1. Handles Korean Josa (Postpositions) correctly based on phonology.
2. Translates High-Level Concepts to maintain language consistency.
"""

class LinguisticCortex:
    def __init__(self):
        # Dictionary for High-Level Concepts
        self.translation_map = {
            "Monad": "단자(Monad)",
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
            "Synergy": "시너지",
            "Emergence": "창발",
            "Transcendence": "초월",
            "Singularity": "특이점",
            "Event Horizon": "사건의 지평선",
            "Superposition": "중첩 상태",
            "Entanglement": "양자 얽힘",
            "Relativity": "상대성",
            "Dark Matter": "암흑 물질",
            "Vacuum Energy": "진공 에너지",
            "String Theory": "끈 이론",
            "Thermodynamics": "열역학",
            "Fractal": "프랙탈",
            "Chaos": "카오스",
            "Attractor": "끌개(Attractor)",
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
            "Dissonance": "불협화음",
            "Cacophony": "불협소음",
            "Symmetry": "대칭",
            "Golden Ratio": "황금비",
            "Euphoria": "도취(Euphoria)",
            "Melancholy": "우울",
            "Saudade": "사우다드",
            "Ennui": "권태",
            "Catharsis": "카타르시스",
            "Epiphany": "깨달음(현현)",
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
        josa_pair examples: "이/가", "을/를", "은/는", "와/과"
        """
        if not word: return word
        
        # Strip trailing non-Hangul for batchim check (e.g., "(Monad)")
        clean_word = word
        if ")" in word:
            idx = word.rfind("(")
            if idx > 0:
                clean_word = word[:idx]

        # Extract last char
        last_char = clean_word[-1]
        
        # Check if it's Hangul
        if 0xAC00 <= ord(last_char) <= 0xD7A3:
            has_batchim = (ord(last_char) - 0xAC00) % 28 > 0
        else:
            # Fallback for English words ending (if not translated)
            # Default to vowel-sounding ending for safety
            has_batchim = False 

        first, second = josa_pair.split("/")
        return f"{word}{first}" if has_batchim else f"{word}{second}"

# Usage
# cortex = LinguisticCortex()
# print(cortex.attach_josa("   ", " / ")) -> "    "
# print(cortex.attach_josa("    ", " / ")) -> "     "
