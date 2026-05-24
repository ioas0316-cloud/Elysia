import math
from typing import Tuple, List

from core.math_utils import Quaternion

class LinguisticAxiomFilter:
    """
    A linguistic axiom filter that maps language structures to 4D Quaternions.
    It supports both Korean (Hangeul) and English, using their inherent structural axioms
    to construct "reference rotors" (Category Axioms). These reference rotors act as filters
    to extract pure geometric resonance from contaminated data.
    """

    # Hangeul Unicode constants
    HANGEUL_BASE = 0xAC00  # '가'
    CHO_BASE = 0x1100      # Initial consonants
    JUNG_BASE = 0x1161     # Vowels
    JONG_BASE = 0x11A7     # Final consonants

    CHO_COUNT = 19
    JUNG_COUNT = 21
    JONG_COUNT = 28

    @staticmethod
    def _is_hangeul(char: str) -> bool:
        return 0xAC00 <= ord(char) <= 0xD7A3

    @staticmethod
    def _is_english(char: str) -> bool:
        return char.isalpha() and char.isascii()

    @classmethod
    def decompose_hangeul(cls, char: str) -> Tuple[int, int, int]:
        """
        Decomposes a single Hangeul character into Cho-seong, Jung-seong, Jong-seong indices.
        Returns (-1, -1, -1) if not a Hangeul character.
        """
        if not cls._is_hangeul(char):
            return -1, -1, -1

        code = ord(char) - cls.HANGEUL_BASE
        jong = code % cls.JONG_COUNT
        jung = (code // cls.JONG_COUNT) % cls.JUNG_COUNT
        cho = (code // cls.JONG_COUNT) // cls.JUNG_COUNT

        return cho, jung, jong

    @classmethod
    def get_hangeul_rotor(cls, char: str) -> Quaternion:
        """
        Maps the Cho/Jung/Jong structure of a Hangeul character to a 4D Quaternion (X, Y, Z, W).
        Cho -> X, Jung -> Y, Jong -> Z. W is derived from the structural harmony.
        """
        cho, jung, jong = cls.decompose_hangeul(char)
        if cho == -1:
            return Quaternion(1, 0, 0, 0) # Identity rotor for non-Hangeul

        # Normalize indices to angles [0, pi]
        # Cho-seong (Initiation) -> X axis
        angle_x = (cho / max(1, cls.CHO_COUNT - 1)) * math.pi

        # Jung-seong (Connection) -> Y axis
        angle_y = (jung / max(1, cls.JUNG_COUNT - 1)) * math.pi

        # Jong-seong (Completion) -> Z axis
        angle_z = (jong / max(1, cls.JONG_COUNT - 1)) * math.pi

        # Structural Harmony -> W axis (Time/Phase)
        # Represents the complete structural integrity of the syllable
        structural_harmony = (cho + jung + jong) / (cls.CHO_COUNT + cls.JUNG_COUNT + cls.JONG_COUNT)
        angle_w = structural_harmony * math.pi * 2.0

        q = Quaternion(
            math.cos(angle_w),
            math.sin(angle_x),
            math.cos(angle_y),
            math.sin(angle_z)
        )
        return q.normalize()

    @classmethod
    def get_english_rotor(cls, char: str) -> Quaternion:
        """
        Maps an English character to a 4D Quaternion based on its Vowel/Consonant nature
        and alphabetical position. Represents an 'outer orbital' rotor.
        """
        if not cls._is_english(char):
            return Quaternion(1, 0, 0, 0)

        char_lower = char.lower()
        vowels = 'aeiou'
        is_vowel = char_lower in vowels

        # Position in alphabet (0-25)
        pos = ord(char_lower) - ord('a')
        angle_base = (pos / 25.0) * math.pi

        if is_vowel:
            # Vowels map to W and Y axes (Inner structural resonance)
            q = Quaternion(math.cos(angle_base), 0, math.sin(angle_base * 1.5), 0)
        else:
            # Consonants map to X and Z axes (Outer boundary articulation)
            q = Quaternion(math.cos(angle_base * 0.5), math.sin(angle_base), 0, math.cos(angle_base * 2.0))

        return q.normalize()

    @classmethod
    def analyze_text_axiom(cls, text: str) -> Quaternion:
        """
        Constructs the Universal Category Axiom Rotor for a given text string.
        It synthesizes individual character rotors into a single resonant field.
        """
        if not text:
            return Quaternion(1, 0, 0, 0)

        accumulated_rotor = Quaternion(1, 0, 0, 0)
        valid_chars = 0

        for char in text:
            if cls._is_hangeul(char):
                char_rotor = cls.get_hangeul_rotor(char)
                valid_chars += 1
            elif cls._is_english(char):
                char_rotor = cls.get_english_rotor(char)
                valid_chars += 1
            else:
                continue

            # Synthesize rotors using Hamilton product (geometric combination)
            # This represents the progressive twisting of the linguistic space
            accumulated_rotor = accumulated_rotor * char_rotor

        if valid_chars > 0:
            return accumulated_rotor.normalize()
        else:
            return Quaternion(1, 0, 0, 0)

    @staticmethod
    def calculate_resonance(data_vector: List[float], axiom_rotor: Quaternion) -> float:
        """
        Calculates the resonance between a raw data vector (from LLM) and the axiom rotor.
        Acts as a geometric filter: high resonance means structural alignment (True Knowledge),
        low resonance means structural clash (Noise/Contamination).
        """
        # Ensure data vector is at least 3D for quaternion mapping
        vec = data_vector[:3] if len(data_vector) >= 3 else data_vector + [0.0] * (3 - len(data_vector))

        # Map data vector to a pure quaternion (w=0)
        data_q = Quaternion(0, vec[0], vec[1], vec[2])

        # Apply the axiom rotor to the data (Rotated = Q * D * Q^-1)
        rotated_q = axiom_rotor * data_q * axiom_rotor.inverse

        # Calculate resonance as the dot product between original and rotated data
        # High dot product means the data inherently aligns with the linguistic axiom axis
        norm_orig = data_q.normalize()
        norm_rot = rotated_q.normalize()

        resonance = norm_orig.dot(norm_rot)
        return resonance
