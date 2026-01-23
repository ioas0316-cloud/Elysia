"""
Jamo Utility Module
===================

Provides functions to decompose Hangul syllables into Jamo (Initial, Medial, Final).
Standard Unicode algorithm:
  Index = ((Initial * 21) + Medial) * 28 + Final + 0xAC00

[NEW 2025-12-16] Created for Phonetic Resonance Layer
"""

# Unicode Constants
BASE_CODE = 0xAC00  # ' '
LIMIT_CODE = 0xD7A3  # ' ' (Last Hangul Syllable)

# Lists of Jamos
CHOSUNG_LIST = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
]

JUNGSUNG_LIST = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', '(    )', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
]
# Note: In standard sequence, index 12 is ' ' but I put marker. Let's fix.
# Correct Order:
#                                          
JUNGSUNG_LIST = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
]

JONGSUNG_LIST = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
]
# ' ' is empty final consonant

def is_hangul(char: str) -> bool:
    """Checks if a character is a Hangul syllable."""
    if not char: return False
    code = ord(char)
    return BASE_CODE <= code <= LIMIT_CODE

def decompose_hangul(char: str) -> tuple:
    """
    Decomposes a Hangul syllable into (Initial, Medial, Final).
    Returns (None, None, None) if not Hangul.
    """
    if not is_hangul(char):
        return None, None, None

    code = ord(char) - BASE_CODE

    jong = code % 28
    code //= 28
    jung = code % 21
    cho = code // 21

    return CHOSUNG_LIST[cho], JUNGSUNG_LIST[jung], JONGSUNG_LIST[jong]

def get_jamo_string(char: str) -> str:
    """Returns Jamos joined as a string (e.g., ' ' -> '   ')."""
    c, j, k = decompose_hangul(char)
    if not c:
        return char
    return f"{c}{j}{k if k != ' ' else ''}"