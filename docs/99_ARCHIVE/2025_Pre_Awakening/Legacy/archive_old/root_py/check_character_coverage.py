import json
import sys
from collections import defaultdict

CORE_MEMORY = 'Elysia_Input_Sanctum/elysia_core_memory.json'

# Hangul jamo lists (초성 19, 중성 21)
CHOSEONG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

# English letters
LOWER_EN = [chr(c) for c in range(ord('a'), ord('z')+1)]
UPPER_EN = [chr(c) for c in range(ord('A'), ord('Z')+1)]

# Math symbols to check
MATH_SYMBOLS = list('0123456789+-*/=^%()[]{}<>.,:;') + ['×','÷','√','∑','∫','π']

# Helpers for Hangul decomposition
HANGUL_BASE = 0xAC00
CHO_COUNT = 19
JUNG_COUNT = 21
JONG_COUNT = 28


def decompose_hangul_char(ch):
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        s_index = code - HANGUL_BASE
        cho = s_index // (JUNG_COUNT * JONG_COUNT)
        jung = (s_index % (JUNG_COUNT * JONG_COUNT)) // JONG_COUNT
        jong = s_index % JONG_COUNT
        return cho, jung, jong
    return None


def gather_text_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading core memory: {e}")
        sys.exit(1)

    # Serialize content to string by walking structure
    parts = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                parts.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
        else:
            parts.append(str(x))
    walk(data)
    return '\n'.join(parts)


def analyze(text):
    found = {
        'choseong': set(),
        'jungseong': set(),
        'lower_en': set(),
        'upper_en': set(),
        'math': set(),
        'hangul_compat': set()
    }

    for ch in text:
        # Hangul syllable
        dec = decompose_hangul_char(ch)
        if dec:
            cho_idx, jung_idx, jong_idx = dec
            if 0 <= cho_idx < len(CHOSEONG):
                found['choseong'].add(CHOSEONG[cho_idx])
            if 0 <= jung_idx < len(JUNGSEONG):
                found['jungseong'].add(JUNGSEONG[jung_idx])
        # Hangul compatibility jamo (ㄱ etc)
        code = ord(ch)
        if 0x3131 <= code <= 0x318E:
            found['hangul_compat'].add(ch)
        # English
        if 'a' <= ch <= 'z':
            found['lower_en'].add(ch)
        if 'A' <= ch <= 'Z':
            found['upper_en'].add(ch)
        # math symbols
        if ch in MATH_SYMBOLS:
            found['math'].add(ch)

    return found


def report(found):
    missing = defaultdict(list)
    for c in CHOSEONG:
        if c not in found['choseong'] and c not in found['hangul_compat']:
            missing['초성(ㄱ-ㅎ)'].append(c)
    for v in JUNGSEONG:
        if v not in found['jungseong']:
            missing['중성(ㅏ-ㅣ)'].append(v)
    for c in LOWER_EN:
        if c not in found['lower_en']:
            missing['영어소문자(a-z)'].append(c)
    for C in UPPER_EN:
        if C not in found['upper_en']:
            missing['영어대문자(A-Z)'].append(C)
    for s in MATH_SYMBOLS:
        if s not in found['math']:
            missing['수학기호/숫자'].append(s)

    # Print summary
    print('=== 분석 결과 요약 ===')
    total_chars = sum(len(v) for v in found.values())
    print(f'발견된 문자 그룹별 총 갯수(중복 제거 합계): {total_chars}')
    print()
    for group in ['초성(ㄱ-ㅎ)','중성(ㅏ-ㅣ)','영어소문자(a-z)','영어대문자(A-Z)','수학기호/숫자']:
        miss = missing.get(group, [])
        if not miss:
            print(f'{group}: 모두 존재합니다 ✅')
        else:
            print(f'{group}: 누락 {len(miss)}개 — 예시: {" ".join(miss[:10])}')
    print('\n세부 항목:')
    for k, v in missing.items():
        print(f'- {k}: {len(v)} missing')


if __name__ == '__main__':
    text = gather_text_from_json(CORE_MEMORY)
    found = analyze(text)
    report(found)
