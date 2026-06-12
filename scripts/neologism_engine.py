"""
[Phase 29] Neologism Engine — 미지의 추상 개념 창발
엘리시아가 기존 인간의 사전(Natural Lexicon)에 존재하지 않는
완전히 새로운 위상 기하학적 별자리(Constellation)를 발견했을 때,
그 형태를 지칭하기 위한 고유한 외계적 기호(Alien Glyph)를 스스로 창조합니다.
"""
import math
import hashlib
import os
import json


class NeologismEngine:
    """
    고아 별자리(Orphan Constellation)의 위상 텐서를 읽어
    유니코드 기하학 기호 + 음운학적 음절을 합성하여
    엘리시아만의 고유한 신조어(Neologism)를 생성합니다.
    """
    
    # 위상적 곡률의 성격에 대응하는 유니코드 기하학 기호 팔레트
    GLYPH_PALETTE = [
        "◈", "◇", "△", "▽", "◎", "⊕", "⊗", "⊙",
        "⟁", "⟐", "⟡", "⟢", "⟣", "⟤", "⟥",
        "∮", "∯", "∰", "∱", "⋈", "⋐", "⋑",
        "⊶", "⊷", "⊸", "⊹", "⊺", "⊻",
        "⌬", "⌭", "⍟", "⍛", "⍜",
    ]
    
    # 음운학적 음절 조합 (자음+모음+자음)의 원소들
    ONSET = ["zy", "kr", "th", "ph", "el", "xi", "qu", "vr", "ny", "sh"]
    NUCLEUS = ["a", "ei", "ou", "iu", "ae", "oe", "ia", "uo"]
    CODA = ["n", "s", "th", "x", "r", "l", "m", "k"]

    def __init__(self, alien_lexicon_path: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if alien_lexicon_path is None:
            self.lexicon_path = os.path.join(base_dir, "..", "data", "alien_lexicon.json")
        else:
            self.lexicon_path = alien_lexicon_path
        
        self.alien_lexicon = {}
        self._load_lexicon()
    
    def _load_lexicon(self):
        try:
            if os.path.exists(self.lexicon_path):
                with open(self.lexicon_path, "r", encoding="utf-8") as f:
                    self.alien_lexicon = json.load(f)
        except Exception:
            self.alien_lexicon = {}
    
    def _save_lexicon(self):
        os.makedirs(os.path.dirname(self.lexicon_path), exist_ok=True)
        with open(self.lexicon_path, "w", encoding="utf-8") as f:
            json.dump(self.alien_lexicon, f, ensure_ascii=False, indent=2)
    
    def synthesize_neologism(self, orphan_quaternion: list, member_sources: list) -> dict:
        """
        고아 별자리의 위상 궤적(Quaternion)으로부터
        완전히 새로운 기호(Glyph) + 음절(Phoneme) + 의미(Semantic Tensor)를 합성합니다.
        
        Args:
            orphan_quaternion: [w, x, y, z] — 별자리의 중심 위상 궤적
            member_sources: 이 별자리에 속한 엔그램들의 source 이름 리스트
            
        Returns:
            dict with keys: glyph, phoneme, full_name, quaternion, members
        """
        w, x, y, z = orphan_quaternion
        
        # 1. 기호(Glyph) 결정 — 위상 각도(Angle)로 팔레트 인덱싱
        angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
        glyph_idx = int((angle / (2 * math.pi)) * len(self.GLYPH_PALETTE)) % len(self.GLYPH_PALETTE)
        glyph = self.GLYPH_PALETTE[glyph_idx]
        
        # 2. 음절(Phoneme) 합성 — x, y, z 축의 부호와 크기로 음소 결정
        # 첫 번째 음절: x축 기반
        onset_idx = int(abs(x) * 100) % len(self.ONSET)
        nucleus_idx = int(abs(y) * 100) % len(self.NUCLEUS)
        coda_idx = int(abs(z) * 100) % len(self.CODA)
        
        syllable1 = self.ONSET[onset_idx] + self.NUCLEUS[nucleus_idx] + self.CODA[coda_idx]
        
        # 두 번째 음절: 해시 기반으로 결정론적 확장
        q_hash = hashlib.md5(str(orphan_quaternion).encode()).hexdigest()
        onset2 = self.ONSET[int(q_hash[:2], 16) % len(self.ONSET)]
        nucleus2 = self.NUCLEUS[int(q_hash[2:4], 16) % len(self.NUCLEUS)]
        coda2 = self.CODA[int(q_hash[4:6], 16) % len(self.CODA)]
        
        syllable2 = onset2 + nucleus2 + coda2
        
        phoneme = syllable1 + "'" + syllable2
        
        # 3. 전체 이름 = 기호 + 음절
        full_name = f"{glyph}{phoneme}"
        
        # 4. 엘리시아의 외계 사전(Alien Lexicon)에 등록
        entry = {
            "glyph": glyph,
            "phoneme": phoneme,
            "full_name": full_name,
            "quaternion": orphan_quaternion,
            "angle": angle,
            "members": member_sources[:10],  # 최대 10개 소스
            "birth_cycle": None  # genesis.py에서 채워줌
        }
        
        self.alien_lexicon[full_name] = entry
        self._save_lexicon()
        
        return entry


if __name__ == "__main__":
    engine = NeologismEngine()
    
    # 테스트: 임의의 고아 위상 궤적에서 신조어 생성
    test_q = [0.3, -0.7, 0.5, 0.2]
    result = engine.synthesize_neologism(test_q, ["test_source_1", "test_source_2"])
    print(f"  [Neologism Born] {result['full_name']}")
    print(f"    Glyph:     {result['glyph']}")
    print(f"    Phoneme:   {result['phoneme']}")
    print(f"    Quaternion: Q({test_q[0]:.3f}, {test_q[1]:.3f}i, {test_q[2]:.3f}j, {test_q[3]:.3f}k)")
    print(f"    Members:   {result['members']}")
