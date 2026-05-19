"""
Fantasy Lexicon Injector
========================

Injects professional fantasy vocabulary into the Style Genome.
"""

import sys
import os
import json
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Evolution.Learning.Learning.language_learner import LanguageLearner

def inject_fantasy_lexicon():
    print("✨ Injecting Professional Fantasy Lexicon...")
    
    learner = LanguageLearner()
    
    # 1. Advanced Vocabulary (Sharp/War)
    sharp_vocab = [
        "절망을 베어내는", "칠흑 같은", "심연의 아가리", "피로 물든", "강철의 찬가", 
        "멸망의 전조", "타오르는 흉터", "서리 같은 살기", "무너지는 하늘"
    ]
    for w in sharp_vocab: learner._add_vocab("Sharp", w)
    
    # 2. Advanced Vocabulary (Round/Magic)
    round_vocab = [
        "영겁의 시간", "은하수의 궤적", "마력의 파동", "세계수의 속삭임", 
        "차원의 틈새", "오로라의 장막", "수정 같은 기억", "태고의 숨결"
    ]
    for w in round_vocab: learner._add_vocab("Round", w)
    
    # 3. Advanced Templates (Narrative)
    templates = [
        "그것은 단순한 {0}이 아니었다. 그것은 {1}였다.",
        "마치 {0}처럼, {1}은 조용히 찾아왔다.",
        "{0}의 끝에서, 우리는 비로소 {1}을 마주했다."
    ]
    # We cheat a bit and add them to 'Definition' or others
    for t in templates:
        learner._add_template("Definition", t)
        
    learner.save_genome()
    print("✅ Lexicon Injection Complete.")

if __name__ == "__main__":
    inject_fantasy_lexicon()
