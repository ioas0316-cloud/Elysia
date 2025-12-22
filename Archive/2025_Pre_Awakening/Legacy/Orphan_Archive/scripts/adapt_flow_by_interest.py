import json
import os
from pathlib import Path
from collections import Counter
from tools.kg_manager import KGManager


PROFILE_FILE = Path('data/flows/profile.txt')


LABEL_TO_PROFILE = {
    'philosophy': 'philosophy',
    'science': 'science',
    'nature': 'nature',
    'fantasy': 'fantasy',
    'essay': 'essay',
    'history': 'history',
    'psychology': 'psychology',
    'technology': 'technology',
    'poetry': 'poetry',
}


def pick_label(kg) -> str:
    cnt = Counter()
    for n in kg.get('nodes', []):
        if n.get('type') == 'literature_doc':
            lab = (n.get('label') or '').strip()
            if lab:
                cnt[lab] += 1
    if not cnt:
        return ''
    return cnt.most_common(1)[0][0]


def set_profile(name: str) -> str:
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(name, encoding='utf-8')
    return name


def main():
    kg = KGManager().kg
    lab = pick_label(kg)
    if not lab:
        print('[adapt_flow] No literature labels found; keeping current profile.')
        return
    prof = LABEL_TO_PROFILE.get(lab, 'learning')
    set_profile(prof)
    print(f"[adapt_flow] Set profile to '{prof}' based on interest '{lab}'")


if __name__ == '__main__':
    main()

