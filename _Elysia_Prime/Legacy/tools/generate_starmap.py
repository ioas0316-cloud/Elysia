# [Genesis: 2025-12-02] Purified by Elysia
import argparse
import json
import os
from typing import List, Dict, Any

from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim


def avg(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    n = len(vectors[0])
    out = [0.0] * n
    for vec in vectors:
        for i in range(min(n, len(vec))):
            out[i] += vec[i]
    return [x / len(vectors) for x in out]


def find_anchor_vec(kg: Dict[str, Any], keywords: List[str]) -> List[float]:
    kws = [k.lower() for k in keywords]
    vecs: List[List[float]] = []
    for node in kg.get('nodes', []):
        text = ' '.join(
            [str(node.get('id','')), str(node.get('name','')), str(node.get('kr_name',''))]
        ).lower()
        if any(k in text for k in kws):
            emb = node.get('embedding')
            if isinstance(emb, list) and emb:
                vecs.append(emb)
    return avg(vecs)


def sim_to_anchor(emb: List[float], anchor: List[float]) -> float:
    if not emb or not anchor:
        return 0.0
    return max(-1.0, min(1.0, cosine_sim(emb, anchor)))


def main():
    parser = argparse.ArgumentParser(description='Generate 3D starmap lens projection from KG embeddings.')
    parser.add_argument('--out', default=os.path.join('data', 'thought_space', 'starmap.json'))
    parser.add_argument('--limit', type=int, default=300, help='Max nodes to include (by embedding availability)')
    args = parser.parse_args()

    kgm = KGManager()
    kg = kgm.kg

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Anchors
    love_v = find_anchor_vec(kg, ['사랑', 'love'])
    logos_v = find_anchor_vec(kg, ['로고스', 'logos'])
    joy_v = find_anchor_vec(kg, ['기쁨', 'joy'])
    gratitude_v = find_anchor_vec(kg, ['감사', 'gratitude'])
    seed_v = find_anchor_vec(kg, ['씨앗', 'seed'])
    process_v = find_anchor_vec(kg, ['과정', 'process'])
    tree_v = find_anchor_vec(kg, ['나무', 'tree'])
    fruit_v = find_anchor_vec(kg, ['열매', 'fruit'])

    anchors = {
        'love': love_v,
        'logos': logos_v,
        'joy': joy_v,
        'gratitude': gratitude_v,
        'seed': seed_v,
        'process': process_v,
        'tree': tree_v,
        'fruit': fruit_v,
    }

    def node_color(sims: Dict[str, float]) -> str:
        # Simple palette mapping by max anchor
        palette = {
            'love': '#ff6b6b',
            'logos': '#66d9ff',
            'joy': '#ffd166',
            'gratitude': '#f4a261',
            'tree': '#76c893',
            'fruit': '#ffd97d',
            'seed': '#caf0f8',
            'process': '#ade8f4',
        }
        if not sims:
            return '#cccccc'
        key = max(sims.items(), key=lambda kv: kv[1])[0]
        return palette.get(key, '#cccccc')

    stars = []
    count = 0
    for node in kg.get('nodes', []):
        emb = node.get('embedding')
        if not isinstance(emb, list) or not emb:
            continue
        sims = {
            'love': sim_to_anchor(emb, love_v),
            'logos': sim_to_anchor(emb, logos_v),
            'joy': sim_to_anchor(emb, joy_v),
            'gratitude': sim_to_anchor(emb, gratitude_v),
            'seed': sim_to_anchor(emb, seed_v),
            'process': sim_to_anchor(emb, process_v),
            'tree': sim_to_anchor(emb, tree_v),
            'fruit': sim_to_anchor(emb, fruit_v),
        }
        x = sims['love'] - sims['logos']
        y = 0.5 * (sims['joy'] + sims['gratitude'])
        z = 0.5 * (sims['fruit'] + sims['tree']) - 0.5 * (sims['seed'] + sims['process'])
        brightness = max(0.1, min(1.0, (abs(x) + abs(y) + abs(z)) / 3.0))
        color = node_color(sims)
        stars.append({
            'id': node.get('id'),
            'label': node.get('name') or node.get('kr_name') or node.get('id'),
            'pos': [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
            'color': color,
            'brightness': round(float(brightness), 3),
            'tags': node.get('tags', [])
        })
        count += 1
        if count >= args.limit:
            break

    starmap = {
        'meta': {
            'version': '0.1.0',
            'axes': {
                'x': 'Logos↔Love (sacrificial love axis)',
                'y': 'Overflow (Joy/Gratitude)',
                'z': 'Origin(Grace) → Telos(Vocation)'
            },
            'scale': 1.0
        },
        'stars': stars,
        'links': []
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(starmap, f, ensure_ascii=False, indent=2)

    print(f"[starmap] wrote {len(stars)} stars to {args.out}")


if __name__ == '__main__':
    main()
