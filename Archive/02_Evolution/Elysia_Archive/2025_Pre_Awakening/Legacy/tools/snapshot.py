import os
import json
from datetime import datetime


def snapshot():
    manifest = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'versions': {},
        'lens': {},
        'kg_summary': {},
        'notes': 'Minimal run snapshot for reproducibility',
    }
    try:
        # Load lens profile if exists
        if os.path.exists('lens_profile.json'):
            with open('lens_profile.json', 'r', encoding='utf-8') as f:
                manifest['lens'] = json.load(f)
    except Exception:
        pass

    try:
        # KG summary
        if os.path.exists('data/kg_with_embeddings.json'):
            with open('data/kg_with_embeddings.json', 'r', encoding='utf-8') as f:
                kg = json.load(f)
            manifest['kg_summary'] = {
                'nodes': len(kg.get('nodes', [])),
                'edges': len(kg.get('edges', [])),
            }
    except Exception:
        pass

    os.makedirs('data/snapshots', exist_ok=True)
    path = os.path.join('data/snapshots', 'run_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('Snapshot saved to', path)


if __name__ == '__main__':
    snapshot()

