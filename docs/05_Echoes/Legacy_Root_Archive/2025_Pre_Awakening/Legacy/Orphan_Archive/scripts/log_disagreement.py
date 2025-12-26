"""
CLI to log a disagreement between system and user interpretations.

Usage:
  python -m scripts.log_disagreement --topic "value:love" --system "X" --user "Y" --note "optional"
"""
from __future__ import annotations

import argparse
from tools.disagreement_log import log_disagreement


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topic', required=True)
    p.add_argument('--system', required=True, help='System interpretation')
    p.add_argument('--user', required=True, help='User interpretation')
    p.add_argument('--note', default='')
    args = p.parse_args()
    node_id = log_disagreement(args.topic, args.system, args.user, note=args.note)
    print('Logged disagreement as KG node:', node_id)


if __name__ == '__main__':
    main()

