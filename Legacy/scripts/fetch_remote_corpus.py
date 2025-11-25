import argparse
import os
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:  # fallback to urllib
    requests = None
    import urllib.request

SOURCE_LIST: List[Tuple[str, str]] = [
    ("https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/README.md", "ml_for_beginners_ml_intro.md"),
    ("https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md", "system_design_primer.md"),
    ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "tiny_shakespeare.txt"),
    ("https://raw.githubusercontent.com/sebastianruder/NLP-progress/master/README.md", "nlp_progress.md"),
    ("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/latest/owid-covid-latest.json", "covid_data.json"),
    ("https://raw.githubusercontent.com/ryanburgert/corpus/master/handbook.md", "developer_handbook.md"),
    ("https://raw.githubusercontent.com/ageron/handson-ml/master/README.md", "hands_on_ml_readme.md"),
]


def download_with_requests(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def download_with_urllib(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8")


def fetch_sources(feed_dir: Path, dry_run: bool = False) -> List[Path]:
    downloaded = []
    feed_dir.mkdir(parents=True, exist_ok=True)
    for url, name in SOURCE_LIST:
        dest = feed_dir / name
        print(f"Fetching {url} -> {dest.name}")
        if dry_run:
            downloaded.append(dest)
            continue
        try:
            if requests:
                content = download_with_requests(url)
            else:
                content = download_with_urllib(url)
        except Exception as exc:
            print(f"  ⚠️ Fetch failed: {exc}")
            continue
        dest.write_text(content, encoding="utf-8")
        downloaded.append(dest)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Fetch remote corpus files into MetaAgent feed.")
    parser.add_argument("--feed-dir", type=str, default="data/corpus_feed", help="Drop zone for new material.")
    parser.add_argument("--dry-run", action="store_true", help="Only print which files will be fetched.")
    args = parser.parse_args()

    print("Starting remote corpus fetch...")
    path = Path(args.feed_dir)
    files = fetch_sources(path, dry_run=args.dry_run)
    print(f"Fetched {len(files)} asset(s) {'(dry run)' if args.dry_run else 'and stored them locally.'}")


if __name__ == "__main__":
    main()
