import argparse
from tools.text_preprocessor import segment_text, extract_content_tokens


def main():
    ap = argparse.ArgumentParser(description="Debug Korean-aware tokenization and segmentation")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    segs = segment_text(args.text)
    print("[segments]")
    print(" ".join([f"{f}/{t}" for f, t in segs]))
    print("[content tokens]")
    print(extract_content_tokens(args.text))


if __name__ == "__main__":
    main()

