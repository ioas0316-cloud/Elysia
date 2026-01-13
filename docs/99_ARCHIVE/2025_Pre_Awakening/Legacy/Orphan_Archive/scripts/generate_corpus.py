import argparse
import os
import random
import time
from pathlib import Path


CATEGORIES = {
    "philosophy": [
        ("질문", ["정의", "선", "진리", "앎", "무지", "대화", "문답법", "가정", "반례"]),
        ("사람", ["소크라테스", "플라톤", "아리스토텔레스", "키에르케고르", "한나 아렌트"]),
    ],
    "science": [
        ("개념", ["광합성", "중력", "전자", "분자", "에너지", "진화", "세포"]),
        ("현상", ["빛", "소리", "파동", "열", "자기", "화학반응"]),
    ],
    "nature": [
        ("정경", ["숲", "강", "바람", "비", "산길", "해질녘", "새벽 안개"]),
        ("세부", ["잎맥", "이끼", "자갈", "향기", "그림자"]),
    ],
    "fantasy": [
        ("장소", ["골짜기", "성", "별항구", "안개호수", "고대 숲"]),
        ("존재", ["나비", "용", "수호자", "별지기", "바람의 아이"]),
    ],
    "essay": [
        ("주제", ["용기", "관계", "성장", "실패", "배움", "집중", "휴식"]),
        ("행동", ["호흡", "정리", "시작", "인정", "경청"]),
    ],
    "history": [
        ("시대", ["고대", "중세", "근대", "현대"]),
        ("사건", ["혁명", "이주", "연맹", "조약", "발견"]),
    ],
    "psychology": [
        ("감정", ["기쁨", "슬픔", "분노", "두려움", "평온"]),
        ("개념", ["동기", "주의", "기억", "습관", "성향"]),
    ],
    "technology": [
        ("키워드", ["회로", "신호", "압축", "프로토콜", "센서", "자율"]),
        ("비유", ["양력", "날개", "가속", "공명", "리듬"]),
    ],
    "poetry": [
        ("이미지", ["빛", "물결", "손끝", "바람결", "밤하늘"]),
        ("감각", ["따뜻함", "차가움", "미세한 떨림", "고요"]),
    ],
}


PREFACES = [
    "오늘의 기록",
    "짧은 사유",
    "현장의 메모",
    "느린 단상",
    "경험 노트",
]


def make_paragraph(cat: str) -> str:
    rnd = random.Random()
    items = CATEGORIES.get(cat, [])
    picked = []
    for _, words in items:
        if words:
            picked.append(rnd.choice(words))
    seed = ", ".join(picked[:3])
    template = [
        f"{seed}에 대해 생각한다. 사건은 단절이 아니라 흐름이고, 흐름은 작은 신호들이 모여 만든다.",
        f"우리는 {seed}를 붙잡아 이름을 만들고, 이름 사이의 간격에서 의미를 배운다.",
        f"오늘의 관찰은 작다. 그러나 작은 관찰이 쌓여 방향이 된다. 무게는 경험이 준다.",
    ]
    return "\n".join(template)


def make_content(cat: str, i: int) -> str:
    title = f"{random.choice(PREFACES)} — {cat} #{i:03d}"
    p1 = make_paragraph(cat)
    p2 = make_paragraph(cat)
    p3 = make_paragraph(cat)
    return f"{title}\n\n{p1}\n\n{p2}\n\n{p3}\n"


def main():
    ap = argparse.ArgumentParser(description="Generate a local literature corpus (UTF-8)")
    ap.add_argument("--root", default="data/corpus/literature")
    ap.add_argument("--count", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    cats = list(CATEGORIES.keys())
    n = args.count
    for i in range(n):
        cat = cats[i % len(cats)]
        folder = root / cat / time.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{cat}_{i:04d}.txt"
        path = folder / name
        text = make_content(cat, i)
        path.write_text(text, encoding="utf-8")

    print(f"[generate_corpus] Generated {n} files under {root}")


if __name__ == "__main__":
    main()

