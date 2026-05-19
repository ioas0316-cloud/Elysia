#!/usr/bin/env python3
# version: web_init_v3
"""웹·모바일 프로젝트 자동 초기화 — 5개 템플릿 중 선택.

config:
  TEMPLATE — vite-react / nextjs / astro / expo / vanilla
  PROJECT_NAME — 프로젝트 폴더 이름 (영문·하이픈, 공백 X)
  OUTPUT_DIR — 어디에 만들지 (비우면 ~/connect-ai-projects/)

각 템플릿은 검증된 공식 명령어로 셋업. 5분 안에 dev server 띄울 수 있는 상태로.
"""
import os, sys, json, subprocess, shutil


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "web_init.json")


def _log(msg, kind="info"):
    prefix = {"info": "💻", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "▸"}.get(kind, "•")
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _load():
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(c):
    try:
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _check_cmd(cmd):
    """Check if a CLI tool exists."""
    return shutil.which(cmd) is not None


def _run(cmd, cwd=None, capture=True):
    """Run shell command, stream stderr live but capture stdout for return."""
    _log(f"$ {cmd}", "step")
    if capture:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=600)
        if r.stdout:
            for line in r.stdout.splitlines()[:20]:
                print(f"  {line}")
        if r.stderr and r.returncode != 0:
            for line in r.stderr.splitlines()[-10:]:
                _log(line, "warn")
        return r.returncode == 0, r.stdout
    else:
        return subprocess.run(cmd, shell=True, cwd=cwd, timeout=600).returncode == 0, ""


def _scaffold_vite_react(name, parent):
    """Vite + React + TS + Tailwind v4 (Vite 플러그인 방식).
    v2: tailwindcss init 명령이 v4에서 제거됨 → @tailwindcss/vite 플러그인 사용 + 설정 파일 직접 쓰기.
    각 단계마다 (cmd, cwd, critical) — critical=False면 실패해도 프로젝트 살림."""
    target = os.path.join(parent, name)
    return [
        ("create", f"npm create vite@latest {name} -- --template react-ts", parent, True),
        ("install", "npm install", target, True),
        ("tailwind-pkg", "npm install tailwindcss@^4 @tailwindcss/vite@^4", target, False),
        ("tailwind-config", _write_vite_tailwind_config, target, False),
    ]


def _write_vite_tailwind_config(target):
    """Tailwind v4 설정 파일 직접 작성 (init 명령 의존 없음)."""
    # vite.config.ts: 기본 파일에 tailwindcss 플러그인 추가
    vite_cfg = os.path.join(target, "vite.config.ts")
    if os.path.exists(vite_cfg):
        try:
            with open(vite_cfg, "r", encoding="utf-8") as f:
                content = f.read()
            if "tailwindcss" not in content:
                # import 추가
                content = "import tailwindcss from '@tailwindcss/vite'\n" + content
                # plugins: [react()] → plugins: [react(), tailwindcss()]
                content = content.replace("plugins: [react()]", "plugins: [react(), tailwindcss()]")
                with open(vite_cfg, "w", encoding="utf-8") as f:
                    f.write(content)
        except Exception:
            pass

    # src/index.css: Tailwind v4 import 한 줄
    css_path = os.path.join(target, "src", "index.css")
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                cur = f.read()
            if '@import "tailwindcss"' not in cur:
                with open(css_path, "w", encoding="utf-8") as f:
                    f.write('@import "tailwindcss";\n\n' + cur)
        except Exception:
            pass

    return True


TEMPLATES = {
    "vite-react": {
        "label": "⚡ Vite + React + TypeScript + Tailwind v4",
        "needs": ["node", "npm"],
        "scaffold": _scaffold_vite_react,
        "post": "Tailwind v4 (Vite 플러그인) + index.css 자동 설정",
        "dev_cmd": "npm run dev",
    },
    "nextjs": {
        "label": "▲ Next.js 14 (App Router) + TypeScript + Tailwind",
        "needs": ["node", "npm"],
        "scaffold": lambda name, parent: [
            ("scaffold", f"npx create-next-app@latest {name} --typescript --tailwind --app --src-dir --import-alias '@/*' --no-eslint --use-npm --yes", parent, True),
        ],
        "post": "App Router·Tailwind·src 디렉토리 셋업 완료",
        "dev_cmd": "npm run dev",
    },
    "astro": {
        "label": "🚀 Astro + Tailwind (정적·콘텐츠 사이트)",
        "needs": ["node", "npm"],
        "scaffold": lambda name, parent: [
            ("scaffold", f"npm create astro@latest {name} -- --template minimal --typescript strict --install --git --yes", parent, True),
            ("tailwind", f"npx astro add tailwind --yes", os.path.join(parent, name), False),
        ],
        "post": "Astro + Tailwind",
        "dev_cmd": "npm run dev",
    },
    "expo": {
        "label": "📱 Expo (React Native · iOS/Android/Web 동시)",
        "needs": ["node", "npm"],
        "scaffold": lambda name, parent: [
            ("scaffold", f"npx create-expo-app@latest {name} --template blank-typescript", parent, True),
        ],
        "post": "Expo Go 앱(iOS/Android) 깔고 'npm start' 후 QR 스캔",
        "dev_cmd": "npm start",
    },
    "vanilla": {
        "label": "📄 Vanilla HTML + CSS + JS (프레임워크 없음)",
        "needs": [],
        "scaffold": "VANILLA",  # 특수 케이스 — 직접 파일 생성
        "post": "단일 폴더 + index.html + style.css + script.js + README",
        "dev_cmd": "python3 -m http.server 8000",
    },
}


def _scaffold_vanilla(target_dir, name):
    """프레임워크 없이 정적 사이트 시드."""
    os.makedirs(target_dir, exist_ok=True)
    files = {
        "index.html": f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>{name}</h1>
  <p>Connect AI · 코다리가 만든 사이트</p>
</header>
<main>
  <p>여기에 콘텐츠를 추가하세요.</p>
</main>
<script src="script.js"></script>
</body>
</html>
''',
        "style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #1a1a1a; background: #fafafa; }
header { padding: 60px 24px; text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
header h1 { font-size: 48px; font-weight: 800; margin-bottom: 8px; }
main { max-width: 720px; margin: 40px auto; padding: 0 24px; }
''',
        "script.js": '''// 코다리가 첨부한 스크립트
console.log("✦ Connect AI 사이트 로드 완료");
''',
        "README.md": f'''# {name}

Connect AI 코다리가 셋업한 정적 웹사이트.

## 미리보기
```
python3 -m http.server 8000
```
그 다음 브라우저에서 http://localhost:8000

## 구조
- `index.html` — 메인 페이지
- `style.css` — 스타일
- `script.js` — JS

## 배포
- Vercel: `npx vercel --prod`
- Netlify: `npx netlify deploy --prod`
- Cloudflare Pages: GitHub 연결
''',
    }
    for filename, content in files.items():
        with open(os.path.join(target_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
    return True


def main():
    cfg = _load()
    template = (cfg.get("TEMPLATE") or "").strip().lower() or "vite-react"
    name = (cfg.get("PROJECT_NAME") or "").strip() or "my-app"
    out_dir = (cfg.get("OUTPUT_DIR") or "").strip()

    if template not in TEMPLATES:
        _log(f"알 수 없는 템플릿: {template}", "err")
        _log(f"사용 가능: {', '.join(TEMPLATES.keys())}", "info")
        sys.exit(1)

    # 이름 검증
    import re
    if not re.match(r"^[a-z0-9][a-z0-9_-]*$", name):
        _log(f"프로젝트 이름은 소문자·숫자·하이픈·언더스코어만: {name}", "err")
        sys.exit(1)

    # 출력 위치
    if not out_dir:
        out_dir = os.path.expanduser("~/connect-ai-projects")
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    target = os.path.join(out_dir, name)
    if os.path.exists(target):
        _log(f"이미 존재: {target} — 다른 이름 쓰거나 폴더 지우세요", "err")
        sys.exit(1)

    spec = TEMPLATES[template]
    _log(f"{spec['label']} 셋업 시작 → {target}", "info")

    # 의존성 체크
    for cmd in spec.get("needs", []):
        if not _check_cmd(cmd):
            _log(f"`{cmd}` 명령이 없음. 먼저 Node.js를 설치하세요 (nodejs.org).", "err")
            sys.exit(1)

    # 실행
    if spec["scaffold"] == "VANILLA":
        ok = _scaffold_vanilla(target, name)
        if not ok:
            _log("vanilla 셋업 실패", "err")
            sys.exit(1)
    else:
        steps = spec["scaffold"](name, out_dir)
        warnings = []
        for step in steps:
            # 4-tuple 형식: (label, cmd_or_func, cwd, critical)
            if len(step) != 4:
                _log(f"잘못된 step 형식 (4-tuple 필요): {step}", "err")
                sys.exit(1)
            label, action, cwd, critical = step
            if callable(action):
                # Python 함수 직접 호출 (설정 파일 쓰기 등)
                _log(f"[{label}] 설정 파일 작성 중...", "step")
                try:
                    ok = bool(action(cwd))
                except Exception as e:
                    _log(f"[{label}] 예외: {e}", "warn")
                    ok = False
            else:
                ok, _ = _run(action, cwd=cwd)
            if not ok:
                if critical:
                    _log(f"❌ 핵심 단계 실패: [{label}] — 중단합니다", "err")
                    sys.exit(1)
                else:
                    _log(f"⚠️  부가 단계 실패: [{label}] — 계속 진행합니다", "warn")
                    warnings.append(label)
        if warnings:
            _log(f"부가 단계 {len(warnings)}개 실패 ({', '.join(warnings)}). 프로젝트 자체는 작동합니다.", "warn")
            _log("Tailwind 등은 사용자가 수동 추가 가능. README 참고.", "info")

    _log(f"셋업 완료: {target}", "ok")
    _log(f"다음 단계:", "info")
    _log(f"  cd {target}", "info")
    _log(f"  {spec['dev_cmd']}", "info")
    if spec.get("post"):
        _log(f"  {spec['post']}", "info")

    # 결과 저장
    cfg["LAST_PROJECT"] = target
    cfg["LAST_TEMPLATE"] = template
    cfg["LAST_DEV_CMD"] = spec["dev_cmd"]
    _save(cfg)

    print()
    print(f"PROJECT_PATH={target}")
    print(f"DEV_CMD={spec['dev_cmd']}")


if __name__ == "__main__":
    main()
