#!/usr/bin/env python3
# version: lint_test_v1
"""프로젝트 자가 검증 — 타입체크·테스트·린트 자동 실행 + 결과 요약.

코다리가 코드를 만든 직후 이 도구를 호출하면:
  1. package.json 의 scripts 자동 감지 (test/lint/typecheck/build)
  2. 또는 .ts/.tsx 파일 있으면 npx tsc --noEmit
  3. .py 파일 있으면 python -m py_compile <각 파일>
  4. 결과 마크다운 리포트

config:
  PROJECT_PATH — 검증할 프로젝트 (비우면 web_init 마지막 결과)
  STRICT       — 'true' 면 첫 실패에서 멈춤. 기본 false (모두 시도)
"""
import os, sys, json, subprocess, glob


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "lint_test.json")
WEB_INIT_CFG = os.path.join(HERE, "web_init.json")


def _log(msg, kind="info"):
    prefix = {"info": "🧪", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "▸"}.get(kind, "•")
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _load(p):
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run(cmd, cwd, timeout=180):
    _log(f"$ {cmd}", "step")
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + "\n" + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"⏱ Timeout ({timeout}s)"
    except Exception as e:
        return -2, str(e)


def main():
    cfg = _load(CONFIG)
    init_cfg = _load(WEB_INIT_CFG)
    project = (cfg.get("PROJECT_PATH") or "").strip()
    if not project:
        project = (init_cfg.get("LAST_PROJECT") or "").strip()
    if not project:
        _log("PROJECT_PATH 비어있고 web_init 기록도 없음", "err")
        sys.exit(1)
    project = os.path.expanduser(project)
    if not os.path.isdir(project):
        _log(f"폴더 없음: {project}", "err")
        sys.exit(1)
    strict = str(cfg.get("STRICT", "")).lower() in ("true", "1", "yes")
    _log(f"검증 대상: {project}", "info")

    results = []  # (label, code, output)

    # 1) package.json scripts 자동 감지
    pkg_path = os.path.join(project, "package.json")
    if os.path.exists(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            scripts = pkg.get("scripts", {})
            for key in ["typecheck", "lint", "test", "build"]:
                if key in scripts:
                    code, out = _run(f"npm run {key}", cwd=project, timeout=300)
                    results.append((f"npm run {key}", code, out))
                    if strict and code != 0:
                        break
        except Exception as e:
            _log(f"package.json 파싱 실패: {e}", "warn")

    # 2) scripts 없으면 직접 tsc/py_compile
    if not results:
        # TS/TSX
        ts_files = glob.glob(os.path.join(project, "**/*.ts"), recursive=True) + \
                   glob.glob(os.path.join(project, "**/*.tsx"), recursive=True)
        ts_files = [f for f in ts_files if "node_modules" not in f and "dist" not in f]
        if ts_files:
            tsconfig = os.path.join(project, "tsconfig.json")
            if os.path.exists(tsconfig):
                code, out = _run("npx tsc --noEmit", cwd=project, timeout=180)
                results.append(("npx tsc --noEmit", code, out))
        # Python
        py_files = glob.glob(os.path.join(project, "**/*.py"), recursive=True)
        py_files = [f for f in py_files if "venv" not in f and ".venv" not in f and "__pycache__" not in f]
        if py_files:
            errs = []
            for pf in py_files[:30]:  # 30개 cap
                code, out = _run(f"python3 -m py_compile {json.dumps(pf)}", cwd=project, timeout=10)
                if code != 0:
                    errs.append((pf, out.strip()[:120]))
            if errs:
                results.append((f"py_compile ({len(errs)}/{len(py_files)} 실패)", 1, "\n".join(f"{f}: {e}" for f, e in errs[:10])))
            else:
                results.append((f"py_compile {len(py_files)} files", 0, "All OK"))

    # 결과 리포트
    print()
    print(f"# 🧪 검증 결과 — {os.path.basename(project)}")
    print()
    if not results:
        print("⚠️ 실행할 검증 없음 (package.json scripts 없고 .ts/.py 파일도 없음)")
        return
    passed = sum(1 for _, c, _ in results if c == 0)
    print(f"**{passed}/{len(results)} 통과**\n")
    for label, code, out in results:
        icon = "✅" if code == 0 else "❌"
        print(f"## {icon} {label}")
        if code == 0:
            print(f"성공 (exit code 0)")
        else:
            print(f"실패 (exit code {code})")
            print()
            print("```")
            for line in out.strip().split("\n")[-15:]:
                print(line)
            print("```")
        print()
    if passed == len(results):
        print("> 🎉 모든 검증 통과. 안전하게 다음 단계로.")
    else:
        print(f"> ⚠️ {len(results) - passed}개 실패 — 위 출력 보고 수정 필요.")


if __name__ == "__main__":
    main()
