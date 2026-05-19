#!/usr/bin/env python3
# version: web_preview_v1
"""웹 프로젝트 dev server 시작 + URL 추출.

config:
  PROJECT_PATH — 프로젝트 폴더 (web_init이 만든 건 자동 감지)
  PORT — 비워두면 자동 (vite=5173, next=3000, astro=4321)
  AUTO_OPEN — 'true' 면 브라우저 자동 열기

특징:
  - package.json scripts.dev 자동 감지
  - 백그라운드 실행 (nohup) + PID 파일 저장
  - 첫 5초 동안 출력에서 localhost URL 파싱
  - 같은 프로젝트 재실행 시 이전 PID 자동 종료
"""
import os, sys, json, subprocess, time, signal, re


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "web_preview.json")
WEB_INIT_CONFIG = os.path.join(HERE, "web_init.json")


def _log(msg, kind="info"):
    prefix = {"info": "💻", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "▸"}.get(kind, "•")
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _load(p):
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(p, c):
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _detect_dev_command(project_path):
    """package.json의 scripts.dev 또는 start를 감지."""
    pkg = os.path.join(project_path, "package.json")
    if not os.path.exists(pkg):
        return None
    try:
        with open(pkg, "r", encoding="utf-8") as f:
            data = json.load(f)
        scripts = data.get("scripts", {})
        for key in ["dev", "start", "develop", "serve"]:
            if key in scripts:
                return f"npm run {key}"
    except Exception:
        pass
    return None


def _kill_old_pid(pid_file):
    """이전 실행이 있으면 종료."""
    if not os.path.exists(pid_file):
        return
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        _log(f"이전 dev server 종료 (PID {pid})", "info")
    except Exception:
        pass
    try:
        os.remove(pid_file)
    except Exception:
        pass


def main():
    cfg = _load(CONFIG)
    init_cfg = _load(WEB_INIT_CONFIG)

    project_path = (cfg.get("PROJECT_PATH") or "").strip()
    if not project_path:
        # web_init 결과 자동 사용
        project_path = (init_cfg.get("LAST_PROJECT") or "").strip()
    if not project_path:
        _log("PROJECT_PATH가 비어있고 web_init 기록도 없음. 프로젝트 경로 지정하세요.", "err")
        sys.exit(1)

    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        _log(f"폴더 없음: {project_path}", "err")
        sys.exit(1)

    # dev 명령 감지
    dev_cmd = (cfg.get("DEV_CMD") or "").strip()
    if not dev_cmd:
        dev_cmd = init_cfg.get("LAST_DEV_CMD", "")
    if not dev_cmd:
        dev_cmd = _detect_dev_command(project_path) or "npm run dev"

    _log(f"프로젝트: {project_path}", "info")
    _log(f"명령: {dev_cmd}", "info")

    # PID 파일
    pid_file = os.path.join(project_path, ".connect-ai-dev.pid")
    log_file = os.path.join(project_path, ".connect-ai-dev.log")
    _kill_old_pid(pid_file)

    # 백그라운드 실행
    try:
        with open(log_file, "w") as logf:
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    dev_cmd, shell=True, cwd=project_path,
                    stdout=logf, stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    dev_cmd, shell=True, cwd=project_path,
                    stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))
        _log(f"dev server 시작됨 (PID {proc.pid})", "ok")
    except Exception as e:
        _log(f"실행 실패: {e}", "err")
        sys.exit(1)

    # 첫 8초 동안 로그에서 URL 파싱
    url = None
    deadline = time.time() + 8
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            with open(log_file, "r") as f:
                content = f.read()
            # 흔한 패턴: "Local:   http://localhost:3000" / "ready - started server on http://localhost:3000"
            m = re.search(r"https?://(?:localhost|127\.0\.0\.1):\d+(?:/\S*)?", content)
            if m:
                url = m.group(0)
                break
        except Exception:
            pass
        if proc.poll() is not None:
            _log("dev server가 일찍 종료됐어요. 로그 확인:", "warn")
            try:
                with open(log_file, "r") as f:
                    for line in f.read().splitlines()[-10:]:
                        print(f"  {line}")
            except Exception:
                pass
            sys.exit(1)

    if url:
        _log(f"미리보기 URL: {url}", "ok")
    else:
        _log("URL을 자동 감지 못 함. 로그 확인:", "warn")
        try:
            with open(log_file, "r") as f:
                for line in f.read().splitlines()[-15:]:
                    print(f"  {line}")
        except Exception:
            pass
        url = "http://localhost:3000"

    auto_open = str(cfg.get("AUTO_OPEN", "")).lower() in ("true", "1", "yes")
    if auto_open and url:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", url])
            elif sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
            else:
                subprocess.Popen(["xdg-open", url])
            _log("브라우저 열림", "ok")
        except Exception:
            pass

    # 결과 저장
    cfg["LAST_PROJECT"] = project_path
    cfg["LAST_PID"] = proc.pid
    cfg["LAST_URL"] = url
    cfg["LAST_LOG"] = log_file
    _save(CONFIG, cfg)

    print()
    print(f"PID={proc.pid}")
    print(f"URL={url}")
    print(f"LOG={log_file}")
    print()
    _log("dev server는 백그라운드에서 계속 실행됩니다.", "info")
    _log(f"종료: kill {proc.pid}  (또는 같은 도구 재실행)", "info")


if __name__ == "__main__":
    main()
