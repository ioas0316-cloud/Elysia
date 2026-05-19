#!/usr/bin/env python3
# version: music_v5
"""음악 스튜디오 — 다중 모델 지원 원클릭 설치.

선택 가능한 모델 (디스크·메모리·품질 트레이드오프):

  ┌────────────────────────┬────────┬───────────┬─────────────┐
  │ MODEL                  │ 디스크 │ 메모리    │ 추천        │
  ├────────────────────────┼────────┼───────────┼─────────────┤
  │ musicgen-small (기본)  │ 300MB  │ 4GB+      │ 모든 기기   │
  │ musicgen-medium        │ 1.5GB  │ 6GB+      │ 8GB+ RAM    │
  │ musicgen-large         │ 3.3GB  │ 12GB+     │ 16GB+ RAM   │
  │ acestep-base           │ 10GB   │ 16GB+     │ 16GB+ Mac   │
  │ acestep-xl             │ 15GB+  │ 24GB+     │ 32GB+ 머신  │
  └────────────────────────┴────────┴───────────┴─────────────┘

기본값: musicgen-small — 300MB만 받고 30초만에 첫 음악. 모든 기기에서 안정적.
큰 모델은 추론 시 명시 RAM의 1.5~2배 실제 압박 발생해서 자동 선택은 무조건 small.
medium/large 쓰고 싶으면 MODEL 필드에 직접 지정.

⚙️ MODEL 필드를 위 5개 중 하나로 설정. 설치는 한 번에 한 모델만 (선택한 거).
"""
import os, sys, json, subprocess, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "music_studio_setup.json")

# 모델 메타 — 디스크·RAM 추천·HuggingFace 경로·설치 방식
MODELS = {
    "musicgen-small": {
        "disk_gb": 0.3, "ram_gb": 4,
        "kind": "transformers", "hf_id": "facebook/musicgen-small",
        "deps": ["torch", "torchaudio", "transformers", "scipy", "soundfile"],
        "label": "MusicGen Small (300MB · 모든 기기)",
    },
    "musicgen-medium": {
        "disk_gb": 1.5, "ram_gb": 6,
        "kind": "transformers", "hf_id": "facebook/musicgen-medium",
        "deps": ["torch", "torchaudio", "transformers", "scipy", "soundfile"],
        "label": "MusicGen Medium (1.5GB · 8GB+ RAM)",
    },
    "musicgen-large": {
        "disk_gb": 3.3, "ram_gb": 12,
        "kind": "transformers", "hf_id": "facebook/musicgen-large",
        "deps": ["torch", "torchaudio", "transformers", "scipy", "soundfile"],
        "label": "MusicGen Large (3.3GB · 16GB+ RAM)",
    },
    "acestep-base": {
        "disk_gb": 10, "ram_gb": 16,
        "kind": "acestep", "hf_id": "ACE-Step/Ace-Step1.5",
        "repo": "https://github.com/ace-step/ACE-Step-1.5.git",
        "label": "ACE-Step 1.5 Base (10GB · 16GB+ Mac/CUDA)",
    },
    "acestep-xl": {
        "disk_gb": 15, "ram_gb": 24,
        "kind": "acestep", "hf_id": "ACE-Step/acestep-v15-xl-base",
        "repo": "https://github.com/ace-step/ACE-Step-1.5.git",
        "label": "ACE-Step 1.5 XL (15GB · 32GB+ 머신)",
    },
}

DEFAULT_INSTALL_DIR = os.path.expanduser("~/connect-ai-music")


def _log(msg, kind="info"):
    prefix = {"info": "🔧", "ok": "✅", "warn": "⚠️ ", "err": "❌"}.get(kind, "•")
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _which(cmd):
    return shutil.which(cmd) is not None


def _system_ram_gb():
    """Detect system RAM. Cross-platform best effort."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(r.stdout.strip()) / (1024 ** 3)
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 16  # 보수적 default


def _recommend_model(ram_gb):
    """RAM 기반 추천 모델. v2.89.78 — 보수적으로 small 우선.
    추론할 때 모델 weight + activation + scratch buffer 합쳐서 명시 RAM의 1.5~2배
    실제 압박 발생. medium은 6GB 명시지만 실제로 12GB+ 압박. 16GB Mac에서 OS·브라우저·
    VS Code 띄운 상태면 medium 추론 중 swap 폭발. small이 모든 환경에서 안정적이고
    품질도 충분. 사용자가 원하면 MODEL 필드로 직접 medium/large 선택."""
    return "musicgen-small"


def _run(cmd, cwd=None):
    _log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            cwd=cwd, check=False, capture_output=True, text=True
        )
        # stderr가 진짜 에러면 표시, 아니면 진행상황으로 간주 (pip 등은 진행상황을 stderr에)
        for stream in (result.stdout, result.stderr):
            if stream and stream.strip():
                for line in stream.splitlines()[-20:]:  # 마지막 20줄만
                    _log(f"  {line}")
        return result.returncode == 0
    except Exception as e:
        _log(f"실행 오류: {e}", "err")
        return False


def _load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _install_transformers_model(model_key, install_dir):
    """MusicGen 류 — pip + huggingface 다운로드. 가벼운 경로."""
    info = MODELS[model_key]
    venv = os.path.join(install_dir, ".venv")

    # venv 생성
    if not os.path.isdir(venv):
        _log("Python venv 생성...")
        if not _run(["python3", "-m", "venv", venv]):
            return False, "venv 생성 실패"

    venv_pip = os.path.join(venv, "bin", "pip")
    venv_python = os.path.join(venv, "bin", "python")
    if not os.path.exists(venv_pip):
        venv_pip = os.path.join(venv, "Scripts", "pip.exe")
        venv_python = os.path.join(venv, "Scripts", "python.exe")

    _log("Python 의존성 설치 (1~3분, ~500MB)...")
    _run([venv_pip, "install", "--upgrade", "pip", "--quiet"])
    if not _run([venv_pip, "install", "--quiet"] + info["deps"]):
        return False, "pip install 실패"

    # 모델 weight 다운로드
    # v2.89.74 — transformers/HF Hub의 verbose 로그 억제. 이전엔 사용자한테
    # "decoder.model.decoder.embed_positions.weights | UNEXPECTED" 같은 내부 로그 노출돼 혼란.
    _log(f"모델 다운로드 중: {info['hf_id']} ({info['disk_gb']}GB)...")
    download_script = f"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import logging, warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
print('🔧 라이브러리 로드 중...', flush=True)
from transformers import MusicgenForConditionalGeneration, AutoProcessor
print('🔧 토크나이저/프로세서 다운로드 중...', flush=True)
AutoProcessor.from_pretrained('{info['hf_id']}')
print('🔧 모델 weight 다운로드 중 (대용량, 시간 걸림)...', flush=True)
MusicgenForConditionalGeneration.from_pretrained('{info['hf_id']}')
print('✅ 모델 다운로드·로드 검증 완료')
"""
    if not _run([venv_python, "-c", download_script]):
        return False, "모델 다운로드 실패 — 인터넷 연결 확인"

    return True, venv_python


def _install_acestep(model_key, install_dir):
    """ACE-Step — git clone + 큰 의존성. 무거운 경로."""
    info = MODELS[model_key]
    repo_dir = os.path.join(install_dir, "ace-step")

    if not os.path.isdir(repo_dir):
        _log(f"ACE-Step 1.5 clone 중 → {repo_dir}")
        if not _run(["git", "clone", "--depth", "1", info["repo"], repo_dir]):
            return False, "git clone 실패"

    venv = os.path.join(repo_dir, ".venv")
    if not os.path.isdir(venv):
        _log("Python venv 생성...")
        if not _run(["python3", "-m", "venv", venv]):
            return False, "venv 생성 실패"

    venv_pip = os.path.join(venv, "bin", "pip")
    venv_python = os.path.join(venv, "bin", "python")
    if not os.path.exists(venv_pip):
        venv_pip = os.path.join(venv, "Scripts", "pip.exe")
        venv_python = os.path.join(venv, "Scripts", "python.exe")

    requirements = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements):
        _log(f"ACE-Step 의존성 설치 중 (5~10분, 큰 패키지 다운로드)...")
        _run([venv_pip, "install", "--upgrade", "pip", "--quiet"])
        if not _run([venv_pip, "install", "-r", requirements]):
            return False, "pip install 일부 실패 — 다시 실행하면 이어짐"

    _log(f"모델 weight (~{info['disk_gb']}GB) 는 첫 음악 생성 때 자동 다운로드", "info")
    return True, venv_python


def main():
    cfg = _load_config()

    # 기본 의존성
    missing = []
    if not _which("python3"):
        missing.append("python3 (https://www.python.org/downloads/)")
    if not _which("git"):
        missing.append("git (https://git-scm.com/downloads)")
    if missing:
        print("❌ 다음 도구 먼저 설치해주세요:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # 모델 선택: config의 MODEL 우선, 없으면 RAM 기반 추천
    requested = (cfg.get("MODEL") or "").strip()
    ram_gb = _system_ram_gb()
    if not requested:
        requested = _recommend_model(ram_gb)
        _log(f"시스템 RAM {ram_gb:.0f}GB → 안전하게 {requested} 선택 (medium/large는 ⚙️ MODEL 필드에서 직접 지정)", "info")

    if requested not in MODELS:
        print(f"❌ 알 수 없는 MODEL: {requested}")
        print(f"  사용 가능: {', '.join(MODELS.keys())}")
        sys.exit(1)

    info = MODELS[requested]
    _log(f"설치 모델: {info['label']}")

    # 이미 설치돼있으면 빠르게 종료
    if cfg.get("INSTALLED_MODEL") == requested and cfg.get("VENV_PYTHON"):
        venv_python = cfg.get("VENV_PYTHON")
        if os.path.exists(venv_python):
            print(f"✅ 이미 설치 완료: {info['label']}")
            print(f"  📁 {cfg.get('INSTALL_DIR')}")
            print(f"  🐍 {venv_python}")
            return

    install_dir = cfg.get("INSTALL_DIR") or DEFAULT_INSTALL_DIR
    os.makedirs(install_dir, exist_ok=True)

    if info["kind"] == "transformers":
        ok, result = _install_transformers_model(requested, install_dir)
    else:
        ok, result = _install_acestep(requested, install_dir)

    if not ok:
        print(f"❌ 설치 실패: {result}")
        sys.exit(1)

    venv_python = result
    cfg["INSTALLED_MODEL"] = requested
    cfg["MODEL"] = requested
    cfg["INSTALL_DIR"] = install_dir
    cfg["VENV_PYTHON"] = venv_python
    cfg["INSTALL_KIND"] = info["kind"]
    cfg["HF_ID"] = info["hf_id"]
    cfg["INSTALLED_AT"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if info["kind"] == "acestep":
        cfg["ACE_STEP_DIR"] = os.path.join(install_dir, "ace-step")
    _save_config(cfg)

    # v2.89.74 — 깔끔한 시각적 완료 카드
    print()
    print("━" * 50)
    print(f"🎉 음악 스튜디오 설치 완료!")
    print("━" * 50)
    print()
    print(f"📦 무엇이 깔렸나:")
    print(f"   • 모델:   {info['label']}")
    print(f"   • 위치:   {install_dir}")
    print(f"   • 디스크: ~{info['disk_gb']}GB 사용 중")
    print()
    print(f"🎼 이제 뭐 할 수 있나:")
    print(f"   • 'music_generate.py' ▶ 클릭 → 30초 BGM 생성")
    print(f"   • 'music_to_video.py' ▶ 클릭 → 영상에 BGM 합성")
    print()
    print(f"⚙️  모델 바꾸고 싶으면 ⚙️ → MODEL 드롭다운에서 선택 → 이 도구 다시 ▶")
    print()
    print(f"💡 위 로그에 'WARNING / UNEXPECTED' 보였어도 무시해도 됩니다 —")
    print(f"   transformers 라이브러리 내부 메시지. 설치는 정상 완료.")


if __name__ == "__main__":
    main()
