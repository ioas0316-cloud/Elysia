import os
import sys
import argparse
from huggingface_hub import snapshot_download

def download_safetensors_only(repo_id: str, local_dir: str):
    """
    HuggingFace에서 특정 모델의 .safetensors 파일만 골라서 다운로드합니다.
    (PyTorch .bin 등 불필요한 파일 제외, 전송 효율 극대화)
    """
    print("=" * 80)
    print(f" 📥 [Elysia Downloader] {repo_id} 모델 위상 추출(Safetensors) 다운로드 시작")
    print("=" * 80)
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 오직 safetensors 포맷과 기본 설정 파일만 다운로드 (대역폭 절약)
        allow_patterns = ["*.safetensors", "config.json"]
        
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True
        )
        print(f"\n🎉 다운로드 완료! 저장 위치: {path}")
        print("이제 MMAPTensorStreamer를 통해 이 파일들의 위상 복제가 가능합니다.")
    except Exception as e:
        print(f"\n❌ 다운로드 중 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elysia Model Safetensors Downloader")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Repository ID (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dir", type=str, default="c:/Elysia/models", help="다운로드할 로컬 폴더 경로")
    
    args = parser.parse_args()
    download_safetensors_only(args.model, args.dir)
