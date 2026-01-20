import os
from huggingface_hub import snapshot_download
import time

def download_model():
    repo_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    # UPDATED PATH: Adhere to Trinity Architecture
    local_dir = "c:/Elysia/data/Weights/DeepSeek-Coder-V2-Lite-Instruct"
    
    print(f"🚀 Starting download for: {repo_id}")
    print(f"📂 Target Directory: {local_dir}")
    print("⚠️  Note: This model is approx 32GB. Ensure you have space.")
    
    max_retries = 100
    for i in range(max_retries):
        try:
            print(f"🔄 Attempt {i+1}/{max_retries}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,
                max_workers=8 # Speed up small files
            )
            print("✅ Download Complete Successfully!")
            return
        except Exception as e:
            print(f"⚠️ Download interrupted: {e}")
            print("⏳ Retrying in 15 seconds...")
            time.sleep(15)

if __name__ == "__main__":
    download_model()
