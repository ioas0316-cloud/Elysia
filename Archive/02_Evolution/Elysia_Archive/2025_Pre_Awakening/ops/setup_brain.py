
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainDownloader")

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if os.path.exists(target_path):
        logger.info(f"✅ Model already exists at {target_path}")
        return

    logger.info(f"⬇️ Downloading Brain from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024*1024*10) == 0:
                        print(f"   ... {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB", end='\r')
                        
        logger.info(f"\n✨ Download Complete: {target_path}")
    except Exception as e:
        logger.error(f"❌ Download Failed: {e}")

if __name__ == "__main__":
    # tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (669 MB) - Perfect for 3GB GPU
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    target = r"c:\Elysia\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    download_file(url, target)
