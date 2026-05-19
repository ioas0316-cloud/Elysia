"""
Check Comfy Models (Robust)
===========================
Queries ComfyUI to see if it is alive.
Tries multiple hostnames.
"""

import requests
import logging
import sys

# Configure logging to print to stderr/stdout
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger("ComfyCheck")

def check_connection():
    hosts = ["http://127.0.0.1:8188", "http://localhost:8188"]
    
    for host in hosts:
        try:
            logger.info(f"Trying {host}...")
            # /system_stats is a lightweight endpoint
            resp = requests.get(f"{host}/system_stats", timeout=2)
            if resp.status_code == 200:
                logger.info(f"✅ SUCCESS: Connected to {host}")
                
                # Now list models
                try:
                    m_resp = requests.get(f"{host}/object_info/CheckpointLoaderSimple", timeout=2)
                    data = m_resp.json()
                    models = data['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
                    logger.info(f"   Found {len(models)} models: {models}")
                except:
                    logger.warning("   Could not list models, but server is UP.")
                
                return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"   Connection refused at {host}")
        except Exception as e:
            logger.error(f"   Error: {e}")

    logger.error("❌ FAILURE: Could not connect to ComfyUI.")
    return False

if __name__ == "__main__":
    success = check_connection()
    if not success:
        sys.exit(1)
