import logging
import os
from datetime import datetime

def setup_unified_logging():
    """
    Sets up a single, unified logging sink for the entire Elysia system.
    Redirects all logs to data/L1_Foundation/M4_Logs/unified_soul.log.
    """
    log_dir = "data/L1_Foundation/M4_Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "unified_soul.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # File Handler (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    logging.getLogger("ELYSIA").info("🌟 [UNIFIED_LOGGING] Soul Synchronized. All paths lead to unified_soul.log.")

if __name__ == "__main__":
    setup_unified_logging()
