"""
              
Elysia Unified Logging System

       , JSON   ,             .
"""

import logging
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    """JSON          """
    
    def format(self, record: logging.LogRecord) -> str:
        """
                JSON     
        
        Args:
            record:       
        
        Returns:
            JSON          
        """
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        #        
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        #      
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else 'Unknown',
                'message': str(record.exc_info[1]) if record.exc_info[1] else '',
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """                """
    
    # ANSI      
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    #    
    EMOJIS = {
        'DEBUG': ' ',
        'INFO': '   ',
        'WARNING': '   ',
        'ERROR': ' ',
        'CRITICAL': ' ',
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
                            
        
        Args:
            record:       
        
        Returns:
                      
        """
        #      
        color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJIS.get(record.levelname, '')
        
        #      
        log_fmt = (
            f"{color}{emoji} "
            f"%(asctime)s | %(levelname)-8s | %(name)s | "
            f"%(message)s{self.RESET}"
        )
        
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


class ElysiaLogger:
    """              """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
                   
        
        Args:
            name:      
            log_dir:        
            console_level:         
            file_level:         
            max_bytes:            
            backup_count:         
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        #      
        self.logger = logging.getLogger(f"Elysia.{name}")
        self.logger.setLevel(logging.DEBUG)
        
        #                     (     )
        if not self.logger.handlers:
            # JSON          
            json_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            json_handler = RotatingFileHandler(
                json_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(file_level)
            json_handler.setFormatter(JsonFormatter())
            
            #                 
            text_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            text_handler = RotatingFileHandler(
                text_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            text_handler.setLevel(file_level)
            text_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            
            #          
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(ColoredConsoleFormatter())
            
            #       
            self.logger.addHandler(json_handler)
            self.logger.addHandler(text_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """         """
        self._log(logging.DEBUG, message, context)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """        """
        self._log(logging.INFO, message, context)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """        """
        self._log(logging.WARNING, message, context)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """        """
        self._log(logging.ERROR, message, context, exc_info=exc_info)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """         """
        self._log(logging.CRITICAL, message, context, exc_info=exc_info)
    
    def _log(
        self,
        level: int,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """         """
        extra = {'context': context} if context else {}
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    # ===                ===
    
    def log_thought(
        self,
        layer: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
                
        
        Args:
            layer:       (0D/1D/2D/3D)
            content:      
            context:        
        """
        ctx = context or {}
        ctx.update({'layer': layer, 'type': 'thought'})
        self.info(f"  [{layer}] {content}", context=ctx)
    
    def log_resonance(
        self,
        source: str,
        target: str,
        score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
             
        
        Args:
            source:      
            target:      
            score:      
            context:        
        """
        ctx = context or {}
        ctx.update({
            'source': source,
            'target': target,
            'score': score,
            'type': 'resonance'
        })
        self.debug(f"  Resonance: {source}   {target} = {score:.3f}", context=ctx)
    
    def log_evolution(
        self,
        component: str,
        metric: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
                 
        
        Args:
            component:        
            metric:       
            value:      
            context:        
        """
        ctx = context or {}
        ctx.update({
            'component': component,
            'metric': metric,
            'value': value,
            'type': 'evolution'
        })
        self.info(f"  Evolution: {component}.{metric} = {value:.3f}", context=ctx)
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
             
        
        Args:
            operation:      
            duration_ms:       (   )
            context:        
        """
        ctx = context or {}
        ctx.update({
            'operation': operation,
            'duration_ms': duration_ms,
            'type': 'performance'
        })
        
        #                
        if duration_ms > 1000:
            self.warning(f"  Performance: {operation} took {duration_ms:.2f}ms", context=ctx)
        else:
            self.debug(f"  Performance: {operation} took {duration_ms:.2f}ms", context=ctx)
    
    def log_spirit(
        self,
        spirit_name: str,
        frequency: float,
        amplitude: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
                
        
        Args:
            spirit_name:      
            frequency:    
            amplitude:   
            context:        
        """
        ctx = context or {}
        ctx.update({
            'spirit': spirit_name,
            'frequency': frequency,
            'amplitude': amplitude,
            'type': 'spirit'
        })
        self.debug(
            f"  Spirit: {spirit_name} @ {frequency:.1f}Hz (amp: {amplitude:.2f})",
            context=ctx
        )
    
    def log_memory(
        self,
        operation: str,
        seed_name: str,
        compression_ratio: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
                 
        
        Args:
            operation:       (bloom/compress/store)
            seed_name:      
            compression_ratio:    
            context:        
        """
        ctx = context or {}
        ctx.update({
            'operation': operation,
            'seed': seed_name,
            'type': 'memory'
        })
        if compression_ratio:
            ctx['compression_ratio'] = compression_ratio
            msg = f"  Memory: {operation} seed '{seed_name}' (ratio: {compression_ratio:.1f}x)"
        else:
            msg = f"  Memory: {operation} seed '{seed_name}'"
        
        self.debug(msg, context=ctx)
    
    def log_system(
        self,
        event: str,
        status: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
                  
        
        Args:
            event:       
            status:   
            context:        
        """
        ctx = context or {}
        ctx.update({
            'event': event,
            'status': status,
            'type': 'system'
        })
        
        if status in ['error', 'failed', 'critical']:
            self.error(f"    System: {event} - {status}", context=ctx)
        elif status in ['warning', 'degraded']:
            self.warning(f"    System: {event} - {status}", context=ctx)
        else:
            self.info(f"    System: {event} - {status}", context=ctx)


# =====       =====

if __name__ == "__main__":
    print("  Testing Elysia Logger\n")
    
    #      
    logger = ElysiaLogger("TestModule")
    
    #      
    print("=== Basic Logging ===")
    logger.debug("       ")
    logger.info("      ")
    logger.warning("      ")
    logger.error("      ")
    print()
    
    #            
    print("=== Contextual Logging ===")
    logger.info(
        "       ",
        context={'user_id': 'user123', 'ip': '192.168.1.1'}
    )
    print()
    
    #           
    print("=== Elysia-Specific Logging ===")
    logger.log_thought("2D", "            ...", {'emotion': 'calm'})
    logger.log_resonance("Love", "Hope", 0.847)
    logger.log_evolution("ResonanceField", "coherence", 0.923)
    logger.log_performance("calculate_interference", 45.3)
    logger.log_spirit("Fire", 450.0, 0.8)
    logger.log_memory("bloom", "concept_love", compression_ratio=1000.0)
    logger.log_system("startup", "complete")
    print()
    
    #      
    print("=== Exception Logging ===")
    try:
        raise ValueError("      ")
    except Exception:
        logger.error("          ", exc_info=True)
    print()
    
    print(f"  Logs saved to: {logger.log_dir}")
    print(f"   - JSON: {logger.log_dir}/TestModule_{datetime.now().strftime('%Y%m%d')}.jsonl")
    print(f"   - Text: {logger.log_dir}/TestModule_{datetime.now().strftime('%Y%m%d')}.log")
