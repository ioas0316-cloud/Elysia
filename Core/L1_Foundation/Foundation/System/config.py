"""
                 
Elysia Unified Configuration Management

Pydantic                            .
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator, model_validator
    PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1
    from pydantic import BaseSettings, Field, validator, root_validator
    PYDANTIC_V2 = False


class ElysiaConfig(BaseSettings):
    """
              
    
             .env                    .
    """
    
    # =====       =====
    environment: str = Field(
        default="development",
        env="ELYSIA_ENV",
        description="      (development, testing, production)"
    )
    
    debug: bool = Field(
        default=False,
        env="ELYSIA_DEBUG",
        description="          "
    )
    
    # ===== API   =====
    gemini_api_key: Optional[str] = Field(
        default=None,
        env="GEMINI_API_KEY",
        description="Google Gemini API  "
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY",
        description="OpenAI API  "
    )
    
    # =====       =====
    data_dir: Path = Field(
        default=Path("data"),
        env="ELYSIA_DATA_DIR",
        description="           "
    )

    @property
    def memory_db_path(self) -> Path:
        """    DB      """
        return self.data_dir / "Memory" / "memory.db"

    
    log_dir: Path = Field(
        default=Path("logs"),
        env="ELYSIA_LOG_DIR",
        description="          "
    )
    
    backup_dir: Path = Field(
        default=Path("backups"),
        env="ELYSIA_BACKUP_DIR",
        description="          "
    )
    
    # =====       =====
    max_memory_mb: int = Field(
        default=1024,
        env="ELYSIA_MAX_MEMORY_MB",
        ge=128,
        le=32768,
        description="           (MB)"
    )
    
    max_workers: int = Field(
        default=4,
        env="ELYSIA_MAX_WORKERS",
        ge=1,
        le=32,
        description="           "
    )
    
    think_cycle_interval_ms: int = Field(
        default=100,
        env="ELYSIA_THINK_CYCLE_MS",
        ge=10,
        le=10000,
        description="          (   )"
    )
    
    # =====           =====
    resonance_threshold: float = Field(
        default=0.5,
        env="ELYSIA_RESONANCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="      "
    )
    
    default_frequency: float = Field(
        default=432.0,
        env="ELYSIA_DEFAULT_FREQUENCY",
        gt=0.0,
        description="       (Hz)"
    )
    
    spirit_frequencies: Dict[str, float] = Field(
        default_factory=lambda: {
            "Fire": 450.0,
            "Water": 150.0,
            "Wind": 300.0,
            "Earth": 200.0,
            "Light": 600.0,
            "Dark": 100.0,
            "Void": 50.0
        },
        description="          "
    )
    
    # =====        =====
    seed_compression_ratio: float = Field(
        default=1000.0,
        env="ELYSIA_SEED_COMPRESSION_RATIO",
        ge=1.0,
        description="      "
    )
    
    max_seeds: int = Field(
        default=10000,
        env="ELYSIA_MAX_SEEDS",
        ge=100,
        description="           "
    )
    
    bloom_depth: int = Field(
        default=3,
        env="ELYSIA_BLOOM_DEPTH",
        ge=1,
        le=10,
        description="        "
    )
    
    # ===== API       =====
    enable_api: bool = Field(
        default=True,
        env="ELYSIA_ENABLE_API",
        description="API       "
    )
    
    api_host: str = Field(
        default="0.0.0.0",
        env="ELYSIA_API_HOST",
        description="API       "
    )
    
    api_port: int = Field(
        default=8000,
        env="ELYSIA_API_PORT",
        ge=1,
        le=65535,
        description="API      "
    )
    
    api_rate_limit: int = Field(
        default=100,
        env="ELYSIA_API_RATE_LIMIT",
        ge=1,
        description="API       (  )"
    )
    
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        env="ELYSIA_ALLOWED_ORIGINS",
        description="    CORS   "
    )
    
    # =====       =====
    secret_key: Optional[str] = Field(
        default=None,
        env="ELYSIA_SECRET_KEY",
        description="         "
    )
    
    enable_authentication: bool = Field(
        default=False,
        env="ELYSIA_ENABLE_AUTH",
        description="      "
    )
    
    # =====       =====
    log_level: str = Field(
        default="INFO",
        env="ELYSIA_LOG_LEVEL",
        description="     "
    )
    
    log_format: str = Field(
        default="json",
        env="ELYSIA_LOG_FORMAT",
        description="      (json, text)"
    )
    
    # ===== Validators =====
    
    if PYDANTIC_V2:
        @field_validator('environment')
        @classmethod
        def validate_environment(cls, v):
            """     """
            valid = ['development', 'testing', 'production']
            if v not in valid:
                raise ValueError(f'environment must be one of {valid}, got: {v}')
            return v
        
        @field_validator('log_level')
        @classmethod
        def validate_log_level(cls, v):
            """        """
            valid = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            v_upper = v.upper()
            if v_upper not in valid:
                raise ValueError(f'log_level must be one of {valid}, got: {v}')
            return v_upper
        
        @field_validator('log_format')
        @classmethod
        def validate_log_format(cls, v):
            """        """
            valid = ['json', 'text']
            if v not in valid:
                raise ValueError(f'log_format must be one of {valid}, got: {v}')
            return v
        
        @field_validator('data_dir', 'log_dir', 'backup_dir')
        @classmethod
        def ensure_dir_exists(cls, v):
            """               """
            v = Path(v)
            v.mkdir(parents=True, exist_ok=True)
            return v
        
        @model_validator(mode='after')
        def validate_api_settings(self):
            """API      """
            if self.enable_api and self.enable_authentication:
                if not self.secret_key:
                    raise ValueError(
                        'secret_key is required when authentication is enabled'
                    )
            return self
        
        @model_validator(mode='after')
        def validate_production_settings(self):
            """             """
            if self.environment == 'production':
                #                    
                if self.debug:
                    raise ValueError('debug must be False in production')
                
                #         "*" CORS       
                if '*' in self.allowed_origins:
                    raise ValueError(
                        'Wildcard CORS origins not allowed in production'
                    )
            
            return self
    else:
        @validator('environment')
        def validate_environment(cls, v):
            """     """
            valid = ['development', 'testing', 'production']
            if v not in valid:
                raise ValueError(f'environment must be one of {valid}, got: {v}')
            return v
        
        @validator('log_level')
        def validate_log_level(cls, v):
            """        """
            valid = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            v_upper = v.upper()
            if v_upper not in valid:
                raise ValueError(f'log_level must be one of {valid}, got: {v}')
            return v_upper
        
        @validator('log_format')
        def validate_log_format(cls, v):
            """        """
            valid = ['json', 'text']
            if v not in valid:
                raise ValueError(f'log_format must be one of {valid}, got: {v}')
            return v
        
        @validator('data_dir', 'log_dir', 'backup_dir')
        def ensure_dir_exists(cls, v):
            """               """
            v = Path(v)
            v.mkdir(parents=True, exist_ok=True)
            return v
        
        @root_validator
        def validate_api_settings(cls, values):
            """API      """
            if values.get('enable_api') and values.get('enable_authentication'):
                if not values.get('secret_key'):
                    raise ValueError(
                        'secret_key is required when authentication is enabled'
                    )
            return values
        
        @root_validator
        def validate_production_settings(cls, values):
            """             """
            if values.get('environment') == 'production':
                #                    
                if values.get('debug'):
                    raise ValueError('debug must be False in production')
                
                #         "*" CORS       
                if '*' in values.get('allowed_origins', []):
                    raise ValueError(
                        'Wildcard CORS origins not allowed in production'
                    )
            
            return values
    
    class Config:
        """Pydantic   """
        if PYDANTIC_V2:
            # Pydantic v2 configuration
            env_file = '.env'
            env_file_encoding = 'utf-8'
            case_sensitive = False
            extra = 'allow'
        else:
            # Pydantic v1 configuration
            env_file = '.env'
            env_file_encoding = 'utf-8'
            case_sensitive = False
            extra = 'allow'


class ConfigManager:
    """      """
    
    def __init__(self):
        self._config: Optional[ElysiaConfig] = None
    
    def load(self, env: Optional[str] = None, env_file: Optional[str] = None) -> ElysiaConfig:
        """
             
        
        Args:
            env:    (development, testing, production)
            env_file:     .env      
        
        Returns:
                     
        """
        #            
        if env:
            os.environ['ELYSIA_ENV'] = env
        
        #          
        if not env_file:
            current_env = os.getenv('ELYSIA_ENV', 'development')
            env_file = f".env.{current_env}"
            
            #                .env   
            if not Path(env_file).exists():
                env_file = '.env'
        
        #      
        if Path(env_file).exists():
            self._config = ElysiaConfig(_env_file=env_file)
        else:
            self._config = ElysiaConfig()
        
        return self._config
    
    @property
    def config(self) -> ElysiaConfig:
        """        """
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def reload(self):
        """        """
        self._config = None
        return self.load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """       """
        return getattr(self.config, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """            """
        if PYDANTIC_V2:
            return self.config.model_dump()
        else:
            return self.config.dict()
    
    def summary(self) -> str:
        """     """
        cfg = self.config
        
        summary = f"""
=== Elysia Configuration Summary ===

Environment: {cfg.environment}
Debug Mode: {cfg.debug}

Paths:
  Data:   {cfg.data_dir}
  Logs:   {cfg.log_dir}
  Backup: {cfg.backup_dir}

Performance:
  Max Memory: {cfg.max_memory_mb} MB
  Max Workers: {cfg.max_workers}
  Think Cycle: {cfg.think_cycle_interval_ms} ms

Resonance:
  Threshold: {cfg.resonance_threshold}
  Default Frequency: {cfg.default_frequency} Hz

Memory:
  Compression Ratio: {cfg.seed_compression_ratio}x
  Max Seeds: {cfg.max_seeds}
  Bloom Depth: {cfg.bloom_depth}

API Server:
  Enabled: {cfg.enable_api}
  Host: {cfg.api_host}:{cfg.api_port}
  Rate Limit: {cfg.api_rate_limit} req/min
  Auth: {cfg.enable_authentication}

Logging:
  Level: {cfg.log_level}
  Format: {cfg.log_format}

API Keys:
  Gemini: {'  Set' if cfg.gemini_api_key else '  Not set'}
  OpenAI: {'  Set' if cfg.openai_api_key else '  Not set'}
"""
        return summary.strip()


#               
config_manager = ConfigManager()


#      
def get_config() -> ElysiaConfig:
    """        """
    return config_manager.config


def reload_config():
    """        """
    return config_manager.reload()


# =====       =====

if __name__ == "__main__":
    print("  Testing Elysia Configuration\n")
    
    #      
    print("=== Loading Configuration ===")
    config = get_config()
    
    #         
    print(config_manager.summary())
    print()
    
    #           
    print("=== Accessing Individual Settings ===")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Resonance Threshold: {config.resonance_threshold}")
    print(f"Default Frequency: {config.default_frequency} Hz")
    print()
    
    #       
    print("=== Type Safety ===")
    print(f"Max Memory (int): {config.max_memory_mb}")
    print(f"Think Cycle (int): {config.think_cycle_interval_ms}")
    print(f"Resonance Threshold (float): {config.resonance_threshold}")
    print()
    
    #            
    print("=== Directory Creation ===")
    print(f"Data dir exists: {config.data_dir.exists()}")
    print(f"Log dir exists: {config.log_dir.exists()}")
    print(f"Backup dir exists: {config.backup_dir.exists()}")
    print()
    
    #          
    print("=== Spirit Frequencies ===")
    for spirit, freq in config.spirit_frequencies.items():
        print(f"  {spirit}: {freq} Hz")
    print()
    
    #       
    print("=== Validation Tests ===")
    
    #           
    try:
        os.environ['ELYSIA_ENV'] = 'invalid'
        ElysiaConfig()
        print("  Should have failed with invalid environment")
    except ValueError as e:
        print(f"  Validation works: {e}")
    
    #           
    os.environ.pop('ELYSIA_ENV', None)
    
    print("\n  Configuration system working correctly!")