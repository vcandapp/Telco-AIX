# Author: Fatih E. NAR
# Agentic AI Framework - MCP Configuration System
#
import configparser
import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

class BackendType(str, Enum):
    """Supported MCP backend types."""
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"

@dataclass
class BackendConfig:
    """Configuration for a specific MCP backend."""
    backend_type: BackendType
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_tokens_per_minute: Optional[int] = None
    supports_create: bool = True
    supports_query: bool = True
    supports_update: bool = True
    supports_delete: bool = True
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

@dataclass
class MCPServerConfig:
    """Complete MCP server configuration."""
    backend_type: BackendType
    backends: Dict[BackendType, BackendConfig]
    enable_backup: bool = True
    backup_backend: BackendType = BackendType.LOCAL
    enable_federation: bool = False
    fallback_backends: List[BackendType] = None
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    debug: bool = False
    
    def __post_init__(self):
        if self.fallback_backends is None:
            self.fallback_backends = []

class MCPConfigManager:
    """Manager for MCP server configuration."""
    
    DEFAULT_CONFIG_PATH = "MCP-Server-Config.cfg"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.logger = logging.getLogger("mcp.config")
        self._config = None
        
    def load_config(self) -> MCPServerConfig:
        """Load configuration from file.
        
        Returns:
            MCPServerConfig object with loaded configuration
        """
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._create_default_config()
            
        config = configparser.ConfigParser()
        config.read(self.config_path)
        
        try:
            # Load default settings
            default_section = config['DEFAULT']
            backend_type = BackendType(default_section.get('backend_type', 'local'))
            timeout = int(default_section.get('timeout', '30'))
            retry_attempts = int(default_section.get('retry_attempts', '3'))
            debug = default_section.getboolean('debug', fallback=False)
            
            # Load backend configurations
            backends = {}
            
            # Local backend
            if 'LOCAL' in config:
                backends[BackendType.LOCAL] = self._load_local_config(config['LOCAL'], timeout, retry_attempts)
            
            # Anthropic backend
            if 'ANTHROPIC' in config:
                backends[BackendType.ANTHROPIC] = self._load_anthropic_config(config['ANTHROPIC'], timeout, retry_attempts)
            
            # HuggingFace backend
            if 'HUGGINGFACE' in config:
                backends[BackendType.HUGGINGFACE] = self._load_huggingface_config(config['HUGGINGFACE'], timeout, retry_attempts)
            
            # OpenAI backend
            if 'OPENAI' in config:
                backends[BackendType.OPENAI] = self._load_openai_config(config['OPENAI'], timeout, retry_attempts)
            
            # Load backup settings
            enable_backup = True
            backup_backend = BackendType.LOCAL
            if 'BACKUP' in config:
                backup_section = config['BACKUP']
                enable_backup = backup_section.getboolean('enable_backup', fallback=True)
                backup_backend = BackendType(backup_section.get('backup_backend', 'local'))
            
            # Load federation settings
            enable_federation = False
            fallback_backends = []
            if 'FEDERATION' in config:
                federation_section = config['FEDERATION']
                enable_federation = federation_section.getboolean('enable_federation', fallback=False)
                fallback_backends_str = federation_section.get('fallback_backends', '')
                fallback_backends = [BackendType(b.strip()) for b in fallback_backends_str.split(',') if b.strip()]
            
            # Load cache settings
            enable_cache = True
            cache_ttl_seconds = 3600
            if 'CACHE' in config:
                cache_section = config['CACHE']
                enable_cache = cache_section.getboolean('enable_cache', fallback=True)
                cache_ttl_seconds = int(cache_section.get('cache_ttl_seconds', '3600'))
            
            self._config = MCPServerConfig(
                backend_type=backend_type,
                backends=backends,
                enable_backup=enable_backup,
                backup_backend=backup_backend,
                enable_federation=enable_federation,
                fallback_backends=fallback_backends,
                enable_cache=enable_cache,
                cache_ttl_seconds=cache_ttl_seconds,
                debug=debug
            )
            
            self.logger.info(f"Loaded MCP configuration: backend={backend_type.value}, backends={len(backends)}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return self._create_default_config()
    
    def _load_local_config(self, section: configparser.SectionProxy, timeout: int, retry_attempts: int) -> BackendConfig:
        """Load local backend configuration."""
        host = section.get('host', 'localhost')
        port = section.get('port', '3000')
        base_url = section.get('base_url', f'http://{host}:{port}')
        
        additional_params = {
            'enable_persistence': section.getboolean('enable_persistence', fallback=True),
            'storage_path': section.get('storage_path', './mcp_storage'),
            'max_contexts': int(section.get('max_contexts', '10000'))
        }
        
        return BackendConfig(
            backend_type=BackendType.LOCAL,
            base_url=base_url,
            timeout=timeout,
            retry_attempts=retry_attempts,
            additional_params=additional_params
        )
    
    def _load_anthropic_config(self, section: configparser.SectionProxy, timeout: int, retry_attempts: int) -> BackendConfig:
        """Load Anthropic backend configuration."""
        api_key = self._resolve_env_var(section.get('api_key', ''))
        
        additional_params = {
            'context_window': int(section.get('context_window', '200000')),
            'organization': self._resolve_env_var(section.get('organization', ''))
        }
        
        return BackendConfig(
            backend_type=BackendType.ANTHROPIC,
            base_url=section.get('base_url', 'https://api.anthropic.com/v1/mcp'),
            api_key=api_key,
            model=section.get('model', 'claude-3-sonnet-20240229'),
            max_tokens=int(section.get('max_tokens', '4096')),
            timeout=timeout,
            retry_attempts=retry_attempts,
            rate_limit_requests_per_minute=int(section.get('rate_limit_requests_per_minute', '100')),
            rate_limit_tokens_per_minute=int(section.get('rate_limit_tokens_per_minute', '50000')),
            supports_create=section.getboolean('supports_create', fallback=True),
            supports_query=section.getboolean('supports_query', fallback=True),
            supports_update=section.getboolean('supports_update', fallback=True),
            supports_delete=section.getboolean('supports_delete', fallback=True),
            additional_params=additional_params
        )
    
    def _load_huggingface_config(self, section: configparser.SectionProxy, timeout: int, retry_attempts: int) -> BackendConfig:
        """Load HuggingFace backend configuration."""
        api_key = self._resolve_env_var(section.get('api_key', ''))
        
        additional_params = {
            'context_model': section.get('context_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            'task': section.get('task', 'text-generation'),
            'use_cache': section.getboolean('use_cache', fallback=True),
            'wait_for_model': section.getboolean('wait_for_model', fallback=True),
            'daily_quota': int(section.get('rate_limit_daily_quota', '1000'))
        }
        
        return BackendConfig(
            backend_type=BackendType.HUGGINGFACE,
            base_url=section.get('base_url', 'https://api-inference.huggingface.co/models'),
            api_key=api_key,
            model=section.get('model', 'microsoft/DialoGPT-large'),
            timeout=timeout,
            retry_attempts=retry_attempts,
            rate_limit_requests_per_minute=int(section.get('rate_limit_requests_per_minute', '30')),
            supports_create=section.getboolean('supports_create', fallback=True),
            supports_query=section.getboolean('supports_query', fallback=True),
            supports_update=section.getboolean('supports_update', fallback=False),
            supports_delete=section.getboolean('supports_delete', fallback=False),
            additional_params=additional_params
        )
    
    def _load_openai_config(self, section: configparser.SectionProxy, timeout: int, retry_attempts: int) -> BackendConfig:
        """Load OpenAI backend configuration."""
        api_key = self._resolve_env_var(section.get('api_key', ''))
        
        additional_params = {
            'organization': self._resolve_env_var(section.get('organization', '')),
            'temperature': float(section.get('temperature', '0.1')),
            'embedding_model': section.get('embedding_model', 'text-embedding-3-large')
        }
        
        return BackendConfig(
            backend_type=BackendType.OPENAI,
            base_url=section.get('base_url', 'https://api.openai.com/v1'),
            api_key=api_key,
            model=section.get('model', 'gpt-4-turbo-preview'),
            max_tokens=int(section.get('max_tokens', '4096')),
            timeout=timeout,
            retry_attempts=retry_attempts,
            rate_limit_requests_per_minute=int(section.get('rate_limit_requests_per_minute', '500')),
            rate_limit_tokens_per_minute=int(section.get('rate_limit_tokens_per_minute', '150000')),
            supports_create=section.getboolean('supports_create', fallback=True),
            supports_query=section.getboolean('supports_query', fallback=True),
            supports_update=section.getboolean('supports_update', fallback=True),
            supports_delete=section.getboolean('supports_delete', fallback=True),
            additional_params=additional_params
        )
    
    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable references in configuration values.
        
        Args:
            value: Configuration value that may contain ${VAR_NAME} references
            
        Returns:
            Resolved value with environment variables substituted
        """
        if not value:
            return value
            
        # Handle ${VAR_NAME} format
        if value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, '')
        
        return value
    
    def _create_default_config(self) -> MCPServerConfig:
        """Create default configuration when config file is not available."""
        default_backend = BackendConfig(
            backend_type=BackendType.LOCAL,
            base_url="http://localhost:3000",
            timeout=30,
            retry_attempts=3,
            additional_params={
                'enable_persistence': True,
                'storage_path': './mcp_storage',
                'max_contexts': 10000
            }
        )
        
        return MCPServerConfig(
            backend_type=BackendType.LOCAL,
            backends={BackendType.LOCAL: default_backend},
            enable_backup=True,
            backup_backend=BackendType.LOCAL,
            enable_federation=False,
            fallback_backends=[],
            enable_cache=True,
            cache_ttl_seconds=3600,
            debug=False
        )
    
    def get_config(self) -> MCPServerConfig:
        """Get the current configuration, loading it if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> MCPServerConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()
    
    def validate_config(self, config: MCPServerConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check if primary backend is configured
        if config.backend_type not in config.backends:
            errors.append(f"Primary backend {config.backend_type.value} is not configured")
        
        # Check backup backend
        if config.enable_backup and config.backup_backend not in config.backends:
            errors.append(f"Backup backend {config.backup_backend.value} is not configured")
        
        # Check fallback backends
        for backend in config.fallback_backends:
            if backend not in config.backends:
                errors.append(f"Fallback backend {backend.value} is not configured")
        
        # Only validate API keys for backends that will actually be used
        backends_to_validate = {config.backend_type}
        if config.enable_backup:
            backends_to_validate.add(config.backup_backend)
        backends_to_validate.update(config.fallback_backends)
        
        # Validate backend-specific requirements only for backends that will be used
        for backend_type in backends_to_validate:
            if backend_type in config.backends:
                backend_config = config.backends[backend_type]
                if backend_type != BackendType.LOCAL and not backend_config.api_key:
                    errors.append(f"{backend_type.value} backend requires an API key")
        
        return errors