# Author: Fatih E. NAR
# Agentic AI Framework - MCP Backend Factory
#
import logging
from typing import Dict, Any, Optional, List

from .config import MCPConfigManager, MCPServerConfig, BackendType, BackendConfig
from .backends.base import BaseMCPBackend
from .backends.local import LocalMCPBackend
from .backends.anthropic import AnthropicMCPBackend
from .backends.huggingface import HuggingFaceMCPBackend
from .backends.openai import OpenAIMCPBackend

class MCPBackendFactory:
    """Factory for creating and managing MCP backends."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the backend factory.
        
        Args:
            config_path: Path to the MCP configuration file
        """
        self.config_manager = MCPConfigManager(config_path)
        self.logger = logging.getLogger("mcp.factory")
        self._backends: Dict[BackendType, BaseMCPBackend] = {}
        self._config: Optional[MCPServerConfig] = None
        
    async def initialize(self) -> None:
        """Initialize the factory and load configuration."""
        self._config = self.config_manager.load_config()
        
        # Validate configuration
        errors = self.config_manager.validate_config(self._config)
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"MCP Backend Factory initialized with {len(self._config.backends)} backends")
    
    async def get_backend(self, backend_type: Optional[BackendType] = None) -> BaseMCPBackend:
        """Get a backend instance.
        
        Args:
            backend_type: Type of backend to get. If None, returns the primary backend.
            
        Returns:
            Backend instance
        """
        if not self._config:
            await self.initialize()
        
        # Use primary backend if not specified
        if backend_type is None:
            backend_type = self._config.backend_type
        
        # Check if backend is configured
        if backend_type not in self._config.backends:
            raise ValueError(f"Backend {backend_type.value} is not configured")
        
        # Create backend if not already created
        if backend_type not in self._backends:
            backend_config = self._config.backends[backend_type]
            backend = self._create_backend(backend_config)
            await backend.initialize()
            self._backends[backend_type] = backend
        
        return self._backends[backend_type]
    
    async def get_primary_backend(self) -> BaseMCPBackend:
        """Get the primary backend."""
        return await self.get_backend()
    
    async def get_backup_backend(self) -> Optional[BaseMCPBackend]:
        """Get the backup backend if configured."""
        if not self._config or not self._config.enable_backup:
            return None
        
        return await self.get_backend(self._config.backup_backend)
    
    async def get_fallback_backends(self) -> List[BaseMCPBackend]:
        """Get all fallback backends."""
        if not self._config or not self._config.enable_federation:
            return []
        
        fallback_backends = []
        for backend_type in self._config.fallback_backends:
            try:
                backend = await self.get_backend(backend_type)
                fallback_backends.append(backend)
            except Exception as e:
                self.logger.warning(f"Failed to initialize fallback backend {backend_type.value}: {str(e)}")
        
        return fallback_backends
    
    def _create_backend(self, config: BackendConfig) -> BaseMCPBackend:
        """Create a backend instance based on configuration.
        
        Args:
            config: Backend configuration
            
        Returns:
            Backend instance
        """
        backend_classes = {
            BackendType.LOCAL: LocalMCPBackend,
            BackendType.ANTHROPIC: AnthropicMCPBackend,
            BackendType.HUGGINGFACE: HuggingFaceMCPBackend,
            BackendType.OPENAI: OpenAIMCPBackend
        }
        
        backend_class = backend_classes.get(config.backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend type: {config.backend_type.value}")
        
        self.logger.info(f"Creating backend: {config.backend_type.value}")
        return backend_class(config)
    
    async def health_check_all_backends(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all configured backends.
        
        Returns:
            Dictionary mapping backend type to health status
        """
        if not self._config:
            await self.initialize()
        
        health_results = {}
        
        for backend_type in self._config.backends.keys():
            try:
                backend = await self.get_backend(backend_type)
                health_info = await backend.health_check()
                health_results[backend_type.value] = health_info
            except Exception as e:
                health_results[backend_type.value] = {
                    "status": "error",
                    "message": f"Failed to get backend: {str(e)}"
                }
        
        return health_results
    
    async def close_all_backends(self) -> None:
        """Close all backend instances."""
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception as e:
                self.logger.warning(f"Error closing backend: {str(e)}")
        
        self._backends.clear()
        self.logger.info("All backends closed")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = self.config_manager.reload_config()
        self.logger.info("Configuration reloaded")
    
    def get_config(self) -> Optional[MCPServerConfig]:
        """Get the current configuration."""
        return self._config
    
    def list_available_backends(self) -> List[str]:
        """List all configured backend types."""
        if not self._config:
            return []
        
        return [backend_type.value for backend_type in self._config.backends.keys()]
    
    def is_backend_available(self, backend_type: BackendType) -> bool:
        """Check if a backend type is configured and available."""
        if not self._config:
            return False
        
        return backend_type in self._config.backends