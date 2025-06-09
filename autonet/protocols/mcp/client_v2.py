# Author: Fatih E. NAR
# Agentic AI Framework - Enhanced MCP Client with Configurable Backends
#
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .schema import MCPContext, MCPHeader, DomainMetadata, ContextType, ConfidenceLevel
from .backend_factory import MCPBackendFactory
from .config import BackendType
from .backends.base import BaseMCPBackend

class MCPClientV2:
    """Enhanced MCP client with configurable backend support."""
    
    def __init__(self, client_id: str, config_path: Optional[str] = None):
        """Initialize the enhanced MCP client.
        
        Args:
            client_id: Unique identifier for this client
            config_path: Path to MCP configuration file
        """
        self.client_id = client_id
        self.logger = logging.getLogger(f"mcp.client.{client_id}")
        self.backend_factory = MCPBackendFactory(config_path)
        self._initialized = False
        self._cache: Dict[str, MCPContext] = {}
        self._cache_enabled = False
        self._cache_ttl = 3600  # 1 hour default
        
    async def initialize(self) -> None:
        """Initialize the client and backend factory."""
        if not self._initialized:
            await self.backend_factory.initialize()
            
            # Configure caching if enabled
            config = self.backend_factory.get_config()
            if config:
                self._cache_enabled = config.enable_cache
                self._cache_ttl = config.cache_ttl_seconds
            
            self._initialized = True
            self.logger.info(f"MCP client {self.client_id} initialized")
    
    async def close(self) -> None:
        """Close the client and all backends."""
        if self._initialized:
            await self.backend_factory.close_all_backends()
            self._cache.clear()
            self._initialized = False
            self.logger.info(f"MCP client {self.client_id} closed")
    
    def create_context(self, 
                     domain_type: ContextType, 
                     payload: Dict[str, Any],
                     priority: int = 5,
                     confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                     correlation_id: Optional[str] = None,
                     intent_tags: Optional[List[str]] = None,
                     scope: Optional[str] = None) -> MCPContext:
        """Create a new MCP context object.
        
        Args:
            domain_type: Type of context being created
            payload: The actual context data
            priority: Priority level (1-10, where 1 is highest)
            confidence: Confidence level for the context
            correlation_id: Optional correlation ID for related contexts
            intent_tags: Optional tags describing intent
            scope: Optional scope identifier
            
        Returns:
            A new MCPContext object
        """
        header = MCPHeader(
            source_id=self.client_id,
            timestamp=datetime.utcnow(),
            priority=priority,
            correlation_id=correlation_id
        )
        
        metadata = DomainMetadata(
            domain_type=domain_type,
            confidence=confidence,
            intent_tags=intent_tags or [],
            scope=scope
        )
        
        return MCPContext(
            header=header,
            metadata=metadata,
            payload=payload
        )
    
    async def send_context(self, context: MCPContext, 
                          backend_type: Optional[BackendType] = None) -> str:
        """Send context to the specified backend.
        
        Args:
            context: The context to send
            backend_type: Specific backend to use. If None, uses primary backend.
            
        Returns:
            The context ID assigned by the backend
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get the appropriate backend
            backend = await self.backend_factory.get_backend(backend_type)
            
            # Store context
            context_id = await backend.store_context(context)
            
            # Cache if enabled
            if self._cache_enabled:
                self._cache[context_id] = context
            
            self.logger.debug(f"Context {context_id} sent to {backend.config.backend_type.value} backend")
            return context_id
            
        except Exception as e:
            # Try backup backend if available and primary failed
            if backend_type is None:  # Only retry for primary backend
                backup_backend = await self.backend_factory.get_backup_backend()
                if backup_backend:
                    try:
                        context_id = await backup_backend.store_context(context)
                        self.logger.warning(f"Used backup backend after primary failure: {str(e)}")
                        return context_id
                    except Exception as backup_error:
                        self.logger.error(f"Backup backend also failed: {str(backup_error)}")
            
            self.logger.error(f"Failed to send context: {str(e)}")
            raise
    
    async def get_context(self, context_id: str, 
                         backend_type: Optional[BackendType] = None) -> Optional[MCPContext]:
        """Retrieve a context by ID.
        
        Args:
            context_id: The unique identifier of the context
            backend_type: Specific backend to query. If None, tries primary then fallbacks.
            
        Returns:
            The context if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        if self._cache_enabled and context_id in self._cache:
            self.logger.debug(f"Context {context_id} found in cache")
            return self._cache[context_id]
        
        # Try specific backend
        if backend_type:
            try:
                backend = await self.backend_factory.get_backend(backend_type)
                context = await backend.get_context(context_id)
                
                # Cache if found and caching enabled
                if context and self._cache_enabled:
                    self._cache[context_id] = context
                
                return context
            except Exception as e:
                self.logger.error(f"Failed to get context from {backend_type.value}: {str(e)}")
                return None
        
        # Try primary backend first
        try:
            primary_backend = await self.backend_factory.get_primary_backend()
            context = await primary_backend.get_context(context_id)
            
            if context:
                if self._cache_enabled:
                    self._cache[context_id] = context
                return context
        except Exception as e:
            self.logger.warning(f"Primary backend failed to get context: {str(e)}")
        
        # Try fallback backends
        fallback_backends = await self.backend_factory.get_fallback_backends()
        for backend in fallback_backends:
            try:
                context = await backend.get_context(context_id)
                if context:
                    if self._cache_enabled:
                        self._cache[context_id] = context
                    self.logger.debug(f"Context {context_id} found in fallback backend {backend.config.backend_type.value}")
                    return context
            except Exception as e:
                self.logger.warning(f"Fallback backend {backend.config.backend_type.value} failed: {str(e)}")
        
        return None
    
    async def query_context(self, query: Dict[str, Any], 
                          backend_type: Optional[BackendType] = None) -> Dict[str, MCPContext]:
        """Query contexts from the specified backend.
        
        Args:
            query: The query parameters
            backend_type: Specific backend to query. If None, queries primary backend.
            
        Returns:
            Dictionary of context_id -> context for matching contexts
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get the appropriate backend
            backend = await self.backend_factory.get_backend(backend_type)
            
            # Query contexts
            contexts = await backend.query_contexts(query)
            
            # Cache results if enabled
            if self._cache_enabled:
                for context_id, context in contexts.items():
                    self._cache[context_id] = context
            
            self.logger.debug(f"Query returned {len(contexts)} contexts from {backend.config.backend_type.value}")
            return contexts
            
        except Exception as e:
            # Try backup backend if available and primary failed
            if backend_type is None:
                backup_backend = await self.backend_factory.get_backup_backend()
                if backup_backend:
                    try:
                        contexts = await backup_backend.query_contexts(query)
                        self.logger.warning(f"Used backup backend for query after primary failure: {str(e)}")
                        return contexts
                    except Exception as backup_error:
                        self.logger.error(f"Backup backend query also failed: {str(backup_error)}")
            
            self.logger.error(f"Failed to query contexts: {str(e)}")
            raise
    
    async def update_context(self, context_id: str, context: MCPContext,
                           backend_type: Optional[BackendType] = None) -> bool:
        """Update an existing context.
        
        Args:
            context_id: The unique identifier of the context
            context: The updated context
            backend_type: Specific backend to update. If None, uses primary backend.
            
        Returns:
            True if the context was updated, False if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get the appropriate backend
            backend = await self.backend_factory.get_backend(backend_type)
            
            # Check if backend supports updates
            if not backend.supports_operation("update"):
                self.logger.warning(f"Backend {backend.config.backend_type.value} does not support updates")
                return False
            
            # Update context
            success = await backend.update_context(context_id, context)
            
            # Update cache if successful and caching enabled
            if success and self._cache_enabled:
                self._cache[context_id] = context
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update context {context_id}: {str(e)}")
            return False
    
    async def delete_context(self, context_id: str,
                           backend_type: Optional[BackendType] = None) -> bool:
        """Delete a context.
        
        Args:
            context_id: The unique identifier of the context
            backend_type: Specific backend to delete from. If None, uses primary backend.
            
        Returns:
            True if the context was deleted, False if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get the appropriate backend
            backend = await self.backend_factory.get_backend(backend_type)
            
            # Check if backend supports deletion
            if not backend.supports_operation("delete"):
                self.logger.warning(f"Backend {backend.config.backend_type.value} does not support deletion")
                return False
            
            # Delete context
            success = await backend.delete_context(context_id)
            
            # Remove from cache if successful
            if success and self._cache_enabled and context_id in self._cache:
                del self._cache[context_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete context {context_id}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all backends.
        
        Returns:
            Health status for all backends
        """
        if not self._initialized:
            await self.initialize()
        
        health_results = await self.backend_factory.health_check_all_backends()
        
        # Add client-specific health info
        health_results["client"] = {
            "client_id": self.client_id,
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "backends_available": self.backend_factory.list_available_backends()
        }
        
        return health_results
    
    def clear_cache(self) -> None:
        """Clear the local cache."""
        self._cache.clear()
        self.logger.debug("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self._cache_enabled,
            "size": len(self._cache),
            "ttl_seconds": self._cache_ttl
        }
    
    async def list_backends(self) -> List[str]:
        """List all available backend types."""
        if not self._initialized:
            await self.initialize()
        
        return self.backend_factory.list_available_backends()
    
    async def switch_primary_backend(self, backend_type: BackendType) -> bool:
        """Switch to a different primary backend.
        
        Args:
            backend_type: New primary backend type
            
        Returns:
            True if switch was successful
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.backend_factory.is_backend_available(backend_type):
            self.logger.error(f"Backend {backend_type.value} is not available")
            return False
        
        try:
            # Test the new backend
            backend = await self.backend_factory.get_backend(backend_type)
            health = await backend.health_check()
            
            if health.get("status") != "healthy":
                self.logger.error(f"Backend {backend_type.value} is not healthy: {health}")
                return False
            
            # Update configuration
            config = self.backend_factory.get_config()
            if config:
                config.backend_type = backend_type
                self.logger.info(f"Switched primary backend to {backend_type.value}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to switch to backend {backend_type.value}: {str(e)}")
            return False