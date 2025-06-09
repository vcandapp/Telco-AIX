# Author: Fatih E. NAR
# Agentic AI Framework - Base MCP Backend
#
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..schema import MCPContext
from ..config import BackendConfig

class BaseMCPBackend(ABC):
    """Abstract base class for MCP backend implementations."""
    
    def __init__(self, config: BackendConfig):
        """Initialize the backend with configuration.
        
        Args:
            config: Backend configuration object
        """
        self.config = config
        self.logger = logging.getLogger(f"mcp.backend.{config.backend_type.value}")
        self._initialized = False
        self._session = None
        
    async def initialize(self) -> None:
        """Initialize the backend."""
        if not self._initialized:
            await self._initialize_impl()
            self._initialized = True
            self.logger.info(f"Backend {self.config.backend_type.value} initialized")
    
    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        if self._initialized:
            await self._close_impl()
            self._initialized = False
            self.logger.info(f"Backend {self.config.backend_type.value} closed")
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Backend-specific initialization implementation."""
        pass
    
    @abstractmethod
    async def _close_impl(self) -> None:
        """Backend-specific cleanup implementation."""
        pass
    
    @abstractmethod
    async def store_context(self, context: MCPContext) -> str:
        """Store a context in the backend.
        
        Args:
            context: The context to store
            
        Returns:
            The unique identifier for the stored context
        """
        pass
    
    @abstractmethod
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve a context by ID.
        
        Args:
            context_id: The unique identifier of the context
            
        Returns:
            The context if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def query_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Query contexts based on criteria.
        
        Args:
            query: Query parameters
            
        Returns:
            Dictionary of context_id -> context for matching contexts
        """
        pass
    
    @abstractmethod
    async def update_context(self, context_id: str, context: MCPContext) -> bool:
        """Update an existing context.
        
        Args:
            context_id: The unique identifier of the context
            context: The updated context
            
        Returns:
            True if the context was updated, False if not found
        """
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context.
        
        Args:
            context_id: The unique identifier of the context
            
        Returns:
            True if the context was deleted, False if not found
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the backend.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self._initialized:
                return {
                    "status": "error",
                    "message": "Backend not initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Perform backend-specific health check
            health_info = await self._health_check_impl()
            health_info.update({
                "backend_type": self.config.backend_type.value,
                "timestamp": datetime.utcnow().isoformat()
            })
            return health_info
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "backend_type": self.config.backend_type.value,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Backend-specific health check implementation.
        
        Returns:
            Dictionary with health status information
        """
        return {"status": "healthy"}
    
    def supports_operation(self, operation: str) -> bool:
        """Check if the backend supports a specific operation.
        
        Args:
            operation: Operation name (create, query, update, delete)
            
        Returns:
            True if the operation is supported
        """
        operation_map = {
            "create": self.config.supports_create,
            "query": self.config.supports_query,
            "update": self.config.supports_update,
            "delete": self.config.supports_delete
        }
        return operation_map.get(operation, False)
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with exponential backoff.
        
        Args:
            operation: The async operation to retry
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Operation failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Operation failed after {self.config.retry_attempts} attempts")
        
        if last_exception:
            raise last_exception
    
    def _validate_context(self, context: MCPContext) -> None:
        """Validate a context object.
        
        Args:
            context: Context to validate
            
        Raises:
            ValueError: If the context is invalid
        """
        if not context.header.source_id:
            raise ValueError("Context must have a source_id")
        
        if not context.payload:
            raise ValueError("Context must have payload data")
    
    def _format_query_error(self, query: Dict[str, Any], error: str) -> str:
        """Format a query error message.
        
        Args:
            query: The query that failed
            error: Error message
            
        Returns:
            Formatted error message
        """
        return f"Query failed: {error}. Query: {query}"