# Author: Fatih E. NAR
# Agentic AI Framework - Local MCP Backend
#
import uuid
import json
import pickle
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseMCPBackend
from ..schema import MCPContext

class LocalMCPBackend(BaseMCPBackend):
    """Local file-based MCP backend implementation."""
    
    def __init__(self, config):
        """Initialize the local backend."""
        super().__init__(config)
        self.storage_path = Path(config.additional_params.get('storage_path', './mcp_storage'))
        self.max_contexts = config.additional_params.get('max_contexts', 10000)
        self.enable_persistence = config.additional_params.get('enable_persistence', True)
        self.context_store: Dict[str, MCPContext] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize the local storage."""
        if self.enable_persistence:
            # Create storage directory if it doesn't exist
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing contexts if available
            await self._load_contexts()
        
        self.logger.info(f"Local backend initialized with {len(self.context_store)} contexts")
    
    async def _close_impl(self) -> None:
        """Save contexts to disk if persistence is enabled."""
        if self.enable_persistence:
            await self._save_contexts()
        
        self.context_store.clear()
    
    async def store_context(self, context: MCPContext) -> str:
        """Store a context locally."""
        self._validate_context(context)
        
        # Generate unique ID
        context_id = str(uuid.uuid4())
        
        # Check if we're at capacity
        if len(self.context_store) >= self.max_contexts:
            # Remove oldest context based on timestamp
            oldest_id = min(self.context_store.keys(), 
                          key=lambda k: self.context_store[k].header.timestamp)
            del self.context_store[oldest_id]
            self.logger.warning(f"Removed oldest context {oldest_id} due to capacity limit")
        
        # Store the context
        self.context_store[context_id] = context
        
        # Persist to disk if enabled
        if self.enable_persistence:
            await self._persist_context(context_id, context)
        
        self.logger.debug(f"Stored context {context_id}")
        return context_id
    
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve a context by ID."""
        return self.context_store.get(context_id)
    
    async def query_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Query contexts based on criteria."""
        filtered_contexts = {}
        
        try:
            # Filter by domain type
            domain_type = query.get("domain_type")
            correlation_id = query.get("correlation_id")
            intent_tags = query.get("intent_tags", [])
            scope = query.get("scope")
            priority_min = query.get("priority_min")
            priority_max = query.get("priority_max")
            
            for context_id, context in self.context_store.items():
                match = True
                
                # Filter by domain type
                if domain_type and context.metadata.domain_type != domain_type:
                    match = False
                
                # Filter by correlation ID
                if correlation_id and context.header.correlation_id != correlation_id:
                    match = False
                
                # Filter by intent tags (any tag must match)
                if intent_tags:
                    if not any(tag in context.metadata.intent_tags for tag in intent_tags):
                        match = False
                
                # Filter by scope
                if scope and context.metadata.scope != scope:
                    match = False
                
                # Filter by priority range
                if priority_min is not None and context.header.priority < priority_min:
                    match = False
                if priority_max is not None and context.header.priority > priority_max:
                    match = False
                
                if match:
                    filtered_contexts[context_id] = context
            
            self.logger.debug(f"Query returned {len(filtered_contexts)} contexts")
            return filtered_contexts
            
        except Exception as e:
            error_msg = self._format_query_error(query, str(e))
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def update_context(self, context_id: str, context: MCPContext) -> bool:
        """Update an existing context."""
        if context_id not in self.context_store:
            return False
        
        self._validate_context(context)
        self.context_store[context_id] = context
        
        # Persist to disk if enabled
        if self.enable_persistence:
            await self._persist_context(context_id, context)
        
        self.logger.debug(f"Updated context {context_id}")
        return True
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        if context_id not in self.context_store:
            return False
        
        del self.context_store[context_id]
        
        # Remove from disk if persistence is enabled
        if self.enable_persistence:
            await self._remove_context_file(context_id)
        
        self.logger.debug(f"Deleted context {context_id}")
        return True
    
    async def _load_contexts(self) -> None:
        """Load contexts from disk."""
        try:
            contexts_file = self.storage_path / "contexts.pkl"
            if contexts_file.exists():
                async with aiofiles.open(contexts_file, 'rb') as f:
                    data = await f.read()
                    self.context_store = pickle.loads(data)
                self.logger.info(f"Loaded {len(self.context_store)} contexts from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load contexts from disk: {str(e)}")
            self.context_store = {}
    
    async def _save_contexts(self) -> None:
        """Save contexts to disk."""
        try:
            contexts_file = self.storage_path / "contexts.pkl"
            async with aiofiles.open(contexts_file, 'wb') as f:
                data = pickle.dumps(self.context_store)
                await f.write(data)
            self.logger.debug(f"Saved {len(self.context_store)} contexts to disk")
        except Exception as e:
            self.logger.error(f"Failed to save contexts to disk: {str(e)}")
    
    async def _persist_context(self, context_id: str, context: MCPContext) -> None:
        """Persist a single context to disk."""
        try:
            context_file = self.storage_path / f"{context_id}.json"
            async with aiofiles.open(context_file, 'w') as f:
                await f.write(context.json(indent=2))
        except Exception as e:
            self.logger.warning(f"Failed to persist context {context_id}: {str(e)}")
    
    async def _remove_context_file(self, context_id: str) -> None:
        """Remove a context file from disk."""
        try:
            context_file = self.storage_path / f"{context_id}.json"
            if context_file.exists():
                context_file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to remove context file {context_id}: {str(e)}")
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Local backend health check."""
        health_info = {
            "status": "healthy",
            "contexts_count": len(self.context_store),
            "max_contexts": self.max_contexts,
            "storage_path": str(self.storage_path),
            "persistence_enabled": self.enable_persistence
        }
        
        # Check storage path accessibility
        if self.enable_persistence:
            try:
                if not self.storage_path.exists():
                    health_info["status"] = "warning"
                    health_info["message"] = "Storage path does not exist"
                elif not self.storage_path.is_dir():
                    health_info["status"] = "error"
                    health_info["message"] = "Storage path is not a directory"
                else:
                    # Check write permissions
                    test_file = self.storage_path / f"health_check_{uuid.uuid4()}.tmp"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        health_info["storage_writable"] = True
                    except Exception:
                        health_info["status"] = "warning" 
                        health_info["storage_writable"] = False
                        health_info["message"] = "Storage path is not writable"
            except Exception as e:
                health_info["status"] = "error"
                health_info["message"] = f"Storage check failed: {str(e)}"
        
        return health_info