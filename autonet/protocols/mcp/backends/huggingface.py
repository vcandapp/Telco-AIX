# Author: Fatih E. NAR
# Agentic AI Framework - HuggingFace MCP Backend
#
import aiohttp
import uuid
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseMCPBackend
from ..schema import MCPContext

class HuggingFaceMCPBackend(BaseMCPBackend):
    """HuggingFace Transformers MCP backend implementation."""
    
    def __init__(self, config):
        """Initialize the HuggingFace backend."""
        super().__init__(config)
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutoNet-MCP-Client/1.0"
        }
        self.context_store: Dict[str, MCPContext] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize the HuggingFace client session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout
        )
        
        # Test connection
        await self._test_connection()
        
    async def _close_impl(self) -> None:
        """Close the HuggingFace client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _test_connection(self) -> None:
        """Test connection to HuggingFace API."""
        try:
            # Test with embedding model first
            context_model = self.config.additional_params.get('context_model', 'sentence-transformers/all-MiniLM-L6-v2')
            test_url = f"{self.config.base_url}/{context_model}"
            
            test_payload = {
                "inputs": "test connection",
                "options": {
                    "use_cache": False,
                    "wait_for_model": True
                }
            }
            
            async with self.session.post(test_url, json=test_payload) as response:
                if response.status == 401:
                    raise ValueError("Invalid HuggingFace API key")
                elif response.status == 429:
                    raise ValueError("HuggingFace API rate limit exceeded")
                elif response.status >= 500:
                    raise ValueError(f"HuggingFace API server error: {response.status}")
                    
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to connect to HuggingFace API: {str(e)}")
    
    async def store_context(self, context: MCPContext) -> str:
        """Store context using HuggingFace embeddings for semantic storage."""
        self._validate_context(context)
        
        if not self.supports_operation("create"):
            raise ValueError("HuggingFace backend does not support context creation")
        
        # Generate unique ID
        context_id = str(uuid.uuid4())
        
        try:
            # Generate embeddings for the context
            embedding = await self._generate_context_embedding(context)
            
            # Store context and embedding
            self.context_store[context_id] = context
            self.embeddings_cache[context_id] = embedding
            
            self.logger.info(f"Stored context {context_id} in HuggingFace backend")
            return context_id
            
        except Exception as e:
            self.logger.error(f"Failed to store context in HuggingFace backend: {str(e)}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve context by ID."""
        return self.context_store.get(context_id)
    
    async def query_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Query contexts using semantic search with embeddings."""
        if not self.supports_operation("query"):
            raise ValueError("HuggingFace backend does not support context querying")
        
        try:
            filtered_contexts = {}
            
            # Handle text-based semantic search
            if "query_text" in query:
                filtered_contexts = await self._semantic_search(query["query_text"], query)
            else:
                # Traditional filtering
                filtered_contexts = await self._filter_contexts(query)
            
            self.logger.debug(f"Query returned {len(filtered_contexts)} contexts from HuggingFace backend")
            return filtered_contexts
            
        except Exception as e:
            error_msg = self._format_query_error(query, str(e))
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def update_context(self, context_id: str, context: MCPContext) -> bool:
        """Update context (not supported by HuggingFace backend)."""
        if not self.supports_operation("update"):
            self.logger.warning("HuggingFace backend does not support context updates")
            return False
        
        if context_id not in self.context_store:
            return False
        
        try:
            self._validate_context(context)
            
            # Regenerate embedding for updated context
            embedding = await self._generate_context_embedding(context)
            
            # Update context and embedding
            self.context_store[context_id] = context
            self.embeddings_cache[context_id] = embedding
            
            self.logger.info(f"Updated context {context_id} in HuggingFace backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update context {context_id}: {str(e)}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context (not supported by HuggingFace backend)."""
        if not self.supports_operation("delete"):
            self.logger.warning("HuggingFace backend does not support context deletion")
            return False
        
        if context_id not in self.context_store:
            return False
        
        # Remove from both stores
        del self.context_store[context_id]
        if context_id in self.embeddings_cache:
            del self.embeddings_cache[context_id]
        
        self.logger.info(f"Deleted context {context_id} from HuggingFace backend")
        return True
    
    async def _generate_context_embedding(self, context: MCPContext) -> np.ndarray:
        """Generate embedding for context using HuggingFace model."""
        # Create text representation of context
        context_text = self._context_to_text(context)
        
        # Use embedding model
        context_model = self.config.additional_params.get('context_model', 'sentence-transformers/all-MiniLM-L6-v2')
        embedding_url = f"{self.config.base_url}/{context_model}"
        
        payload = {
            "inputs": context_text,
            "options": {
                "use_cache": self.config.additional_params.get('use_cache', True),
                "wait_for_model": self.config.additional_params.get('wait_for_model', True)
            }
        }
        
        async with self.session.post(embedding_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Embedding generation failed: {error_text}")
            
            result = await response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Direct embedding array
                    return np.array(result[0])
                elif isinstance(result[0], dict) and 'embedding' in result[0]:
                    # Embedding in object
                    return np.array(result[0]['embedding'])
            
            raise ValueError("Unexpected embedding response format")
    
    async def _semantic_search(self, query_text: str, query_params: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Perform semantic search using embeddings."""
        if not self.embeddings_cache:
            return {}
        
        # Generate embedding for query
        query_embedding = await self._generate_text_embedding(query_text)
        
        # Calculate similarities
        similarities = {}
        for context_id, context_embedding in self.embeddings_cache.items():
            similarity = self._cosine_similarity(query_embedding, context_embedding)
            similarities[context_id] = similarity
        
        # Sort by similarity and get top results
        similarity_threshold = query_params.get("similarity_threshold", 0.5)
        max_results = query_params.get("max_results", 10)
        
        sorted_contexts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        filtered_contexts = {}
        for context_id, similarity in sorted_contexts[:max_results]:
            if similarity >= similarity_threshold:
                context = self.context_store.get(context_id)
                if context and self._matches_additional_criteria(context, query_params):
                    filtered_contexts[context_id] = context
        
        return filtered_contexts
    
    async def _filter_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Filter contexts using traditional criteria."""
        filtered_contexts = {}
        
        domain_type = query.get("domain_type")
        correlation_id = query.get("correlation_id")
        intent_tags = query.get("intent_tags", [])
        scope = query.get("scope")
        priority_min = query.get("priority_min")
        priority_max = query.get("priority_max")
        
        for context_id, context in self.context_store.items():
            match = True
            
            # Apply filters
            if domain_type and context.metadata.domain_type != domain_type:
                match = False
            if correlation_id and context.header.correlation_id != correlation_id:
                match = False
            if intent_tags and not any(tag in context.metadata.intent_tags for tag in intent_tags):
                match = False
            if scope and context.metadata.scope != scope:
                match = False
            if priority_min is not None and context.header.priority < priority_min:
                match = False
            if priority_max is not None and context.header.priority > priority_max:
                match = False
            
            if match:
                filtered_contexts[context_id] = context
        
        return filtered_contexts
    
    def _matches_additional_criteria(self, context: MCPContext, query_params: Dict[str, Any]) -> bool:
        """Check if context matches additional query criteria."""
        # Domain type filter
        if "domain_type" in query_params and context.metadata.domain_type != query_params["domain_type"]:
            return False
        
        # Priority filter
        if "priority_min" in query_params and context.header.priority < query_params["priority_min"]:
            return False
        if "priority_max" in query_params and context.header.priority > query_params["priority_max"]:
            return False
        
        # Intent tags filter
        if "intent_tags" in query_params:
            required_tags = query_params["intent_tags"]
            if not any(tag in context.metadata.intent_tags for tag in required_tags):
                return False
        
        return True
    
    async def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for arbitrary text."""
        context_model = self.config.additional_params.get('context_model', 'sentence-transformers/all-MiniLM-L6-v2')
        embedding_url = f"{self.config.base_url}/{context_model}"
        
        payload = {
            "inputs": text,
            "options": {
                "use_cache": self.config.additional_params.get('use_cache', True),
                "wait_for_model": self.config.additional_params.get('wait_for_model', True)
            }
        }
        
        async with self.session.post(embedding_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Text embedding generation failed: {error_text}")
            
            result = await response.json()
            
            # Handle response format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return np.array(result[0])
                elif isinstance(result[0], dict) and 'embedding' in result[0]:
                    return np.array(result[0]['embedding'])
            
            raise ValueError("Unexpected text embedding response format")
    
    def _context_to_text(self, context: MCPContext) -> str:
        """Convert context to text representation for embedding."""
        text_parts = [
            f"Domain: {context.metadata.domain_type}",
            f"Source: {context.header.source_id}",
            f"Priority: {context.header.priority}",
            f"Confidence: {context.metadata.confidence}",
            f"Tags: {', '.join(context.metadata.intent_tags)}"
        ]
        
        if context.metadata.scope:
            text_parts.append(f"Scope: {context.metadata.scope}")
        
        if context.header.correlation_id:
            text_parts.append(f"Correlation: {context.header.correlation_id}")
        
        # Add payload content
        payload_text = json.dumps(context.payload, indent=None)
        text_parts.append(f"Data: {payload_text}")
        
        return " | ".join(text_parts)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """HuggingFace backend health check."""
        try:
            # Test embedding model
            context_model = self.config.additional_params.get('context_model', 'sentence-transformers/all-MiniLM-L6-v2')
            test_url = f"{self.config.base_url}/{context_model}"
            
            test_payload = {
                "inputs": "health check",
                "options": {"use_cache": False}
            }
            
            async with self.session.post(test_url, json=test_payload) as response:
                health_info = {
                    "status": "healthy" if response.status == 200 else "error",
                    "api_status": response.status,
                    "embedding_model": context_model,
                    "contexts_count": len(self.context_store),
                    "embeddings_count": len(self.embeddings_cache),
                    "rate_limits": {
                        "requests_per_minute": self.config.rate_limit_requests_per_minute,
                        "daily_quota": self.config.additional_params.get('daily_quota')
                    }
                }
                
                if response.status != 200:
                    error_text = await response.text()
                    health_info["message"] = error_text
                
                return health_info
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "contexts_count": len(self.context_store),
                "embeddings_count": len(self.embeddings_cache)
            }