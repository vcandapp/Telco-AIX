# Author: Fatih E. NAR
# Agentic AI Framework - OpenAI MCP Backend
#
import aiohttp
import uuid
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseMCPBackend
from ..schema import MCPContext

class OpenAIMCPBackend(BaseMCPBackend):
    """OpenAI GPT MCP backend implementation."""
    
    def __init__(self, config):
        """Initialize the OpenAI backend."""
        super().__init__(config)
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutoNet-MCP-Client/1.0"
        }
        
        # Add organization header if provided
        org_id = config.additional_params.get('organization')
        if org_id:
            self.headers["OpenAI-Organization"] = org_id
            
        self.context_store: Dict[str, MCPContext] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    async def _initialize_impl(self) -> None:
        """Initialize the OpenAI client session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout
        )
        
        # Test connection
        await self._test_connection()
        
    async def _close_impl(self) -> None:
        """Close the OpenAI client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _test_connection(self) -> None:
        """Test connection to OpenAI API."""
        try:
            # Test with a simple completion
            test_payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=test_payload
            ) as response:
                if response.status == 401:
                    raise ValueError("Invalid OpenAI API key")
                elif response.status == 429:
                    raise ValueError("OpenAI API rate limit exceeded")
                elif response.status >= 500:
                    raise ValueError(f"OpenAI API server error: {response.status}")
                    
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")
    
    async def store_context(self, context: MCPContext) -> str:
        """Store context using OpenAI GPT for processing and embeddings for search."""
        self._validate_context(context)
        
        if not self.supports_operation("create"):
            raise ValueError("OpenAI backend does not support context creation")
        
        # Generate unique ID
        context_id = str(uuid.uuid4())
        
        try:
            # Generate embeddings for semantic search
            embedding = await self._generate_context_embedding(context)
            
            # Use GPT to process and summarize the context
            summary = await self._generate_context_summary(context)
            
            # Store context, embedding, and summary
            enriched_context = context.copy(deep=True)
            enriched_context.payload["_ai_summary"] = summary
            enriched_context.payload["_context_id"] = context_id
            
            self.context_store[context_id] = enriched_context
            self.embeddings_cache[context_id] = embedding
            
            self.logger.info(f"Stored context {context_id} in OpenAI backend")
            return context_id
            
        except Exception as e:
            self.logger.error(f"Failed to store context in OpenAI backend: {str(e)}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve context by ID."""
        return self.context_store.get(context_id)
    
    async def query_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Query contexts using GPT understanding and semantic search."""
        if not self.supports_operation("query"):
            raise ValueError("OpenAI backend does not support context querying")
        
        try:
            filtered_contexts = {}
            
            # Handle natural language queries with GPT
            if "query_text" in query:
                # Use GPT to understand the query intent
                query_analysis = await self._analyze_query_with_gpt(query["query_text"])
                
                # Combine with semantic search
                filtered_contexts = await self._intelligent_search(query["query_text"], query_analysis, query)
            else:
                # Traditional filtering
                filtered_contexts = await self._filter_contexts(query)
            
            self.logger.debug(f"Query returned {len(filtered_contexts)} contexts from OpenAI backend")
            return filtered_contexts
            
        except Exception as e:
            error_msg = self._format_query_error(query, str(e))
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def update_context(self, context_id: str, context: MCPContext) -> bool:
        """Update context with GPT analysis."""
        if not self.supports_operation("update"):
            raise ValueError("OpenAI backend does not support context updates")
        
        if context_id not in self.context_store:
            return False
        
        try:
            self._validate_context(context)
            
            # Regenerate embedding and summary
            embedding = await self._generate_context_embedding(context)
            summary = await self._generate_context_summary(context)
            
            # Update context with new AI analysis
            enriched_context = context.copy(deep=True)
            enriched_context.payload["_ai_summary"] = summary
            enriched_context.payload["_context_id"] = context_id
            enriched_context.payload["_updated_at"] = datetime.utcnow().isoformat()
            
            # Store updated context
            self.context_store[context_id] = enriched_context
            self.embeddings_cache[context_id] = embedding
            
            self.logger.info(f"Updated context {context_id} in OpenAI backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update context {context_id}: {str(e)}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context."""
        if not self.supports_operation("delete"):
            raise ValueError("OpenAI backend does not support context deletion")
        
        if context_id not in self.context_store:
            return False
        
        # Remove from both stores
        del self.context_store[context_id]
        if context_id in self.embeddings_cache:
            del self.embeddings_cache[context_id]
        
        self.logger.info(f"Deleted context {context_id} from OpenAI backend")
        return True
    
    async def _generate_context_embedding(self, context: MCPContext) -> np.ndarray:
        """Generate embedding for context using OpenAI embeddings."""
        context_text = self._context_to_text(context)
        
        embedding_model = self.config.additional_params.get('embedding_model', 'text-embedding-3-large')
        
        payload = {
            "model": embedding_model,
            "input": context_text,
            "encoding_format": "float"
        }
        
        async with self.session.post(
            f"{self.config.base_url}/embeddings",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Embedding generation failed: {error_text}")
            
            result = await response.json()
            return np.array(result["data"][0]["embedding"])
    
    async def _generate_context_summary(self, context: MCPContext) -> str:
        """Generate AI summary of context using GPT."""
        context_text = self._context_to_text(context)
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an AI assistant that analyzes network operation contexts. 
                    Provide a concise summary highlighting key insights, anomalies, and actionable information."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this network context and provide a brief summary:

{context_text}

Focus on:
1. Key metrics and their significance
2. Any anomalies or concerning patterns
3. Potential impact and urgency
4. Recommended actions or follow-up"""
                }
            ],
            "max_tokens": 300,
            "temperature": self.config.additional_params.get('temperature', 0.1)
        }
        
        async with self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Summary generation failed: {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _analyze_query_with_gpt(self, query_text: str) -> Dict[str, Any]:
        """Analyze natural language query using GPT."""
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a query analyzer for network operations. 
                    Extract structured search criteria from natural language queries.
                    Return JSON with extracted criteria."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this search query and extract relevant criteria:
"{query_text}"

Return JSON with these fields (only include if mentioned):
{{
  "domain_type": "network_data|incident|procedural|...",
  "priority_level": "high|medium|low",
  "intent_tags": ["tag1", "tag2"],
  "time_range": "recent|last_hour|today|...",
  "severity": "critical|high|medium|low",
  "component": "amf|smf|upf|...",
  "search_keywords": ["keyword1", "keyword2"]
}}"""
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        try:
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Try to parse JSON response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                
                return {}
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze query with GPT: {str(e)}")
            return {}
    
    async def _intelligent_search(self, query_text: str, query_analysis: Dict[str, Any], 
                                query_params: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Perform intelligent search combining semantic search and GPT analysis."""
        if not self.embeddings_cache:
            return {}
        
        # Generate embedding for query
        query_embedding = await self._generate_text_embedding(query_text)
        
        # Calculate similarities
        similarities = {}
        for context_id, context_embedding in self.embeddings_cache.items():
            similarity = self._cosine_similarity(query_embedding, context_embedding)
            similarities[context_id] = similarity
        
        # Sort by similarity
        sorted_contexts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Filter based on GPT analysis and query parameters
        similarity_threshold = query_params.get("similarity_threshold", 0.3)
        max_results = query_params.get("max_results", 10)
        
        filtered_contexts = {}
        for context_id, similarity in sorted_contexts[:max_results * 2]:  # Get more candidates
            if similarity >= similarity_threshold:
                context = self.context_store.get(context_id)
                if context and self._matches_intelligent_criteria(context, query_analysis, query_params):
                    filtered_contexts[context_id] = context
                    if len(filtered_contexts) >= max_results:
                        break
        
        return filtered_contexts
    
    def _matches_intelligent_criteria(self, context: MCPContext, query_analysis: Dict[str, Any], 
                                    query_params: Dict[str, Any]) -> bool:
        """Check if context matches intelligent search criteria."""
        # Traditional filters
        if not self._matches_traditional_criteria(context, query_params):
            return False
        
        # GPT analysis-based filters
        if "domain_type" in query_analysis and context.metadata.domain_type != query_analysis["domain_type"]:
            return False
        
        if "priority_level" in query_analysis:
            priority_mapping = {"high": (1, 3), "medium": (4, 6), "low": (7, 10)}
            min_p, max_p = priority_mapping.get(query_analysis["priority_level"], (1, 10))
            if not (min_p <= context.header.priority <= max_p):
                return False
        
        # Check for keyword matches in payload or summary
        search_keywords = query_analysis.get("search_keywords", [])
        if search_keywords:
            context_text = json.dumps(context.payload).lower()
            summary = context.payload.get("_ai_summary", "").lower()
            combined_text = f"{context_text} {summary}"
            
            if not any(keyword.lower() in combined_text for keyword in search_keywords):
                return False
        
        return True
    
    def _matches_traditional_criteria(self, context: MCPContext, query_params: Dict[str, Any]) -> bool:
        """Check traditional filtering criteria."""
        if "domain_type" in query_params and context.metadata.domain_type != query_params["domain_type"]:
            return False
        if "correlation_id" in query_params and context.header.correlation_id != query_params["correlation_id"]:
            return False
        if "priority_min" in query_params and context.header.priority < query_params["priority_min"]:
            return False
        if "priority_max" in query_params and context.header.priority > query_params["priority_max"]:
            return False
        
        return True
    
    async def _filter_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Traditional context filtering."""
        filtered_contexts = {}
        
        for context_id, context in self.context_store.items():
            if self._matches_traditional_criteria(context, query):
                filtered_contexts[context_id] = context
        
        return filtered_contexts
    
    async def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for arbitrary text."""
        embedding_model = self.config.additional_params.get('embedding_model', 'text-embedding-3-large')
        
        payload = {
            "model": embedding_model,
            "input": text,
            "encoding_format": "float"
        }
        
        async with self.session.post(
            f"{self.config.base_url}/embeddings",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Text embedding generation failed: {error_text}")
            
            result = await response.json()
            return np.array(result["data"][0]["embedding"])
    
    def _context_to_text(self, context: MCPContext) -> str:
        """Convert context to text representation."""
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
        payload_copy = context.payload.copy()
        # Remove internal AI fields for embedding
        payload_copy.pop("_ai_summary", None)
        payload_copy.pop("_context_id", None)
        payload_copy.pop("_updated_at", None)
        
        payload_text = json.dumps(payload_copy, indent=None)
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
        """OpenAI backend health check."""
        try:
            # Test completion endpoint
            test_payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "health check"}],
                "max_tokens": 5
            }
            
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=test_payload
            ) as response:
                health_info = {
                    "status": "healthy" if response.status == 200 else "error",
                    "api_status": response.status,
                    "model": self.config.model,
                    "embedding_model": self.config.additional_params.get('embedding_model'),
                    "contexts_count": len(self.context_store),
                    "embeddings_count": len(self.embeddings_cache),
                    "rate_limits": {
                        "requests_per_minute": self.config.rate_limit_requests_per_minute,
                        "tokens_per_minute": self.config.rate_limit_tokens_per_minute
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