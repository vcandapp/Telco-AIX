# Author: Fatih E. NAR
# Agentic AI Framework - Anthropic MCP Backend
#
import aiohttp
import uuid
import json
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseMCPBackend
from ..schema import MCPContext

class AnthropicMCPBackend(BaseMCPBackend):
    """Anthropic Claude MCP backend implementation."""
    
    def __init__(self, config):
        """Initialize the Anthropic backend."""
        super().__init__(config)
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutoNet-MCP-Client/1.0",
            "anthropic-version": "2023-06-01"
        }
        
    async def _initialize_impl(self) -> None:
        """Initialize the Anthropic client session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout
        )
        
        # Test connection
        await self._test_connection()
        
    async def _close_impl(self) -> None:
        """Close the Anthropic client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _test_connection(self) -> None:
        """Test connection to Anthropic API."""
        try:
            # Use a simple message completion to test the connection
            test_payload = {
                "model": self.config.model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            async with self.session.post(
                f"{self.config.base_url}/messages", 
                json=test_payload
            ) as response:
                if response.status == 401:
                    raise ValueError("Invalid Anthropic API key")
                elif response.status == 429:
                    raise ValueError("Anthropic API rate limit exceeded")
                elif response.status >= 500:
                    raise ValueError(f"Anthropic API server error: {response.status}")
                    
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to connect to Anthropic API: {str(e)}")
    
    async def store_context(self, context: MCPContext) -> str:
        """Store context using Anthropic's Claude API for processing and storage."""
        self._validate_context(context)
        
        if not self.supports_operation("create"):
            raise ValueError("Anthropic backend does not support context creation")
        
        # Generate unique ID
        context_id = str(uuid.uuid4())
        
        # Prepare context for Claude processing
        context_data = self._prepare_context_for_claude(context)
        
        # Use Claude to process and store the context
        storage_payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Please process and store this network context data with ID {context_id}:

Context Type: {context.metadata.domain_type}
Source: {context.header.source_id}
Priority: {context.header.priority}
Confidence: {context.metadata.confidence}
Tags: {', '.join(context.metadata.intent_tags)}
Correlation ID: {context.header.correlation_id}

Data: {json.dumps(context_data, indent=2)}

Please acknowledge storage and provide a summary of the key information."""
                }
            ],
            "metadata": {
                "context_id": context_id,
                "operation": "store_context",
                "domain_type": context.metadata.domain_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            response = await self._retry_operation(self._make_claude_request, storage_payload)
            
            # Log successful storage
            self.logger.info(f"Stored context {context_id} in Anthropic backend")
            return context_id
            
        except Exception as e:
            self.logger.error(f"Failed to store context in Anthropic backend: {str(e)}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Retrieve context by ID using Claude's memory and processing."""
        if not self.supports_operation("query"):
            raise ValueError("Anthropic backend does not support context querying")
        
        retrieval_payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user", 
                    "content": f"""Please retrieve the network context data with ID: {context_id}

If you have this context stored, please return it in JSON format with the following structure:
{{
  "found": true,
  "context_id": "{context_id}",
  "header": {{
    "source_id": "...",
    "timestamp": "...",
    "priority": ...,
    "correlation_id": "..."
  }},
  "metadata": {{
    "domain_type": "...",
    "confidence": "...",
    "intent_tags": [...],
    "scope": "..."
  }},
  "payload": {{...}}
}}

If not found, return: {{"found": false, "context_id": "{context_id}"}}"""
                }
            ]
        }
        
        try:
            response = await self._retry_operation(self._make_claude_request, retrieval_payload)
            
            # Parse Claude's response to extract context
            context = self._parse_claude_context_response(response, context_id)
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve context {context_id} from Anthropic backend: {str(e)}")
            return None
    
    async def query_contexts(self, query: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Query contexts using Claude's understanding and memory."""
        if not self.supports_operation("query"):
            raise ValueError("Anthropic backend does not support context querying")
        
        # Convert query to natural language for Claude
        query_description = self._build_query_description(query)
        
        query_payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Please search for network contexts matching these criteria:

{query_description}

Return results in JSON format as an array of context objects:
{{
  "results": [
    {{
      "context_id": "...",
      "header": {{...}},
      "metadata": {{...}},
      "payload": {{...}}
    }}
  ],
  "count": ...
}}

If no contexts match, return: {{"results": [], "count": 0}}"""
                }
            ]
        }
        
        try:
            response = await self._retry_operation(self._make_claude_request, query_payload)
            contexts = self._parse_claude_query_response(response)
            
            self.logger.debug(f"Query returned {len(contexts)} contexts from Anthropic backend")
            return contexts
            
        except Exception as e:
            error_msg = self._format_query_error(query, str(e))
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def update_context(self, context_id: str, context: MCPContext) -> bool:
        """Update context using Claude's processing."""
        if not self.supports_operation("update"):
            raise ValueError("Anthropic backend does not support context updates")
        
        self._validate_context(context)
        
        context_data = self._prepare_context_for_claude(context)
        
        update_payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Please update the network context with ID {context_id} with this new data:

Updated Context Type: {context.metadata.domain_type}
Source: {context.header.source_id}
Priority: {context.header.priority}
Confidence: {context.metadata.confidence}
Tags: {', '.join(context.metadata.intent_tags)}

Updated Data: {json.dumps(context_data, indent=2)}

Please confirm the update was successful and provide a brief summary of changes."""
                }
            ]
        }
        
        try:
            await self._retry_operation(self._make_claude_request, update_payload)
            self.logger.info(f"Updated context {context_id} in Anthropic backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update context {context_id}: {str(e)}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context using Claude's processing."""
        if not self.supports_operation("delete"):
            raise ValueError("Anthropic backend does not support context deletion")
        
        delete_payload = {
            "model": self.config.model,
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Please delete the network context with ID: {context_id}

Confirm deletion by responding with: {{"deleted": true, "context_id": "{context_id}"}}
If context not found, respond with: {{"deleted": false, "context_id": "{context_id}", "reason": "not_found"}}"""
                }
            ]
        }
        
        try:
            response = await self._retry_operation(self._make_claude_request, delete_payload)
            
            # Parse deletion confirmation
            deleted = self._parse_claude_deletion_response(response, context_id)
            if deleted:
                self.logger.info(f"Deleted context {context_id} from Anthropic backend")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete context {context_id}: {str(e)}")
            return False
    
    async def _make_claude_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Claude API."""
        async with self.session.post(
            f"{self.config.base_url}/messages",
            json=payload
        ) as response:
            if response.status == 429:
                raise ValueError("Rate limit exceeded")
            elif response.status == 401:
                raise ValueError("Invalid API key")
            elif response.status >= 400:
                error_text = await response.text()
                raise ValueError(f"API error {response.status}: {error_text}")
            
            return await response.json()
    
    def _prepare_context_for_claude(self, context: MCPContext) -> Dict[str, Any]:
        """Prepare context data for Claude processing."""
        return {
            "header": context.header.dict(),
            "metadata": context.metadata.dict(),
            "payload": context.payload
        }
    
    def _parse_claude_context_response(self, response: Dict[str, Any], context_id: str) -> Optional[MCPContext]:
        """Parse Claude's response to extract context data."""
        try:
            content = response.get("content", [{}])[0].get("text", "")
            
            # Try to extract JSON from Claude's response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                if data.get("found"):
                    # Convert back to MCPContext
                    return MCPContext.parse_obj(data)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Claude context response: {str(e)}")
            return None
    
    def _parse_claude_query_response(self, response: Dict[str, Any]) -> Dict[str, MCPContext]:
        """Parse Claude's query response."""
        contexts = {}
        
        try:
            content = response.get("content", [{}])[0].get("text", "")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                for result in data.get("results", []):
                    context_id = result.get("context_id")
                    if context_id:
                        try:
                            context = MCPContext.parse_obj(result)
                            contexts[context_id] = context
                        except Exception as e:
                            self.logger.warning(f"Failed to parse context {context_id}: {str(e)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Claude query response: {str(e)}")
        
        return contexts
    
    def _parse_claude_deletion_response(self, response: Dict[str, Any], context_id: str) -> bool:
        """Parse Claude's deletion response."""
        try:
            content = response.get("content", [{}])[0].get("text", "")
            
            # Look for deletion confirmation
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("deleted", False)
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Claude deletion response: {str(e)}")
            return False
    
    def _build_query_description(self, query: Dict[str, Any]) -> str:
        """Build natural language query description for Claude."""
        criteria = []
        
        if "domain_type" in query:
            criteria.append(f"Domain type: {query['domain_type']}")
        
        if "correlation_id" in query:
            criteria.append(f"Correlation ID: {query['correlation_id']}")
        
        if "intent_tags" in query:
            criteria.append(f"Intent tags containing: {', '.join(query['intent_tags'])}")
        
        if "scope" in query:
            criteria.append(f"Scope: {query['scope']}")
        
        if "priority_min" in query or "priority_max" in query:
            priority_range = []
            if "priority_min" in query:
                priority_range.append(f"minimum priority {query['priority_min']}")
            if "priority_max" in query:
                priority_range.append(f"maximum priority {query['priority_max']}")
            criteria.append(f"Priority: {' and '.join(priority_range)}")
        
        return "\n".join(f"- {criterion}" for criterion in criteria)
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Anthropic backend health check."""
        try:
            # Simple API test
            test_payload = {
                "model": self.config.model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            async with self.session.post(
                f"{self.config.base_url}/messages",
                json=test_payload
            ) as response:
                return {
                    "status": "healthy" if response.status == 200 else "error",
                    "api_status": response.status,
                    "model": self.config.model,
                    "rate_limits": {
                        "requests_per_minute": self.config.rate_limit_requests_per_minute,
                        "tokens_per_minute": self.config.rate_limit_tokens_per_minute
                    }
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }