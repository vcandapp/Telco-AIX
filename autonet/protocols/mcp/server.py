# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .schema import MCPContext
from .client import MCPClient

class MCPServer:
    """Server for the Model Context Protocol (MCP).
    
    This server provides endpoints for storing and retrieving context data
    according to the Model Context Protocol specification.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the MCP server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger("mcp.server")
        self.app = FastAPI(title="Model Context Protocol Server")
        self.context_store: Dict[str, MCPContext] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.server = None
        self.server_task = None
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up the API routes."""
        
        @self.app.post("/context", response_model=Dict[str, Any])
        async def store_context(context: Dict[str, Any]):
            """Store a new context."""
            try:
                # Validate and parse the context
                mcp_context = MCPContext.parse_obj(context)
                
                # Generate a unique ID if not provided
                context_id = str(uuid.uuid4())
                
                # Store the context
                self.context_store[context_id] = mcp_context
                
                # Notify subscribers
                domain_type = mcp_context.metadata.domain_type
                if domain_type in self.subscribers:
                    for callback in self.subscribers[domain_type]:
                        asyncio.create_task(callback(mcp_context))
                
                return {"status": "success", "context_id": context_id}
            except ValidationError as e:
                self.logger.error(f"Invalid context format: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid context format: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error storing context: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/context/{context_id}", response_model=Dict[str, Any])
        async def get_context(context_id: str):
            """Get a specific context by ID."""
            if context_id not in self.context_store:
                raise HTTPException(status_code=404, detail=f"Context with ID {context_id} not found")
            
            return self.context_store[context_id].dict()
        
        @self.app.post("/context/query", response_model=Dict[str, Any])
        async def query_context(query: Dict[str, Any]):
            """Query contexts based on criteria."""
            filtered_contexts = {}
            
            try:
                # Filter by domain type if specified
                domain_type = query.get("domain_type")
                if domain_type:
                    for context_id, context in self.context_store.items():
                        if context.metadata.domain_type == domain_type:
                            filtered_contexts[context_id] = context.dict()
                else:
                    # Return all contexts if no filter specified
                    for context_id, context in self.context_store.items():
                        filtered_contexts[context_id] = context.dict()
                
                return {"status": "success", "contexts": filtered_contexts}
            except Exception as e:
                self.logger.error(f"Error querying contexts: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def start(self):
        """Start the MCP server in background."""
        import uvicorn
        
        self.logger.info(f"Starting MCP server on {self.host}:{self.port}")
        config = uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())
        # Give server time to start
        await asyncio.sleep(0.5)
        
    async def stop(self):
        """Stop the MCP server."""
        if self.server:
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
        self.logger.info("MCP server stopped")
    
    def register_subscriber(self, domain_type: str, callback: Callable):
        """Register a subscriber for context updates.
        
        Args:
            domain_type: The domain type to subscribe to
            callback: A callback function that will be called when a new context of the specified type is stored
        """
        if domain_type not in self.subscribers:
            self.subscribers[domain_type] = []
        
        self.subscribers[domain_type].append(callback)
        self.logger.info(f"Registered subscriber for domain type {domain_type}")
    
    def unregister_subscriber(self, domain_type: str, callback: Callable):
        """Unregister a subscriber for context updates.
        
        Args:
            domain_type: The domain type to unsubscribe from
            callback: The callback function to unregister
        """
        if domain_type in self.subscribers and callback in self.subscribers[domain_type]:
            self.subscribers[domain_type].remove(callback)
            self.logger.info(f"Unregistered subscriber for domain type {domain_type}")
            
    def create_client(self, client_id: str = None) -> MCPClient:
        """Create a new MCP client connected to this server.
        
        Args:
            client_id: Optional client ID, will be generated if not provided
        
        Returns:
            MCPClient instance configured to connect to this server
        """
        import uuid
        if not client_id:
            client_id = f"client-{str(uuid.uuid4())[:8]}"
        return MCPClient(client_id=client_id, base_url=f"http://{self.host}:{self.port}")
