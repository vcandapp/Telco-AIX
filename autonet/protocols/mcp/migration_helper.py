# Author: Fatih E. NAR
# Agentic AI Framework - MCP Migration Helper
#
import logging
from typing import Optional, Dict, Any
from .client_v2 import MCPClientV2
from .client import MCPClient

class MCPMigrationHelper:
    """Helper class to migrate from old MCP client to new configurable backend client."""
    
    @staticmethod
    def create_enhanced_client(client_id: str, 
                             base_url: Optional[str] = None, 
                             config_path: Optional[str] = None) -> MCPClientV2:
        """Create an enhanced MCP client with backward compatibility.
        
        Args:
            client_id: Unique identifier for the client
            base_url: Legacy base URL (ignored in new implementation)
            config_path: Path to MCP configuration file
            
        Returns:
            MCPClientV2 instance
        """
        logger = logging.getLogger("mcp.migration")
        
        if base_url:
            logger.warning(f"base_url parameter ({base_url}) is deprecated. "
                         "Use MCP-Server-Config.cfg to configure backends.")
        
        return MCPClientV2(client_id=client_id, config_path=config_path)
    
    @staticmethod
    def migrate_agent_mcp_client(agent, config_path: Optional[str] = None) -> None:
        """Migrate an agent's MCP client to the new enhanced version.
        
        Args:
            agent: Agent instance with mcp_client attribute
            config_path: Path to MCP configuration file
        """
        logger = logging.getLogger("mcp.migration")
        
        if not hasattr(agent, 'mcp_client'):
            logger.warning(f"Agent {agent.agent_id} does not have an mcp_client attribute")
            return
        
        old_client = agent.mcp_client
        
        # Create new enhanced client
        new_client = MCPClientV2(client_id=agent.agent_id, config_path=config_path)
        
        # Replace the client
        agent.mcp_client = new_client
        
        logger.info(f"Migrated MCP client for agent {agent.agent_id}")
        
        # Close old client if it has a close method
        if hasattr(old_client, 'close'):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(old_client.close):
                    # Schedule the close for later
                    asyncio.create_task(old_client.close())
                else:
                    old_client.close()
            except Exception as e:
                logger.warning(f"Failed to close old MCP client: {str(e)}")
    
    @staticmethod
    def get_backward_compatible_client(client_id: str, 
                                     base_url: str, 
                                     config_path: Optional[str] = None) -> MCPClientV2:
        """Get a backward compatible MCP client.
        
        This method provides the same interface as the old MCPClient constructor
        but returns the new enhanced client.
        
        Args:
            client_id: Unique identifier for the client
            base_url: Legacy base URL (logged but ignored)
            config_path: Path to MCP configuration file
            
        Returns:
            MCPClientV2 instance configured for backward compatibility
        """
        logger = logging.getLogger("mcp.migration")
        logger.info(f"Creating backward compatible MCP client for {client_id}")
        logger.debug(f"Legacy base_url {base_url} replaced with configurable backends")
        
        return MCPClientV2(client_id=client_id, config_path=config_path)


# Backward compatibility alias
def create_mcp_client(client_id: str, base_url: str, config_path: Optional[str] = None) -> MCPClientV2:
    """Backward compatibility function for creating MCP clients.
    
    Args:
        client_id: Unique identifier for the client
        base_url: Legacy base URL (ignored)
        config_path: Path to MCP configuration file
        
    Returns:
        MCPClientV2 instance
    """
    return MCPMigrationHelper.get_backward_compatible_client(client_id, base_url, config_path)