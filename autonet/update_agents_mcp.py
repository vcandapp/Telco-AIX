#!/usr/bin/env python3
# Author: Fatih E. NAR
# Agentic AI Framework - Update Agents for New MCP Backend System
#
"""
Utility script to update all agent implementations to use the new configurable MCP backend system.
"""

import os
import re
from pathlib import Path

def update_agent_file(file_path: Path) -> bool:
    """Update a single agent file to use the new MCP system.
    
    Args:
        file_path: Path to the agent file
        
    Returns:
        True if file was updated, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update constructor to add mcp_config_path parameter
        constructor_pattern = r'(def __init__\(self[^)]*?)\):'
        
        def add_mcp_config_param(match):
            params = match.group(1)
            if 'mcp_config_path' not in params:
                # Add the parameter before the closing parenthesis
                return params + ',\n                 mcp_config_path: Optional[str] = None):'
            return match.group(0)
        
        content = re.sub(constructor_pattern, add_mcp_config_param, content, flags=re.DOTALL)
        
        # Update the initialization of mcp_config_path in constructor body
        if 'self.mcp_config_path = mcp_config_path' not in content:
            # Find where other URL assignments are made and add our assignment
            url_assignment_pattern = r'(self\.acp_broker_url = acp_broker_url\n)'
            replacement = r'\1        self.mcp_config_path = mcp_config_path\n'
            content = re.sub(url_assignment_pattern, replacement, content)
        
        # Update MCP client initialization
        old_mcp_init = r'self\.mcp_client = MCPClient\(\s*client_id=self\.agent_id,\s*base_url=self\.mcp_url\s*\)'
        new_mcp_init = '''from protocols.mcp.migration_helper import create_mcp_client
        self.mcp_client = create_mcp_client(
            client_id=self.agent_id,
            base_url=self.mcp_url,  # Legacy parameter for compatibility
            config_path=self.mcp_config_path
        )'''
        
        content = re.sub(old_mcp_init, new_mcp_init, content, flags=re.MULTILINE)
        
        # Only write if content actually changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {str(e)}")
        return False

def main():
    """Main function to update all agent files."""
    print("🔄 Updating agents to use configurable MCP backends...")
    
    # Find all agent files
    agent_files = []
    agents_dir = Path("agents")
    
    if agents_dir.exists():
        for agent_file in agents_dir.rglob("*.py"):
            if "agent.py" in agent_file.name:
                agent_files.append(agent_file)
    
    if not agent_files:
        print("❌ No agent files found")
        return
    
    print(f"📁 Found {len(agent_files)} agent files to update")
    
    updated_count = 0
    for agent_file in agent_files:
        if update_agent_file(agent_file):
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count} out of {len(agent_files)} agent files")
    print("\n📝 Manual steps required:")
    print("1. Update agent constructor calls in main.py to include mcp_config_path parameter")
    print("2. Test the updated agents with different MCP backends")
    print("3. Update agent documentation with new configuration options")
    
    # Show example usage
    print("\n💡 Example configuration usage:")
    print("Set environment variables for different backends:")
    print("export ANTHROPIC_API_KEY='your-key-here'")
    print("export OPENAI_API_KEY='your-key-here'") 
    print("export HUGGINGFACE_API_KEY='your-key-here'")
    print("\nThen edit MCP-Server-Config.cfg to change backend_type to:")
    print("- local (default)")
    print("- anthropic")
    print("- huggingface") 
    print("- openai")

if __name__ == "__main__":
    main()