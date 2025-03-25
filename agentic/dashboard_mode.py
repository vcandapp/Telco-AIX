# dashboard_mode.py

import asyncio
import logging
import argparse
import sys
from datetime import datetime

from dashboard.server import DashboardServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("agentic")

async def run_dashboard(dashboard_host="0.0.0.0", dashboard_port=8080):
    """Run just the dashboard with simulated data."""
    
    # Start dashboard server without dependencies
    dashboard_server = DashboardServer(
        host=dashboard_host,
        port=dashboard_port,
        mcp_url="http://localhost:8000",  # Dummy URLs since we're not using them
        acp_url="http://localhost:8002",
        orchestration_service=None  # No orchestration
    )
    
    dashboard_task = asyncio.create_task(dashboard_server.start())
    logger.info(f"Dashboard server starting on {dashboard_host}:{dashboard_port}")
    
    # Wait for dashboard to initialize
    await asyncio.sleep(2)
    logger.info("Dashboard ready at http://localhost:8080")
    
    # Simulate demo data for the dashboard
    while True:
        # Simulate telemetry data
        telemetry = {
            "packet_loss": round(0.01 + (asyncio.get_event_loop().time() % 10) / 100, 3),
            "latency": 45 + (asyncio.get_event_loop().time() % 30),
            "throughput": 950 + (asyncio.get_event_loop().time() % 200),
            "jitter": 3 + (asyncio.get_event_loop().time() % 5),
            "signal_strength": -70 + (asyncio.get_event_loop().time() % 15),
            "timestamp": datetime.utcnow().isoformat(),
            "cell_id": "SITE_123",
            "region": "REGION_A"
        }
        
        # Send to dashboard
        await dashboard_server.broadcast_update("telemetry_update", telemetry)
        
        # Simulate events every 10 seconds
        if int(asyncio.get_event_loop().time()) % 10 == 0:
            agent_types = ["diagnostic", "planning", "execution", "validation"]
            agent_type = agent_types[int(asyncio.get_event_loop().time() / 10) % 4]
            
            message = {
                "agent_type": agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Simulated {agent_type} event on cell site SITE_123"
            }
            
            await dashboard_server.broadcast_update("event", message)
        
        await asyncio.sleep(1)

async def main():
    """Main entry point for dashboard-only mode."""
    
    parser = argparse.ArgumentParser(description="Telecom AI Framework Dashboard-Only Mode")
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="Dashboard server host")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard server port")
    args = parser.parse_args()
    
    logger.info("Starting Telecom AI Agent Framework in Dashboard-Only Mode")
    
    try:
        await run_dashboard(
            dashboard_host=args.dashboard_host,
            dashboard_port=args.dashboard_port
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        logger.info("Dashboard stopped")

if __name__ == "__main__":
    asyncio.run(main())