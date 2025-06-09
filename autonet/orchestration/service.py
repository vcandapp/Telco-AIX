# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from protocols.acp.schema import (
    ACPMessage, AgentDescription, MessageType, ActionType, CapabilityInfo
)
from protocols.acp.adapter import ACPBrokerAdapter
from protocols.mcp.server import MCPServer

class OrchestrationService:
    """Service for orchestrating agent interactions.
    
    The orchestration service manages agent discovery, task decomposition, 
    workflow management, and resource allocation.
    """
    
    def __init__(self, acp_broker: ACPBrokerAdapter, mcp_server: MCPServer, service_id: Optional[str] = None):
        """Initialize a new orchestration service.
        
        Args:
            acp_broker: ACP broker for agent communication
            mcp_server: MCP server for context management
            service_id: Unique identifier for this service instance
        """
        self.service_id = service_id or f"orchestrator-{str(uuid.uuid4())[:8]}"
        self.logger = logging.getLogger(f"orchestration.{self.service_id}")
        self.acp_broker = acp_broker
        self.mcp_server = mcp_server
        
        # Registry of known agents
        self.agent_registry: Dict[str, AgentDescription] = {}
        
        # Active workflows
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        # Event subscribers
        self.event_subscribers: Dict[str, Set[str]] = {}
        
        # Running tasks
        self._tasks = []
        
    async def start(self) -> None:
        """Start the orchestration service."""
        self.logger.info(f"Starting orchestration service {self.service_id}")
        # Subscribe to orchestration-related messages
        await self.acp_broker.subscribe("orchestration.*", self._handle_orchestration_message)
        
    async def stop(self) -> None:
        """Stop the orchestration service."""
        await self.shutdown()
        
    async def _handle_orchestration_message(self, message: ACPMessage) -> None:
        """Handle orchestration-related messages."""
        self.logger.debug(f"Received orchestration message: {message.message_type}")
        
    async def shutdown(self) -> None:
        """Shut down the orchestration service."""
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to be cancelled
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        self.logger.info(f"Shut down orchestration service {self.service_id}")
        
    async def register_agent(self, agent: Any) -> None:
        """Register an agent with the orchestration service.
        
        Args:
            agent: Agent instance to register
        """
        # Determine agent capabilities based on type
        agent_type = agent.__class__.__name__.replace('Network', '').replace('Agent', '').lower()
        
        if agent_type == 'diagnostic':
            action_types = [ActionType.DIAGNOSE, ActionType.QUERY]
            domains = ['network', 'telemetry', 'anomaly_detection']
            description = 'Diagnoses network issues and detects anomalies'
        elif agent_type == 'planning':
            action_types = [ActionType.PLAN]
            domains = ['network', 'remediation', 'planning']
            description = 'Generates remediation plans for network issues'
        elif agent_type == 'execution':
            action_types = [ActionType.EXECUTE]
            domains = ['network', 'automation', 'playbooks']
            description = 'Executes remediation plans and automation tasks'
        elif agent_type == 'validation':
            action_types = [ActionType.VALIDATE]
            domains = ['network', 'verification', 'testing']
            description = 'Validates execution results and system state'
        else:
            action_types = []
            domains = ['general']
            description = 'General purpose agent'
            
        agent_description = AgentDescription(
            agent_id=agent.agent_id,
            agent_type=agent_type,
            name=agent.name,
            capabilities=CapabilityInfo(
                action_types=action_types,
                domains=domains,
                description=description
            ),
            network_location={
                "host": "localhost",
                "ports": {
                    "mcp": 8080,
                    "acp": 8081
                }
            },
            status="active",
            last_seen=datetime.utcnow()
        )
        self.agent_registry[agent.agent_id] = agent_description
        self.logger.info(f"Registered agent {agent.agent_id} ({agent_description.agent_type})")
        
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestration service.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
            
    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update the status of an agent.
        
        Args:
            agent_id: ID of the agent
            status: New status
        """
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id].status = status
            self.agent_registry[agent_id].last_seen = datetime.utcnow()
            self.logger.debug(f"Updated status of agent {agent_id} to {status}")
            
    async def find_agents_by_type(self, agent_type: str) -> List[AgentDescription]:
        """Find agents of a specific type.
        
        Args:
            agent_type: Type of agent to find
            
        Returns:
            List of matching agent descriptions
        """
        return [
            agent for agent in self.agent_registry.values()
            if agent.agent_type == agent_type
        ]
        
    async def find_agents_by_capability(self, capability: str) -> List[AgentDescription]:
        """Find agents with a specific capability.
        
        Args:
            capability: Capability to look for
            
        Returns:
            List of matching agent descriptions
        """
        return [
            agent for agent in self.agent_registry.values()
            if capability in agent.capabilities.action_types
        ]
        
    async def create_workflow(self, 
                            workflow_type: str, 
                            context: Dict[str, Any],
                            initiator_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow.
        
        Args:
            workflow_type: Type of workflow to create
            context: Workflow context
            initiator_id: ID of the agent initiating the workflow
            
        Returns:
            ID of the created workflow
        """
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "context": context,
            "initiator_id": initiator_id,
            "status": "created",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "steps": [],
            "results": {}
        }
        
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow {workflow_id} of type {workflow_type}")
        
        # Start workflow execution
        task = asyncio.create_task(self._execute_workflow(workflow_id))
        self._tasks.append(task)
        
        return workflow
        
    async def _execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
        """
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return
            
        workflow = self.workflows[workflow_id]
        workflow_type = workflow["workflow_type"]
        
        self.logger.info(f"Executing workflow {workflow_id} of type {workflow_type}")
        
        try:
            # Update workflow status
            workflow["status"] = "running"
            workflow["updated_at"] = datetime.utcnow()
            
            # Execute workflow based on type
            if workflow_type == "anomaly_resolution":
                await self._execute_anomaly_resolution_workflow(workflow)
            elif workflow_type == "anomaly_remediation":
                await self._execute_anomaly_remediation_workflow(workflow)
            elif workflow_type == "network_optimization":
                await self._execute_network_optimization_workflow(workflow)
            elif workflow_type == "customer_issue_resolution":
                await self._execute_customer_issue_workflow(workflow)
            else:
                self.logger.error(f"Unknown workflow type: {workflow_type}")
                workflow["status"] = "failed"
                workflow["error"] = f"Unknown workflow type: {workflow_type}"
                
            # Update workflow status if not already failed
            if workflow["status"] != "failed":
                workflow["status"] = "completed"
                workflow["completed_at"] = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            workflow["status"] = "failed"
            workflow["error"] = str(e)
        finally:
            workflow["updated_at"] = datetime.utcnow()
            
    async def _execute_anomaly_resolution_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute an anomaly resolution workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for resolving network anomalies
        # For this example, we'll define a simple workflow
        
        # Step 1: Find diagnostic agents
        diagnostic_agents = await self.find_agents_by_type("diagnostic")
        if not diagnostic_agents:
            raise RuntimeError("No diagnostic agents available")
            
        # Step 2: Get detailed diagnostics
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "diagnostics",
            "status": "completed",
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "agents": [agent.agent_id for agent in diagnostic_agents]
        })
        
        # Step 3: Find planning agents
        planning_agents = await self.find_agents_by_type("planning")
        if not planning_agents:
            raise RuntimeError("No planning agents available")
            
        # Step 4: Generate resolution plan
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "planning",
            "status": "completed",
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "agents": [agent.agent_id for agent in planning_agents]
        })
        
        # Step 5: Find execution agents
        execution_agents = await self.find_agents_by_type("execution")
        if not execution_agents:
            raise RuntimeError("No execution agents available")
            
        # Step 6: Execute resolution plan
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "execution",
            "status": "completed",
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "agents": [agent.agent_id for agent in execution_agents]
        })
        
        # Step 7: Find validation agents
        validation_agents = await self.find_agents_by_type("validation")
        if not validation_agents:
            raise RuntimeError("No validation agents available")
            
        # Step 8: Validate resolution
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "validation",
            "status": "completed",
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "agents": [agent.agent_id for agent in validation_agents]
        })
        
        # Record results
        workflow["results"] = {
            "resolution_status": "success",
            "resolution_time": (datetime.utcnow() - workflow["created_at"]).total_seconds(),
            "agents_involved": len(diagnostic_agents) + len(planning_agents) + len(execution_agents) + len(validation_agents)
        }
        
    async def _execute_network_optimization_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute a network optimization workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for network optimization
        # Similar structure to the anomaly resolution workflow
        workflow["results"] = {
            "optimization_status": "success",
            "optimization_gain": "15%",
            "completion_time": datetime.utcnow().isoformat()
        }
        
    async def _execute_customer_issue_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute a customer issue resolution workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for customer issue resolution
        # Similar structure to the anomaly resolution workflow
        workflow["results"] = {
            "resolution_status": "success",
            "customer_id": workflow["parameters"].get("customer_id"),
            "resolution_time": (datetime.utcnow() - workflow["created_at"]).total_seconds()
        }
        
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Workflow status information, or None if not found
        """
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow["workflow_id"],
            "workflow_type": workflow["workflow_type"],
            "status": workflow["status"],
            "created_at": workflow["created_at"].isoformat(),
            "updated_at": workflow["updated_at"].isoformat(),
            "steps_completed": len([step for step in workflow["steps"] if step["status"] == "completed"]),
            "steps_total": len(workflow["steps"]),
            "results": workflow.get("results", {})
        }
        
    async def subscribe_to_events(self, event_type: str, subscriber_id: str) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            subscriber_id: ID of the subscribing agent
        """
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = set()
            
        self.event_subscribers[event_type].add(subscriber_id)
        self.logger.info(f"Agent {subscriber_id} subscribed to {event_type} events")
        
    async def unsubscribe_from_events(self, event_type: str, subscriber_id: str) -> None:
        """Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of event to unsubscribe from
            subscriber_id: ID of the subscribing agent
        """
        if event_type in self.event_subscribers and subscriber_id in self.event_subscribers[event_type]:
            self.event_subscribers[event_type].remove(subscriber_id)
            self.logger.info(f"Agent {subscriber_id} unsubscribed from {event_type} events")
            
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an event to subscribers.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type not in self.event_subscribers:
            return
            
        subscribers = self.event_subscribers[event_type]
        self.logger.info(f"Publishing {event_type} event to {len(subscribers)} subscribers")
        
        # In a real implementation, this would send messages to the subscribers
        # For this example, we'll just log the event
        event_data["published_at"] = datetime.utcnow().isoformat()
        event_data["subscribers"] = list(subscribers)
        
        self.logger.info(f"Published {event_type} event: {event_data}")
        
    async def subscribe_workflow_events(self, workflow_id: str):
        """Subscribe to events for a specific workflow.
        
        Args:
            workflow_id: ID of the workflow to monitor
            
        Yields:
            Workflow events as they occur
        """
        # Create a queue for workflow events
        event_queue = asyncio.Queue()
        
        # Subscribe to workflow events
        async def workflow_event_handler(message: ACPMessage):
            if message.payload.get("workflow_id") == workflow_id:
                await event_queue.put(message.payload)
                
        await self.acp_broker.subscribe(f"workflow.{workflow_id}.*", workflow_event_handler)
        
        try:
            while True:
                event = await event_queue.get()
                yield event
        finally:
            # Unsubscribe when done
            await self.acp_broker.unsubscribe(f"workflow.{workflow_id}.*", workflow_event_handler)
            
    async def _execute_anomaly_remediation_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute an anomaly remediation workflow.
        
        Args:
            workflow: The workflow to execute
        """
        workflow_id = workflow["workflow_id"]
        context = workflow["context"]
        
        # Step 1: Diagnostic phase
        diagnostic_agents = await self.find_agents_by_type("diagnostic")
        if not diagnostic_agents:
            raise RuntimeError("No diagnostic agents available")
            
        # Send diagnostic request via ACP
        diagnostic_message = ACPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=self.service_id,
            receiver_id=diagnostic_agents[0].agent_id,
            timestamp=datetime.utcnow(),
            payload={
                "action": "analyze_anomaly",
                "workflow_id": workflow_id,
                "anomaly_context": context
            }
        )
        
        await self.acp_broker.publish(diagnostic_message)
        
        # Broadcast workflow event
        await self._broadcast_workflow_event(workflow_id, {
            "agent_id": diagnostic_agents[0].agent_id,
            "action": "diagnostic_started",
            "status": "in_progress",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Wait for diagnostic response (simplified for this example)
        await asyncio.sleep(2)  # In real implementation, wait for actual response
        
        # Step 2: Planning phase
        planning_agents = await self.find_agents_by_type("planning")
        if not planning_agents:
            raise RuntimeError("No planning agents available")
            
        # Continue with planning, execution, and validation phases...
        # (Similar pattern as diagnostic phase)
        
    async def _broadcast_workflow_event(self, workflow_id: str, event: Dict[str, Any]) -> None:
        """Broadcast a workflow event.
        
        Args:
            workflow_id: ID of the workflow
            event: Event data
        """
        event["workflow_id"] = workflow_id
        message = ACPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=self.service_id,
            receiver_id="*",  # Broadcast to all
            timestamp=datetime.utcnow(),
            payload=event
        )
        
        await self.acp_broker.publish(message)
    
    async def delegate_task(self, agent_id: str, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a task to a specific agent.
        
        Args:
            agent_id: ID of the agent to delegate to
            task_type: Type of task to delegate
            context: Task context and parameters
            
        Returns:
            Task execution result
        """
        try:
            self.logger.info(f"Delegating {task_type} task to agent {agent_id}")
            
            # Find the agent
            agent = None
            for registered_agent in self.agent_registry.values():
                if registered_agent.agent_id == agent_id:
                    agent = registered_agent
                    break
            
            if not agent:
                # Try to find by agent object reference
                for agent_obj in self.agent_registry.keys():
                    if hasattr(agent_obj, 'agent_id') and agent_obj.agent_id == agent_id:
                        agent = agent_obj
                        break
            
            if not agent:
                raise RuntimeError(f"Agent {agent_id} not found in registry")
            
            # Simulate task execution based on task type
            await asyncio.sleep(0.5)  # Simulate processing time
            
            if task_type == "anomaly_analysis":
                return {
                    "success": True,
                    "summary": f"Anomaly analysis completed for {context.get('component', 'unknown')} component",
                    "details": {
                        "anomaly_type": "performance_degradation",
                        "severity": context.get('severity', 'medium'),
                        "confidence": 0.85
                    },
                    "duration": 0.5,
                    "agent_id": agent_id
                }
            elif task_type == "remediation_planning":
                return {
                    "success": True,
                    "summary": f"Remediation plan created for {context.get('component', 'unknown')} component",
                    "details": {
                        "plan_type": "automated_remediation",
                        "estimated_time": "5 minutes",
                        "success_probability": 0.90
                    },
                    "duration": 0.5,
                    "agent_id": agent_id
                }
            elif task_type == "playbook_execution":
                return {
                    "success": True,
                    "summary": f"Playbook {context.get('playbook_name', 'unknown')} executed successfully",
                    "details": {
                        "playbook": context.get('playbook_name'),
                        "execution_mode": "autonomous",
                        "changes_applied": True
                    },
                    "duration": 2.0,
                    "agent_id": agent_id
                }
            elif task_type == "execution_validation":
                return {
                    "success": True,
                    "summary": f"Execution validation completed for {context.get('component', 'unknown')}",
                    "details": {
                        "validation_result": "passed",
                        "metrics_improved": True,
                        "system_stable": True
                    },
                    "duration": 0.5,
                    "agent_id": agent_id
                }
            else:
                return {
                    "success": True,
                    "summary": f"Generic task {task_type} completed",
                    "details": {},
                    "duration": 0.5,
                    "agent_id": agent_id
                }
                
        except Exception as e:
            self.logger.error(f"Error delegating task {task_type} to agent {agent_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"Task {task_type} failed",
                "agent_id": agent_id
            }
