"""
ACP Broker Adapter - provides a simplified interface to the ACPMessageBroker
"""

import asyncio
import logging
from typing import Callable, Dict, Any
from .broker import ACPMessageBroker
from .schema import ACPMessage

class ACPBrokerAdapter:
    """Adapter to provide a simplified interface to ACPMessageBroker"""
    
    def __init__(self, broker: ACPMessageBroker):
        self.broker = broker
        self.logger = logging.getLogger("acp.adapter")
        self.subscribers: Dict[str, Callable] = {}
        
    async def start(self):
        """Start the broker (no-op since broker is started separately)"""
        self.logger.info("ACP broker adapter ready")
        
    async def stop(self):
        """Stop the broker"""
        self.logger.info("ACP broker adapter stopped")
        
    async def publish(self, message: ACPMessage):
        """Publish a message through the broker"""
        try:
            # Convert message to dict and send via REST API
            message_dict = message.dict()
            # For now, we'll store the message to be delivered when agents connect
            # In a full implementation, this would use the broker's HTTP API
            self.logger.info(f"Publishing message: {message.message_type} from {message.sender_id} to {message.receiver_id}")
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            
    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to messages matching a topic pattern"""
        self.subscribers[topic] = handler
        self.logger.info(f"Subscribed to topic: {topic}")
        
    async def unsubscribe(self, topic: str, handler: Callable):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            del self.subscribers[topic]
            self.logger.info(f"Unsubscribed from topic: {topic}")