from typing import Any
import time
import asyncio as aio
import logging
from agentopy import WithStateMixin, WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo

from frankenstein.lib.networking.communication import IMessaging

logger = logging.getLogger(__name__)

class Messenger(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a communication component that uses websockets to send and receive messages."""

    def __init__(self, messaging: IMessaging):
        super().__init__()
        self.action_space.register_actions([
            Action(
                "send_message", "use this action to send a text message to the user, use it as the main way to interact with the user (input param: 'message' of string type)", self.send_message),
            Action(
                "fetch_new_messages", "use this action to check messages user sent to you, prioritise acting on those", self.get_new_messages),
            Action("get_message_history", "use this action to get the history of messages sent and received", self.get_message_history)
        ])

        self._messaging: IMessaging = messaging
        self._num_new_messages: int = 0
        self._messages = []
                    
    async def process_command(self, message: dict) -> None:
        """Processes the command message."""
        command = message.get("command")
        uuid = message.get("uuid")
        if command == "get_message_history":
            await self._messaging.send_message(
                {
                    "type": "command",
                    "data": {
                        "messages": self._messages,
                        "response_for": uuid,
                        "command": "get_message_history"
                    }
                    
                }
            )        
            
    async def send_message(self, message: Any) -> ActionResult:
        """Sends a message to the user. This is a blocking operation."""
        if not await self._messaging.is_connected():
            return ActionResult(value="No connection", success=False)
        
        msg = {
            "type": "text",
            "data": {
                "from": "agent",
                "ts": time.time(),
                "data": message    
            }
        }
        
        await self._messaging.send_message(msg)
        self._messages.append(msg["data"])
        
        return ActionResult(value="OK", success=True)

    async def get_new_messages(self) -> ActionResult:
        """Gets unread messages. This is a blocking operation."""
        new_messages = self._messages[-self._num_new_messages:]
        self._num_new_messages = 0
        return ActionResult(value={f"Message {i}": msg.get("data") for i, msg in enumerate(new_messages)}, success=True)
        
    async def get_message_history(self) -> ActionResult:
        """Gets the message history. This is a blocking operation."""
        self._num_new_messages = 0
        return ActionResult(value=self._messages, success=True)

    async def tick(self) -> None:
        self._state.set_item(
            "status", {"num_new_messages": self._num_new_messages})
        message = await self._messaging.get(0.1)
        if message is None:
            return
        if message.get("type") == "text":
            self._messages.append(message.get("data"))
            self._num_new_messages += 1
        elif message.get("type") == "command":
            await self.process_command(message.get("data"))
        
    def info(self) -> EntityInfo:
        return EntityInfo(
            name="Messenger",
            version="0.1.0",
            params={
                "address": self._messaging.address()
            }
        )
